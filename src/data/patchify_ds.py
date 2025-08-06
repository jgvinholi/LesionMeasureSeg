# patchify_utils.py
"""Utility functions for patchifying images (and their corresponding label masks) with
optional overlap and on‑disk dataset preparation.

This module covers four common scenarios:

1. **Pure numpy patch extraction** – `patchify_array` takes a single ndarray and
yields (or returns) overlapping crops.

2. **Image/label pair patchification** – `patchify_image_label` applies the same
   crop coordinates to both arrays so that the *n*-th image patch aligns with
   the *n*-th label patch.

3. **Folder‑level batch patchification** – `patchify_folder` walks over two
   parallel directory trees (images / labels) and writes the resulting patches
   to new folder(s).

4. **Dataset preparation with train/val split** – `prepare_dataset_from_dict`
   reproduces the behaviour of the original `patchify_and_save` function but
   additionally performs a random split **before** patchification, placing the
   resulting patches – and also the *full* images/masks – into `train/` and
   `val/` sub‑folders.

All high‑level functions ultimately rely on `patchify_array`, so the patching
logic lives in a single place.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

__all__ = [
    "PatchCoords",
    "patchify_array",
    "patchify_image_label",
    "patchify_folder",
    "prepare_dataset_from_dict",
]


@dataclass(frozen=True, slots=True)
class PatchCoords:
    """Simple (y, x, h, w) container so we know where a patch came from."""

    y: int
    x: int
    h: int
    w: int

    def as_slice(self) -> Tuple[slice, slice]:
        return slice(self.y, self.y + self.h), slice(self.x, self.x + self.w)


# -----------------------------------------------------------------------------
# 1. Generic numpy patchification
# -----------------------------------------------------------------------------

def patchify_array(
    arr: np.ndarray,
    patch_size: int,
    overlap: int = 0,
    *,
    return_coords: bool = False,
):
    """Return **overlapping** crops from *arr*.

    Parameters
    ----------
    arr
        The array to crop. Expected shape: (H, W, ...) or (H, W).
    patch_size
        The length of the square patch, i.e. `crop.shape[0] = crop.shape[1] = patch_size`.
    overlap
        Number of pixels by which adjacent patches should overlap. Must satisfy
        `0 <= overlap < patch_size`.
    return_coords
        If *True*, return a `(patch, PatchCoords)` tuple for each crop.

    Returns
    -------
    list
        List of patches (or of `(patch, coords)` tuples).
    """
    if not (0 <= overlap < patch_size):
        raise ValueError("'overlap' must be in the interval [0, patch_size).")

    stride = patch_size - overlap
    H, W = arr.shape[:2]

    patches: List[np.ndarray] = []
    coords: List[PatchCoords] = []

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end = min(y + patch_size, H)
            x_end = min(x + patch_size, W)
            patch = arr[y:y_end, x:x_end].copy()
            patches.append(patch)
            coords.append(PatchCoords(y, x, y_end - y, x_end - x))

    if return_coords:
        return list(zip(patches, coords))
    return patches


# -----------------------------------------------------------------------------
# 2. Joint image/label patchification
# -----------------------------------------------------------------------------

def patchify_image_label(
    img: np.ndarray,
    lbl: np.ndarray,
    patch_size: int,
    overlap: int = 0,
    *,
    return_coords: bool = False,
):
    """Apply identical cropping to *img* and *lbl* and return patch pairs."""
    if img.shape[:2] != lbl.shape[:2]:
        raise ValueError("Image and label must have identical spatial dimensions.")

    img_patches, coords = zip(*patchify_array(img, patch_size, overlap, return_coords=True))
    lbl_patches = [lbl[c.as_slice()].copy() for c in coords]

    if return_coords:
        return list(zip(img_patches, lbl_patches, coords))
    return list(zip(img_patches, lbl_patches))


# -----------------------------------------------------------------------------
# 3. Folder‑level patchification
# -----------------------------------------------------------------------------

def _save_patch(patch: np.ndarray, dest: Path, mode: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(patch)
    if mode:
        img = img.convert(mode)
    img.save(dest)


def patchify_folder(
    images_dir: Path | str,
    labels_dir: Path | str,
    patch_size: int,
    overlap: int = 0,
    *,
    output_root: Path | str = "patches",
    extensions: Sequence[str] = (".png", ".jpg", ".jpeg"),
    progress: bool = True,
) -> None:
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_root = Path(output_root)

    out_img_dir = output_root / "images"
    out_lbl_dir = output_root / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    label_lookup = {p.stem: p for ext in extensions for p in labels_dir.glob(f"*{ext}")}
    img_files = [p for ext in extensions for p in images_dir.glob(f"*{ext}")]
    iterator = tqdm(img_files, desc="Patchifying", unit="img") if progress else img_files

    for img_path in iterator:
        stem = img_path.stem
        lbl_path = label_lookup.get(stem)
        if lbl_path is None:
            print(f"⚠️  No matching label for '{img_path.name}'. Skipping.")
            continue

        img = np.asarray(Image.open(img_path))
        lbl = np.asarray(Image.open(lbl_path))

        for idx, (ip, lp) in enumerate(patchify_image_label(img, lbl, patch_size, overlap)):
            _save_patch(ip, out_img_dir / f"{stem}_patch{idx}.png")
            _save_patch(lp, out_lbl_dir / f"{stem}_patch{idx}.png", mode="L")


# -----------------------------------------------------------------------------
# 4. Train/val split *before* patchification (dict‑style)
# -----------------------------------------------------------------------------
def downsample(
    arr: np.ndarray,   # image or label
    factor: int = 8,
    *,
    is_label: bool = False,
) -> np.ndarray:
    """
    Downsample an array.  Lanczos for images, nearest for label maps.
    """
    if factor <= 1:
        return arr

    h, w = arr.shape[:2]
    new_size = (w // factor, h // factor)

    pil_kwargs = (
        dict(resample=Image.Resampling.NEAREST)
        if is_label else
        dict(resample=Image.Resampling.LANCZOS, reducing_gap=3.0)
    )

    mode = "L" if is_label else None        # RGB inferred if mode=None
    arr_pil = Image.fromarray(arr, mode=mode)
    return np.asarray(arr_pil.resize(new_size, **pil_kwargs))


def prepare_dataset_from_dict(
    simplified: dict[str, dict[str, np.ndarray]],
    patch_size: int,
    overlap: int = 0,
    train_ratio: float = 0.8,
    downsampling_factor: int = 8,
    seed: int | None = 42,
    output_root: Path | str = "dataset",
) -> None:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("'train_ratio' must be between 0 and 1 (exclusive).")

    rng = random.Random(seed)
    filenames = list(simplified.keys())
    rng.shuffle(filenames)

    split_idx = int(len(filenames) * train_ratio)
    train_files = set(filenames[:split_idx])

    output_root = Path(output_root)
    subdirs = {
        "full_train": output_root / "full" / "train",
        "full_val": output_root / "full" / "val",
        "patch_img_train": output_root / "images" / "train",
        "patch_img_val": output_root / "images" / "val",
        "patch_lbl_train": output_root / "labels" / "train",
        "patch_lbl_val": output_root / "labels" / "val",
    }
    for d in subdirs.values():
        d.mkdir(parents=True, exist_ok=True)

    for fname in tqdm(filenames, desc="Preparing dataset", unit="img"):
        item = simplified[fname]
        img = downsample(item["image"], factor=downsampling_factor)
        lbl = downsample(item["annotation"], factor=downsampling_factor, is_label=True)

        target_full_dir = subdirs["full_train"] if fname in train_files else subdirs["full_val"]
        _save_patch(img, target_full_dir / f"{fname}.png")
        _save_patch(lbl, target_full_dir / f"{fname}_label.png", mode="L")

        patch_dest_img = subdirs["patch_img_train"] if fname in train_files else subdirs["patch_img_val"]
        patch_dest_lbl = subdirs["patch_lbl_train"] if fname in train_files else subdirs["patch_lbl_val"]

        for idx, (ip, lp) in enumerate(patchify_image_label(img, lbl, patch_size, overlap, return_coords=False)):
            _save_patch(ip, patch_dest_img / f"{fname}_patch{idx}.png")
            _save_patch(lp, patch_dest_lbl / f"{fname}_patch{idx}.png", mode="L")


# -----------------------------------------------------------------------------
# Example CLI usage (optional)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, pickle

    parser = argparse.ArgumentParser(description="Patchify utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sub‑command: folder
    p_folder = subparsers.add_parser("folder", help="Patchify two directories of images/labels")
    p_folder.add_argument("images_dir", type=Path)
    p_folder.add_argument("labels_dir", type=Path)
    p_folder.add_argument("--patch_size", type=int, default=512)
    p_folder.add_argument("--overlap", type=int, default=0)
    p_folder.add_argument("--output_root", type=Path, default=Path("patches"))
    p_folder.add_argument("--no_progress", action="store_true")

    # Sub‑command: pkl (expects a pickle file holding the 'simplified' dict)
    p_pkl = subparsers.add_parser("pkl", help="Prepare dataset from a pickle-serialized 'simplified' mapping")
    p_pkl.add_argument("simplified_pkl", type=Path)
    p_pkl.add_argument("--patch_size", type=int, default=512)
    p_pkl.add_argument("--overlap", type=int, default=384)
    p_pkl.add_argument("--train_ratio", type=float, default=0.8)
    p_pkl.add_argument("--output_root", type=Path, default=Path("dataset"))

    args = parser.parse_args()

    if args.command == "folder":
        patchify_folder(
            args.images_dir,
            args.labels_dir,
            patch_size=args.patch_size,
            overlap=args.overlap,
            output_root=args.output_root,
            progress=not args.no_progress,
        )
    elif args.command == "pkl":
        with open(args.simplified_pkl, "rb") as fh:
            data = pickle.load(fh)
        # The pickle is expected to already store numpy arrays, but ensure dtype
        for v in data.values():
            v["image"] = np.asarray(v["image"], dtype=np.uint8)
            v["annotation"] = np.asarray(v["annotation"], dtype=np.uint8)
        prepare_dataset_from_dict(
            data,
            patch_size=args.patch_size,
            overlap=args.overlap,
            train_ratio=args.train_ratio,
            output_root=args.output_root,
        )
