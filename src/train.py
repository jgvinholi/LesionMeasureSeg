import os
from pathlib import Path
import re
from datasets import Dataset
from PIL import Image
import numpy as np
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
)
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.data.data_collator import default_data_collator
from collections import Counter
from datasets import enable_caching
enable_caching()
import torch.nn.functional as F

class SegLossTrainer(Trainer):
    def __init__(
        self, *args,
        ignore_index: int = 255,
        smooth: float = 1.0,
        ce_weight: float = 0.3,      # ← α
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ignore_index = ignore_index
        self.smooth       = smooth
        self.ce_weight    = ce_weight      # 0.3 CE  +  0.7 Dice

    # ------------------------------------------------------------------ #
    def _one_hot(self, y: torch.Tensor, C: int) -> torch.Tensor:
        return F.one_hot(y, C).permute(0, 3, 1, 2)     # (B,C,H,W)

    # ------------------------------------------------------------------ #
    def compute_loss(
        self, model, inputs,
        return_outputs: bool = False,
        num_items_in_batch = None,
        **_
    ):
        labels  = inputs.pop("labels")                 # (B,H,W)
        outputs = model(**inputs)
        logits  = outputs.logits                       # (B,C,H,W)

        # ---------- resize gt to logits ------------------------------- #
        if logits.shape[-2:] != labels.shape[-2:]:
            labels = F.interpolate(
                labels.unsqueeze(1).float(),
                size=logits.shape[-2:], mode="nearest"
            ).squeeze(1).long()

        valid = labels.ne(self.ignore_index)           # mask (B,H,W)
        if not valid.any():
            loss = logits.sum() * 0.0
            return (loss, outputs) if return_outputs else loss

        labels = labels.masked_fill(~valid, 0)
        probs  = logits.softmax(1)                     # (B,C,H,W)
        target = self._one_hot(labels, probs.shape[1]).float()

        valid   = valid.unsqueeze(1)                   # (B,1,H,W)
        probs  = probs* valid
        target = target* valid

        # ---------- Dice ------------------------------------------------ #
        dims = (0, 2, 3)
        intersect = (probs * target).sum(dims)
        cardinal  = probs.sum(dims) + target.sum(dims)

        # simple inverse-frequency weights (clipped)
        with torch.no_grad():
            freq     = target.sum(dims)
            weights  = 1.0 / (freq + 1e-6)
            weights[freq == 0] = 0.0
            w_sum = weights.sum()
            if w_sum > 0:
                weights /= w_sum

        dice_per_class = (2 * intersect + self.smooth) / (cardinal + self.smooth)
        dice_loss = 1.0 - dice_per_class.mean()

        # ---------- Cross-entropy -------------------------------------- #
        ce_loss = F.cross_entropy(
            logits, labels,
            ignore_index=self.ignore_index, reduction="mean", weight=weights
        )

        # ---------- Blend ---------------------------------------------- #
        loss = self.ce_weight * ce_loss + (1.0 - self.ce_weight) * dice_loss
        return (loss, outputs) if return_outputs else loss



class SegFormerTrainer:
    def __init__(
        self,
        train_image_dir: str | Path,
        train_mask_dir: str | Path,
        pretrained_model_name: str,
        output_dir: str,
        label2id: dict,
        val_image_dir: str | Path | None = None,
        val_mask_dir: str | Path | None = None,
        train_batch_size: int = 4,
        eval_batch_size: int = 4,
        num_train_epochs: int = 5,
        target_size: tuple[int,int] = (512, 512),
        trainval_split_seed: int = 42,
    ):
        self.train_image_dir = train_image_dir
        self.train_mask_dir = train_mask_dir
        self.pretrained_model_name = pretrained_model_name
        self.output_dir = output_dir
        self.label2id = label2id
        self.val_image_dir = val_image_dir
        self.val_mask_dir = val_mask_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_train_epochs = num_train_epochs
        self.target_size = target_size
        self.trainval_split_seed = trainval_split_seed
        self.id2label = {int(v): k for k, v in label2id.items()}
        # Load processor & model
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name, use_fast=True)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            pretrained_model_name,
            ignore_mismatched_sizes=True,
            id2label=self.id2label,
            label2id=self.label2id,
        ).to("cuda")

    def _pair_key(self, filename: str) -> str:
        match = re.search(r"(.+_patch\d+)", filename)
        return match.group(1) if match else None

    def load_dataset(self, image_dir: str | Path, mask_dir: str | Path):
        """
        Scan image_dir and mask_dir, pair files by key, and split into train/test.
        """
        imgs = {
            self._pair_key(f): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if self._pair_key(f)
        }
        masks = {
            self._pair_key(f): os.path.join(mask_dir, f)
            for f in os.listdir(mask_dir)
            if self._pair_key(f)
        }
        examples = [
            {"pixel_values": imgs[k], "labels": masks[k]}
            for k in imgs.keys() if k in masks
        ]
        ds = Dataset.from_list(examples)
        return ds

    def pad_to_size(self, image: Image.Image, mask: Image.Image):
        """
        Pad both image and mask to self.target_size (width, height).
        """
        target_w, target_h = self.target_size
        w, h = image.size
        if (w, h) == (target_w, target_h):
            return image, mask

        # Create new canvases
        new_image = Image.new("RGB", (target_w, target_h), color=0)
        new_image.paste(image, (0, 0))

        new_mask = Image.new(mask.mode, (target_w, target_h), color=0)
        new_mask.paste(mask, (0, 0))

        return new_image, new_mask

    def _transform(self, examples):
        imgs, msks = [], []
        for img_path, msk_path in zip(examples["pixel_values"], examples["labels"]):
            img = Image.open(img_path).convert("RGB")
            raw = Image.open(msk_path)

            m = np.array(raw, dtype=np.uint8)
            m[m == 255] = 0

            valid_ids = set(self.label2id.values())
            m[~np.isin(m, list(valid_ids))] = 0
            clean = Image.fromarray(m, mode="L")

            img, clean = self.pad_to_size(img, clean)

            imgs.append(img)
            msks.append(clean)

        proc = self.processor(images=imgs,
                            segmentation_maps=msks,
                            return_tensors="pt")

        return {"pixel_values": proc.pixel_values,
                "labels":       proc.labels}
    

    def prepare_data(self):
        # 1) load or split raw (paths-only) dataset as before…
        self.dataset = {}
        self.dataset["train"] = self.load_dataset(self.train_image_dir, self.train_mask_dir)
        self.dataset["test"] = None
        if self.val_image_dir is not None and self.val_mask_dir is not None:
            self.dataset["test"] = self.load_dataset(self.val_image_dir, self.val_mask_dir)

        # 2) attach lazy processing
        for split in ["train", "test"]:
            ds = self.dataset[split].with_transform(self._transform)
            # 3) strip away the old columns *and* cast to torch
            self.dataset[split] = ds

        return self.dataset
    
    @torch.inference_mode()
    def predict_image(
        self,
        image: str | Path | Image.Image,
        *,
        patch_size: int | None = None,
        overlap: int = 0,
        palette: dict[int, tuple[int, int, int]] | None = None,
        device: torch.device | str | None = None,
    ):
        """
        Patch-wise inference for an *arbitrary-sized* RGB image.

        Parameters
        ----------
        image
            Path or PIL.Image.
        patch_size
            Square crop size fed to the model.  Defaults to `self.target_size[0]`.
        overlap
            Pixels of overlap between adjacent crops (helps to reduce
            border artefacts). 0 = no overlap, `patch_size//2` is common.
        palette
            Optional mapping {label_id: (R,G,B)}.  If *None* a simple palette
            is generated (stable colours derived from label id).
        device
            Where to run the model.  `None` → use `self.model.device`.

        Returns
        -------
        pred_ids : np.ndarray
            2-D array (HxW) of label ids.
        pred_img : PIL.Image
            Colour-mapped version of *pred_ids* (mode ``"RGB"``).
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError("`image` must be a path or PIL.Image")

        patch_size = patch_size or self.target_size[0]
        stride = patch_size - overlap
        if stride <= 0:
            raise ValueError("`overlap` must be smaller than `patch_size`")

        # ── prepare accumulator ────────────────────────────────────────────
        img_np = np.asarray(image)
        H, W = img_np.shape[:2]
        C = len(self.label2id)                       # #classes
        logit_acc = torch.zeros((C, H, W), dtype=torch.float32)
        count_acc = torch.zeros((H, W), dtype=torch.float32)

        device = device or self.model.device
        self.model.eval()

        # ── sliding-window inference ───────────────────────────────────────
        for top in range(0, H, stride):
            for left in range(0, W, stride):
                bottom = min(top + patch_size, H)
                right  = min(left + patch_size, W)

                crop = image.crop((left, top, right, bottom))
                # pad to square if necessary (bottom/right borders)
                if crop.size != (patch_size, patch_size):
                    pad_img = Image.new("RGB", (patch_size, patch_size), color=0)
                    pad_img.paste(crop, (0, 0))
                    crop = pad_img

                inp = self.processor(images=crop, return_tensors="pt").to(device)
                logits = self.model(**inp).logits.squeeze(0)      # C x h x w
                logits = torch.nn.functional.interpolate(
                    logits.unsqueeze(0),
                    size=(bottom - top, right - left),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)                                       # C x h' x w'

                logit_acc[:, top:bottom, left:right] += logits.cpu()
                count_acc[top:bottom, left:right] += 1

        # ── normalise & argmax ─────────────────────────────────────────────
        logit_acc /= count_acc.unsqueeze(0).clamp(min=1.0)           # broadcast (… / N)
        pred_ids = logit_acc.argmax(0).numpy().astype(np.uint8)   # HxW

        # ── colour map ────────────────────────────────────────────────────
        if palette is None:
            # deterministic “nice” colours
            rng = np.random.default_rng(0)
            palette = {i: tuple(int(x) for x in rng.integers(0, 255, 3))
                    for i in range(C)}
            palette[0] = (0, 0, 0)                    # background = black

        colour = np.zeros((H, W, 3), dtype=np.uint8)
        for lbl, rgb in palette.items():
            colour[pred_ids == lbl] = rgb

        return pred_ids, Image.fromarray(colour, mode="RGB")

    def train(self):
        """
        Fine-tune the segmentation model.
        """
        args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            eval_strategy="steps",
            num_train_epochs=self.num_train_epochs,
            save_total_limit=2,
            bf16=True,
            bf16_full_eval=True,
            eval_on_start=True,
            dataloader_num_workers=4,
            learning_rate=1e-4,
            weight_decay=1e-2,
            remove_unused_columns=False,
            logging_steps=10,
            torch_compile=True,
            max_grad_norm=1.0,
            load_best_model_at_end=True,
            save_steps=4096,
            eval_steps=256,
        )
        trainer = SegLossTrainer(
            model=self.model,
            args=args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            data_collator=default_data_collator,
            processing_class=self.processor,
        )
        trainer.train()


def run_train():
    root = Path("../smart_dataset/Lesions/patches/")
    train_image_dir = root / "images/train"
    train_mask_dir = root / "labels/train"
    val_image_dir = root / "images/val"
    val_mask_dir = root / "labels/val"
    pretrained_model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    # pretrained_model_name = "/home/vinholi/Documenti/Dev/vithoria/smart/out/checkpoint-18900"
    output_dir = "./out"
    label2id = {'bg': 0, 'DE_1.5_1.5': 1, 'DE_20_30': 2, 'DE_2_2': 3, 'DE_3_1.5': 4, 'DE_3_4': 5, 'DE_50_70': 6, 'lesão': 7}

    trainer = SegFormerTrainer(
        train_image_dir=train_image_dir,
        train_mask_dir=train_mask_dir,
        val_image_dir=val_image_dir,
        val_mask_dir=val_mask_dir,
        pretrained_model_name=pretrained_model_name,
        output_dir=output_dir,
        label2id=label2id,
        train_batch_size=28,
        eval_batch_size=28,
        num_train_epochs=5000,
        trainval_split_seed=42,
    )
    trainer.prepare_data() 
    trainer.train()

def run_predict():
    root = Path("../smart_dataset/Lesions/patches/")
    train_image_dir = root / "images/train"
    train_mask_dir = root / "labels/train"
    val_image_dir = root / "images/val"
    val_mask_dir = root / "labels/val"
    pretrained_model_name = "./out/checkpoint-512"
    output_dir = "./out"
    label2id = {'bg': 0, 'DE_1.5_1.5': 1, 'DE_20_30': 2, 'DE_2_2': 3, 'DE_3_1.5': 4, 'DE_3_4': 5, 'DE_50_70': 6, 'lesão': 7}

    trainer = SegFormerTrainer(
        train_image_dir=train_image_dir,
        train_mask_dir=train_mask_dir,
        val_image_dir=val_image_dir,
        val_mask_dir=val_mask_dir,
        pretrained_model_name=pretrained_model_name,
        output_dir=output_dir,
        label2id=label2id,
        train_batch_size=28,
        eval_batch_size=28,
        num_train_epochs=500,
        trainval_split_seed=42,
    )
    ids, colour = trainer.predict_image(
        "/home/vinholi/Documenti/Dev/vithoria/smart_dataset/Lesions/patches/full/train/5b9976f6-M.M.M_1_3.jpg.png",
        patch_size=512,
        device="cuda",
        overlap=384,
    )
    colour.save("image_001_pred.png")

if __name__ == "__main__":
    run_train()
    # run_predict()