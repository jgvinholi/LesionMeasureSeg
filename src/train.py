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
from datasets import enable_caching
enable_caching()


class SegLossTrainer(Trainer):
    """
    Same API as `Trainer`, but uses a custom cross-entropy with ignore_index.
    """
    def __init__(self, *args, ignore_index=255, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_index = ignore_index

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")              # B × H × W
        outputs = model(**inputs)                  # forward pass
        logits  = outputs.logits                  # B × C × H × W

        if logits.shape[-2:] != labels.shape[-2:]:
            logits = torch.nn.functional.interpolate(logits, labels.shape[-2:], align_corners=False,mode="bilinear")

        if torch.all(labels == self.ignore_index):
            # a zero tensor on the correct device & dtype, gradients flow
            loss = logits.sum() * 0.0
            return (loss, outputs) if return_outputs else loss

        loss = torch.nn.functional.cross_entropy(
            logits,
            labels.long(),
            ignore_index=self.ignore_index,
            reduction="mean",
        )
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
        )

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
        new_image = Image.new("RGB", (target_w, target_h), color=255)
        new_image.paste(image, (0, 0))

        new_mask = Image.new(mask.mode, (target_w, target_h), color=255)
        new_mask.paste(mask, (0, 0))

        return new_image, new_mask

    def _transform(self, examples):
        imgs, msks = [], []
        for img_path, msk_path in zip(examples["pixel_values"], examples["labels"]):
            # -------- load -------------------------------------------------
            img = Image.open(img_path).convert("RGB")
            raw = Image.open(msk_path)          # PNG palette or L mode

            # -------- clean label ids -------------------------------------
            m = np.array(raw, dtype=np.uint8)
            m[m == 255] = 0                     # 255 inside image → bg

            valid_ids = set(self.label2id.values())
            m[~np.isin(m, list(valid_ids))] = 0
            clean = Image.fromarray(m, mode="L")

            # -------- pad AFTER remap -------------------------------------
            img, clean = self.pad_to_size(img, clean)  # pads mask with 255

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
            save_steps=2048,
            eval_steps=1024,
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

# Example usage:
if __name__ == "__main__":
    # Replace these with your actual paths and label mappings
    root = Path("../smart_dataset/Lesions/patches/")
    train_image_dir = root / "images/train"
    train_mask_dir = root / "labels/train"
    val_image_dir = root / "images/val"
    val_mask_dir = root / "labels/val"
    pretrained_model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
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
        num_train_epochs=100,
        trainval_split_seed=42,
    )
    trainer.prepare_data() 
    from collections import Counter
    bad = Counter()
    trainer.train()
