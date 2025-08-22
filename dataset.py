import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import random


class DeBlurDataset(data.Dataset):
    def __init__(self, root, cfg, split="train"):
        self.root = root
        self.cfg = cfg
        self.split = split

        # Setup paths
        if split == "train":
            self.input_dir = os.path.join(root, cfg["train_dir"], cfg["input_dir"])
            self.target_dir = os.path.join(root, cfg["train_dir"], cfg["target_dir"])
        else:
            self.input_dir = os.path.join(root, cfg["test_dir"], cfg["input_dir"])
            self.target_dir = os.path.join(root, cfg["test_dir"], cfg["target_dir"])

        # Get image files
        self.image_files = self._get_image_files()

        # Setup transforms
        self.transform = self._get_transforms()

        print(f"Dataset {split}: {len(self.image_files)} images loaded")
        print(f"Input dir: {self.input_dir}")
        print(f"Target dir: {self.target_dir}")

    def _get_image_files(self):
        """Get list of image files"""
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        image_files = []

        if os.path.exists(self.input_dir) and os.path.exists(self.target_dir):
            input_files = sorted(
                [
                    f
                    for f in os.listdir(self.input_dir)
                    if f.lower().endswith(valid_extensions)
                ]
            )

            for file in input_files:
                target_file = os.path.join(self.target_dir, file)
                if os.path.exists(target_file):
                    image_files.append(file)
        else:
            print(
                f"Warning: Directory not found - Input: {self.input_dir}, Target: {self.target_dir}"
            )

        return image_files

    def _get_transforms(self):
        """Get image transforms with proper augmentation"""
        image_size = self.cfg.get("image_size", 256)
        crop_size = self.cfg.get("crop_size", image_size)

        if self.split == "train":
            transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomCrop(crop_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.1),
                    transforms.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

        return transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """Get a single sample"""
        filename = self.image_files[idx]

        # Load images
        input_path = os.path.join(self.input_dir, filename)
        target_path = os.path.join(self.target_dir, filename)

        try:
            input_image = Image.open(input_path).convert("RGB")
            target_image = Image.open(target_path).convert("RGB")
        except Exception as e:
            print(f"Error loading images for {filename}: {e}")
            # Return a dummy sample if image loading fails
            dummy_tensor = torch.zeros(3, 256, 256)
            return {
                "input": dummy_tensor,
                "target": dummy_tensor,
                "filename": filename,
                "idx": idx,
            }

        # Apply transforms
        input_tensor = self.transform(input_image)
        target_tensor = self.transform(target_image)

        return {
            "input": input_tensor,
            "target": target_tensor,
            "filename": filename,
            "idx": idx,
        }

    def data_collator(self, batch):
        """Custom collate function for batching - compatible with training pipeline"""
        # Filter out any None or invalid samples
        valid_batch = [item for item in batch if item is not None and "input" in item]

        if not valid_batch:
            print("Warning: Empty batch after filtering")
            return None

        try:
            inputs = torch.stack([item["input"] for item in valid_batch])
            targets = torch.stack([item["target"] for item in valid_batch])
            filenames = [item["filename"] for item in valid_batch]
            indices = [item["idx"] for item in valid_batch]

            return {
                "inputs": inputs,  # Changed from "input" to "inputs" to match pipeline
                "targets": targets,  # Changed from "target" to "targets" to match pipeline
                "filenames": filenames,
                "indices": indices,
            }
        except Exception as e:
            print(f"Error in data collator: {e}")
            print(f"Batch items: {[type(item) for item in batch]}")
            return None


def get_training_set(root, cfg):
    """Get training dataset"""
    return DeBlurDataset(root, cfg, split="train")


def get_test_set(root, cfg):
    """Get test dataset"""
    return DeBlurDataset(root, cfg, split="test")


# Additional utility functions for data loading
def create_dataloader(
    dataset, batch_size, num_workers=4, shuffle=True, pin_memory=True
):
    """Create dataloader with proper error handling"""
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.data_collator,
        pin_memory=pin_memory,
        drop_last=shuffle,  # Drop last batch during training
        persistent_workers=num_workers > 0,
    )


def validate_dataset(dataset):
    """Validate dataset integrity"""
    print(f"Validating dataset with {len(dataset)} samples...")

    valid_samples = 0
    for i in range(min(10, len(dataset))):  # Check first 10 samples
        try:
            sample = dataset[i]
            if sample is not None and "input" in sample and "target" in sample:
                valid_samples += 1
        except Exception as e:
            print(f"Error loading sample {i}: {e}")

    print(f"Dataset validation: {valid_samples}/10 samples loaded successfully")
    return valid_samples == 10
