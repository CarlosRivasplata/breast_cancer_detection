from torch.utils.data import Dataset
import torch
from pathlib import Path
import pyarrow.parquet as pq
from utils.dicom import load_dicom_image
from utils.constants import IMAGES_ABS_PATH, LABEL_MAP


class CBISDDSMDataset(Dataset):
    def __init__(self, parquet_path, transform=None, label_map=None, images_base_path=None):
        self.table = pq.read_table(parquet_path)
        self.num_rows = self.table.num_rows
        self.transform = transform
        self.label_map = label_map or LABEL_MAP
        self.images_base_path = Path(images_base_path or IMAGES_ABS_PATH)

    def __len__(self):
        return self.num_rows

    def __getitem__(self, idx):
        row = self.table.slice(idx, 1).to_pydict()

        try:
            img1 = load_dicom_image(self.images_base_path / row["image_path"][0].lstrip('/'))
            img2 = load_dicom_image(self.images_base_path / row["cropped_path"][0].lstrip('/'))
            img3 = load_dicom_image(self.images_base_path / row["roi_path"][0].lstrip('/'))
        except Exception as e:
            raise RuntimeError(f"Image load failed at index {idx}: {e}")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        # Stack as channels: shape will be (3, H, W)
        image = torch.cat([img1, img2, img3], dim=0)

        label = self.label_map[row["pathology"][0]]
        return image, torch.tensor(label, dtype=torch.long)