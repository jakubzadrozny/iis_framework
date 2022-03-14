from pathlib import Path
import numpy as np
from PIL import Image

from data.iis_dataset import SegDataset


class InriaAerialDataset(SegDataset):
    def __init__(self, dataset_path, split="train"):
        dataset_path = Path(dataset_path)
        self._split_path = dataset_path / split
        self.split = split
        self._images_path = self._split_path / "images"
        self._masks_path = self._split_path / "gt"
        
        self.dataset_samples = [f.stem for f in self._images_path.iterdir()]

    def get_sample(self, index):
        image_id = self.dataset_samples[index]
        image_path = self._images_path / f"{image_id}.tif"
        mask_path = self._masks_path / f"{image_id}.tif"

        image = np.array(Image.open(str(image_path)))
        mask = np.array(Image.open(str(mask_path)).convert('1'))
        info = {}
        return image, mask[:, :, None], info
