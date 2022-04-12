from pathlib import Path
from PIL import Image
import numpy as np


DATASET_DIR = Path("/content/AerialImageDataset")
CUT_SIZE = 350
STEP = 256

img_dir = DATASET_DIR / "train" / "images"
mask_dir = DATASET_DIR / "train" / "gt"
out_img_dir = DATASET_DIR / "train_cut" / "images"
out_mask_dir = DATASET_DIR / "train_cut" / "gt"

out_img_dir.mkdir(parents=True, exist_ok=True)
out_mask_dir.mkdir(exist_ok=True)

for fpath in img_dir.glob("*.tif"):
    fname = fpath.stem
    image = Image.open(str(fpath))
    
    mask_path = mask_dir / f"{fname}.tif"
    mask = Image.open(str(mask_path)).convert('1')

    w, h = image.size
    cnt = 1
    for left in range(0, w-CUT_SIZE, STEP):
        for upper in range(0, h-CUT_SIZE, STEP):
            _image = image.crop((left, upper, left+CUT_SIZE, upper+CUT_SIZE))
            _mask = mask.crop((left, upper, left+CUT_SIZE, upper+CUT_SIZE))

            _mask_np = np.array(_mask)
            if np.sum(_mask_np) / _mask_np.size < 0.05:
                # print(_fname, "empty")
                continue

            _fname = f"{fname}_{cnt:04d}.tif"
            # print(_fname)
            cnt += 1
            _image.save(str(out_img_dir / _fname))
            _mask.save(str(out_mask_dir / _fname))