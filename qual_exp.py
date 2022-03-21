from functools import partial
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import albumentations as A
from tqdm import tqdm

from data.datasets.coco_lvis import CocoLvisDataset
from data.datasets.inria_aerial import InriaAerialDataset
from data.region_selector import random_single, dummy
from data.iis_dataset import RegionDataset
from data.transformations import RandomCrop
from engine.mas import compute_importance_l2
from engine.cont_adapt import AdaptLoss, interact
from models.iis_models.ritm import HRNetISModel


norm_fn = lambda x: (x - x.min()) / (x.max() - x.min())

def visualize_error(ax, image, error_mask, alpha, pc_list, nc_list):
    image = norm_fn(np.array(image))
    mean_color = image.sum((0, 1)) / image.size
    min_rgb = np.argmin(mean_color)
    mask_color = np.ones((1, 1, 3)) * np.eye(3)[min_rgb][None, None, :]
    out = (
        image / image.max() * (1 - alpha) + error_mask[:, :, None] * mask_color * alpha
    )  # add inverted mean color mask
    ax.imshow(out)
    ax.axis("off")
    ax.grid()
    for pc in pc_list:
        ax.scatter(pc[1], pc[0], s=10, color="g")
    for nc in nc_list:
        ax.scatter(nc[1], nc[0], s=10, color="r")


if __name__ == "__main__":
    # train data
    seg_dataset = InriaAerialDataset(
        "/Users/kubaz/ENS-offline/satellites/project/data/AerialImageDataset", 
        split="train"
    )
    region_selector = dummy
    # seg_dataset = CocoLvisDataset(
    #     "/Users/kubaz/ENS-offline/satellites/project/data/CocoLvis",
    #     split="val"
    # )
    # region_selector = random_single
    augmentator = A.Compose([
        RandomCrop(out_size=(224,224)),
        A.Normalize(),
    ])

    iis_dataset = RegionDataset(seg_dataset, region_selector, augmentator)
    iis_loader = DataLoader(
        iis_dataset, batch_size=1, num_workers=1, shuffle=True
    )

    model = HRNetISModel.load_from_checkpoint(
        '/Users/kubaz/ENS-offline/satellites/project/iis_framework/checkpoints/coco_lvis_h18_baseline.pth',
    )
    model.eval()
    model.train()
    # iis_loader_batched = DataLoader(
    #     iis_dataset, batch_size=8, num_workers=8, shuffle=True
    # )
    # omega = compute_importance_l2(model, iis_loader_batched)
    # torch.save(omega, 'omega.pth')
    omega = torch.load('omega.pth')
    # omega_ones = [torch.ones_like(p) for p in model.parameters()]
    optim = Adam(model.parameters(), lr=1e-6)
    crit = AdaptLoss(model, omega)

    batch = next(iter(iis_loader))
    scores, preds, pcs, ncs = interact(crit, batch, interaction_steps=20, clicks_per_step=1, optim=optim, grad_steps=10)
    # breakpoint()
    fig, axs = plt.subplots(4, 5, sharex=True, sharey=True, figsize=(10, 10), tight_layout=True)
    img = batch['image'][0]
    gt_mask = batch['mask'][0]
    for i in range(4):
        for j in range(5):
            idx = i*5 + j
            pred = gt_mask.numpy() if idx == 0 else preds[idx][0, 0].numpy()
            visualize_error(axs[i, j], img, pred, 0.3, pcs[idx][0], ncs[idx][0])
    plt.savefig("test.png")
