from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import albumentations as A

from data.datasets.coco_lvis import CocoLvisDataset
from data.datasets.inria_aerial import InriaAerialDataset
from data.region_selector import random_single, dummy
from data.iis_dataset import RegionDataset
from data.transformations import RandomCrop
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
        RandomCrop(out_size=(300,300)),
        A.Normalize(),
    ])

    iis_dataset = RegionDataset(seg_dataset, region_selector, augmentator)
    iis_loader = DataLoader(
        iis_dataset, batch_size=1, num_workers=1, shuffle=True
    )

    batch = next(iter(iis_loader))

    model = HRNetISModel.load_from_checkpoint(
        '/Users/kubaz/ENS-offline/satellites/project/iis_framework/checkpoints/coco_lvis_h18_baseline.pth',
    )
    model.eval()
    model.train()
    optim = Adam(model.parameters(), lr=1e-6)

    omega = torch.load('omega.pth')
    omega_ones = [torch.ones_like(p) for p in omega]
    crit = AdaptLoss(model, omega_ones, gamma=3e4)

    scores, preds, pcs, ncs = interact(crit, batch, interaction_steps=30, clicks_per_step=1, optim=optim, grad_steps=5)
    fig, axs = plt.subplots(5, 6, sharex=True, sharey=True, figsize=(12, 10), tight_layout=True)
    img = batch['image'][0]
    gt_mask = batch['mask'][0]
    for i in range(5):
        for j in range(6):
            idx = i*6 + j
            pred = gt_mask.numpy() if idx == 0 else preds[idx][0, 0].numpy()
            visualize_error(axs[i, j], img, pred, 0.3, pcs[idx][0], ncs[idx][0])
            axs[i, j].set_title("IoU={}".format(round(scores[idx], 4)))
    axs[0, 0].set_title("ground truth")
    plt.savefig("test_adapt.png")

    model = HRNetISModel.load_from_checkpoint(
        '/Users/kubaz/ENS-offline/satellites/project/iis_framework/checkpoints/coco_lvis_h18_baseline.pth',
    )
    model.eval()
    crit = AdaptLoss(model, omega)

    scores, preds, pcs, ncs = interact(crit, batch, interaction_steps=30, clicks_per_step=1, grad_steps=0)
    fig, axs = plt.subplots(5, 6, sharex=True, sharey=True, figsize=(12, 10), tight_layout=True)
    img = batch['image'][0]
    gt_mask = batch['mask'][0]
    for i in range(5):
        for j in range(6):
            idx = i*6 + j
            pred = gt_mask.numpy() if idx == 0 else preds[idx][0, 0].numpy()
            visualize_error(axs[i, j], img, pred, 0.3, pcs[idx][0], ncs[idx][0])
            axs[i, j].set_title("IoU={}".format(round(scores[idx], 4)))
    axs[0, 0].set_title("ground truth")
    plt.savefig("test_frozen.png")
