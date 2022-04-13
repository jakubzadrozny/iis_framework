from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import albumentations as A

from data.datasets.inria_aerial import InriaAerialDataset
from data.region_selector import dummy
from data.transformations import RandomCrop
from data.iis_dataset import RegionDataset
from engine.cont_adapt import AdaptLoss, interact
from models.iis_models.ritm import HRNetISModel


norm_fn = lambda x: (x - x.min()) / (x.max() - x.min())

def visualize(ax, image, gt_mask, pred, alpha, pc_list, nc_list):
    image = norm_fn(np.array(image))
    fn = np.maximum(0, gt_mask - pred)
    fp = np.maximum(0, pred - gt_mask)
    tp = gt_mask * pred
    fn_color = np.array([0, 0, 1])[None, None, :]
    fp_color = np.array([1, 0, 0])[None, None, :]
    tp_color = np.array([0, 1, 0])[None, None, :]
    out = (
        image / image.max() * (1 - alpha) 
        + fn[:, :, None] * fn_color * alpha 
        + fp[:, :, None] * fp_color * alpha
        + tp[:, :, None] * tp_color * alpha
    )
    ax.imshow(out)
    ax.axis("off")

    for idx, clicks in enumerate(pc_list):
        for pc in clicks[0]:
            if idx >= len(pc_list) - 1:
                ax.scatter(pc[1], pc[0], s=26, color="#00ccff", marker='x')
            else:
                ax.scatter(pc[1], pc[0], s=10, color="#00ccff", marker='x')
    
    for idx, clicks in enumerate(nc_list):
        for nc in clicks[0]:
            if idx >= len(nc_list) - 1:
                ax.scatter(nc[1], nc[0], s=26, color="r", marker='x')
            else:
                ax.scatter(nc[1], nc[0], s=10, color="r", marker='x')


def plot_result(img, gt_mask, scores, preds, pcs, ncs, path):
    fig, axs = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(12, 8), tight_layout=True)
    
    for i in range(3):
        for j in range(4):
            plot_idx = i*4 + j
            step_idx = 2*plot_idx + 1
            pred = preds[step_idx][0, 0].numpy()
            visualize(axs[i, j], img, gt_mask.numpy(), pred, 0.5, pcs[:step_idx+1], ncs[:step_idx+1])
            axs[i, j].set_title("IoU={}".format(round(scores[step_idx], 4)))
    plt.savefig(path, dpi=300)


if __name__ == "__main__":
    # train data
    seg_dataset = InriaAerialDataset(
        "/content/AerialImageDataset", 
        split="train"
    )
    region_selector = dummy
    augmentator = A.Compose([
        RandomCrop((350, 350)),
        A.Normalize(),
    ])

    iis_dataset = RegionDataset(seg_dataset, region_selector, augmentator)
    iis_loader = DataLoader(
        iis_dataset, batch_size=1, num_workers=1, shuffle=True
    )

    batch = next(iter(iis_loader))
    img = batch['image'][0]
    gt_mask = batch['mask'][0]

    model = HRNetISModel.load_from_checkpoint(
        '/content/coco_lvis_h18_baseline.pth',
    )
    model.eval()
    model.train()
    optim = Adam(model.parameters(), lr=1e-5)

    omega = torch.load('/content/omega_bce.pth', map_location='cpu')
    crit = AdaptLoss(model, omega, gamma=1e7)

    scores, preds, pcs, ncs = interact(crit, batch, interaction_steps=25, clicks_per_step=1, optim=optim, grad_steps=4)
    plot_result(img, gt_mask, scores, preds, pcs, ncs, "test_adapt.png")

    model = HRNetISModel.load_from_checkpoint(
        '/content/coco_lvis_h18_baseline.pth',
    )
    model.eval()
    crit = AdaptLoss(model, omega)
    scores, preds, pcs, ncs = interact(crit, batch, interaction_steps=25, clicks_per_step=1, grad_steps=0)
    plot_result(img, gt_mask, scores, preds, pcs, ncs, "test_frozen.png")
