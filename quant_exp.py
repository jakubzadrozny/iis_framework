from functools import partial
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import albumentations as A
from tqdm.auto import tqdm

from data.datasets.coco_lvis import CocoLvisDataset
from data.datasets.inria_aerial import InriaAerialDataset
from data.region_selector import random_single, dummy
from data.iis_dataset import RegionDataset
from data.transformations import RandomCrop
from engine.mas import compute_importance_l2
from engine.cont_adapt import AdaptLoss, interact
from models.iis_models.ritm import HRNetISModel


def fill_scores(scores, iters):
    while len(scores) < iters:
        scores.append(scores[-1])
    return scores


def pipeline(batch, model_path, weights, grad_steps, lbd=1.0, gamma=1.0, device='cpu', interaction_steps=20):
    for k in batch:
        batch[k] = batch[k].to(device)
    model = HRNetISModel.load_from_checkpoint(
        model_path,
    )
    model.to(device)
    model.eval() # set BatchNorm to eval
    if grad_steps > 0:
        model.train()
        optim = Adam(model.parameters(), lr=1e-6)
    else:
        optim = None
    crit = AdaptLoss(model, weights, lbd=lbd, gamma=gamma)
    scores, _, _, _ = interact(crit, batch, interaction_steps=interaction_steps, clicks_per_step=1, optim=optim, grad_steps=grad_steps)
    fill_scores(scores, interaction_steps+1)
    return scores


def main(loader, to_test, num_batches=10):
    results = {k: [] for k in to_test}
    for batch_idx, batch in tqdm(enumerate(loader), total=num_batches):
        for name, f in tqdm(to_test.items(), leave=False):
            scores = f(batch)
            results[name].append(scores)
        if batch_idx == num_batches-1:
            break

    iou_targets = [0.7, 0.75, 0.8]
    clicks_at_iou = [{}] * len(iou_targets)
    results_mean = {}
    results_std = {}

    plt.figure(figsize=(10, 6))
    for name, scores in results.items():
        scores = np.array(scores)
        results_mean[name] = np.mean(scores, axis=0)
        results_std[name] = np.std(scores, axis=0)
        xs = np.arange(0, scores.shape[1])
        plt.plot(xs, results_mean[name], label=name)
        # plt.fill_between(xs, mean-std, mean+std, label=name, alpha=0.2)
        scores = np.concatenate((scores, np.ones((scores.shape[0], 1))), axis=1)
        for i, t in enumerate(iou_targets):
            clicks = np.argmax(scores > t, axis=1)
            clicks_at_iou[i][name] = np.mean(clicks)

    print(results_mean)
    print()
    print(results_std)
    print()
    print(iou_targets)
    print(clicks_at_iou)
    
    plt.legend()
    plt.ylim(0.2, 0.9)
    plt.ylabel("mean IoU")
    plt.xlabel("# of clicks")
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig("comparison_inria.png", dpi=300)

    return results_mean, results_std, iou_targets, clicks_at_iou


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
        RandomCrop(out_size=(256,256)),
        A.Normalize(),
    ])

    model_path = '/Users/kubaz/ENS-offline/satellites/project/iis_framework/checkpoints/coco_lvis_h18_baseline.pth'

    iis_dataset = RegionDataset(seg_dataset, region_selector, augmentator)
    iis_loader = DataLoader(
        iis_dataset, batch_size=1, num_workers=1, shuffle=True
    )

    # iis_loader_batched = DataLoader(
    #     iis_dataset, batch_size=8, num_workers=8, shuffle=True
    # )
    # omega = compute_importance_l2(model, iis_loader_batched)
    # torch.save(omega, 'omega.pth')
    omega = torch.load('omega.pth')
    model = HRNetISModel.load_from_checkpoint(
        model_path,
    )
    omega_ones = [torch.ones_like(p) for p in model.parameters()]
    
    _pipeline = partial(pipeline, model_path=model_path)
    to_test = {
        'adapt_mas_1e4': partial(_pipeline, weights=omega, grad_steps=3, gamma=1e4),
        'adapt_mas_1e5': partial(_pipeline, weights=omega, grad_steps=3, gamma=1e5),
        'adapt_mas_1e6': partial(_pipeline, weights=omega, grad_steps=3, gamma=1e6),
        'adapt_mas_1e7': partial(_pipeline, weights=omega, grad_steps=3, gamma=1e7),
        'adapt_mas_1e8': partial(_pipeline, weights=omega, grad_steps=3, gamma=1e8),
        'adapt_mas_1e9': partial(_pipeline, weights=omega, grad_steps=3, gamma=1e9),
        'frozen': partial(_pipeline, weights=omega_ones, grad_steps=0),
    }
    
    main(iis_loader, to_test, num_batches=20)
