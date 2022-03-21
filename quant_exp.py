from functools import partial
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
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


def fill_scores(scores, iters):
    while len(scores) < iters:
        scores.append(scores[-1])
    return scores


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
    # iis_loader_batched = DataLoader(
    #     iis_dataset, batch_size=8, num_workers=8, shuffle=True
    # )
    # omega = compute_importance_l2(model, iis_loader_batched)
    # torch.save(omega, 'omega.pth')
    omega = torch.load('omega.pth')
    omega_ones = [torch.ones_like(p) for p in model.parameters()]

    def pipeline(batch, weights, grad_steps, lbd=1.0, gamma=1.0):
        model = HRNetISModel.load_from_checkpoint(
           '/Users/kubaz/ENS-offline/satellites/project/iis_framework/checkpoints/coco_lvis_h18_baseline.pth',
        )
        model.eval() # set BatchNorm to eval
        if grad_steps > 0:
            model.train()
            optim = Adam(model.parameters(), lr=1e-6)
        else:
            optim = None
        crit = AdaptLoss(model, weights, lbd=lbd, gamma=gamma)
        scores, _, _, _ = interact(crit, batch, interaction_steps=20, clicks_per_step=1, optim=optim, grad_steps=grad_steps)
        fill_scores(scores, 20)
        return scores


    to_test = {
        'adaptive_mas_1.0': partial(pipeline, weights=omega, grad_steps=10, gamma=1.0),
        'adaptive_mas_0.1': partial(pipeline, weights=omega, grad_steps=10, gamma=0.1),
        'adaptive_mas_10': partial(pipeline, weights=omega, grad_steps=10, gamma=10.0),
        'adaptive_unif_1.0': partial(pipeline, weights=omega_ones, grad_steps=10, gamma=1.0),
        'adaptive_unif_0.1': partial(pipeline, weights=omega_ones, grad_steps=10, gamma=0.1),
        'adaptive_unif_10': partial(pipeline, weights=omega_ones, grad_steps=10, gamma=10.0),
        'frozen': partial(pipeline, weights=omega_ones, grad_steps=0),
    }
    results = {k: [] for k in to_test}
    num_batches = 25
    for batch_idx, batch in tqdm(enumerate(iis_loader), total=num_batches):
        for name, f in tqdm(to_test.items(), leave=False):
            scores = f(batch)
            results[name].append(scores)
        if batch_idx == num_batches-1:
            break

    results_mean = {name: np.mean(np.array(val), axis=0) for name, val in results.items()}
    results_std = {name: np.std(np.array(val), axis=0) for name, val in results.items()}
    print(results_mean)
    print()
    print(results_std)

    for name, scores in results.items():
        scores = np.array(scores)
        mean = np.mean(scores, axis=0)
        std = np.std(scores, axis=0)
        xs = np.arange(1, mean.shape[0]+1)
        plt.plot(xs, mean, label=name)
        plt.fill_between(xs, mean-std, mean+std, label=name, alpha=0.2)
    
    plt.legend()
    plt.ylim(0., 1.)
    plt.ylabel("mean IoU")
    plt.xlabel("# of clicks")
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig("comparison_inria.png")
