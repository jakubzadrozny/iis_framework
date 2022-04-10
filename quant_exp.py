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


def pipeline(batch, model_path, weights, grad_steps, lbd=1.0, gamma=1.0, lr=1e-6, device='cpu', interaction_steps=20):
    for k in batch:
        batch[k] = batch[k].to(device)
    model = HRNetISModel.load_from_checkpoint(
        model_path,
    )
    model.to(device)
    model.eval() # set BatchNorm to eval
    if grad_steps > 0:
        model.train()
        optim = Adam(model.parameters(), lr=lr)
    else:
        optim = None
    crit = AdaptLoss(model, weights, lbd=lbd, gamma=gamma)
    scores, _, _, _ = interact(crit, batch, interaction_steps=interaction_steps, clicks_per_step=1, optim=optim, grad_steps=grad_steps)
    fill_scores(scores, interaction_steps+1)
    return scores


def main(loader, to_test, num_batches=10):
    results = {k: [] for k in to_test}
    batch_idx = 0
    while batch_idx < num_batches:
        for batch in tqdm(loader, total=num_batches):
            for name, f in to_test.items():
                scores = f(batch)
                results[name].append(scores)
            batch_idx += 1
            if batch_idx == num_batches:
                break

    iou_targets = [0.7, 0.75, 0.8]
    clicks_at_iou = [{} for _ in range(len(iou_targets))]
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

    return results, results_mean, results_std, iou_targets, clicks_at_iou


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

    model_path = '/Users/kubaz/ENS-offline/satellites/project/iis_framework/checkpoints/coco_lvis_h18_baseline.pth'
    seq_model_path = '/Users/kubaz/ENS-offline/satellites/project/iis_framework/checkpoints/seq_adapt.pth'

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
    omega_ones = [torch.ones_like(p) for p in omega]
    
    _pipeline = partial(pipeline, interaction_steps=30)
    to_test = {
        'ia': partial(_pipeline, weights=omega_ones, grad_steps=5, gamma=3e4, model_path=model_path),
        'sa': partial(_pipeline, weights=omega_ones, grad_steps=5, gamma=3e4, model_path=seq_model_path),
        'frozen': partial(_pipeline, weights=omega_ones, grad_steps=0, model_path=model_path),
    }
    
    main(iis_loader, to_test, num_batches=2)
