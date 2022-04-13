import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm.auto import tqdm
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
import albumentations as A

from data.clicking import disk_mask_from_coords_batch
from data.datasets.inria_aerial import InriaAerialDataset
from data.region_selector import random_single, dummy
from data.iis_dataset import RegionDataset
from data.transformations import RandomCrop
from engine.mas import compute_importance_l2
from engine.cont_adapt import AdaptLoss, interact
from models.iis_models.ritm import HRNetISModel


def merge_clicks(pcs, ncs):
    all_clicks = []
    for _pcs, _ncs in zip(pcs, ncs):
        clicks = [(c, 1) for c in _pcs]
        clicks += [(c, 0) for c in _ncs]
        all_clicks.append(clicks)
    return all_clicks


def split_clicks(all_clicks):
    pcs, ncs = [], []
    for clicks in all_clicks:
        _pcs, _ncs = [], []
        for c, _type in clicks:
            if _type == 0:
                _ncs.append(c)
            else:
                _pcs.append(c)
        pcs.append(_pcs)
        ncs.append(_ncs)
    return pcs, ncs


def sample_clicks(all_clicks, p=0.5):
    sampled = []
    for clicks in all_clicks:
        N = len(clicks)
        n = int(N * p)
        pi = np.random.permutation(N)
        _sampled = [clicks[i] for i in pi[:n]]
        sampled.append(_sampled)
    return sampled


def to_device(batch, device):
    for k in batch:
        batch[k] = batch[k].to(device)
    return batch


def seq_adapt(num_epochs, train_loader, test_loader, crit, baseline, optim, save_checkpoint, interaction_steps, 
              target_iou=0.75, report_clicks=6, device='cpu', log_every=10):
    test_epoch_log = []
    test_clicks_needed = []
    for batch in tqdm(test_loader, leave=False):
        batch = to_device(batch, device)
        scores = interact(baseline, batch, interaction_steps, grad_steps=0, target_iou=target_iou)[0]
        score = scores[report_clicks] if len(scores) > report_clicks else scores[-1]
        test_epoch_log.append(scores)
        test_clicks_needed.append(len(scores) - 1)
    print("Baseline test: iou={}, clicks={}".format(
        round(np.mean(np.array(test_epoch_log)), 4),
        round(np.mean(np.array(test_clicks_needed)), 2),
    ))

    train_scores_log = []
    baseline_scores_log = []
    clicks_needed = []
    baseline_clicks_needed = []
    test_scores_log = []
    test_clicks_log = []
    for epoch_idx in range(1, num_epochs+1):
        for batch_idx, batch in tqdm(enumerate(train_loader)):
            batch = to_device(batch, device)
            image = batch["image"]
            image = image.permute(0, 3, 1, 2) if image.shape[-1] == 3 else image
            bs, _, h, w = image.shape
            scores, _, pcs, ncs = interact(crit, batch, interaction_steps, grad_steps=0, target_iou=target_iou)
            baseline_scores = interact(baseline, batch, interaction_steps, grad_steps=0, target_iou=target_iou)[0]
            all_pcs = [sum([iter_clicks[i] for iter_clicks in pcs], []) for i in range(bs)]
            all_ncs = [sum([iter_clicks[i] for iter_clicks in ncs], []) for i in range(bs)]

            all_clicks = merge_clicks(all_pcs, all_ncs)
            sampled = sample_clicks(all_clicks, p=0.7)
            pcs, ncs = split_clicks(sampled)

            pc_mask = disk_mask_from_coords_batch(
                all_pcs, 
                torch.zeros(bs, 1, h, w, device=device),
            )[:, None, :, :]
            nc_mask = disk_mask_from_coords_batch(
                all_ncs, 
                torch.zeros(bs, 1, h, w, device=device)
            )[:, None, :, :]
            pc_mask_subsampled = disk_mask_from_coords_batch(
                pcs, 
                torch.zeros(bs, 1, h, w, device=device),
            )[:, None, :, :]
            nc_mask_sumbsampled = disk_mask_from_coords_batch(
                ncs, 
                torch.zeros(bs, 1, h, w, device=device)
            )[:, None, :, :]

            aux = torch.cat((pc_mask_subsampled, nc_mask_sumbsampled), dim=1)
            _, loss = crit(image, aux, pc_mask, nc_mask)
            optim.zero_grad()
            loss.backward()
            optim.step()

            score = scores[report_clicks] if len(scores) > report_clicks else scores[-1]
            baseline_score = baseline_scores[report_clicks] if len(baseline_scores) > report_clicks else baseline_scores[-1]
            train_scores_log.append(score)
            baseline_scores_log.append(baseline_score)
            clicks_needed.append(len(scores) - 1)
            baseline_clicks_needed.append(len(baseline_scores) - 1)
            if len(train_scores_log) % log_every == 1:
                episode_iou = np.mean(np.array(train_scores_log[-log_every:]))
                episode_clicks = np.mean(np.array(clicks_needed[-log_every:]))
                baseline_episode_iou = np.mean(np.array(baseline_scores_log[-log_every:]))
                baseline_episode_clicks = np.mean(np.array(baseline_clicks_needed[-log_every:]))
                msg = "After {iter} images iou={iou} (baseline iou={baseline_iou}), clicks={clicks} (baseline clicks={baseline_clicks})"
                print(msg.format(
                    iter=len(train_scores_log),
                    iou=round(episode_iou, 4),
                    baseline_iou=round(baseline_episode_iou, 4),
                    clicks=round(episode_clicks, 2),
                    baseline_clicks=round(baseline_episode_clicks, 2),
                ))
                save_checkpoint()

            if len(train_scores_log) % (5*log_every) == 1:
                test_epoch_log = []
                test_clicks_needed = []
                for batch in tqdm(test_loader, leave=False):
                    batch = to_device(batch, device)
                    scores, _, pcs, ncs = interact(crit, batch, interaction_steps, grad_steps=0, target_iou=target_iou)
                    score = scores[report_clicks] if len(scores) > report_clicks else scores[-1]
                    test_epoch_log.append(scores)
                    test_clicks_needed.append(len(scores) - 1)
                test_scores_log.append(np.mean(np.array(test_epoch_log)))
                test_clicks_log.append(np.mean(np.array(test_clicks_needed)))
                print()
                print("Test: iou={}, clicks={}".format(
                    round(test_scores_log[-1], 4),
                    round(test_clicks_log[-1], 2),
                ))
                print()

    return train_scores_log, clicks_needed, baseline_scores_log, baseline_clicks_needed


if __name__ == "__main__":
    # train data
    seg_dataset = InriaAerialDataset(
        "/Users/kubaz/ENS-offline/satellites/project/data/AerialImageDataset", 
        split="train"
    )
    torch.manual_seed(0)
    pi = torch.randperm(len(seg_dataset))
    N_test = 100
    train_seg_dataset = Subset(seg_dataset, pi[:-N_test])
    # train_seg_dataset = seg_dataset
    test_seg_dataset = Subset(seg_dataset, pi[-N_test:])
    region_selector = dummy
    augmentator = A.Compose([
        # RandomCrop(out_size=(350,350)),
        A.Normalize(),
    ])

    train_dataset = RegionDataset(train_seg_dataset, region_selector, augmentator)
    test_dataset = RegionDataset(test_seg_dataset, region_selector, augmentator)
    train_loader = DataLoader(
        train_dataset, batch_size=4, num_workers=2, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=4, num_workers=2,
    )

    model_path = '/Users/kubaz/ENS-offline/satellites/project/iis_framework/checkpoints/coco_lvis_h18_baseline.pth'
    new_model_path = '/Users/kubaz/ENS-offline/satellites/project/iis_framework/checkpoints/seq_adapt.pth'
    model = HRNetISModel.load_from_checkpoint(
        model_path,
    )
    model.eval()
    model.train()
    optim = Adam(model.parameters(), lr=1e-6) # increase
    
    omega = torch.load('omega_bce.pth')
    # omega_ones = [torch.ones_like(w) for w in omega]
    crit = AdaptLoss(model, omega, lbd=0.5, gamma=3e7)
    save_checkpoint = lambda: model.save_checkpoint(model_path, new_model_path)

    baseline_model = HRNetISModel.load_from_checkpoint(
        model_path,
    )
    baseline_model.eval()
    baseline = AdaptLoss(baseline_model, omega)

    res = seq_adapt(
        num_epochs=1, 
        train_loader=train_loader,
        test_loader=test_loader,
        crit=crit,
        baseline=baseline,
        optim=optim, 
        save_checkpoint=save_checkpoint,
        interaction_steps=15,
        report_clicks=8,
        log_every=20,
        target_iou=0.75,
    )
