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


def seq_adapt(num_epochs, train_loader, crit, optim, save_checkpoint, interaction_steps, 
              target_iou=0.75, report_clicks=6, device='cpu', log_every=10):
    train_scores_log = []
    clicks_needed = []
    # test_scores_log = []
    for epoch_idx in tqdm(range(1, num_epochs+1)):
        # train_epoch_iou = 0.

        for batch in tqdm(train_loader, leave=False):
            batch = to_device(batch, device)
            image = batch["image"]
            image = image.permute(0, 3, 1, 2) if image.shape[-1] == 3 else image
            bs, _, h, w = image.shape

            scores, _, pcs, ncs = interact(crit, batch, interaction_steps, grad_steps=0, target_iou=target_iou)
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
            train_scores_log.append(score)
            clicks_needed.append(len(scores) - 1)
            if len(train_scores_log) % log_every == 1:
                episode_iou = np.mean(np.array(train_scores_log[-log_every:]))
                episode_clicks = np.mean(np.array(clicks_needed[-log_every:]))
                print("After {} images iou @ {} clicks = {}, clicks @ {} iou = {}".format(
                    len(train_scores_log),
                    report_clicks,
                    round(episode_iou, 4),
                    round(target_iou, 2),
                    round(episode_clicks, 2),
                ))
                save_checkpoint()

        # test_epoch_log = []
        # for _ in range(4):
        #     for batch in tqdm(test_loader, leave=False):
        #         batch = to_device(batch, device)
        #         scores, _, pcs, ncs = interact(crit, batch, interaction_steps, grad_steps=0)
        #         test_epoch_log.append(scores)
        # test_epoch_mean = np.mean(np.stack(test_epoch_log, axis=0), axis=0)
        # test_scores_log.append(test_epoch_mean)

        # print()
        # print("Epoch {}: train iou={}, test iou={}".format(
        #     epoch_idx,
        #     round(train_epoch_iou / len(train_loader), 4),
        #     round(test_epoch_mean[-1], 4),
        # ))

    return train_scores_log, clicks_needed


if __name__ == "__main__":
    # train data
    seg_dataset = InriaAerialDataset(
        "/Users/kubaz/ENS-offline/satellites/project/data/AerialImageDataset", 
        split="train"
    )
    # seg_dataset = Subset(seg_dataset, torch.arange(100))
    # pi = torch.randperm(len(seg_dataset))
    # N_train = int(len(seg_dataset) * 0.8)
    # train_seg_dataset = Subset(seg_dataset, pi[:N_train])
    train_seg_dataset = seg_dataset
    # test_seg_dataset = Subset(seg_dataset, pi[N_train:])
    region_selector = dummy
    augmentator = A.Compose([
        RandomCrop(out_size=(350,350)),
        A.Normalize(),
    ])

    train_dataset = RegionDataset(train_seg_dataset, region_selector, augmentator)
    # test_dataset = RegionDataset(test_seg_dataset, region_selector, augmentator)
    train_loader = DataLoader(
        train_dataset, batch_size=1, num_workers=1, shuffle=True
    )
    # test_loader = DataLoader(
    #     test_dataset, batch_size=1, num_workers=1,
    # )

    model_path = '/Users/kubaz/ENS-offline/satellites/project/iis_framework/checkpoints/coco_lvis_h18_baseline.pth'
    new_model_path = '/Users/kubaz/ENS-offline/satellites/project/iis_framework/checkpoints/seq_adapt.pth'
    save_checkpoint = lambda: model.save_checkpoint(model_path, new_model_path)
    
    model = HRNetISModel.load_from_checkpoint(
        model_path,
    )
    model.eval()
    model.train()
    optim = Adam(model.parameters(), lr=1e-6)
    
    omega = torch.load('omega.pth')
    omega_ones = [torch.ones_like(w) for w in omega]
    crit = AdaptLoss(model, omega_ones, gamma=1e5)

    train_scores_log, clicks_needed = seq_adapt(
        5, 
        train_loader, 
        crit, 
        optim, 
        save_checkpoint,
        interaction_steps=25,
        log_every=10,
    )
    
    plt.plot(train_scores_log)
    plt.ylim(0., 1.)
    plt.ylabel("IoU @ 6 clicks")
    plt.xlabel("# of images")
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig("seq_adapt1.png")
    plt.plot()

    plt.plot(clicks_needed)
    plt.ylabel("Clicks to 0.75 IoU")
    plt.xlabel("# of images")
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig("seq_adapt2.png")
    plt.plot()

    
