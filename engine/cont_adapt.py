from copy import deepcopy
import numpy as np
import torch
from torch import nn

from data.clicking import get_error_clicks_batch, disk_mask_from_coords_batch
from engine.metrics import iou


def get_next_points_1(
    n_points,
    gt_mask,
    pred_mask,
    prev_pc_mask=None,
    prev_nc_mask=None,
):
    prev_pc_mask = torch.zeros_like(gt_mask, dtype=torch.float) if prev_pc_mask is None else prev_pc_mask
    prev_nc_mask = torch.zeros_like(gt_mask, dtype=torch.float) if prev_nc_mask is None else prev_nc_mask

    pos_clicks, neg_clicks = get_error_clicks_batch(
        n_points, 
        gt_mask[:, 0, :, :].cpu().numpy(), 
        pred_mask[:, 0, :, :].cpu().numpy(),
        t=0.5,
    )
    pc_mask = disk_mask_from_coords_batch(pos_clicks, prev_pc_mask)
    nc_mask = (
        disk_mask_from_coords_batch(neg_clicks, prev_nc_mask)
        if neg_clicks
        else torch.zeros_like(pc_mask)
    )

    return (
        pc_mask[:, None, :, :],
        nc_mask[:, None, :, :],
        pos_clicks,
        neg_clicks,
    )


class AdaptLoss(nn.Module):
    def __init__(self, model, omega, lbd=1.0, gamma=1.0):
        super().__init__()
        self.model = model
        self.init_model = deepcopy(model)
        for p in self.init_model.parameters():
            p.requires_grad = False
        # self.init_vals = [param.detach().clone() for param in model.parameters()]
        Z = sum([torch.sum(w) for w in omega])
        self.omega = [w / Z for w in  omega]
        
        self.lbd = lbd
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, image, aux, pc_mask, nc_mask):
        out = self.model(image, aux)
        if self.lbd < 1:
            with torch.no_grad():
                init_logits = self.init_model(image, aux)
                init_mask = (init_logits > 0).float()
        else:
            init_mask = torch.zeros_like(out)

        all_mask = torch.clip(pc_mask + nc_mask, min=0, max=1)
        mask_cnt = torch.sum(all_mask, dim=[2, 3])
        sparse_signal_per_item = torch.sum(all_mask * self.bce(out, pc_mask)) / mask_cnt
        sparse_signal = torch.mean(sparse_signal_per_item)
        dense_signal = torch.mean(self.bce(out, init_mask))

        params_change_signal = 0
        for idx, (p_new, p_old) in enumerate(zip(self.model.parameters(), self.init_model.parameters())):
            params_change_signal += torch.sum(self.omega[idx] * ((p_new - p_old) ** 2))

        loss = self.lbd * sparse_signal + (1-self.lbd) * dense_signal + self.gamma * params_change_signal
        return out, loss


def interact(crit, batch, interaction_steps, optim=None, clicks_per_step=1, grad_steps=10, verbose=False, target_iou=1.0):
    image, gt_mask = (
        batch["image"],
        batch["mask"],
    )

    image = image.permute(0, 3, 1, 2) if image.shape[-1] == 3 else image
    gt_mask = gt_mask[:, None, :, :] if gt_mask.ndim < 4 else gt_mask

    prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
    pc_mask, nc_mask, _pcs, _ncs = get_next_points_1(
        clicks_per_step,
        gt_mask,
        prev_output,
    )
    pcs = [_pcs]
    ncs = [_ncs]
    preds = [prev_output.cpu()]

    scores = [0.]
    for iter_idx in range(interaction_steps):
        if grad_steps > 0:
            for grad_idx in range(grad_steps):
                aux = torch.cat((pc_mask, nc_mask), dim=1)
                logits, loss = crit(image, aux, pc_mask, nc_mask)
                optim.zero_grad()
                loss.backward()
                optim.step()
                if verbose:
                    print('Loss:', loss.item())
        else:
            with torch.no_grad():
                aux = torch.cat((pc_mask, nc_mask), dim=1)
                logits, _ = crit(image, aux, pc_mask, nc_mask)
        
        prev_output = (logits.detach() > 0).float()

        score = torch.mean(iou(prev_output, gt_mask))
        scores.append(score.item())
        if verbose:
            print()
            print('IoU:', score.item())
            print()
        if score.item() > target_iou:
            break

        pc_mask, nc_mask, _pcs, _ncs = get_next_points_1(
            clicks_per_step,
            gt_mask,
            prev_output,
            prev_pc_mask=pc_mask,
            prev_nc_mask=nc_mask,
        )
        preds.append(prev_output.cpu())
        pcs.append(_pcs)
        ncs.append(_ncs)

    return np.array(scores), preds, pcs, ncs
