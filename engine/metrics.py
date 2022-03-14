import torch

def mse(output, gt_mask):
    '''mean squared error'''
    return ((output - gt_mask)**2).sum() / output.numel()

def iou(output, gt_mask, eps=1e-6):
    gt_sum = torch.sum(gt_mask, dim=[-1, -2])
    _prod = output * gt_mask
    _sum = torch.clip(output + gt_mask, max=1)
    _iou = 1 * (gt_sum == 0) + torch.sum(_prod, dim=[-1, -2]) / (torch.sum(_sum, dim=[-1, -2]) + eps)
    return _iou
