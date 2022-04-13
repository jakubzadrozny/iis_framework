# Adapted from https://github.com/rahafaljundi/MAS-Memory-Aware-Synapses

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A

from data.datasets.inria_aerial import InriaAerialDataset
from data.iis_dataset import RegionDataset
from data.region_selector import dummy
from data.clicking import get_positive_clicks_batch, disk_mask_from_coords_batch
from models.iis_models.ritm import HRNetISModel


def initialize_reg_params(model,freeze_layers=[]):
    """initialize an omega for each parameter to zero"""
    omegas = []
    for param in model.parameters():
        omegas.append(torch.zeros_like(param))
    return omegas


class MAS_Omega_update(optim.SGD):
    """
    Update the paramerter importance using the gradient of the function output norm. To be used at deployment time.
    reg_params:parameters omega to be updated
    batch_index,batch_size:used to keep a running average over the seen samples
    """

    def __init__(self, params, lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        
        super(MAS_Omega_update, self).__init__(params, lr,momentum,dampening,weight_decay,nesterov)
        
    def __setstate__(self, state):
        super(MAS_Omega_update, self).__setstate__(state)
       

    def step(self, omegas, batch_index, batch_size, closure=None):
        """
        Performs a single parameters importance update setp
        """

        #print('************************DOING A STEP************************')
 
        loss = None
        if closure is not None:
            loss = closure()
             
        group = self.param_groups[0]
        for idx, p in enumerate(group['params']):
          
            #print('************************ONE PARAM************************')
            
            if p.grad is None:
                continue
            
            d_p = p.grad.data
                
            #HERE MAS IMPOERANCE UPDATE GOES
            #get the gradient
            unreg_dp = p.grad.data.clone()
            omega = omegas[idx]
                
            zero = torch.zeros_like(p)
                
            #sum up the magnitude of the gradient
            prev_size = batch_index*batch_size
            curr_size=(batch_index+1)*batch_size
            omega=omega.mul(prev_size)
                
            omega=omega.add(unreg_dp.abs_())
            #update omega value
            omega=omega.div(curr_size)

            omegas[idx] = omega
            #HERE MAS IMPOERANCE UPDATE ENDS
        return loss#HAS NOTHING TO DO


def compute_importance_l2(model, loader, device='cpu'):
    """Mimic the depoloyment setup where the model is applied on some samples and those are used to update the importance params
       Uses the L2norm of the function output. This is what we MAS uses as default
    """
    # model.eval()  # Set model to training mode so we get the gradient
    reg_params = initialize_reg_params(model)
    optim = MAS_Omega_update(model.parameters(), lr=0.0003, momentum=0.9)

    crit = nn.BCEWithLogitsLoss(reduction='mean')

    # Iterate over data.
    for index, batch in tqdm(enumerate(loader)):
        # get the inputs
        image, gt_mask = (
            batch["image"],
            batch["mask"],
        )

        image = image.permute(0, 3, 1, 2) if image.shape[-1] == 3 else image
        gt_mask = gt_mask[:, None, :, :] if gt_mask.ndim < 4 else gt_mask

        pos_clicks = get_positive_clicks_batch(5, gt_mask, near_border=False, uniform_probs=True, erode_iters=0)
        neg_clicks = get_positive_clicks_batch(5, 1-gt_mask, near_border=False, uniform_probs=True, erode_iters=0)
        pc_mask = disk_mask_from_coords_batch(pos_clicks, torch.zeros_like(gt_mask))[:, None, :, :]
        nc_mask = disk_mask_from_coords_batch(neg_clicks, torch.zeros_like(gt_mask))[:, None, :, :]
        aux = torch.cat((pc_mask, nc_mask), dim=1).float()

        # forward
        image = image.to(device)
        aux = aux.to(device)
        gt_mask = gt_mask.to(device)
        logits = model(image, aux)
        loss = crit(logits, gt_mask.float()) 

        optim.zero_grad()
        loss.backward()
        optim.step(reg_params, index, image.size(0))

        if index % 100 == 0:
            torch.save(reg_params, 'omega_bce.pth')
   
    return reg_params

if __name__ == "__main__":
    seg_dataset = InriaAerialDataset(
        "/content/AerialImageDataset", 
        split="train_cut"
    )
    region_selector = dummy
    augmentator = A.Normalize()
    iis_dataset = RegionDataset(seg_dataset, region_selector, augmentator)
    iis_loader = DataLoader(
        iis_dataset, batch_size=10, num_workers=2, shuffle=True
    )

    model_path = '/content/coco_lvis_h18_baseline.pth'
    model = HRNetISModel.load_from_checkpoint(
        model_path,
    )
    model.to('cuda')

    omega = compute_importance_l2(model, iis_loader, device='cuda')
    torch.save(omega, 'omega_bce.pth')
