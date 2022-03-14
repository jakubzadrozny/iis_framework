from random import uniform
import torch
from torch import nn
from torch import optim
from tqdm import tqdm

from data.clicking import get_positive_clicks_batch, disk_mask_from_coords_batch


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
            if omega.equal(zero):
                print('omega after zero')

            omegas[idx] = omega
            #HERE MAS IMPOERANCE UPDATE ENDS
        return loss#HAS NOTHING TO DO


def compute_importance_l2(model, loader):
    """Mimic the depoloyment setup where the model is applied on some samples and those are used to update the importance params
       Uses the L2norm of the function output. This is what we MAS uses as default
    """
    # model.eval()  # Set model to training mode so we get the gradient
    reg_params = initialize_reg_params(model)
    optim = MAS_Omega_update(model.parameters(), lr=0.0001, momentum=0.9)

    crit = nn.MSELoss(reduction='sum')

    # Iterate over data.
    for index, batch in tqdm(enumerate(loader)):
        # get the inputs
        image, gt_mask = (
            batch["image"],
            batch["mask"],
        )

        image = image.permute(0, 3, 1, 2) if image.shape[-1] == 3 else image
        gt_mask = gt_mask[:, None, :, :] if gt_mask.ndim < 4 else gt_mask

        pos_clicks = get_positive_clicks_batch(4, gt_mask, near_border=False, uniform_probs=True, erode_iters=0)
        neg_clicks = get_positive_clicks_batch(4, 1-gt_mask, near_border=False, uniform_probs=True, erode_iters=0)
        pc_mask = disk_mask_from_coords_batch(pos_clicks, torch.zeros_like(gt_mask))[:, None, :, :]
        nc_mask = disk_mask_from_coords_batch(neg_clicks, torch.zeros_like(gt_mask))[:, None, :, :]
        aux = torch.cat((pc_mask, nc_mask), dim=1).float()

        # forward
        logits = model(image, aux)
        probs = torch.sigmoid(logits)
        target_zeros = torch.zeros_like(probs)
        loss = crit(probs, target_zeros) 

        optim.zero_grad()
        loss.backward()
        optim.step(reg_params, index, image.size(0))
   
    return reg_params