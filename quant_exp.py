from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from torch.optim import Adam
from tqdm.auto import tqdm

from engine.cont_adapt import AdaptLoss, interact
from models.iis_models.ritm import HRNetISModel


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
    scores, _, _, _ = interact(
        crit, 
        batch, 
        interaction_steps=interaction_steps, 
        clicks_per_step=1, 
        optim=optim, 
        grad_steps=grad_steps
    )
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

    iou_targets = [0.7, 0.75, 0.8, 0.85, 0.9]
    clicks_at_iou = [{} for _ in range(len(iou_targets))]
    results_mean = {}
    area_under_curve = {}

    plt.figure(figsize=(10, 6))
    for name, scores in results.items():
        scores = np.array(scores)
        mean = np.mean(scores, axis=0)
        results_mean[name] = mean
        area_under_curve[name] = 0.5*(np.sum(mean[1:]) + np.sum(mean[2:-1]))

        xs = np.arange(1, scores.shape[1])
        plt.plot(xs, results_mean[name][1:], label=name)

        scores = np.concatenate((scores, np.ones((scores.shape[0], 1))), axis=1)
        for i, t in enumerate(iou_targets):
            clicks = np.argmax(scores > t, axis=1)
            clicks_at_iou[i][name] = np.mean(clicks)

    print("means")
    print(results_mean)
    print("area under curve")
    print(area_under_curve)
    print("clicks@iou")
    print(iou_targets)
    print(clicks_at_iou)

    plt.legend()
    plt.ylim(0.2, 0.9)
    plt.ylabel("mean IoU")
    plt.xlabel("# of clicks")
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig("comparison.png", dpi=300)

    return results, results_mean, iou_targets, clicks_at_iou
