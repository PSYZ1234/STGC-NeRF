import torch

def sceneflow_loss_func(flow_preds, flow_gt, 
                   gamma=0.9,
                   **kwargs,
                   ):
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        level_target = flow_gt
        if level_target.shape[1] == 4:
            flow_mask = level_target[:, 3, :] > 0
            diff = flow_preds[i] - level_target[:, :3, :]
            epe_l1 = torch.pow(diff.abs().sum(dim=1) + 0.01, 0.4)[flow_mask].mean()
        else:
            diff = flow_preds[i] - level_target
            epe_l1 = torch.pow(diff.abs().sum(dim=1) + 0.01, 0.4).mean()
        
        i_weight = gamma ** (n_predictions - i - 1)
        flow_loss += i_weight * epe_l1

    return flow_loss