import torch
import numpy as np

def focal_DRN(predications, labels, alpha=0.1):
    # compute the focal loss
    alpha_factor = torch.ones_like(labels) * alpha
    alpha_factor = torch.where(labels == 0 , alpha_factor, 1-alpha_factor)

    # compute smooth L1 loss
    # f(x) = 0.5 * (x)^2            if |x| < 1
    #        |x| - 0.5              otherwise

    reg_diff = torch.abs(labels - predications)
    reg_loss_pow = 0.5*(torch.pow(reg_diff, 2))
    reg_loss_sub = reg_diff - 0.5
    reg_loss = torch.where(reg_diff <= 1, reg_loss_pow, reg_loss_sub)

    cls_loss = alpha_factor*reg_loss

    return torch.sum(cls_loss)

def calc_metrices(gt_pred_pairs, monitor_metric):
    '''

    :param gt_pred_pairs:
    :return:
    '''
    DiC_list = [x-y for x,y in zip(gt_pred_pairs['gt'], gt_pred_pairs['pred'])]
    absDiC_list = [torch.abs(x-y) for x,y in zip(gt_pred_pairs['gt'], gt_pred_pairs['pred'])]
    agreement_list = [1 for x,y in zip(gt_pred_pairs['gt'], gt_pred_pairs['pred']) if x == y]
    MSE_list = [torch.pow(x-y, 2) for x,y in zip(gt_pred_pairs['gt'], gt_pred_pairs['pred'])]

    DiC = torch.mean(torch.tensor(DiC_list))
    absDiC = torch.mean(torch.tensor(absDiC_list))
    agreement = torch.mean(torch.tensor(agreement_list))
    MSE = torch.mean(torch.tensor(MSE_list))

    print(f'DiC: {DiC}, absDiC: {absDiC}, %: {agreement}, MSE: {MSE}')

    if monitor_metric == 'absDiC':
        return absDiC
    elif monitor_metric == 'agreement':
        return -agreement
    elif monitor_metric == 'MSE':
        return MSE