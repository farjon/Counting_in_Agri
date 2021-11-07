import torch

def focal_DRN(predications, labels, alpha=0.1):
    # compute the focal loss
    alpha_factor = torch.ones_like(labels) * alpha
    alpha_factor = torch.where(labels == 0 , alpha_factor, 1-alpha_factor)

    # compute smooth L1 loss
    # f(x) = 0.5 * (x)^2            if |x| < 1
    #        |x| - 0.5              otherwise

    reg_diff = torch.abs(labels - predications)
    reg_loss = torch.where(reg_diff <= 1, 0.5*(torch.pow(reg_diff, 2), reg_diff - 0.5))

    cls_loss = alpha_factor*reg_loss

    return torch.sum(cls_loss)