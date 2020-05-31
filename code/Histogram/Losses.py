import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_function(prediction, true_y, phi_cand, alpha_0, hidden_dim, annealing_rate, how_many_samps, kl_term):

    BCE = F.binary_cross_entropy(prediction, true_y, reduction='mean')

    if kl_term:
        # KLD term
        alpha_0 = torch.Tensor([alpha_0])
        hidden_dim = torch.Tensor([hidden_dim])
        trm1 = torch.lgamma(torch.sum(phi_cand)) - torch.lgamma(hidden_dim*alpha_0)
        trm2 = - torch.sum(torch.lgamma(phi_cand)) + hidden_dim*torch.lgamma(alpha_0)
        trm3 = torch.sum((phi_cand-alpha_0)*(torch.digamma(phi_cand)-torch.digamma(torch.sum(phi_cand))))
        KLD = trm1 + trm2 + trm3

        return BCE + annealing_rate * KLD / how_many_samps

    else:

        return BCE
