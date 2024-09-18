import torch
from torch import nn
import torch.nn.functional as F

# Output vector indices
# 0, 1, 2 are phi_a^alpha, phi_b^alpha, phi_c^alpha (c depends on a and b, not considered)
# 3, 4, 5 are phi_a^beta, phi_b^beta, phi_c^beta
# 6, 7, 8 are phi_a^gamma, phi_b^gamma, phi_c^gamma
# 9, 10, 11 are w^alpha, w^beta, w^gamma

EPS = 1e-10

class ChainSoftmax(nn.Module):
    "Chain softmax model for phase prediction and regression"

    def __init__(self, device, mask=1, dim=128):
        super().__init__()
        self.device = device
        self.mask = mask
        self.dim = dim
        self.input_linear = nn.Linear(8, dim)
        self.input_bn = nn.BatchNorm1d(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.final_linear = nn.Linear(dim, 3)

        self.reg_alpha = nn.Linear(dim, 3)
        self.bn_alpha = nn.BatchNorm1d(3)

        self.reg_beta = nn.Linear(dim, 3)
        self.bn_beta = nn.BatchNorm1d(3)

        self.reg_gamma = nn.Linear(dim, 3)
        self.bn_gamma = nn.BatchNorm1d(3)

        self.reg_abundance = nn.Linear(dim, 3)
        self.bn_abundance = nn.BatchNorm1d(3)

        self.bn_cls = nn.BatchNorm1d(3)
        self.bn_first = nn.BatchNorm1d(8)

    def forward(self, x):
        x = x.to(self.device)
        identity = F.relu(self.input_bn(self.input_linear(self.bn_first(x))))
        out = F.relu(self.bn1(self.linear1(identity)))
        out = F.relu(self.bn2(self.linear2(out)))

        class_out = self.bn_cls(self.final_linear(out))
        class_idx = torch.argmax(class_out, dim=1)

        alpha_comp = F.softmax(self.bn_alpha(self.reg_alpha(out)), dim=-1)
        beta_comp = F.softmax(self.bn_beta(self.reg_beta(out)), dim=-1)
        gamma_comp = F.softmax(self.bn_gamma(self.reg_gamma(out)), dim=-1)
        abundance = F.softmax(self.bn_abundance(self.reg_abundance(out)), dim=-1)

        if self.mask == 1:
            beta_comp, gamma_comp, abundance = self.apply_mask(
                beta_comp, gamma_comp, abundance, class_idx
            )

        regression_out = torch.cat(
            (alpha_comp, beta_comp, gamma_comp, abundance), dim=1
        )

        return class_out, regression_out

    def apply_mask(self, beta_comp, gamma_comp, abundance, class_idx):
        beta_comp_temp = beta_comp.clone()
        gamma_comp_temp = gamma_comp.clone()
        abundance_temp = abundance.clone()

        beta_comp_temp[class_idx == 0, :] = 1 / 3
        beta_comp_temp[class_idx == 1, :] = 1 / 3
        gamma_comp_temp[class_idx == 1, :] = 1 / 3

        abund_0 = abundance_temp[class_idx == 0, :].clone()
        abund_0_norm = abund_0[:, :-2] / torch.sum(abund_0[:, :-2], dim=1, keepdim=True)
        abund_0_out = torch.zeros_like(abund_0) + EPS
        abund_0_out[..., :-2] = abund_0_norm

        abund_1 = abundance_temp[class_idx == 1, :].clone()
        abund_1_norm = abund_1[:, :-1] / torch.sum(abund_1[:, :-1], dim=1, keepdim=True)
        abund_1_out = torch.zeros_like(abund_1) + EPS
        abund_1_out[..., :-1] = abund_1_norm

        abundance_temp[class_idx == 0, :] = abund_0_out
        abundance_temp[class_idx == 1, :] = abund_1_out

        return beta_comp_temp.clone(), gamma_comp_temp.clone(), abundance_temp.clone()


class ChainLinear(nn.Module):
    "Chain linear model for phase prediction and regression"

    def __init__(self, device, mask=1, dim=128):
        super().__init__()
        self.device = device
        self.mask = mask
        self.dim = dim
        self.input_linear = nn.Linear(8, dim)
        self.input_bn = nn.BatchNorm1d(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.final_linear = nn.Linear(dim, 3)

        self.reg_alpha = nn.Linear(dim, 3)
        self.bn_alpha = nn.BatchNorm1d(3)

        self.reg_beta = nn.Linear(dim, 3)
        self.bn_beta = nn.BatchNorm1d(3)

        self.reg_gamma = nn.Linear(dim, 3)
        self.bn_gamma = nn.BatchNorm1d(3)

        self.reg_abundance = nn.Linear(dim, 3)
        self.bn_abundance = nn.BatchNorm1d(3)

        self.bn_cls = nn.BatchNorm1d(3)
        self.bn_first = nn.BatchNorm1d(8)

    def forward(self, x):
        x = x.to(self.device)
        identity = F.relu(self.input_bn(self.input_linear(self.bn_first(x))))
        out = F.relu(self.bn1(self.linear1(identity)))
        out = F.relu(self.bn2(self.linear2(out)))

        class_out = self.bn_cls(self.final_linear(out))
        class_idx = torch.argmax(class_out, dim=1)

        alpha_comp = F.sigmoid(self.bn_alpha(self.reg_alpha(out)))
        beta_comp = F.sigmoid(self.bn_beta(self.reg_beta(out)))
        gamma_comp = F.sigmoid(self.bn_gamma(self.reg_gamma(out)))
        abundance = F.sigmoid(self.bn_abundance(self.reg_abundance(out)))

        if self.mask == 1:
            beta_comp, gamma_comp, abundance = self.apply_mask(
                beta_comp, gamma_comp, abundance, class_idx
            )

        regression_out = torch.cat(
            (alpha_comp, beta_comp, gamma_comp, abundance), dim=1
        )

        return class_out, regression_out

    def apply_mask(self, beta_comp, gamma_comp, abundance, class_idx):
        beta_comp_temp = beta_comp.clone()
        gamma_comp_temp = gamma_comp.clone()
        abundance_temp = abundance.clone()

        beta_comp_temp[class_idx == 0, :] = 1 / 3
        beta_comp_temp[class_idx == 1, :] = 1 / 3
        gamma_comp_temp[class_idx == 1, :] = 1 / 3

        abund_0 = abundance_temp[class_idx == 0, :].clone()
        abund_0_norm = abund_0[:, :-2]
        abund_0_out = torch.zeros_like(abund_0) + EPS
        abund_0_out[..., :-2] = abund_0_norm

        abund_1 = abundance_temp[class_idx == 1, :].clone()
        abund_1_norm = abund_1[:, :-1]
        abund_1_out = torch.zeros_like(abund_1) + EPS
        abund_1_out[..., :-1] = abund_1_norm

        abundance_temp[class_idx == 0, :] = abund_0_out
        abundance_temp[class_idx == 1, :] = abund_1_out

        return beta_comp_temp.clone(), gamma_comp_temp.clone(), abundance_temp.clone()
