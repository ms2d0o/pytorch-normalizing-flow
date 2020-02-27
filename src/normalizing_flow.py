import torch
from torch import nn
import torch.nn.functional as F


class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        device = 'cuda:7' if torch.cuda.is_available() else 'cpu'

        self.w = nn.Parameter(torch.randn(1, dim).normal_(0, 0.1))
        self.b = nn.Parameter(torch.randn(1).normal_(0, 0.1))
        self.u = nn.Parameter(torch.randn(1, dim).normal_(0, 0.1))

    # 順伝播の計算
    def forward(self, z):
        if torch.mm(self.u, self.w.t()) < -1:
            self.u.data = self.u_hat()
        zk = z + self.u * nn.Tanh()(torch.mm(z, self.w.t()) + self.b)
        return zk

    def u_hat(self):
        wtu = torch.mm(self.u, self.w.t())
        m_wtu = -1 + torch.log(1 + torch.exp(wtu))

        return (
            self.u + (m_wtu - wtu) * self.w / torch.norm(self.w, p=2, dim=1)**2
        )

    def log_det_J(self, z):
        if torch.mm(self.u, self.w.t()) < -1:
            print(f"Normalizing u to u_hat. Old w.T.dot(u)={torch.mm(self.u, self.w.t())}")
            self.u.data = self.u_hat()
            print(f"New w.Tdot(u): {torch.mm(self.u, self.w.t())}")

        a = torch.mm(z, self.w.t()) + self.b
        psi = (1 - nn.Tanh()(a)**2) * self.w
        abs_det = (1 + torch.mm(self.u, psi.t())).abs()
        log_det = torch.log(1e-4 + abs_det)

        if torch.isnan(log_det).sum() > 0:
            print(f"u: {self.u}")
            print(f"w: {self.w}")
            print(f"b: {self.b}")
            print(f"abs_det: {abs_det}")

        return log_det


class NormalizingFlow(nn.Module):
    def __init__(self, K, dim):
        super().__init__()
        self.K = K
        self.dim = dim
        self.planarflow = [PlanarFlow(self.dim) for i in range(self.K)]
        print(self.planarflow)
        self.model = nn.Sequential(*self.planarflow)

    def forward(self, z):
        zk = self.planarflow[0].forward(z)
        log_qk = self.planarflow[0].log_det_J(z)
        for pf in self.planarflow[1:]:
            zk = pf.forward(zk)
            log_qk += pf.log_det_J(zk)
        return zk, log_qk


def calc_loss(z_k, log_q_k, target_density):
    log_p = torch.log(target_density.calc_prob_torch(z_k) + 1e-7)

    loss = F.mse_loss(-log_p, log_q_k)

    return loss


def get_optimizer(parameters):
    return torch.optim.Adam(parameters)
