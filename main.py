import torch
from normalizingflow import NormalDistribution2D, TargetDistribution1, TargetDistribution2
from normalizingflow import NormalizingFlow, calc_loss, get_optimizer
import matplotlib.pyplot as plt


if __name__ == '__main__':
    device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
    normal_distribution = NormalDistribution2D()
    # かっこみたいなやつ
    # target_density = TargetDistribution1()
    # target_density.plot(fig_name='figures/target_dist_1.png')

    # sin波っぽいやつ
    target_density = TargetDistribution2()
    target_density.plot(fig_name='figures/target_dist_2.png')

    normalizing_flow = NormalizingFlow(K=15, dim=2)
    optimizer = get_optimizer(normalizing_flow.model.parameters())
    invisible_axis = True

    losses = []
    for epoch in range(10000 + 1):
        z_0_batch = normal_distribution.sample_torch(10000)
        log_q_0_batch = torch.log(normal_distribution.calc_prob_torch(z_0_batch))
        z_k, log_q_k = normalizing_flow.forward(z_0_batch)

        optimizer.zero_grad()
        loss = calc_loss(z_k, log_q_k, target_density)
        losses.append(loss.cpu().data.item())
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}   Loss: {loss}")

        if epoch % 1000 == 0:
            plt.figure(figsize=(6, 6))
            z_k_value = z_k.data
            plt.scatter(z_k_value[:, 0], z_k_value[:, 1], alpha=0.7)
            if invisible_axis:
                plt.tick_params(bottom=False, left=False, right=False, top=False)
                plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            plt.savefig(f'./figures/epoch_{epoch}.png')