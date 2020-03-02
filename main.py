import torch
from src import NormalDistribution2D, TargetDistribution1, TargetDistribution2
from src import NormalizingFlow, calc_loss, get_optimizer
import matplotlib.pyplot as plt

N_SAMPLE = 10000
EPOCHS = 10000
K_FLOW = 5
N_DIM = 2

if __name__ == '__main__':
    normal_distribution = NormalDistribution2D()

    # かっこみたいなやつ
    target_density = TargetDistribution1()
    target_density.plot(fig_name='figures/target_dist_1.png')

    # sin波
    # target_density = TargetDistribution2()
    # target_density.plot(fig_name='figures/target_dist_2.png')

    normalizing_flow = NormalizingFlow(K=K_FLOW, dim=N_DIM)
    optimizer = get_optimizer(normalizing_flow.model.parameters())
    invisible_axis = True

    losses = []
    for epoch in range(EPOCHS + 1):
        z_0_batch = normal_distribution.sample_torch(N_SAMPLE)
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
