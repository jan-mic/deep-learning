import torch
from torch_geometric.datasets import QM9

if __name__ == "__main__":
    dataset = QM9(root='./data_qm9')
    dipole_moments = dataset.data.y[:, 0]

    min_val = torch.min(dipole_moments).item()
    max_val = torch.max(dipole_moments).item()
    mean_val = torch.mean(dipole_moments).item()
    std_val = torch.std(dipole_moments).item()

    print(f"Min: {min_val:.4f}")
    print(f"Max: {max_val:.4f}")
    print(f"Średnia: {mean_val:.4f}")
    print(f"Odchylenie standardowe: {std_val:.4f}")
