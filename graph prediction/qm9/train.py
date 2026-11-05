import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from model import QM9Regressor

SEED = 42
BATCH_SIZE = 64
EPOCHS = 200
PATIENCE = 10
DATA_PATH = './data_qm9'

GNN_TYPE = 'Transformer'  # [GCN, Transformer]
HEAD_TYPE = 'linear'  # [linear, mlp]
NUM_LAYERS = 3
EMBEDDING_DIM = 1
HIDDEN_DIM = 32
MLP_HIDDEN_DIM = 32
LR = 0.001
DROPOUT = 0.1


def generate_charts(file_path, exp_name):
    if not os.path.exists(file_path):
        return
    metrics = pd.read_csv(file_path)

    train_loss = metrics[['epoch', 'train_loss_eval_mode']].dropna().set_index('epoch')
    val_loss = metrics[['epoch', 'val_loss']].dropna().set_index('epoch')

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss.index, train_loss['train_loss_eval_mode'], label='Train Loss (Eval Mode)', marker='o')
    plt.plot(val_loss.index, val_loss['val_loss'], label='Validation Loss', marker='s')

    plt.title('Training vs Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    charts_dir = os.path.join(os.path.dirname(__file__), "charts")
    os.makedirs(charts_dir, exist_ok=True)
    plt.savefig(os.path.join(charts_dir, f"loss_curve_{exp_name}_{timestamp}.png"))
    plt.close()


def visualize_regression_3d(model, loader, device='cpu', title='Regression (3D)'):
    model.eval()
    model.to(device)

    all_embeddings = []
    all_true_values = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out, embedding = model(batch.x, batch.edge_index, batch.batch)

            all_embeddings.append(embedding.cpu().numpy())
            all_true_values.append(batch.y[:, 0].cpu().numpy())

    X_emb = np.concatenate(all_embeddings, axis=0)
    Y_true = np.concatenate(all_true_values, axis=0)

    x_min, x_max = X_emb[:, 0].min() - 0.5, X_emb[:, 0].max() + 0.5
    y_min, y_max = X_emb[:, 1].min() - 0.5, X_emb[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.from_numpy(grid_points).float().to(device)

    with torch.no_grad():
        predictions_on_grid = model.head(grid_tensor)

    zz = predictions_on_grid.cpu().numpy().reshape(xx.shape)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(xx, yy, zz, cmap='viridis', alpha=0.6, edgecolor='none')

    ax.scatter(X_emb[:, 0], X_emb[:, 1], Y_true, c='red', s=20, label='Dane rzeczywiste (QM9)', depthshade=True)

    ax.set_xlabel('Embedding Dim 1')
    ax.set_ylabel('Embedding Dim 2')
    ax.set_zlabel('Wartość fizyczna (np. moment dipolowy)')
    ax.set_title('Wizualizacja Regresji: Powierzchnia aproksymacji vs Dane')
    ax.view_init(elev=20, azim=45)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Wartość przewidywana przez model')
    ax.legend()
    os.makedirs("./images", exist_ok=True)
    plt.savefig(f"./images/{title.replace(' ', '_')}.png")
    plt.show()


def visualize_regression_1d(model, loader, device='cpu', title='Regression (1D)'):
    model.eval()
    model.to(device)

    all_embeddings = []
    all_true_values = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out, embedding = model(batch.x, batch.edge_index, batch.batch)

            all_embeddings.append(embedding.cpu().numpy())
            all_true_values.append(batch.y[:, 0].cpu().numpy())

    X_emb = np.concatenate(all_embeddings).flatten()
    Y_true = np.concatenate(all_true_values).flatten()

    x_min, x_max = X_emb.min() - 0.5, X_emb.max() + 0.5

    x_grid_np = np.linspace(x_min, x_max, 500)
    x_grid = torch.from_numpy(x_grid_np).float().view(-1, 1).to(device)

    with torch.no_grad():
        y_grid = model.head(x_grid).cpu().numpy().flatten()

    plt.figure(figsize=(10, 6))

    plt.scatter(X_emb, Y_true, color='red', alpha=0.5, label='Wartość Prawdziwa (QM9)', s=20)

    plt.plot(x_grid_np, y_grid, color='blue', linewidth=3, label='Funkcja Aproksymacji (Model)')

    indices = np.random.choice(len(X_emb), size=min(50, len(X_emb)), replace=False)
    for i in indices:
        emb_val = torch.tensor([[X_emb[i]]]).float().to(device)
        with torch.no_grad():
            pred_val = model.head(emb_val).item()

        plt.plot([X_emb[i], X_emb[i]], [Y_true[i], pred_val],
                 color='gray', linestyle='--', alpha=0.6, linewidth=1)

    plt.title(f'Regresja QM9 w przestrzeni osadzeń 1D')
    plt.xlabel('Wartość Osadzenia (1D)')
    plt.ylabel('Wartość Właściwości')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.plot([], [], color='gray', linestyle='--', label='Błąd (Residuum)')
    plt.legend()
    os.makedirs("./images", exist_ok=True)
    plt.savefig(f"./images/{title.replace(' ', '_')}.png")
    plt.show()


def process_data():
    dataset = QM9(root=DATA_PATH)

    target_idx = 0
    dataset.data.y = dataset.data.y[:, target_idx].unsqueeze(1)
    total = len(dataset)
    print(f"Rozmiar zbioru: {total} cząsteczek")

    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size

    gen = torch.Generator().manual_seed(SEED)
    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size], generator=gen
    )
    loaders = {
        'train': DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, persistent_workers=True),
        'val': DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, persistent_workers=True),
        'test': DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)
    }
    return loaders, dataset.num_node_features


def main():
    loaders, num_features = process_data()
    exp_name = f"qm9_{GNN_TYPE}_dim{EMBEDDING_DIM}_{HEAD_TYPE}"

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints_qm9',
        filename=f'{exp_name}-{{epoch:02d}}-{{val_mae:.4f}}',
        save_top_k=1,
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        mode='min',
        verbose=True
    )

    logger = CSVLogger("logs_qm9", name=exp_name)

    model = QM9Regressor(
        num_node_features=num_features,
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        gnn_type=GNN_TYPE,
        head_type=HEAD_TYPE,
        num_layers=NUM_LAYERS,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        lr=LR,
        dropout=DROPOUT,
        after_train_eval=True
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator="auto",
        devices=1,
        log_every_n_steps=50
    )

    trainer.fit(model, loaders['train'], loaders['val'])

    print("\n--- Wyniki Testu ---")
    trainer.test(dataloaders=loaders['test'], ckpt_path='best')
    model = QM9Regressor.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path)
    if EMBEDDING_DIM == 2:
        try:
            visualize_regression_3d(model, loaders['test'], title=exp_name)
        except Exception as e:
            print(f"Nie można wygenerować wykresu: {e}")
    elif EMBEDDING_DIM == 1:
        try:
            visualize_regression_1d(model, loaders['test'], title=exp_name)
        except Exception as e:
            print(f"Nie można wygenerować wykresu 1D: {e}")

    metrics_file = logger.log_dir + "/metrics.csv"
    generate_charts(metrics_file, exp_name)


if __name__ == "__main__":
    main()
