import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from model import MoleculeGNNLightningModel

SEED = 42
BATCH_SIZE = 32
EPOCHS = 200
PATIENCE = 20
DATA_PATH = './data'

GNN_TYPE = 'Transformer'  # [GCN, Transformer]
HEAD_TYPE = 'linear'  # [linear, mlp]
NUM_LAYERS = 3
EMBEDDING_DIM = 32
HIDDEN_DIM = 32
MLP_HIDDEN_DIM = 32
LR = 0.001
DROPOUT = 0.1


def generate_charts(file_path, exp_name, y_min=0.4, y_max=0.8):
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

    plt.ylim(y_min, y_max)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    charts_dir = os.path.join(os.path.dirname(__file__), "charts")
    os.makedirs(charts_dir, exist_ok=True)
    plt.savefig(os.path.join(charts_dir, f"loss_curve_{exp_name}_{timestamp}.png"))
    plt.close()


def calculate_and_plot_dec_boundary(model, test_loader, title="2d Visualization"):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model.eval()
    model.to(device)

    x = []
    y = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits, features = model(batch.x, batch.edge_index, batch.batch)

            x.append(features.detach().cpu())
            y.append(batch.y.view(-1).detach().cpu())

    X = torch.cat(x, dim=0).numpy()
    y = torch.cat(y, dim=0).long().numpy()

    indices = np.arange(0, len(X), 1)
    X = X[indices]
    y = y[indices]

    plot_decision_boundaries(model, X, y, device, title=title)


def visualize_classification_1d(model, loader, device='cpu', title='Classification (1D)'):
    model.eval()
    model.to(device)

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, embedding = model(batch.x, batch.edge_index, batch.batch)

            all_embeddings.append(embedding.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

    X_emb = np.concatenate(all_embeddings).flatten()  # Kształt [N]
    Y_true = np.concatenate(all_labels).flatten()  # Kształt [N] (0 lub 1)

    x_min, x_max = X_emb.min() - 0.5, X_emb.max() + 0.5

    x_grid_np = np.linspace(x_min, x_max, 300)
    x_grid = torch.from_numpy(x_grid_np).float().view(-1, 1).to(device)

    with torch.no_grad():
        logits_grid = model.head(x_grid)

        if logits_grid.shape[1] == 1:
            probs_grid = torch.sigmoid(logits_grid).cpu().numpy().flatten()
        else:
            probs_grid = torch.softmax(logits_grid, dim=1)[:, 1].cpu().numpy().flatten()

    plt.figure(figsize=(10, 6))

    jitter = np.random.normal(0, 0.02, size=Y_true.shape)

    mask_0 = (Y_true == 0)
    plt.scatter(X_emb[mask_0], Y_true[mask_0] + jitter[mask_0],
                color='blue', alpha=0.5, label='Klasa 0 (Nie-inhibitor)', marker='o')

    mask_1 = (Y_true == 1)
    plt.scatter(X_emb[mask_1], Y_true[mask_1] + jitter[mask_1],
                color='red', alpha=0.5, label='Klasa 1 (Inhibitor)', marker='^')

    plt.plot(x_grid_np, probs_grid, color='green', linewidth=3, label='Prawdopodobieństwo (Model)')

    idx_boundary = np.argmin(np.abs(probs_grid - 0.5))
    decision_boundary_x = x_grid_np[idx_boundary]

    plt.axvline(x=decision_boundary_x, color='black', linestyle='--', label='Granica Decyzyjna')
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    plt.title(title)
    plt.xlabel('Wartość Osadzenia (1D)')
    plt.ylabel('Prawdopodobieństwo klasy 1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs("./images", exist_ok=True)
    plt.savefig(f"./images/{title.replace(' ', '_')}.png")
    plt.show()


def plot_decision_boundaries(model, x, y, device='cpu', title='Decision Boundary'):
    model.eval()
    model.to(device)

    if x.shape[1] != 2:
        raise ValueError(f"Wizualizacja wymaga embedding_dim=2, dostałem: {x.shape[1]}")

    x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model.head(grid_tensor)
        predicted = outputs.argmax(dim=1)

    predicted = predicted.cpu().numpy()
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    cmap = plt.get_cmap('tab10', np.unique(y).size)
    levels = np.arange(num_classes + 1) - 0.5
    plt.figure(figsize=(8, 6))
    cntr = plt.contourf(
        xx, yy, predicted.reshape(xx.shape),
        levels=levels,
        alpha=0.3,
        cmap=cmap,
        vmin=-0.5,
        vmax=num_classes - 0.5
    )
    scatter = plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=cmap)
    plt.title(title)
    plt.xlabel('Cecha 1')
    plt.ylabel('Cecha 2')
    plt.legend(*scatter.legend_elements(), title="Klasa")
    colorbar = plt.colorbar(cntr, ticks=range(num_classes))
    colorbar.set_label('Klasa')
    plt.tight_layout()
    os.makedirs("./images", exist_ok=True)
    plt.savefig(f"./images/{title.replace(' ', '_')}.png")
    plt.show()


def process_data():
    dataset = MoleculeNet(DATA_PATH, name='BACE')

    y = dataset.y.squeeze().long()
    neg = (y == 0).sum().item()
    pos = (y == 1).sum().item()
    total = len(y)

    w0 = total / (2 * neg)
    w1 = total / (2 * pos)
    class_weights = torch.tensor([w0, w1], dtype=torch.float)

    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size

    gen = torch.Generator().manual_seed(SEED)
    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size], generator=gen
    )

    loaders = {
        'train': DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, persistent_workers=True),
        'val': DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, persistent_workers=True),
        'test': DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }
    return loaders, dataset.num_node_features, class_weights


def main():
    loaders, num_features, class_weights = process_data()

    exp_name = f"bace_{GNN_TYPE}_L{NUM_LAYERS}_{HEAD_TYPE}_dim{EMBEDDING_DIM}"

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename=f'{exp_name}-{{epoch:02d}}-{{val_loss:.2f}}',
        save_top_k=1,
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        mode='min',
        verbose=True
    )

    logger = CSVLogger("logs", name=exp_name)

    model = MoleculeGNNLightningModel(
        num_node_features=num_features,
        class_weights=class_weights,
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        num_classes=2,
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
        log_every_n_steps=10
    )

    trainer.fit(model, loaders['train'], loaders['val'])

    print("\n--- Wyniki Testu Najlepszego Modelu ---")
    test_results = trainer.test(dataloaders=loaders['test'], ckpt_path='best')
    print(f"Test Loss: {test_results[0]['test_loss']:.4f}")
    model = MoleculeGNNLightningModel.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path)
    if EMBEDDING_DIM == 2:
        try:
            calculate_and_plot_dec_boundary(model, loaders['test'], title=exp_name)
        except Exception as e:
            print(f"Nie można wygenerować wykresu: {e}")
    elif EMBEDDING_DIM == 1:
        try:
            visualize_classification_1d(model, loaders['test'], title=exp_name)
        except Exception as e:
            print(f"Nie można wygenerować wykresu 1D: {e}")

    metrics_file = logger.log_dir + "/metrics.csv"
    generate_charts(metrics_file, exp_name)


if __name__ == "__main__":
    main()
