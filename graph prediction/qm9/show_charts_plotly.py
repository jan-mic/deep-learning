import torch
import numpy as np
import plotly.graph_objects as go
from model import QM9Regressor
from train import process_data

CKPT_PATH = "/home/jd/PycharmProjects/deep-learning/task3/qm9/final_models/transformer_linear_dim2/qm9_Transformer_dim2_linear-epoch=192-val_mae=0.6179.ckpt"
EMBEDDING_DIM = 2


# --- NOWE FUNKCJE PLOTLY ---
def visualize_regression_3d_plotly(model, loader, device='cpu', title='Regression (3D)'):
    model.eval()
    model.to(device)
    all_embeddings, all_true_values = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, embedding = model(batch.x, batch.edge_index, batch.batch)
            all_embeddings.append(embedding.cpu().numpy())
            all_true_values.append(batch.y[:, 0].cpu().numpy())

    X_emb = np.concatenate(all_embeddings, axis=0)
    Y_true = np.concatenate(all_true_values, axis=0)

    # Tworzenie siatki do płaszczyzny aproksymacji
    x_min, x_max = X_emb[:, 0].min() - 0.5, X_emb[:, 0].max() + 0.5
    y_min, y_max = X_emb[:, 1].min() - 0.5, X_emb[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    grid_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float().to(device)
    with torch.no_grad():
        zz = model.head(grid_tensor).cpu().numpy().reshape(xx.shape)

    # Rysowanie w Plotly
    fig = go.Figure()
    fig.add_trace(go.Surface(x=xx, y=yy, z=zz, colorscale='Viridis', opacity=0.8, name='Model'))
    fig.add_trace(go.Scatter3d(x=X_emb[:, 0], y=X_emb[:, 1], z=Y_true,
                               mode='markers', marker=dict(size=3, color='red'), name='Dane QM9'))

    fig.update_layout(title=title,
                      scene=dict(xaxis_title='Embedding 1', yaxis_title='Embedding 2', zaxis_title='Moment Dipolowy'),
                      margin=dict(l=0, r=0, b=0, t=40))
    fig.show()


def visualize_regression_1d_plotly(model, loader, device='cpu', title='Regression (1D)'):
    model.eval()
    model.to(device)
    all_embeddings, all_true_values = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, embedding = model(batch.x, batch.edge_index, batch.batch)
            all_embeddings.append(embedding.cpu().numpy())
            all_true_values.append(batch.y[:, 0].cpu().numpy())

    X_emb = np.concatenate(all_embeddings).flatten()
    Y_true = np.concatenate(all_true_values).flatten()

    x_grid_np = np.linspace(X_emb.min() - 0.5, X_emb.max() + 0.5, 500)
    x_grid = torch.from_numpy(x_grid_np).float().view(-1, 1).to(device)

    with torch.no_grad():
        y_grid = model.head(x_grid).cpu().numpy().flatten()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_emb, y=Y_true, mode='markers', marker=dict(color='red', opacity=0.5), name='Dane QM9'))
    fig.add_trace(
        go.Scatter(x=x_grid_np, y=y_grid, mode='lines', line=dict(color='blue', width=3), name='Aproksymacja'))

    fig.update_layout(title=title, xaxis_title='Wartość Osadzenia (1D)', yaxis_title='Wartość Właściwości')
    fig.show()


# ---------------------------

if __name__ == "__main__":
    loaders, num_features = process_data()
    test_loader = loaders['test']

    # Urządzenie (szybsze generowanie na GPU, jeśli masz)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = QM9Regressor.load_from_checkpoint(CKPT_PATH)
    model.to(device)
    model.eval()

    if EMBEDDING_DIM == 2:
        title = f"Regression_3D_Checkpoint"
        visualize_regression_3d_plotly(model, test_loader, device=device, title=title)

    elif EMBEDDING_DIM == 1:
        title = f"Regression_1D_Checkpoint"
        visualize_regression_1d_plotly(model, test_loader, device=device, title=title)