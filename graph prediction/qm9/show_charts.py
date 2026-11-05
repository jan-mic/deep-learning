import matplotlib
matplotlib.use('TkAgg') # Wymusza użycie osobnego, interaktywnego okna
from model import QM9Regressor
from train import process_data, visualize_regression_3d, visualize_regression_1d

CKPT_PATH = "/home/jd/PycharmProjects/deep-learning/task3/qm9/final_models/transformer_linear_dim2/qm9_Transformer_dim2_linear-epoch=192-val_mae=0.6179.ckpt"
EMBEDDING_DIM = 2

if __name__ == "__main__":
    loaders, num_features = process_data()
    test_loader = loaders['test']

    model = QM9Regressor.load_from_checkpoint(CKPT_PATH)
    model.eval()

    if EMBEDDING_DIM == 2:
        title = f"Regression_3D_Checkpoint"
        visualize_regression_3d(model, test_loader, title=title)

    elif EMBEDDING_DIM == 1:
        title = f"Regression_1D_Checkpoint"
        visualize_regression_1d(model, test_loader, title=title)
