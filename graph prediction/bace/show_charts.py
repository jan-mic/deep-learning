from model import MoleculeGNNLightningModel
from train import process_data, calculate_and_plot_dec_boundary, visualize_classification_1d

CKPT_PATH = "/home/jd/PycharmProjects/deep-learning/task3/bace/checkpoints/bace_Transformer_L3_linear_dim1-epoch=73-val_loss=0.60.ckpt"
EMBEDDING_DIM = 1

if __name__ == "__main__":
    loaders, num_features, class_weights = process_data()
    test_loader = loaders['test']

    model = MoleculeGNNLightningModel.load_from_checkpoint(CKPT_PATH)
    model.eval()

    if EMBEDDING_DIM == 2:
        title = f"Decision Boundary (Checkpoint 2D)"
        calculate_and_plot_dec_boundary(model, test_loader, title=title)

    elif EMBEDDING_DIM == 1:
        title = f"Classification 1D (Checkpoint 1D)"
        visualize_classification_1d(model, test_loader, title=title)
