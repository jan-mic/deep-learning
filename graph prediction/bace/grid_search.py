import numpy as np
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import lightning as pl
from lightning.pytorch.loggers import CSVLogger

from model import MoleculeGNNLightningModel
from train import process_data


def grid_search():
    loaders, num_features, class_weights = process_data()

    gnn_types = ['Transformer']
    head_types = ['mlp', 'linear']
    num_layers = [2]
    embedding_dim = [2]
    hidden_dim = [16, 24, 32]
    # mlp_hidden_dim = [16, 24, 32]
    learning_rates = [0.005, 0.001, 0.0001]
    DROPOUT = 0.1
    PATIENCE = 10
    EPOCHS = 100
    best_params = {
        "GNN_TYPE": None,
        "HEAD_TYPE": None,
        "NUM_LAYERS": None,
        "EMBEDDING_DIM": None,
        "HIDDEN_DIM": None,
        "MLP_HIDDEN_DIM": None,
        "LR": None
    }
    best_params_embed2 = {
        "GNN_TYPE": None,
        "HEAD_TYPE": None,
        "NUM_LAYERS": None,
        "EMBEDDING_DIM": None,
        "HIDDEN_DIM": None,
        "MLP_HIDDEN_DIM": None,
        "LR": None
    }
    best_params_embed1 = {
        "GNN_TYPE": None,
        "HEAD_TYPE": None,
        "NUM_LAYERS": None,
        "EMBEDDING_DIM": None,
        "HIDDEN_DIM": None,
        "MLP_HIDDEN_DIM": None,
        "LR": None
    }
    best_test_acc2 = 0.0
    best_test_acc = 0.0
    best_exp_name = None
    best_test_acc1 = 0.0

    for GNN_TYPE in gnn_types:
        for HEAD_TYPE in head_types:
            for NUM_LAYERS in num_layers:
                for EMBEDDING_DIM in embedding_dim:
                    for HIDDEN_DIM in hidden_dim:
                        # for MLP_HIDDEN_DIM in mlp_hidden_dim:
                        for LR in learning_rates:
                            exp_name = f"bace_{GNN_TYPE}_L{NUM_LAYERS}_{HEAD_TYPE}_dim{EMBEDDING_DIM}_hidden_dim{HIDDEN_DIM}_mlphiddendim_{HIDDEN_DIM}_lr{LR}"
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
                            print(f"--- Uruchamianie eksperymentu: {exp_name} ---")
                            model = MoleculeGNNLightningModel(
                                num_node_features=num_features,
                                class_weights=class_weights,
                                hidden_dim=HIDDEN_DIM,
                                embedding_dim=EMBEDDING_DIM,
                                num_classes=2,
                                gnn_type=GNN_TYPE,
                                head_type=HEAD_TYPE,
                                num_layers=NUM_LAYERS,
                                mlp_hidden_dim=HIDDEN_DIM,
                                lr=LR,
                                dropout=DROPOUT
                            )
                            # 5. Trainer
                            trainer = pl.Trainer(
                                max_epochs=EPOCHS,
                                callbacks=[checkpoint_callback, early_stop_callback],
                                logger=logger,
                                accelerator="auto",
                                devices=1,
                                log_every_n_steps=10
                            )
                            # 6. Trening
                            trainer.fit(model, loaders['train'], loaders['val'])

                            # 7. Testowanie
                            ckpt_path = checkpoint_callback.best_model_path
                            print("\n--- Wyniki Testu (model po treningu) ---")
                            test_results = trainer.test(model=model, dataloaders=loaders['test'], ckpt_path=ckpt_path,
                                                        weights_only=False)

                            # Lightning zwraca listę dictów (zwykle długości 1)
                            test_dict = test_results[0] if test_results else {}
                            test_acc = (
                                    test_dict.get('test_acc')
                                    or test_dict.get('acc')
                                    or test_dict.get('accuracy')
                            )

                            if test_acc is None:
                                print(f"Brak metryki accuracy w wynikach testu: {test_dict.keys()}")
                                continue

                            # Najlepszy ogólny (bez dim=1/2)
                            if test_acc > best_test_acc and EMBEDDING_DIM not in (1, 2):
                                best_test_acc = float(test_acc)
                                best_exp_name = exp_name
                                best_params = {
                                    "GNN_TYPE": GNN_TYPE,
                                    "HEAD_TYPE": HEAD_TYPE,
                                    "NUM_LAYERS": NUM_LAYERS,
                                    "EMBEDDING_DIM": EMBEDDING_DIM,
                                    "HIDDEN_DIM": HIDDEN_DIM,
                                    "MLP_HIDDEN_DIM": HIDDEN_DIM,
                                    "LR": float(LR)
                                }

                            if EMBEDDING_DIM == 1:
                                if test_acc > best_test_acc1:
                                    best_test_acc1 = float(test_acc)
                                    best_params_embed1 = {
                                        "GNN_TYPE": GNN_TYPE,
                                        "HEAD_TYPE": HEAD_TYPE,
                                        "NUM_LAYERS": NUM_LAYERS,
                                        "EMBEDDING_DIM": EMBEDDING_DIM,
                                        "HIDDEN_DIM": HIDDEN_DIM,
                                        "MLP_HIDDEN_DIM": HIDDEN_DIM,
                                        "LR": float(LR)
                                    }

                            if EMBEDDING_DIM == 2:
                                if test_acc > best_test_acc2:
                                    best_test_acc2 = float(test_acc)
                                    best_params_embed2 = {
                                        "GNN_TYPE": GNN_TYPE,
                                        "HEAD_TYPE": HEAD_TYPE,
                                        "NUM_LAYERS": NUM_LAYERS,
                                        "EMBEDDING_DIM": EMBEDDING_DIM,
                                        "HIDDEN_DIM": HIDDEN_DIM,
                                        "MLP_HIDDEN_DIM": HIDDEN_DIM,
                                        "LR": float(LR)
                                    }

    print(f"Best Test Accuracy: {best_test_acc} with params: {best_params}")
    # save to file
    with open("best_params.txt", "w") as f:
        f.write(f"Best Experiment Name: {best_exp_name}\n")
        f.write(f"Best Test Accuracy (General): {best_test_acc} with params: {best_params}\n")
        f.write(f"Best Test Accuracy (Embedding Dim 1): {best_test_acc1} with params: {best_params_embed1}\n")
        f.write(f"Best Test Accuracy (Embedding Dim 2): {best_test_acc2} with params: {best_params_embed2}\n")
    return best_params, best_params_embed1, best_params_embed2


if __name__ == "__main__":
    results = grid_search()
    print(results)
