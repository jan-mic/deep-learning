import torch
from torch.nn import Linear, ReLU, Sequential, ModuleList, MSELoss
from torch_geometric.nn import GCNConv, TransformerConv, global_mean_pool
import lightning.pytorch as pl
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

class QM9Regressor(pl.LightningModule):
    def __init__(self,
                 num_node_features,
                 hidden_dim=32,
                 embedding_dim=2,
                 gnn_type='GCN',
                 head_type='mlp',
                 num_layers=2,
                 mlp_hidden_dim=32,
                 lr=0.001,
                 dropout=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.dropout = torch.nn.Dropout(p=dropout)

        self.convs = ModuleList()
        if gnn_type == 'GCN':
            LayerClass = GCNConv
            e_dim = None
        elif gnn_type == 'Transformer':
            LayerClass = TransformerConv
            e_dim = 4

        for i in range(num_layers):
            layer_kwargs = {}
            if e_dim is not None:
                layer_kwargs['edge_dim'] = e_dim

            if i == 0:
                self.convs.append(LayerClass(num_node_features, hidden_dim, **layer_kwargs))
            elif i == num_layers - 1:
                self.convs.append(LayerClass(hidden_dim, embedding_dim, **layer_kwargs))
            else:
                self.convs.append(LayerClass(hidden_dim, hidden_dim, **layer_kwargs))

        output_dim = 1

        if head_type == 'linear':
            self.head = Linear(embedding_dim, output_dim)
        elif head_type == 'mlp':
            self.head = Sequential(
                Linear(embedding_dim, mlp_hidden_dim),
                ReLU(),
                torch.nn.Dropout(p=dropout),
                Linear(mlp_hidden_dim, output_dim)
            )

        self.criterion = MSELoss()

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()
        self.test_mse = MeanSquaredError()
        self.test_r2 = R2Score()

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = x.float()
        if edge_attr is not None:
            edge_attr = edge_attr.float()

        for i, conv in enumerate(self.convs):
            if edge_attr is not None and isinstance(conv, TransformerConv):
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            x = x.relu()
            x = self.dropout(x)

        pooled = global_mean_pool(x, batch)
        out = self.head(pooled)

        return out, pooled

    def training_step(self, batch, batch_idx):
        y = batch.y.float().view(-1, 1)
        out, _ = self(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        loss = self.criterion(out, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        self.train_mae(out, y)
        self.log('train_mae', self.train_mae, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def on_train_epoch_end(self):
        if not self.hparams.after_train_eval:
            return
        self.eval()
        with torch.no_grad():
            train_loader = self.trainer.train_dataloader
            total_loss = 0
            num_batches = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                out, _ = self(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                y = batch.y.float().view(-1, 1)
                loss = self.criterion(out, y)
                total_loss += loss.item()
                num_batches += 1
            if num_batches > 0:
                avg_train_loss = total_loss / num_batches
                self.log("train_loss_eval_mode", avg_train_loss, prog_bar=False, logger=True)
        self.train()

    def validation_step(self, batch, batch_idx):
        y = batch.y.float().view(-1, 1)
        out, _ = self(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        loss = self.criterion(out, y)

        self.val_mae(out, y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, batch_size=batch.num_graphs)
        self.log('val_mae', self.val_mae, prog_bar=True, on_epoch=True, batch_size=batch.num_graphs)

    def test_step(self, batch, batch_idx):
        y = batch.y.float().view(-1, 1)
        out, _ = self(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        loss = self.criterion(out, y)

        self.test_mae(out, y)
        self.test_mse(out, y)
        self.test_r2(out, y)

        self.log_dict({
            "test_loss": loss,
            "test_mae": self.test_mae,
            "test_mse": self.test_mse,
        }, prog_bar=True, on_epoch=True, batch_size=batch.num_graphs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }