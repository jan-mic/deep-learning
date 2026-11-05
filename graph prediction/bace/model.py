import lightning.pytorch as pl
import torch
from torch.nn import Linear, ReLU, Sequential, ModuleList
from torch_geometric.nn import GCNConv, TransformerConv, global_mean_pool
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score


class MoleculeGNNLightningModel(pl.LightningModule):
    def __init__(self,
                 num_node_features,
                 class_weights=None,
                 hidden_dim=32,
                 embedding_dim=2,
                 num_classes=2,
                 gnn_type='GCN',
                 head_type='mlp',
                 num_layers=2,
                 mlp_hidden_dim=16,
                 lr=0.001,
                 dropout=0.3,
                 after_train_eval=False):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.dropout = torch.nn.Dropout(p=dropout)

        self.convs = ModuleList()

        if gnn_type == 'GCN':
            LayerClass = GCNConv
        elif gnn_type == 'Transformer':
            LayerClass = TransformerConv

        for i in range(num_layers):
            if i == 0:
                self.convs.append(LayerClass(num_node_features, hidden_dim))
            elif i == num_layers - 1:
                self.convs.append(LayerClass(hidden_dim, embedding_dim))
            else:
                self.convs.append(LayerClass(hidden_dim, hidden_dim))

        if head_type == 'linear':
            self.head = Linear(embedding_dim, num_classes)
        elif head_type == 'mlp':
            self.head = Sequential(
                Linear(embedding_dim, mlp_hidden_dim),
                ReLU(),
                torch.nn.Dropout(p=dropout),
                Linear(mlp_hidden_dim, num_classes)
            )

        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.val_prec = BinaryPrecision()
        self.val_rec = BinaryRecall()
        self.test_acc = BinaryAccuracy()
        self.test_prec = BinaryPrecision()
        self.test_rec = BinaryRecall()
        self.test_f1 = BinaryF1Score()

    def forward(self, x, edge_index, batch):
        x = x.float()

        for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                x = x.relu()
                x = self.dropout(x)

        pooled = global_mean_pool(x, batch)
        out = self.head(pooled)

        return out, pooled

    def training_step(self, batch, batch_idx):
        out, _ = self(batch.x, batch.edge_index, batch.batch)
        loss = self.criterion(out, batch.y.squeeze().long())

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        self.train_acc(out.argmax(dim=1), batch.y.squeeze().long())
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
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
                batch.to(self.device)
                out, _ = self(batch.x, batch.edge_index, batch.batch)
                loss = self.criterion(out, batch.y.squeeze().long())
                total_loss += loss.item()
                num_batches += 1
            if num_batches > 0:
                avg_train_loss = total_loss / num_batches
                self.log("train_loss_eval_mode", avg_train_loss, prog_bar=False, logger=True)
        self.train()

    def validation_step(self, batch, batch_idx):
        out, _ = self(batch.x, batch.edge_index, batch.batch)
        loss = self.criterion(out, batch.y.squeeze().long())
        preds = out.argmax(dim=1)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True, batch_size=batch.num_graphs)
        self.val_acc(preds, batch.y.squeeze().long())
        self.log('val_acc', self.val_acc, prog_bar=True, on_epoch=True, batch_size=batch.num_graphs)

    def test_step(self, batch, batch_idx):
        out, _ = self(batch.x, batch.edge_index, batch.batch)
        y = batch.y.squeeze().long()
        preds = out.argmax(dim=1)
        loss = self.criterion(out, y)

        self.test_acc(preds, y)
        self.test_prec(preds, y)
        self.test_rec(preds, y)
        self.test_f1(preds, y)

        self.log_dict({
            "test_loss": loss,
            "test_acc": self.test_acc,
            "test_prec": self.test_prec,
            "test_rec": self.test_rec,
            "test_f1": self.test_f1,
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
