import torch
import lightning as pl
from torch import nn
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.text import Perplexity


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, ninp, nhid, nlayers=2, dropout=0.2):
        super().__init__()
        self.encoder = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=ninp)
        self.lstm = torch.nn.LSTM(input_size=ninp, hidden_size=nhid, num_layers=nlayers, dropout=dropout,
                                  batch_first=True)
        self.decoder = torch.nn.Linear(in_features=nhid, out_features=vocab_size)

        self.nlayers = nlayers
        self.nhid = nhid
        self.ninp = ninp
        self.vocab_size = vocab_size

    def forward(self, x, hidden=None):
        emb = self.encoder(x)
        output, hidden = self.lstm(emb, hidden)
        decoded = self.decoder(output)
        return decoded, hidden


class TadeuszLSTM(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        # self.hidden = None
        self.criterion = torch.nn.CrossEntropyLoss()

        self.val_perplexity = Perplexity()
        self.val_top3_acc = MulticlassAccuracy(
            num_classes=self.model.vocab_size,
            top_k=3,
            average='micro'
        )
        self.val_acc = MulticlassAccuracy(
            num_classes=self.model.vocab_size,
            top_k=1,
            average='micro'
        )

        self.test_perplexity = Perplexity()
        self.test_top3_acc = MulticlassAccuracy(num_classes=self.model.vocab_size, top_k=3, average='micro')
        self.test_acc = MulticlassAccuracy(num_classes=self.model.vocab_size, top_k=1, average='micro')

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits, _ = self.model(x)

        loss = self.criterion(logits.view(-1, self.model.vocab_size), y.view(-1))

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits, _ = self.model(x)

        flat_logits = logits.reshape(-1, self.model.vocab_size)
        flat_y = y.reshape(-1)

        loss = self.criterion(flat_logits, flat_y)

        self.val_perplexity(logits, y)
        self.val_top3_acc(flat_logits, flat_y)
        self.val_acc(flat_logits, flat_y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_ppl", self.val_perplexity, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_top3_acc", self.val_top3_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits, _ = self.model(x)

        flat_logits = logits.reshape(-1, self.model.vocab_size)
        flat_y = y.reshape(-1)

        loss = self.criterion(flat_logits, flat_y)

        self.test_perplexity(logits, y)
        self.test_top3_acc(flat_logits, flat_y)
        self.test_acc(flat_logits, flat_y)

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_ppl", self.test_perplexity, on_epoch=True)
        self.log("test_top3_acc", self.test_top3_acc, on_epoch=True)
        self.log("test_acc", self.test_acc, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "val_loss",
                }
                }
