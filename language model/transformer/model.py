import numpy as np
import torch
import lightning as pl
from torch import nn
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.text import Perplexity


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        embedding_index = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(embedding_index * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerTadeusz(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.encoder = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.vocab_size = vocab_size
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        sz = x.size(1)
        mask = torch.triu(torch.ones((sz, sz), device=x.device), diagonal=1).bool()
        x = self.encoder(x) * np.sqrt(self.d_model)
        x = self.pos_encoder(x)
        output = self.transformer_decoder(
            tgt=x,
            memory=x,
            tgt_mask=mask,
            memory_mask=mask
        )
        return self.decoder(output)


class TadeuszTransformerModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
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
        logits = self.model(x)

        loss = self.criterion(logits.view(-1, self.model.vocab_size), y.view(-1))

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self.model(x)

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

        logits = self.model(x)

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
