import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule
from torch import optim
from torchmetrics import F1
from transformers import ViTFeatureExtractor, ViTModel


class DescriptionModel(LightningModule):
    def __init__(self, hidden_dim=768, individual_logs=None):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
   
        self.emotion_head = nn.Linear(hidden_dim, 7)

        self.loss = nn.CrossEntropyLoss()
        self.train_f1 = F1()
        self.val_f1 = F1()
        self.test_f1 = F1()
        self.individual_logs = individual_logs
        self.tta_logs = defaultdict(list)

    def forward(self, x):
        x = self.vit(x).pooler_output
        # x = self.sigmoid(x)
        emotion = self.emotion_head(x)

        return emotion

    def run_batch(self, batch, batch_idx, metric, training=False):
        name, image_features, labels = batch[0], batch[1], batch[2:]
        print("name, " , name)
        print("labels", labels[0])
        image_features = image_features
        # labels = [label. for label in labels]

        emotion_label = labels[0]
        emotion = self(image_features)

        try:
            emotion_loss = self.loss(emotion, emotion_label)
            loss = emotion_loss 
            metric.update(emotion, emotion_label)
            
            f1 = metric.compute()
            print("f1", f1)
            tp, fp, tn, fn = metric._get_final_stats()
            # precision = tp / (tp + fp) if tp+fp else 0
            # recall = tp / (tp + fn) if tp+fn else 0
            self.tta_logs[name[0]].append((tp, fp, fn))
        except Exception as e:
            print(e)
            loss = 0

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.run_batch(batch, batch_idx, self.train_f1, training=True)
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, training_step_outputs):
        self.log("train_f1", self.train_f1.compute())
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        loss = self.run_batch(batch, batch_idx, self.val_f1)
        self.log("val_loss", loss)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        self.log("val_f1", self.val_f1.compute())
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        loss = self.run_batch(batch, batch_idx, self.test_f1)
        self.log("test_loss", loss)
        return loss

    def test_epoch_end(self, outputs):
        f1 = self.test_f1.compute()
        self.log("test_f1", f1)
        print(f"Test f1: {f1}")
        tp, fp, tn, fn = self.test_f1._get_final_stats()
        print(f"\nTest f1: {f1}, TP: {tp}, FP: {fp}, TN: {tn}, fn: {fn}")
        self.test_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--argument", help="Example argument.")
    args = parser.parse_args()
