#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trainer for BiLSTM Trajectory Classification
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import opt
from models.lossFun import FocalLoss
import utils.metrics as metrics
import pytorch_warmup as warmup
import json


class Trainer():
    def __init__(self, model, mode='trajectory_only'):
        self.save_dir = opt.save_dir
        self.data_dir = opt.data_dir
        self.batch_size = opt.batch_size
        self.number_workers = opt.number_workers
        self.time_tri = opt.time_tri
        self.device = opt.device
        self.model_name = opt.NAME
        self.accumulation_step = opt.accumulation_step
        self.max_epoch = opt.epochs
        self.filename = opt.filename
        self.mode = mode

        os.makedirs(self.save_dir, exist_ok=True)

        # Load data based on mode
        if mode == 'trajectory_only':
            from utils.loader import get_loader_trajectory
            self.train_loader, self.valid_loader, self.test_loader, _ = get_loader_trajectory(
                self.data_dir, self.time_tri, self.batch_size, num_workers=self.number_workers
            )
        else:  # multimodal
            from utils.loader import get_loader_multimodal
            self.train_loader, self.valid_loader, self.test_loader, _ = get_loader_multimodal(
                self.data_dir, self.time_tri, self.batch_size, num_workers=self.number_workers
            )

        self.num_train = len(self.train_loader.dataset)
        self.num_valid = len(self.valid_loader.dataset)
        self.num_test = len(self.test_loader.dataset)

        print(f'Find {self.num_train} train samples, {self.num_valid} validation samples, {self.num_test} test samples')
        print(f'Batch size: {self.batch_size}')

        self.model = model.to(self.device)

        # Optimizer
        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=opt.LEARNING_RATE,
                weight_decay=opt.WEIGHT_DECAY
            )
        elif opt.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt.LEARNING_RATE,
                betas=(0.9, 0.999),
                weight_decay=opt.WEIGHT_DECAY
            )
        else:
            raise NotImplementedError(f"Optimizer {opt.optimizer} not implemented")

        # Learning rate scheduler
        if opt.lrsc == "warmup":
            self.lrsc = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=len(self.train_loader) * self.max_epoch
            )
        else:
            self.lrsc = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.1, patience=3,
                verbose=True, threshold=0.0001, threshold_mode='rel'
            )

        # Loss function
        if opt.loss == "Focal":
            self.loss_fn = FocalLoss(logits=True, alpha=opt.ALPHA, gamma=opt.GAMMA)
        elif opt.loss == "CE":
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"Loss {opt.loss} not implemented")

        self.start_epoch = 0
        self.best_loss = 1e10
        self.best_acc = -1

        # History tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        if opt.resume:
            if os.path.isfile(opt.resume_path):
                self.resume(opt.resume_path, load_optimizer=True)
            else:
                print("⚠ Checkpoint not found")

    def resume(self, path, load_optimizer=True):
        print(f"Resuming from {path}")
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.best_acc = checkpoint.get('best_acc', -1)
        self.model.load_state_dict(checkpoint['state_dict'])
        if "optimizer" in checkpoint.keys() and load_optimizer:
            print("Loading optimizer state dict")
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if "history" in checkpoint.keys():
            self.history = checkpoint['history']

    def save_checkpoint(self, epoch, save_optimizer=True, suffix=""):
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "best_loss": self.best_loss,
            "best_acc": self.best_acc,
            "history": self.history
        }
        if save_optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()

        save_path = os.path.join(self.save_dir, 'weights', f"{self.model_name}_{suffix}.pth")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f"✓ Saved checkpoint: {save_path}")

    def train(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()

        if opt.lrsc == "warmup":
            warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)

        y_predict = torch.tensor(()).to(self.device)
        y_true = torch.tensor(()).to(self.device)
        total_losses = 0

        tbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.max_epoch} [Train]")
        for batch_idx, data in enumerate(tbar):
            if self.mode == 'trajectory_only':
                X, y = data
            else:  # multimodal
                X_traj, X_img, y = data
                X = (X_traj, X_img)

            # One-hot encoding for Focal Loss
            onehot_target = torch.eye(opt.n_classes)[y.long().cpu(), :].to(self.device)

            # Forward pass
            pred = self.model(X) if self.mode == 'trajectory_only' else self.model(X_traj, X_img)
            y_predict = torch.cat([y_predict, pred.argmax(1)], dim=0)
            y_true = torch.cat([y_true, y], dim=0)

            # Compute loss
            loss = self.loss_fn(pred, onehot_target)
            total_losses += float(loss)

            # Gradient accumulation
            loss /= self.accumulation_step
            loss.backward()

            if (batch_idx + 1) % self.accumulation_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            tbar.set_postfix({'loss': total_losses / (batch_idx + 1)})

        # Compute metrics
        met = metrics.scores(y_true.cpu().numpy(), y_predict.cpu().numpy())
        return total_losses / len(self.train_loader), met

    def valid(self, dataloader, epoch, phase='Val'):
        self.model.eval()
        test_loss = 0
        y_predict = torch.tensor(()).to(self.device)
        y_true = torch.tensor(()).to(self.device)

        tbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.max_epoch} [{phase}]")
        for batch_idx, data in enumerate(tbar):
            if self.mode == 'trajectory_only':
                X, y = data
            else:  # multimodal
                X_traj, X_img, y = data
                X = (X_traj, X_img)

            with torch.no_grad():
                onehot_target = torch.eye(opt.n_classes)[y.long().cpu(), :].to(self.device)
                pred = self.model(X) if self.mode == 'trajectory_only' else self.model(X_traj, X_img)
                y_predict = torch.cat([y_predict, pred.argmax(1)], dim=0)
                y_true = torch.cat([y_true, y], dim=0)
                test_loss += self.loss_fn(pred, onehot_target).item()

        test_loss /= len(dataloader)
        met = metrics.scores(y_true.cpu().numpy(), y_predict.cpu().numpy())

        return test_loss, met

    def start_train(self):
        print("\n" + "="*70)
        print(f"Training: {self.model_name}")
        print(f"Mode: {self.mode}")
        print("="*70 + "\n")

        pbar = tqdm(total=self.max_epoch - self.start_epoch)

        if opt.lrsc == "warmup":
            warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)

        for epoch in range(self.start_epoch, self.max_epoch):
            # Training
            train_loss, train_met = self.train(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_met['accuracy'])

            # Validation
            val_loss, val_met = self.valid(self.valid_loader, epoch, phase='Val')
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_met['accuracy'])

            # Learning rate scheduling
            if opt.lrsc == "warmup":
                with warmup_scheduler.dampening():
                    self.lrsc.step()
            else:
                self.lrsc.step(val_loss)

            # Save best model
            if val_met['accuracy'] > self.best_acc:
                self.best_acc = val_met['accuracy']
                self.best_loss = val_loss
                self.save_checkpoint(epoch, save_optimizer=True, suffix="best")

            # Periodic checkpoint
            if epoch >= self.max_epoch // 2 and epoch % 10 == 0:
                self.save_checkpoint(epoch, save_optimizer=True, suffix=f"epoch{epoch}")

            pbar.set_postfix({
                'train_acc': f"{train_met['accuracy']:.2f}%",
                'val_acc': f"{val_met['accuracy']:.2f}%"
            })
            pbar.update()

        # Test evaluation
        test_loss, test_met = self.valid(self.test_loader, 0, phase='Test')

        # Save results
        results = {
            'mode': self.mode,
            'best_val_acc': self.best_acc,
            'final_train_acc': train_met['accuracy'],
            'final_val_acc': val_met['accuracy'],
            'test_acc': test_met['accuracy'],
            'test_metrics': test_met,
            'history': self.history,
            'args': {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v
                    for k, v in vars(opt).items() if not k.startswith('_')}
        }

        results_file = os.path.join(self.save_dir, f'results_{self.mode}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {results_file}")

        # Print final results
        print("\n" + "="*70)
        print("Final Results:")
        print("="*70)
        print(f"Train Accuracy: {train_met['accuracy']:.2f}%")
        print(f"Validation Accuracy: {val_met['accuracy']:.2f}%")
        print(f"Test Accuracy: {test_met['accuracy']:.2f}%")
        print(f"Best Validation Accuracy: {self.best_acc:.2f}%")
        print("="*70 + "\n")
