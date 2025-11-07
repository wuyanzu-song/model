
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
from datetime import datetime


class Trainer:

    def __init__(self, model, train_loader, val_loader, optimizer, criterion,
                 device, config, results_dir=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.results_dir = results_dir  

        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []  
        self.best_val_loss = float('inf')

    def train_epoch(self):
   
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc="Training")

        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)

            seq_len = data.size(1)
            mask = self._create_causal_mask(seq_len)

            self.optimizer.zero_grad()

            outputs = self.model(data, mask)
            loss = self.criterion(outputs.view(-1, outputs.size(-1)),
                                  targets.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                seq_len = data.size(1)
                mask = self._create_causal_mask(seq_len)

                outputs = self.model(data, mask)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)),
                                      targets.view(-1))
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def _create_causal_mask(self, seq_len):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
        return mask.unsqueeze(0)  # [1, seq_len, seq_len]

    def train(self, num_epochs, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)

        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')

            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            val_loss = self.validate()
            self.val_losses.append(val_loss)

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            self.learning_rates.append(current_lr)

            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}')

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, 'best_model.pt'))

            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pt'))

    def plot_losses(self, save_path=None):
        if save_path is None and self.results_dir:
            save_path = os.path.join(self.results_dir, "training_curves.png")
        elif save_path is None:
            save_path = 'training_loss.png'

        plt.figure(figsize=(12, 8))

        # 绘制损失曲线
        plt.subplot(2, 1, 1)
        plt.plot(self.train_losses, label='Training Loss', linewidth=2)
        plt.plot(self.val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 绘制学习率变化
        plt.subplot(2, 1, 2)
        plt.plot(self.learning_rates, label='Learning Rate', color='red', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # 对数尺度更好地显示学习率变化

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"训练曲线已保存至: {save_path}")

    def save_training_results(self):
        """保存训练结果到CSV文件"""
        if not self.results_dir:
            print("警告: 未设置results_dir，无法保存训练结果")
            return None

        results_data = {
            'epoch': list(range(1, len(self.train_losses) + 1)),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'learning_rate': self.learning_rates
        }

        df = pd.DataFrame(results_data)
        csv_path = os.path.join(self.results_dir, "training_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"训练结果表格已保存至: {csv_path}")
        return df

    def save_experiment_config(self):
        """保存实验配置到JSON文件"""
        if not self.results_dir:
            return

        config_path = os.path.join(self.results_dir, "experiment_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        print(f"实验配置已保存至: {config_path}")

    def save_final_model(self, model):
        """保存最终模型"""
        if not self.results_dir:
            return

        model_path = os.path.join(self.results_dir, "final_model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"最终模型已保存至: {model_path}")

    def save_training_summary(self):
        """保存训练摘要"""
        if not self.results_dir:
            return

        summary = {
            'final_train_loss': float(self.train_losses[-1]) if self.train_losses else None,
            'final_val_loss': float(self.val_losses[-1]) if self.val_losses else None,
            'best_val_loss': float(self.best_val_loss),
            'final_learning_rate': float(self.learning_rates[-1]) if self.learning_rates else None,
            'total_epochs': len(self.train_losses),
            'config': self.config
        }

        summary_path = os.path.join(self.results_dir, "training_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"训练摘要已保存至: {summary_path}")
        return summary
