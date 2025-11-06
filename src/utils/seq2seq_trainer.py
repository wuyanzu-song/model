import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import json
import pandas as pd
from datetime import datetime


class Seq2SeqTrainer:
    """
    Sequence-to-Sequence Training utility class
    """

    def __init__(self, model, train_loader, val_loader, optimizer, criterion,
                 device, config, results_dir):  # 新增 results_dir 参数
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.results_dir = results_dir  # 新增：结果目录

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self):
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc="Seq2Seq Training")

        for batch_idx, batch in enumerate(progress_bar):
            src = batch['src'].to(self.device)
            tgt_input = batch['tgt_input'].to(self.device)
            tgt_output = batch['tgt_output'].to(self.device)
            src_mask = batch['src_mask'].to(self.device)
            tgt_mask = batch['tgt_mask'].to(self.device)

            # 调试信息（只在第一个batch显示）
            if batch_idx == 0:
                print(f"Batch {batch_idx}: src shape {src.shape}, tgt_input shape {tgt_input.shape}")
                print(f"src_mask shape {src_mask.shape}, tgt_mask shape {tgt_mask.shape}")

            self.optimizer.zero_grad()

            # Forward pass - 注意：解码器输入要去掉最后一个token
            outputs = self.model(
                src,
                tgt_input[:, :-1],  # 去掉EOS token
                src_mask,
                tgt_mask[:, :, :-1, :-1]  # 调整掩码大小
            )

            if batch_idx == 0:
                print(f"Output shape: {outputs.shape}")

            # Reshape for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))
            tgt_output = tgt_output[:, 1:].contiguous().view(-1)  # 去掉SOS token

            if batch_idx == 0:
                print(f"After reshape - outputs: {outputs.shape}, tgt_output: {tgt_output.shape}")

            # Compute loss (忽略padding token)
            loss = self.criterion(outputs, tgt_output)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(self.train_loader)

    def validate(self):
        """
        Validate the model
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                src = batch['src'].to(self.device)
                tgt_input = batch['tgt_input'].to(self.device)
                tgt_output = batch['tgt_output'].to(self.device)
                src_mask = batch['src_mask'].to(self.device)
                tgt_mask = batch['tgt_mask'].to(self.device)

                # 与训练时相同的处理
                outputs = self.model(
                    src,
                    tgt_input[:, :-1],  # 去掉EOS token
                    src_mask,
                    tgt_mask[:, :, :-1, :-1]  # 调整掩码大小
                )

                # Reshape for loss calculation
                outputs = outputs.view(-1, outputs.size(-1))
                tgt_output = tgt_output[:, 1:].contiguous().view(-1)  # 去掉SOS token

                loss = self.criterion(outputs, tgt_output)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self, num_epochs, save_dir='checkpoints_seq2seq'):
        """
        Full training loop
        """
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, 'best_model.pt'))

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pt'))

    def plot_losses(self, save_path=None):
        """
        Plot training and validation losses
        """
        if save_path is None:
            save_path = os.path.join(self.results_dir, "training_curves.png")

        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Seq2Seq Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"训练曲线已保存至: {save_path}")

    def save_training_results(self):
        """保存训练结果到CSV文件"""
        results_data = {
            'epoch': list(range(1, len(self.train_losses) + 1)),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }

        df = pd.DataFrame(results_data)
        csv_path = os.path.join(self.results_dir, "training_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"训练结果表格已保存至: {csv_path}")
        return df

    def save_experiment_config(self):
        """保存实验配置到JSON文件"""
        config_path = os.path.join(self.results_dir, "experiment_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        print(f"实验配置已保存至: {config_path}")

    def save_final_model(self, model):
        """保存最终模型"""
        model_path = os.path.join(self.results_dir, "final_model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"最终模型已保存至: {model_path}")