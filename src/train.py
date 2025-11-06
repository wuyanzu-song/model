import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import yaml
import os
import sys
import matplotlib.pyplot as plt
import json
import pandas as pd
from datetime import datetime

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（src的父目录）
project_root = os.path.dirname(current_dir)
# 添加项目根目录到Python路径
sys.path.append(project_root)
sys.path.append(current_dir)

from models.transformer import Transformer
from data.dataloader import get_dataset_loader
from utils.trainer import Trainer

# Configuration as Python dictionary
CONFIG = {
    # Model configuration
    'd_model': 64,
    'num_heads': 2,
    'num_layers': 1,
    'd_ff': 256,
    'max_seq_len': 128,
    'dropout': 0.1,

    # Training configuration
    'batch_size': 32,
    'seq_len': 128,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'num_epochs': 5,

    # Data configuration - 使用新数据集
    'dataset': "ag_news",
}


def setup_results_dir(config, experiment_type="main"):
    """创建结果目录结构"""
    # 基础结果目录
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # 按时间创建子目录，避免覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = config.get('dataset', 'ag_news')
    exp_dir = os.path.join(results_dir, f"{experiment_type}_{dataset_name}_exp_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # 创建子目录
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "ablation_studies"), exist_ok=True)

    return exp_dir


def run_ablation_study(base_config, dataset, results_dir):
    """
    运行消融实验，比较不同配置的性能
    """
    print("\n" + "=" * 60)
    print("开始消融实验")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 消融实验配置
    ablation_configs = {
        "base_model": {
            "d_model": base_config['d_model'],
            "num_heads": base_config['num_heads'],
            "num_layers": base_config['num_layers'],
            "use_pos_encoding": True,
            "description": "基准模型（完整配置）"
        },
        "no_positional_encoding": {
            "d_model": base_config['d_model'],
            "num_heads": base_config['num_heads'],
            "num_layers": base_config['num_layers'],
            "use_pos_encoding": False,
            "description": "无位置编码"
        },
        "single_head": {
            "d_model": base_config['d_model'],
            "num_heads": 1,
            "num_layers": base_config['num_layers'],
            "use_pos_encoding": True,
            "description": "单头注意力"
        },
        "more_heads": {
            "d_model": base_config['d_model'],
            "num_heads": base_config['num_heads'] * 2,
            "num_layers": base_config['num_layers'],
            "use_pos_encoding": True,
            "description": "更多注意力头"
        }
    }

    results = {}
    ablation_results_dir = os.path.join(results_dir, "ablation_studies")

    for name, model_config in ablation_configs.items():
        print(f"\n--- 训练模型: {model_config['description']} ---")

        # 创建模型
        model = Transformer(
            vocab_size=dataset.vocab_size,
            d_model=model_config['d_model'],
            num_heads=model_config['num_heads'],
            num_layers=model_config['num_layers'],
            d_ff=base_config['d_ff'],
            max_seq_len=base_config['max_seq_len'],
            dropout=base_config['dropout'],
            use_pos_encoding=model_config['use_pos_encoding']
        ).to(device)

        print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

        # 训练配置（使用更少的epochs进行快速实验）
        fast_config = base_config.copy()
        fast_config['num_epochs'] = 3  # 消融实验只训练3轮
        fast_config['batch_size'] = 16  # 减小批大小

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=fast_config['learning_rate'],
            weight_decay=fast_config.get('weight_decay', 0.01)
        )

        # 加载数据 - 使用新的数据集加载器
        dataset_loader = get_dataset_loader(base_config.get('dataset', 'ag_news'))
        train_loader, val_loader, _ = dataset_loader(
            batch_size=fast_config['batch_size'],
            seq_len=fast_config['seq_len']
        )

        # 为每个消融实验创建单独的结果目录
        ablation_exp_dir = os.path.join(ablation_results_dir, name)
        os.makedirs(ablation_exp_dir, exist_ok=True)

        # 创建trainer并训练
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            config=fast_config,
            results_dir=ablation_exp_dir  # 传入结果目录
        )

        # 保存消融实验配置
        trainer.save_experiment_config()

        # 训练模型
        trainer.train(num_epochs=fast_config['num_epochs'],
                      save_dir=os.path.join(ablation_exp_dir, "checkpoints"))

        # 保存消融实验结果
        trainer.plot_losses()
        trainer.save_training_results()
        trainer.save_training_summary()

        # 记录结果
        results[name] = {
            'final_train_loss': trainer.train_losses[-1],
            'final_val_loss': trainer.val_losses[-1],
            'best_val_loss': trainer.best_val_loss,
            'config': model_config,
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'results_dir': ablation_exp_dir
        }

        print(f"{model_config['description']} - 最终验证损失: {trainer.val_losses[-1]:.4f}")

    return results


def plot_ablation_results(results, save_path=None):
    """
    绘制消融实验结果图
    """
    if save_path is None:
        save_path = 'ablation_study.png'

    plt.figure(figsize=(12, 8))

    # 绘制验证损失曲线
    plt.subplot(2, 1, 1)
    for name, result in results.items():
        description = result['config']['description']
        plt.plot(result['val_losses'], label=description, linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Ablation Study - Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 绘制最终性能对比
    plt.subplot(2, 1, 2)
    model_names = []
    final_losses = []

    for name, result in results.items():
        model_names.append(result['config']['description'])
        final_losses.append(result['final_val_loss'])

    bars = plt.bar(model_names, final_losses, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    plt.ylabel('Final Validation Loss')
    plt.title('Ablation Study - Final Performance Comparison')
    plt.xticks(rotation=15)

    # 在柱状图上添加数值
    for bar, loss in zip(bars, final_losses):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{loss:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"消融实验图表已保存: {save_path}")


def generate_ablation_report(results, save_path='ablation_report.txt'):
    """
    生成消融实验报告
    """
    report = """消融实验报告
============

实验配置对比:
-------------
"""

    for name, result in results.items():
        config = result['config']
        report += f"""
{config['description']}:
  - 嵌入维度: {config['d_model']}
  - 注意力头数: {config['num_heads']}
  - 编码器层数: {config['num_layers']}
  - 使用位置编码: {config['use_pos_encoding']}
  - 最终验证损失: {result['final_val_loss']:.4f}
  - 最佳验证损失: {result['best_val_loss']:.4f}
  - 困惑度: {torch.exp(torch.tensor(result['final_val_loss'])).item():.2f}
  - 结果目录: {result.get('results_dir', 'N/A')}

"""

    # 性能对比分析
    report += """
性能分析:
---------
"""
    base_loss = results['base_model']['final_val_loss']

    for name, result in results.items():
        if name != 'base_model':
            relative_perf = (result['final_val_loss'] - base_loss) / base_loss * 100
            report += f"- {result['config']['description']}: "
            report += f"相对基准模型性能下降 {relative_perf:+.1f}%\n"

    report += f"""
关键发现:
---------
1. 位置编码的重要性: {('极高' if results['no_positional_encoding']['final_val_loss'] > base_loss + 0.1 else '较高')}
2. 多头注意力的效果: {('显著' if results['single_head']['final_val_loss'] > base_loss + 0.05 else '适中')}
3. 模型配置对性能的影响: {('很大' if max(r['final_val_loss'] for r in results.values()) - min(r['final_val_loss'] for r in results.values()) > 0.1 else '适中')}

图表文件:
---------
- 消融实验曲线: ablation_study.png
- 本报告: ablation_report.txt
"""

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"消融实验报告已保存: {save_path}")
    return report


def generate_sample_text(model, dataset, start_string="The ", length=100):
    """生成样本文本"""
    model.eval()
    chars = [ch for ch in start_string]

    with torch.no_grad():
        for _ in range(length):
            # 转换当前序列为tensor，处理未知字符
            input_chars = chars[-dataset.seq_len:]
            input_ids = []
            for ch in input_chars:
                if ch in dataset.char_to_idx:
                    input_ids.append(dataset.char_to_idx[ch])
                else:
                    # 如果字符不在词汇表中，使用空格代替
                    input_ids.append(dataset.char_to_idx.get(' ', 0))

            input_tensor = torch.tensor(input_ids, device=model.device).unsqueeze(0)

            # 创建mask
            seq_len = input_tensor.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=model.device)).unsqueeze(0)

            # 获取模型预测
            output = model(input_tensor, mask)
            last_logits = output[0, -1, :]

            # 从分布中采样
            probs = torch.softmax(last_logits, dim=-1)
            next_char_idx = torch.multinomial(probs, 1).item()

            # 确保生成的字符在词汇表内
            if next_char_idx < len(dataset.idx_to_char):
                next_char = dataset.idx_to_char[next_char_idx]
                chars.append(next_char)
            else:
                # 如果索引超出范围，使用空格
                chars.append(' ')

    return ''.join(chars)


def save_sample_texts(model, dataset, results_dir, num_samples=5):
    """保存生成的样本文本到文件"""
    samples = []

    print("\n生成样本文本...")
    for i in range(num_samples):
        start_text = "The " if i == 0 else f"Sample {i + 1}: "
        sample = generate_sample_text(model, dataset, start_text, 150)
        samples.append({
            'sample_id': i + 1,
            'start_text': start_text,
            'generated_text': sample
        })
        print(f"示例 {i + 1}: {sample[:100]}...")

    # 保存到JSON文件
    samples_path = os.path.join(results_dir, "generated_samples.json")
    with open(samples_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"生成的样本文本已保存至: {samples_path}")
    return samples


def main(config):
    # 设置结果目录
    results_dir = setup_results_dir(config, "transformer")
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    print(f"结果将保存至: {results_dir}")
    print(f"数据集: {config.get('dataset', 'ag_news')}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data - 使用新的数据集加载器
    print(f"Loading {config.get('dataset', 'ag_news')} dataset...")
    dataset_loader = get_dataset_loader(config.get('dataset', 'ag_news'))
    train_loader, val_loader, dataset = dataset_loader(
        batch_size=config['batch_size'],
        seq_len=config['seq_len']
    )

    # Create model
    print("Creating model...")
    model = Transformer(
        vocab_size=dataset.vocab_size,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
        use_pos_encoding=True  # 基准模型使用位置编码
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01)
    )

    # Create trainer (传入results_dir)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config,
        results_dir=results_dir  # 传入结果目录
    )

    # 保存实验配置
    trainer.save_experiment_config()

    # Train model
    print("Starting training...")
    trainer.train(
        num_epochs=config['num_epochs'],
        save_dir=checkpoints_dir  # 使用results目录下的checkpoints
    )

    # 保存所有结果
    trainer.plot_losses()
    trainer.save_training_results()
    trainer.save_final_model(model)
    trainer.save_training_summary()

    # 保存生成的样本文本
    samples = save_sample_texts(model, dataset, results_dir)

    # 训练统计
    print("\n" + "=" * 50)
    print("训练统计:")
    print("=" * 50)
    print(f"最终训练损失: {trainer.train_losses[-1]:.4f}")
    print(f"最终验证损失: {trainer.val_losses[-1]:.4f}")
    print(f"最佳验证损失: {trainer.best_val_loss:.4f}")
    print(f"困惑度: {torch.exp(torch.tensor(trainer.val_losses[-1])).item():.2f}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 运行消融实验
    ablation_results = run_ablation_study(config, dataset, results_dir)

    # 绘制消融实验结果并保存到结果目录
    ablation_plot_path = os.path.join(results_dir, "ablation_study.png")
    plot_ablation_results(ablation_results, ablation_plot_path)

    # 生成消融实验报告并保存到结果目录
    ablation_report_path = os.path.join(results_dir, "ablation_report.txt")
    ablation_report = generate_ablation_report(ablation_results, ablation_report_path)

    print("\n" + "=" * 60)
    print("所有实验完成!")
    print("=" * 60)
    print(f"主要结果目录: {results_dir}")
    print(f"消融实验报告:\n{ablation_report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset', type=str, default='ag_news',
                       choices=['ag_news', 'imdb'],
                       help='Dataset to use')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # 使用配置
    config = CONFIG.copy()
    config['dataset'] = args.dataset
    config['num_epochs'] = args.num_epochs
    config['batch_size'] = args.batch_size
    config['seq_len'] = args.seq_len

    main(config)