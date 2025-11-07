import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys
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

from models.transformer import EncoderDecoderTransformer
from data.seq2seq_dataloader import get_seq2seq_dataset_loader
from utils.seq2seq_trainer import Seq2SeqTrainer

# Seq2Seq配置
SEQ2SEQ_CONFIG = {
    # Model configuration
    'd_model': 64,
    'num_heads': 4,
    'num_encoder_layers': 2,
    'num_decoder_layers': 2,
    'd_ff': 256,
    'max_seq_len': 30,
    'dropout': 0.1,

    # Training configuration
    'batch_size': 16,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'num_epochs': 10, 

    'dataset': "reliable",
    'num_samples': 800,
}


def setup_results_dir(config):
    """创建结果目录结构"""
    # 基础结果目录
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # 按时间创建子目录，避免覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = config.get('dataset', 'reliable')
    exp_dir = os.path.join(results_dir, f"seq2seq_{dataset_name}_exp_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # 创建子目录
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)

    return exp_dir


def decode_sequence(seq, dataset):
    """解码序列为可读字符串"""
    # 对于翻译数据集，直接返回字符
    if hasattr(dataset, 'tgt_tokenizer'):
        chars = []
        for idx in seq.tolist():
            if idx in [0, 1, 2]:  # 忽略特殊token
                continue
            if idx in dataset.tgt_tokenizer.idx_to_char:
                chars.append(dataset.tgt_tokenizer.idx_to_char[idx])
        return ''.join(chars)
    else:
        # 对于原来的数字序列数据集
        return ' '.join(str(x) for x in seq.tolist() if x not in [0, 1, 2])


def test_seq2seq_model(model, dataset, device, results_dir, num_tests=5):
    """测试seq2seq模型并保存结果"""
    model.eval()

    test_results = []

    print("\n" + "=" * 50)
    print("Seq2Seq模型测试结果:")
    print("=" * 50)

    with torch.no_grad():
        for i in range(min(num_tests, len(dataset))):
            sample = dataset[i]

            # 提取源序列和目标序列
            src_seq = sample['src'].tolist() if hasattr(sample['src'], 'tolist') else sample['src']
            tgt_seq = sample['tgt_output'][1:-1].tolist() if hasattr(sample['tgt_output'], 'tolist') else sample['tgt_output'][1:-1]

            # 转换为tensor
            src = torch.tensor(src_seq, device=device).unsqueeze(0)
            tgt_input = torch.tensor([dataset.sos_token] + tgt_seq, device=device).unsqueeze(0)

            # 创建掩码
            src_mask = (src != dataset.pad_token).unsqueeze(1).unsqueeze(2)
            tgt_len = tgt_input.size(1)
            tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).unsqueeze(0).unsqueeze(0)

            # 模型预测
            output = model(
                src,
                tgt_input[:, :-1],
                src_mask,
                tgt_mask[:, :, :-1, :-1]
            )

            # 获取预测结果
            predicted = output.argmax(dim=-1).squeeze(0)

            input_str = decode_sequence(src.squeeze(0), dataset)
            expected_str = decode_sequence(torch.tensor(tgt_seq), dataset)
            predicted_str = decode_sequence(predicted, dataset)

            # 相似度计算
            if hasattr(dataset, 'tgt_tokenizer'):
                expected_chars = set(expected_str.lower().replace(' ', ''))
                predicted_chars = set(predicted_str.lower().replace(' ', ''))
                common_chars = expected_chars.intersection(predicted_chars)
                similarity = len(common_chars) / max(len(expected_chars), 1)
                is_correct = similarity > 0.5
            else:
                is_correct = (expected_str == predicted_str)
                similarity = 1.0 if is_correct else 0.0

            test_results.append({
                'sample_id': i + 1,
                'input': input_str,
                'expected': expected_str,
                'predicted': predicted_str,
                'correct': is_correct,
                'similarity': similarity
            })

            print(f"样本 {i + 1}:")
            print(f"  输入: {input_str}")
            print(f"  期望输出: {expected_str}")
            print(f"  模型预测: {predicted_str}")
            if hasattr(dataset, 'tgt_tokenizer'):
                print(f"  相似度: {similarity:.2%}")
            print(f"  是否正确: {'✓' if is_correct else '✗'}")
            print()

    # 保存测试结果
    test_df = pd.DataFrame(test_results)
    test_csv_path = os.path.join(results_dir, "test_results.csv")
    test_df.to_csv(test_csv_path, index=False)

    accuracy = sum(test_df['correct']) / len(test_df)
    summary = {
        'test_accuracy': accuracy,
        'total_test_samples': len(test_df),
        'correct_predictions': sum(test_df['correct']),
        'test_details_path': test_csv_path
    }

    if 'similarity' in test_df.columns:
        summary['average_similarity'] = test_df['similarity'].mean()

    summary_path = os.path.join(results_dir, "test_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"测试结果已保存至: {test_csv_path}")
    print(f"测试准确率: {accuracy:.2%}")
    if 'similarity' in test_df.columns:
        print(f"平均相似度: {test_df['similarity'].mean():.2%}")

    return test_df, accuracy


def main(config):
    # 设置结果目录
    results_dir = setup_results_dir(config)
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    print(f"结果将保存至: {results_dir}")
    print(f"数据集: {config.get('dataset', 'reliable')}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading {config.get('dataset', 'reliable')} dataset...")
    dataset_loader = get_seq2seq_dataset_loader(config.get('dataset', 'reliable'))

    # 使用与数据集加载器相同的参数
    train_loader, val_loader, dataset = dataset_loader(
        batch_size=config['batch_size'],
        max_len=config['max_seq_len'],
        num_samples=config['num_samples']
    )

    # 获取词汇表大小
    vocab_size = dataset.vocab_size

    # Create model
    print("Creating Encoder-Decoder Transformer...")
    model = EncoderDecoderTransformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Encoder layers: {config['num_encoder_layers']}")
    print(f"Decoder layers: {config['num_decoder_layers']}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Max sequence length: {config['max_seq_len']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Training samples: {config['num_samples']}")

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01)
    )

    trainer = Seq2SeqTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config,
        results_dir=results_dir
    )

    # 保存实验配置
    trainer.save_experiment_config()

    print("Starting Seq2Seq training...")
    trainer.train(
        num_epochs=config['num_epochs'],
        save_dir=checkpoints_dir
    )

    # 保存所有结果
    trainer.plot_losses()
    trainer.save_training_results()
    trainer.save_final_model(model)

    # 测试模型
    test_df, accuracy = test_seq2seq_model(model, dataset, device, results_dir)

    # 训练统计
    print("\n" + "=" * 50)
    print("Seq2Seq训练统计:")
    print("=" * 50)
    print(f"最终训练损失: {trainer.train_losses[-1]:.4f}")
    print(f"最终验证损失: {trainer.val_losses[-1]:.4f}")
    print(f"最佳验证损失: {trainer.best_val_loss:.4f}")
    print(f"测试准确率: {accuracy:.2%}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n所有结果已保存至: {results_dir}")

    # 保存训练统计摘要
    training_summary = {
        'final_train_loss': float(trainer.train_losses[-1]),
        'final_val_loss': float(trainer.val_losses[-1]),
        'best_val_loss': float(trainer.best_val_loss),
        'test_accuracy': float(accuracy),
        'model_parameters': int(sum(p.numel() for p in model.parameters())),
        'training_epochs': len(trainer.train_losses),
        'config': config
    }

    summary_path = os.path.join(results_dir, "training_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(training_summary, f, indent=2, ensure_ascii=False)

    print(f"训练摘要已保存至: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset', type=str, default='reliable',
                        choices=['reliable', 'builtin'],
                        help='Dataset to use')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=800, help='Number of samples')
    parser.add_argument('--max_seq_len', type=int, default=30, help='Max sequence length')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # 使用Seq2Seq配置
    config = SEQ2SEQ_CONFIG.copy()
    config['dataset'] = args.dataset
    config['num_epochs'] = args.num_epochs
    config['batch_size'] = args.batch_size
    config['num_samples'] = args.num_samples
    config['max_seq_len'] = args.max_seq_len

    main(config)
