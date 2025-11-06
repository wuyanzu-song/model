import torch
from torch.utils.data import Dataset, DataLoader
import os
import requests
import random


class CharDataset(Dataset):
    """
    Character-level dataset for language modeling
    """

    def __init__(self, text, seq_len=128):
        self.seq_len = seq_len
        self.text = text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        # Encode the entire text
        self.data = torch.tensor([self.char_to_idx[ch] for ch in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y


def load_ag_news_titles(batch_size=32, seq_len=64, split_ratio=0.9):
    """
    加载AG News标题数据集 - 优化为快速训练版本
    """
    print("Loading AG News Titles dataset (optimized for fast training)...")

    try:
        from datasets import load_dataset

        # 加载AG News数据集
        dataset = load_dataset("ag_news")

        texts = []
        # 大幅减少样本数量以加快训练
        max_samples = 800  # 从5000减少到800，大幅加快训练

        for split in ['train', 'test']:
            for example in dataset[split]:
                if len(texts) >= max_samples:
                    break

                # 使用'text'字段
                content = example['text']

                # 选择较短的内容，进一步加快训练
                if 20 < len(content) < 200:  # 选择中等长度的文本
                    texts.append(content + " ")

        # 合并所有文本
        text = "".join(texts)
        print(f"AG News Titles: {len(texts)} samples, {len(text)} characters")

    except Exception as e:
        print(f"Error loading AG News: {e}")
        # 回退文本 - 新闻标题风格（也减少数量）
        text = """
        Stocks rise amid economic recovery signs. 
        Tech companies report strong earnings.
        Climate summit reaches new agreement.
        Researchers make breakthrough in energy.
        Global markets show mixed results.
        New study reveals health benefits.
        Artificial intelligence transforms healthcare.
        Electric vehicle sales reach records.
        Scientists discover new species.
        Space agency launches satellite.
        Education adopts digital learning.
        Healthcare workers develop treatments.
        Renewable energy gets funding.
        Smartphone sales increase.
        Film industry celebrates awards.
        Tourism sector rebounds.
        Automotive companies invest.
        Financial institutions update security.
        Retail industry adapts.
        Agriculture implements practices.
        """ * 10  # 从20减少到10
        print("Using fallback news-style text")

    # 创建数据集
    full_dataset = CharDataset(text, seq_len)

    # 分割数据集
    total_size = len(full_dataset)
    train_size = int(total_size * split_ratio)
    val_size = total_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(f"Vocabulary size: {full_dataset.vocab_size}")

    return train_loader, val_loader, full_dataset


def load_imdb_reviews(batch_size=32, seq_len=64, split_ratio=0.9):
    """
    加载IMDB电影评论数据集 - 优化为快速训练版本
    """
    print("Loading IMDB Reviews dataset (optimized for fast training)...")

    try:
        from datasets import load_dataset

        dataset = load_dataset("imdb")

        texts = []
        max_samples = 600  # 从4000减少到600

        for split in ['train', 'test']:
            for example in dataset[split]:
                if len(texts) >= max_samples:
                    break
                review = example['text']
                if 30 < len(review) < 150:  # 选择更短的评论
                    texts.append(review + " ")

        text = "".join(texts)
        print(f"IMDB Reviews: {len(texts)} samples, {len(text)} characters")

    except Exception as e:
        print(f"Error loading IMDB: {e}")
        # 回退文本 - 电影评论风格（减少数量）
        text = """
        This movie was absolutely fantastic. 
        The plot was engaging and interesting.
        I would recommend this film.
        The cinematography was beautiful.
        Outstanding performance by cast.
        The story was heartwarming.
        This sequel improves the original.
        A must see film for cinema.
        The direction was superb.
        Visual effects were stunning.
        Character development was excellent.
        Emotional depth is remarkable.
        Comedy elements were humorous.
        Action sequences were thrilling.
        Romantic storyline was touching.
        Historical accuracy was authentic.
        Science fiction was creative.
        Horror scenes were scary.
        Documentary provided insights.
        Animation was visually stunning.
        """ * 8  # 从15减少到8
        print("Using fallback review-style text")

    full_dataset = CharDataset(text, seq_len)

    total_size = len(full_dataset)
    train_size = int(total_size * split_ratio)
    val_size = total_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(f"Vocabulary size: {full_dataset.vocab_size}")

    return train_loader, val_loader, full_dataset


def create_causal_mask(seq_len, device):
    """
    Create causal mask for autoregressive training
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0)  # [1, seq_len, seq_len]


# 数据集映射字典
DATASET_LOADERS = {
    'ag_news': load_ag_news_titles,
    'imdb': load_imdb_reviews,
}


def get_dataset_loader(dataset_name):
    """获取数据集加载函数"""
    return DATASET_LOADERS.get(dataset_name, load_ag_news_titles)