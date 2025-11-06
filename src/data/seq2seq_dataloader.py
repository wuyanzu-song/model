# import torch
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import random
#
#
# class ReverseStringDataset(Dataset):
#     """
#     简单的字符串反转任务数据集，用于测试Encoder-Decoder模型
#     """
#
#     def __init__(self, num_samples=1000, min_len=5, max_len=20, vocab_size=50):
#         self.num_samples = num_samples
#         self.min_len = min_len
#         self.max_len = max_len
#         self.vocab_size = vocab_size
#
#         # 特殊token
#         self.pad_token = 0
#         self.sos_token = 1  # Start of Sequence
#         self.eos_token = 2  # End of Sequence
#
#         # 生成数据
#         self.samples = self._generate_samples()
#
#     def _generate_samples(self):
#         samples = []
#         for _ in range(self.num_samples):
#             # 随机生成长度
#             length = random.randint(self.min_len, self.max_len)
#
#             # 生成源序列（随机整数序列）
#             src_seq = [random.randint(3, self.vocab_size - 1) for _ in range(length)]
#
#             # 目标序列是源序列的反转
#             tgt_seq = src_seq[::-1]
#
#             samples.append({
#                 'src': src_seq,
#                 'tgt': tgt_seq,
#                 'src_len': length,
#                 'tgt_len': length
#             })
#
#         return samples
#
#     def __len__(self):
#         return self.num_samples
#
#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#
#         # 源序列：直接使用
#         src = torch.tensor(sample['src'], dtype=torch.long)
#
#         # 目标序列：添加SOS和EOS token
#         tgt_input = torch.tensor([self.sos_token] + sample['tgt'], dtype=torch.long)
#         tgt_output = torch.tensor(sample['tgt'] + [self.eos_token], dtype=torch.long)
#
#         return {
#             'src': src,
#             'tgt_input': tgt_input,
#             'tgt_output': tgt_output,
#             'src_len': sample['src_len'],
#             'tgt_len': sample['tgt_len'] + 1  # 因为加了EOS
#         }
#
#
# def collate_seq2seq_batch(batch):
#     """
#     处理seq2seq数据的批处理函数
#     """
#     src_seqs = [item['src'] for item in batch]
#     tgt_input_seqs = [item['tgt_input'] for item in batch]
#     tgt_output_seqs = [item['tgt_output'] for item in batch]
#
#     # 获取序列长度
#     src_lens = [len(seq) for seq in src_seqs]
#     tgt_lens = [len(seq) for seq in tgt_input_seqs]
#
#     # 填充序列
#     src_padded = torch.nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=0)
#     tgt_input_padded = torch.nn.utils.rnn.pad_sequence(tgt_input_seqs, batch_first=True, padding_value=0)
#     tgt_output_padded = torch.nn.utils.rnn.pad_sequence(tgt_output_seqs, batch_first=True, padding_value=0)
#
#     # 创建掩码
#     src_mask = _create_padding_mask(src_padded, pad_token=0)
#     tgt_mask = _create_causal_mask(tgt_input_padded.size(1))
#
#     return {
#         'src': src_padded,
#         'tgt_input': tgt_input_padded,
#         'tgt_output': tgt_output_padded,
#         'src_mask': src_mask,
#         'tgt_mask': tgt_mask,
#         'src_lens': src_lens,
#         'tgt_lens': tgt_lens
#     }
#
#
# def _create_padding_mask(seq, pad_token=0):
#     """
#     创建填充掩码
#     """
#     # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
#     mask = (seq != pad_token).unsqueeze(1).unsqueeze(2)
#     return mask
#
#
# def _create_causal_mask(seq_len):
#     """
#     创建因果掩码（防止看到未来信息）
#     """
#     # [1, 1, seq_len, seq_len]
#     mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
#     return mask
#
#
# def load_reverse_string_dataset(batch_size=32, num_samples=1000):
#     """
#     加载字符串反转数据集
#     """
#     # 创建原始数据集
#     original_dataset = ReverseStringDataset(num_samples=num_samples)
#
#     # 划分训练集和验证集
#     train_size = int(0.9 * len(original_dataset))
#     val_size = len(original_dataset) - train_size
#     train_dataset, val_dataset = torch.utils.data.random_split(
#         original_dataset, [train_size, val_size]
#     )
#
#     # 创建包装类来保留原始数据集的属性
#     class DatasetWrapper:
#         def __init__(self, subset, original_dataset):
#             self.subset = subset
#             self.original_dataset = original_dataset
#             self.vocab_size = original_dataset.vocab_size
#             self.pad_token = original_dataset.pad_token
#             self.sos_token = original_dataset.sos_token
#             self.eos_token = original_dataset.eos_token
#
#         def __getitem__(self, idx):
#             return self.subset[idx]
#
#         def __len__(self):
#             return len(self.subset)
#
#     # 包装训练集和验证集
#     train_wrapped = DatasetWrapper(train_dataset, original_dataset)
#     val_wrapped = DatasetWrapper(val_dataset, original_dataset)
#
#     train_loader = DataLoader(
#         train_wrapped,
#         batch_size=batch_size,
#         shuffle=True,
#         collate_fn=collate_seq2seq_batch
#     )
#     val_loader = DataLoader(
#         val_wrapped,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=collate_seq2seq_batch
#     )
#
#     print(f"Reverse String Dataset: {len(original_dataset)} samples")
#     print(f"Vocabulary size: {original_dataset.vocab_size}")
#     print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
#
#     # 返回训练loader、验证loader和原始数据集（用于测试）
#     return train_loader, val_loader, original_dataset


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from datasets import load_dataset
import os


class TranslationDataset(Dataset):
    """
    机器翻译数据集，用于Encoder-Decoder任务
    """

    def __init__(self, pairs, src_tokenizer, tgt_tokenizer, max_len=50):
        self.pairs = pairs
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

        # 特殊token
        self.pad_token = 0
        self.sos_token = 1
        self.eos_token = 2

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]

        # 简单的字符级编码
        src_encoded = [self.src_tokenizer.char_to_idx.get(c, 3) for c in src_text[:self.max_len]]
        tgt_encoded = [self.tgt_tokenizer.char_to_idx.get(c, 3) for c in tgt_text[:self.max_len]]

        # 源序列
        src = torch.tensor(src_encoded, dtype=torch.long)

        # 目标序列（添加SOS和EOS）
        tgt_input = torch.tensor([self.sos_token] + tgt_encoded, dtype=torch.long)
        tgt_output = torch.tensor(tgt_encoded + [self.eos_token], dtype=torch.long)

        return {
            'src': src,
            'tgt_input': tgt_input,
            'tgt_output': tgt_output,
            'src_len': len(src_encoded),
            'tgt_len': len(tgt_encoded) + 1
        }


def collate_seq2seq_batch(batch):
    """
    处理seq2seq数据的批处理函数
    """
    src_seqs = [item['src'] for item in batch]
    tgt_input_seqs = [item['tgt_input'] for item in batch]
    tgt_output_seqs = [item['tgt_output'] for item in batch]

    # 获取序列长度
    src_lens = [len(seq) for seq in src_seqs]
    tgt_lens = [len(seq) for seq in tgt_input_seqs]

    # 填充序列
    src_padded = torch.nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=0)
    tgt_input_padded = torch.nn.utils.rnn.pad_sequence(tgt_input_seqs, batch_first=True, padding_value=0)
    tgt_output_padded = torch.nn.utils.rnn.pad_sequence(tgt_output_seqs, batch_first=True, padding_value=0)

    # 创建掩码
    src_mask = _create_padding_mask(src_padded, pad_token=0)
    tgt_mask = _create_causal_mask(tgt_input_padded.size(1))

    return {
        'src': src_padded,
        'tgt_input': tgt_input_padded,
        'tgt_output': tgt_output_padded,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask,
        'src_lens': src_lens,
        'tgt_lens': tgt_lens
    }


def _create_padding_mask(seq, pad_token=0):
    """
    创建填充掩码
    """
    # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
    mask = (seq != pad_token).unsqueeze(1).unsqueeze(2)
    return mask


def _create_causal_mask(seq_len):
    """
    创建因果掩码（防止看到未来信息）
    """
    # [1, 1, seq_len, seq_len]
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    return mask


def create_comprehensive_translation_data(batch_size=16, max_len=30):
    """
    创建全面的翻译数据 - 直接使用内置数据，避免网络问题
    """
    print("Creating comprehensive translation dataset...")

    # 丰富的英德翻译对
    base_pairs = [
        # 基础问候
        ("hello", "hallo"),
        ("hello world", "hallo welt"),
        ("good morning", "guten morgen"),
        ("good evening", "guten abend"),
        ("good night", "gute nacht"),
        ("how are you", "wie geht es dir"),
        ("i am fine", "mir geht es gut"),
        ("thank you", "danke"),
        ("thank you very much", "vielen dank"),
        ("you are welcome", "bitte schön"),

        # 个人信息
        ("what is your name", "wie ist dein name"),
        ("my name is john", "mein name ist john"),
        ("how old are you", "wie alt bist du"),
        ("i am 25 years old", "ich bin 25 jahre alt"),
        ("where are you from", "woher kommst du"),
        ("i am from berlin", "ich komme aus berlin"),
        ("where do you live", "wo wohnst du"),
        ("i live in munich", "ich wohne in münchen"),

        # 日常对话
        ("i am hungry", "ich habe hunger"),
        ("i am thirsty", "ich habe durst"),
        ("i am tired", "ich bin müde"),
        ("i am happy", "ich bin glücklich"),
        ("i am sad", "ich bin traurig"),
        ("i dont understand", "ich verstehe nicht"),
        ("can you help me", "kannst du mir helfen"),
        ("i need help", "ich brauche hilfe"),
        ("what time is it", "wie spät ist es"),
        ("see you later", "bis später"),

        # 购物和餐饮
        ("how much does it cost", "wie viel kostet es"),
        ("i would like coffee", "ich möchte kaffee"),
        ("i would like tea", "ich möchte tee"),
        ("the food is delicious", "das essen ist köstlich"),
        ("the water is cold", "das wasser ist kalt"),
        ("i want to buy this", "ich möchte das kaufen"),

        # 方向和位置
        ("where is the station", "wo ist der bahnhof"),
        ("where is the hotel", "wo ist das hotel"),
        ("where is the restaurant", "wo ist das restaurant"),
        ("go straight ahead", "geradeaus gehen"),
        ("turn left", "links abbiegen"),
        ("turn right", "rechts abbiegen"),

        # 简单句子
        ("this is a test", "das ist ein test"),
        ("i love programming", "ich liebe programmierung"),
        ("the weather is nice", "das wetter ist schön"),
        ("the book is on the table", "das buch ist auf dem tisch"),
        ("i like music", "ich mag musik"),
        ("i want to learn", "ich möchte lernen"),
        ("this is important", "das ist wichtig"),
        ("that is interesting", "das ist interessant"),
    ]

    # 通过变换增加数据多样性
    train_pairs = []
    for en, de in base_pairs:
        # 原始版本
        train_pairs.append((en, de))
        # 大写版本
        train_pairs.append((en.upper(), de.upper()))
        # 首字母大写版本
        train_pairs.append((en.capitalize(), de.capitalize()))

    # 验证集使用原始版本的前20个
    val_pairs = base_pairs[:20]

    print(f"Comprehensive translation data: {len(train_pairs)} training pairs, {len(val_pairs)} validation pairs")

    # 打印样本检查
    print("数据样本示例:")
    for i in range(min(5, len(train_pairs))):
        print(f"  英文: '{train_pairs[i][0]}'")
        print(f"  德文: '{train_pairs[i][1]}'")
        print("  ---")

    # 创建字符级分词器
    all_src_text = " ".join([pair[0] for pair in train_pairs])
    all_tgt_text = " ".join([pair[1] for pair in train_pairs])

    class CharTokenizer:
        def __init__(self, text):
            chars = sorted(list(set(text)))
            self.char_to_idx = {ch: i + 3 for i, ch in enumerate(chars)}
            self.char_to_idx['<pad>'] = 0
            self.char_to_idx['<sos>'] = 1
            self.char_to_idx['<eos>'] = 2
            self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
            self.vocab_size = len(self.char_to_idx)

    src_tokenizer = CharTokenizer(all_src_text)
    tgt_tokenizer = CharTokenizer(all_tgt_text)

    # 创建数据集
    train_dataset = TranslationDataset(train_pairs, src_tokenizer, tgt_tokenizer, max_len)
    val_dataset = TranslationDataset(val_pairs, src_tokenizer, tgt_tokenizer, max_len)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_seq2seq_batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_seq2seq_batch
    )

    print(f"Source vocab size: {src_tokenizer.vocab_size}, Target vocab size: {tgt_tokenizer.vocab_size}")

    # 包装器
    class DatasetWrapper:
        def __init__(self, dataset, src_tokenizer, tgt_tokenizer):
            self.dataset = dataset
            self.src_tokenizer = src_tokenizer
            self.tgt_tokenizer = tgt_tokenizer
            self.vocab_size = tgt_tokenizer.vocab_size
            self.pad_token = 0
            self.sos_token = 1
            self.eos_token = 2

        def __getitem__(self, idx):
            return self.dataset[idx]

        def __len__(self):
            return len(self.dataset)

    wrapped_dataset = DatasetWrapper(train_dataset, src_tokenizer, tgt_tokenizer)

    return train_loader, val_loader, wrapped_dataset


def _create_data_loaders(train_pairs, val_pairs, batch_size, max_len):
    """创建数据加载器的公共函数"""
    # 创建分词器
    all_src_text = " ".join([pair[0] for pair in train_pairs])
    all_tgt_text = " ".join([pair[1] for pair in train_pairs])

    class CharTokenizer:
        def __init__(self, text):
            chars = sorted(list(set(text)))
            self.char_to_idx = {ch: i + 3 for i, ch in enumerate(chars)}
            self.char_to_idx['<pad>'] = 0
            self.char_to_idx['<sos>'] = 1
            self.char_to_idx['<eos>'] = 2
            self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
            self.vocab_size = len(self.char_to_idx)

    src_tokenizer = CharTokenizer(all_src_text)
    tgt_tokenizer = CharTokenizer(all_tgt_text)

    # 创建数据集
    train_dataset = TranslationDataset(train_pairs, src_tokenizer, tgt_tokenizer, max_len)
    val_dataset = TranslationDataset(val_pairs, src_tokenizer, tgt_tokenizer, max_len)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_seq2seq_batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_seq2seq_batch
    )

    print(f"  Source vocab: {src_tokenizer.vocab_size}, Target vocab: {tgt_tokenizer.vocab_size}")

    # 包装器
    class DatasetWrapper:
        def __init__(self, dataset, src_tokenizer, tgt_tokenizer):
            self.dataset = dataset
            self.src_tokenizer = src_tokenizer
            self.tgt_tokenizer = tgt_tokenizer
            self.vocab_size = tgt_tokenizer.vocab_size
            self.pad_token = 0
            self.sos_token = 1
            self.eos_token = 2

        def __getitem__(self, idx):
            return self.dataset[idx]

        def __len__(self):
            return len(self.dataset)

    wrapped_dataset = DatasetWrapper(train_dataset, src_tokenizer, tgt_tokenizer)
    return train_loader, val_loader, wrapped_dataset


def load_reliable_translation_dataset(batch_size=16, max_len=30, num_samples=800):
    """
    可靠的多重尝试翻译数据集加载 - 修复版
    """
    print("Loading reliable translation dataset (multiple attempts)...")

    # 设置镜像源和网络配置
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

    # 更可靠的数据集列表（按稳定性排序）
    datasets_to_try = [
        # 1. 最稳定的多语言数据集
        ("bentrevett/multi30k", None),
        # 2. OPUS数据集的不同配置
        ("opus100", "en-de"),
        # 3. Tatoeba的新版本
        ("Helsinki-NLP/tatoeba_mt", "eng-deu"),
        # 4. 备选数据集
        ("cfilt/iitb-english-hindi", None),  # 英-印地语，但我们可以用
    ]

    for dataset_name, config_name in datasets_to_try:
        try:
            print(f"Attempting to load: {dataset_name} {config_name if config_name else ''}")

            # 加载数据集（添加超时和重试）
            if config_name:
                dataset = load_dataset(dataset_name, config_name, trust_remote_code=True)
            else:
                dataset = load_dataset(dataset_name, trust_remote_code=True)

            # 提取翻译对
            train_pairs = []

            # 处理不同的数据集格式
            for example in dataset['train']:
                if len(train_pairs) >= num_samples:
                    break

                src_text, tgt_text = None, None

                # 处理不同的数据格式
                if 'translation' in example:
                    # OPUS格式
                    if 'en' in example['translation'] and 'de' in example['translation']:
                        src_text = example['translation']['en'].lower().strip()
                        tgt_text = example['translation']['de'].lower().strip()
                    elif 'english' in example['translation'] and 'hindi' in example['translation']:
                        # 对于英-印地语数据集
                        src_text = example['translation']['english'].lower().strip()
                        tgt_text = example['translation']['hindi'].lower().strip()
                elif 'en' in example and 'de' in example:
                    # multi30k格式
                    src_text = example['en'].lower().strip()
                    tgt_text = example['de'].lower().strip()
                elif 'source' in example and 'target' in example:
                    # 某些数据集的格式
                    src_text = example['source'].lower().strip()
                    tgt_text = example['target'].lower().strip()
                elif 'text' in example and 'translation' in example:
                    # 其他格式
                    src_text = example['text']['en'].lower().strip()
                    tgt_text = example['text']['de'].lower().strip()
                else:
                    # 打印数据结构以便调试
                    if len(train_pairs) == 0:
                        print(f"  Dataset structure: {list(example.keys())}")
                    continue

                # 验证数据质量
                if (src_text and tgt_text and
                        len(src_text) > 3 and len(tgt_text) > 3 and
                        len(src_text) < 100 and len(tgt_text) < 100 and
                        src_text.count(' ') < 20 and tgt_text.count(' ') < 20):
                    train_pairs.append((src_text, tgt_text))

            if len(train_pairs) > 50:  # 降低阈值
                print(f"✓ Successfully loaded {dataset_name}")
                print(f"  Found {len(train_pairs)} training pairs")

                # 打印一些样本
                print("  Data samples:")
                for i in range(min(3, len(train_pairs))):
                    print(f"    EN: '{train_pairs[i][0]}'")
                    print(f"    DE: '{train_pairs[i][1]}'")
                    print("    ---")

                # 分割训练集和验证集
                val_size = min(200, len(train_pairs) // 5)
                val_pairs = train_pairs[:val_size]
                train_pairs = train_pairs[val_size:]

                print(f"  Final split: {len(train_pairs)} training, {len(val_pairs)} validation")

                # 创建分词器和数据集
                return _create_data_loaders(train_pairs, val_pairs, batch_size, max_len)
            else:
                print(f"  ✗ {dataset_name} has insufficient data: {len(train_pairs)} pairs")
                continue

        except Exception as e:
            print(f"  ✗ Failed to load {dataset_name}: {str(e)[:150]}...")
            continue

    print("All online datasets failed, falling back to built-in data...")
    return create_comprehensive_translation_data(batch_size, max_len)


# 序列到序列数据集映射
SEQ2SEQ_DATASET_LOADERS = {
    'reliable': load_reliable_translation_dataset,  # 多重尝试的可靠版本
    'builtin': create_comprehensive_translation_data,  # 内置数据
}


def get_seq2seq_dataset_loader(dataset_name):
    """获取序列到序列数据集加载函数"""
    return SEQ2SEQ_DATASET_LOADERS.get(dataset_name, load_reliable_translation_dataset)