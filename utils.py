import torch
import jieba
import jieba.posseg as pseg
from collections import Counter

def tokenize(text):
    """分词"""
    return list(jieba.lcut(text))

def remove_stopwords(tokens, stopwords):
    """去除停用词"""
    return [word for word in tokens if word not in stopwords]

def build_vocab(all_tokens):
    """构建词典"""
    # 统计词频
    word_counts = Counter(all_tokens)
    # 构建词汇表（手动实现）
    vocab = {word: idx for idx, (word, _) in enumerate(word_counts.most_common(), start=1)}
    vocab['<unk>'] = 0  # 未知词
    return vocab

def load_stopwords(stopwords_file):
    """加载停用词"""
    return set(open(stopwords_file, encoding="utf-8").read().splitlines())

def pad_sequence(sequences, padding_value=0, max_length=None):
    """对序列进行填充"""
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            padded_sequences.append(seq + [padding_value] * (max_length - len(seq)))
        else:
            padded_sequences.append(seq[:max_length])
    return torch.tensor(padded_sequences)
