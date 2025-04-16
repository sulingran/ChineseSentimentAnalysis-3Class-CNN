import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes):
        super(CNNTextClassifier, self).__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (size, embedding_dim)) for size in filter_sizes
        ])

        # 全连接层
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        # x: (batch_size, sequence_length)
        embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        print("Embedded shape:", embedded.shape)  # 打印嵌入后的形状
        embedded = embedded.unsqueeze(1)  # (batch_size, 1, sequence_length, embedding_dim)
        print("Embedded shape after unsqueeze:", embedded.shape)  # 打印unsqueeze后的形状

        # 卷积操作
        convs = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]

        # 拼接所有卷积层的输出
        concat = torch.cat(pools, dim=1)

        # 全连接层
        logits = self.fc(concat)
        return logits
