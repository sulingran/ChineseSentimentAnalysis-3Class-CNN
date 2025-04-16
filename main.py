import pandas as pd
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from utils import tokenize, remove_stopwords, build_vocab, load_stopwords, pad_sequence
from model import CNNTextClassifier
from torch.optim.lr_scheduler import StepLR

# 定义 TextDataset 类
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label

def load_datasets():
    # 加载正面、负面和中性数据
    try:
        pos_data = pd.read_csv("data/pos.csv", encoding='utf-8', header=None, names=['句子'], on_bad_lines='skip')
    except pd.errors.ParserError as e:
        print(f"加载 pos.csv 出错: {e}")
        return None
    try:
        neg_data = pd.read_csv("data/neg.csv", encoding='utf-8', header=None, names=['句子'], on_bad_lines='skip')
    except pd.errors.ParserError as e:
        print(f"加载 neg.csv 出错: {e}")
        return None
    try:
        neutral_data = pd.read_csv("data/neutral.csv", encoding='utf-8', header=None, names=['句子'], on_bad_lines='skip')
    except pd.errors.ParserError as e:
        print(f"加载 neutral.csv 出错: {e}")
        return None

    # 设置标签
    pos_data["类别"] = 1  # 正面
    neg_data["类别"] = 0  # 负面
    neutral_data["类别"] = 2  # 中性

    # 合并数据集
    data = pd.concat([pos_data, neg_data, neutral_data], ignore_index=True)

    return data

# 加载数据
data = load_datasets()

# 去重
data = data.drop_duplicates()

# 检查类别分布
print(data["类别"].value_counts())

# 分词
data["分词后"] = data["句子"].apply(tokenize)

# 加载停用词
stopwords = load_stopwords("stopwords.txt")
data["分词后"] = data["分词后"].apply(lambda tokens: remove_stopwords(tokens, stopwords))

# 构建词典
all_tokens = [token for tokens in data["分词后"] for token in tokens]
vocab = build_vocab(all_tokens)
vocab_size = len(vocab)

# 将词语转换为索引
data["索引"] = data["分词后"].apply(lambda tokens: [vocab[token] if token in vocab else vocab["<unk>"] for token in tokens])

# 类别编码
label_encoder = LabelEncoder()
data["标签"] = label_encoder.fit_transform(data["类别"])

# 划分训练集、验证集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# 提取特征和标签
train_texts = [text for text in train_data["索引"].tolist()]
train_labels = train_data["标签"].tolist()
val_texts = [text for text in val_data["索引"].tolist()]
val_labels = val_data["标签"].tolist()
test_texts = [text for text in test_data["索引"].tolist()]
test_labels = test_data["标签"].tolist()

# 填充训练数据
train_texts_padded = pad_sequence(train_texts, max_length=50)  # 假设最大长度为 50
train_dataset = TextDataset(train_texts_padded, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_texts_padded = pad_sequence(val_texts, max_length=50)
val_dataset = TextDataset(val_texts_padded, val_labels)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_texts_padded = pad_sequence(test_texts, max_length=50)
test_dataset = TextDataset(test_texts_padded, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 初始化模型
embedding_dim = 200  # 嵌入层维度
num_filters = 100
filter_sizes = [2, 3, 4, 5]  # 卷积核尺寸
num_classes = 3  # 三分类任务

model = CNNTextClassifier(vocab_size, embedding_dim, num_filters, filter_sizes, num_classes)

# 显式设置为 CPU
device = torch.device("cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 学习率调度器
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# 训练和评估
num_epochs = 20

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch in loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader), accuracy, y_true, y_pred

# 训练和评估
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )

    # 打印分类报告
    print(classification_report(y_true, y_pred))

# 评估模型
# 测试预测
def predict(model, texts, vocab, label_encoder, device, max_length=50):
    model.eval()
    predictions = []
    for text in texts:
        tokens = tokenize(text)
        tokens = remove_stopwords(tokens, stopwords)
        indices = [vocab[token] if token in vocab else vocab["<unk>"] for token in tokens]
        indices = pad_sequence([indices], max_length=max_length)  # 对单个预测进行填充
        indices = indices.to(device)
        with torch.no_grad():
            output = model(indices)
            _, predicted = torch.max(output, 1)
            category = label_encoder.inverse_transform(predicted.cpu().numpy())[0]
            predictions.append(category)
    return predictions

# 测试用例
test_texts = [
    "张锦威要起飞了",
    "这部电影真是太糟糕了，浪费我的时间。",
    "我感觉还行"
]

# 进行批量预测
predicted_categories = predict(model, test_texts, vocab, label_encoder, device)

# 打印预测结果
for text, category in zip(test_texts, predicted_categories):
    print(f"文本：'{text}'，预测情感：'{category}'")
