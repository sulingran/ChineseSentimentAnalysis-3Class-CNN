import pandas as pd
import random


# 模拟数据生成函数
def generate_comment():
    # 定义一些示例评论模板
    templates = [
        "这次住宿体验很好，房间干净整洁，服务态度非常好。",
        "酒店位置很好，交通便利，但房间隔音效果不佳。",
        "这家餐厅的菜品非常美味，服务也很热情。",
        "酒店的房间很小，设施老旧，服务态度一般。",
        "这次购物体验很好，商品种类丰富，价格合理。",
        "这部电影非常好看，演员表演出色，情节紧凑。",
        "餐厅的环境很好，但菜品口味一般。",
        "这次旅行体验非常糟糕，酒店服务差，交通不便。",
        "这本书内容丰富，语言流畅，值得一看。",
        "这次购物体验一般，服务态度还可以，但商品质量一般。",
    ]

    # 随机选择一个模板
    template = random.choice(templates)

    # 添加一些随机噪声（模拟真实数据的多样性）
    if random.random() < 0.5:
        template = template.replace("很好", "非常不错")
    if random.random() < 0.5:
        template = template.replace("很好", "还可以")

    return template


def generate_label(comment):
    # 根据评论文本简单判断情感标签
    if "很好" in comment or "出色" in comment or "丰富" in comment:
        return "正面"
    elif "糟糕" in comment or "一般" in comment or "老旧" in comment:
        return "负面"
    else:
        return "中性"


# 生成 1000 条数据
data = []
for _ in range(1000):
    comment = generate_comment()
    label = generate_label(comment)
    data.append({"句子": comment, "类别": label})

# 转换为 DataFrame
df = pd.DataFrame(data)

# 保存为 CSV 文件
df.to_csv("data.csv", index=False, encoding="utf-8-sig")

print("数据生成完成，已保存为 data.csv")
