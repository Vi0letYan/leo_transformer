import json

path = "../data/translation_dataset/train.json"  # 需要从查看的数据集
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)  # 自动把 \uXXXX 还原

# 看前3条，ensure_ascii=False 用于“打印中文”
print(json.dumps(data[:3], ensure_ascii=False, indent=2))

# data 的结构应当是：[[en, zh], [en, zh], ...]
print(type(data), len(data), type(data[0]), len(data[0]))
