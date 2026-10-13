'''数据集划分脚本'''
import json
import random
random.seed(42)#固定随机种子，保证每次划分完全一致，确保可重复性

all_json_path = r"D:\kc_bird\datasets\all_annotations.json"
with open(all_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
all_samples = data["annotations"]
random.shuffle(all_samples)#打乱样本顺序

total = len(all_samples)
train_num = int(total * 0.8)
val_num = int(total * 0.1)

train_smaples = all_samples[:train_num]
val_samples = all_samples[train_num:train_num+val_num]
test_samples = all_samples[train_num+val_num:]

def save_json(samples, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"annotations": samples}, f, indent=2, ensure_ascii=False)

save_json(train_smaples, r"D:\kc_bird\datasets\train_annotations.json")
save_json(val_samples, r"D:\kc_bird\datasets\val_annotations.json")
save_json(test_samples, r"D:\kc_bird\datasets\test_annotations.json")
print(f"train: {len(train_smaples)} | val: {len(val_samples)} | test: {len(test_samples)}")