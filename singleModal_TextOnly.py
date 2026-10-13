# 最顶部必须设置hf镜像，解决国内网络超时
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from text_encoder import TextEncoder
from sklearn.metrics import accuracy_score, f1_score

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "bert-base-uncased"

# 自定义数据集：只读取文本caption
class TextBirdDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_len=32):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.samples = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        captions = item["text"]
        # 随机选取10条caption里其中一句
        cap = captions[torch.randint(0, len(captions), ()).item()]
        label = item["label"]
        # tokenizer编码文本
        enc = self.tokenizer(
            cap, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "label": torch.tensor(label, dtype=torch.long)
        }

# 单模态文本分类模型
class TextOnlyModel(nn.Module):
    def __init__(self, num_classes, out_dim=512):
        super().__init__()
        self.text_encoder = TextEncoder(out_dim=out_dim)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        text_feat = self.text_encoder(input_ids, attention_mask) # [B,512]
        logits = self.classifier(text_feat) # [B, num_classes]
        return logits

# =========训练流程=========
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    tokenizer = AutoTokenizer.from_pretrained(model_name, timeout=30)
    train_ds = TextBirdDataset(r"D:\kc_bird\datasets\annotations_train.json", tokenizer)
    val_ds = TextBirdDataset(r"D:\kc_bird\datasets\annotations_val.json", tokenizer)
    test_ds = TextBirdDataset(r"D:\kc_bird\datasets\annotations_test.json", tokenizer)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)
    test_loader = DataLoader(test_ds, batch_size=4)

    # 获取类别总数
    num_classes = max([s["label"] for s in train_ds.samples]) + 1
    model = TextOnlyModel(num_classes).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    epoch_num = 10
    best_acc = 0

    # ========== 新增：记录曲线数据 ==========
    train_loss_list = []
    val_acc_list = []
    val_f1_list = []
    epoch_list = list(range(epoch_num))

    for epoch in range(epoch_num):
        # train
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attn_mask)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)

        # val
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                logits = model(input_ids, attn_mask)
                pred = torch.argmax(logits, dim=1)
                preds_all.extend(pred.cpu().numpy())
                labels_all.extend(labels.cpu().numpy())
        acc_val = accuracy_score(labels_all, preds_all)
        f1_val = f1_score(labels_all, preds_all, average="macro")
        val_acc_list.append(acc_val)
        val_f1_list.append(f1_val)

        print(f"Epoch {epoch} | Avg Loss:{avg_train_loss:.4f} | Val Acc:{acc_val:.4f} | Val MacroF1:{f1_val:.4f}")
        if acc_val>best_acc:
            best_acc = acc_val
            torch.save(model.state_dict(), "text_best.pth")

    # ========== 新增：绘制训练曲线 ==========
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 支持中文
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # 子图1：训练Loss
    ax1.plot(epoch_list, train_loss_list, marker='o', color='#d62728', label='Train Loss')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("TextOnly 训练损失曲线")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 子图2：验证集Acc & F1
    ax2.plot(epoch_list, val_acc_list, marker='o', color='#1f77b4', label='Val Acc')
    ax2.plot(epoch_list, val_f1_list, marker='s', color='#2ca02c', label='Val Macro-F1')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("指标")
    ax2.set_title("TextOnly 验证集Acc / F1曲线")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("text_only_curve.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("✅ 曲线图片已保存：text_only_curve.png")

    # 测试集评估
    model.load_state_dict(torch.load("text_best.pth", map_location=device))
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attn_mask)
            pred = torch.argmax(logits, dim=1)
            preds_all.extend(pred.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    acc_test = accuracy_score(labels_all, preds_all)
    f1_test = f1_score(labels_all, preds_all, average="macro")
    print("====Test Result====")
    print(f"Test Acc: {acc_test:.4f}, Test Macro-F1: {f1_test:.4f}")


    # 测试集评估
    model.load_state_dict(torch.load("text_best.pth", map_location=device))
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attn_mask)
            pred = torch.argmax(logits, dim=1)
            preds_all.extend(pred.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    acc_test = accuracy_score(labels_all, preds_all)
    f1_test = f1_score(labels_all, preds_all, average="macro")
    print("====Test Result====")
    print(f"Test Acc: {acc_test:.4f}, Test Macro-F1: {f1_test:.4f}")
