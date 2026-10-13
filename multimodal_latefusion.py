import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from sklearn.metrics import accuracy_score, f1_score
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "bert-base-uncased"
img_root = r"D:\kc_bird\CUB_200_subset\images"

# 图像预处理流水线
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
val_test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# 多模态数据集：同时加载图片+文本caption
class MultiBirdDataset(Dataset):
    def __init__(self, json_path, tokenizer, transform, max_len=32):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.samples = data
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_full_path = item["image_path"]
        image = Image.open(img_full_path).convert("RGB")
        image = self.transform(image)

        all_caps = [c.strip() for c in item["text"].split("\n") if c.strip()]
        cap = all_caps[torch.randint(0, len(all_caps), ()).item()]
        label = item["label"]
        enc = self.tokenizer(
            cap, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
        )
        return {
            "image": image,
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "label": torch.tensor(label, dtype=torch.long)
        }

# Late Fusion多模态模型
class MultiModalLateFusion(nn.Module):
    def __init__(self, num_classes, feat_dim=512):
        super().__init__()
        self.img_encoder = ImageEncoder(out_dim=feat_dim)
        self.txt_encoder = TextEncoder(out_dim=feat_dim)
        # 融合：拼接图像特征+文本特征，512+512 =1024
        fusion_dim = feat_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.img_encoder(image) # [B,512]
        txt_feat = self.txt_encoder(input_ids, attention_mask) # [B,512]
        fused = torch.cat([img_feat, txt_feat], dim=1) # 拼接 [B,1024]
        logits = self.classifier(fused)
        return logits

# =========训练主流程=========
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    tokenizer = AutoTokenizer.from_pretrained(model_name, timeout=30)
    train_ds = MultiBirdDataset(r"D:\kc_bird\datasets\annotations_train.json", tokenizer, train_transform)
    val_ds = MultiBirdDataset(r"D:\kc_bird\datasets\annotations_val.json", tokenizer, val_test_transform)
    test_ds = MultiBirdDataset(r"D:\kc_bird\datasets\annotations_test.json", tokenizer, val_test_transform)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)
    test_loader = DataLoader(test_ds, batch_size=4)

    num_classes = max([s["label"] for s in train_ds.samples]) + 1
    model = MultiModalLateFusion(num_classes).to(device)

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
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(image, input_ids, attn_mask)
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
                image = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                logits = model(image, input_ids, attn_mask)
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
            torch.save(model.state_dict(), "multimodal_best.pth")

    # ========== 新增：绘制训练曲线 ==========
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    ax1.plot(epoch_list, train_loss_list, marker='o', color='#d62728', label='Train Loss')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("多模态LateFusion 训练损失曲线")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epoch_list, val_acc_list, marker='o', color='#1f77b4', label='Val Acc')
    ax2.plot(epoch_list, val_f1_list, marker='s', color='#2ca02c', label='Val Macro-F1')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("指标")
    ax2.set_title("多模态LateFusion 验证集Acc / F1曲线")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("multimodal_curve.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("✅ 曲线图片已保存：multimodal_curve.png")

    # test评估
    model.load_state_dict(torch.load("multimodal_best.pth", map_location=device))
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(image, input_ids, attn_mask)
            pred = torch.argmax(logits, dim=1)
            preds_all.extend(pred.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    acc_test = accuracy_score(labels_all, preds_all)
    f1_test = f1_score(labels_all, preds_all, average="macro")
    print("====MultiModal Test Result====")
    print(f"Test Acc: {acc_test:.4f}, Test Macro-F1: {f1_test:.4f}")