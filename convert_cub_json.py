import os
import json
import random

image_root = r"D:\kc_bird\CUB_200_subset\images"
text_root = r"D:\kc_bird\CUB_200_subset\text"
output_dir = r"D:\kc_bird\datasets"
os.makedirs(output_dir, exist_ok=True)

class_list = os.listdir(image_root)
print("检测到类别：", class_list)

samples = []

for cls_name in class_list:
    cls_img_dir = os.path.join(image_root, cls_name)
    cls_text_dir = os.path.join(text_root, cls_name)

    # 遍历该类别下全部图片
    for img_name in os.listdir(cls_img_dir):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_full_path = os.path.join(cls_img_dir, img_name)
        # txt和图片同名
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_full_path = os.path.join(cls_text_dir, txt_name)

        # 读取文本
        if os.path.exists(txt_full_path):
            with open(txt_full_path, "r", encoding="utf-8") as f:
                text_content = f.read().strip()
        else:
            print(f"警告：缺失文本 {txt_full_path}")
            continue

        sample = {
            "image_path": img_full_path,
            "text": text_content,
            "label": class_list.index(cls_name)
        }
        samples.append(sample)

# 划分训练集验证集测试集 8:1:1
random.seed(42)
random.shuffle(samples)
total = len(samples)
train_data = samples[:int(total*0.8)]
val_data = samples[int(total*0.8):int(total*0.9)]
test_data = samples[int(total*0.9):]

with open(os.path.join(output_dir, "annotations_train.json"), "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)
with open(os.path.join(output_dir, "annotations_val.json"), "w", encoding="utf-8") as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)
with open(os.path.join(output_dir, "annotations_test.json"), "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)
# 新增：输出全部样本到 all_annotations.json
out_data = {
    "annotations": samples
}
with open(os.path.join(output_dir, "all_annotations.json"), "w", encoding="utf-8") as f:
    json.dump(out_data, f, ensure_ascii=False, indent=2)

print(f"已生成全部标注文件 all_annotations.json，总样本数：{len(samples)}")


print(f"生成完成！训练集:{len(train_data)} 验证集:{len(val_data)} 测试集:{len(test_data)}")
