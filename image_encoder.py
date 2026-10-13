'''图像编码器ImageEncoder'''
import torch
import torchvision.models as models
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self,out_dim=512):
        super().__init__()
        #加载预训练模型ResNet-18
        self.backbone = models.resnet18(pretrained=True)
        #去掉ResNet最后的全局池化+分类头，只保留卷积主干
        self.backbone = nn.Sequential(*list(self.backbone.children()))[:-2]
        #添加一个全局池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        #投影层，将特征维度从512降到out_dim
        self.fc = nn.Linear(512, out_dim)
    def forward(self, x):
        #x: [B,3,224,224]
        feat_map = self.backbone(x)# [B,512,7,7],卷积输出特征图
        pooled = self.avgpool(feat_map)# [B,512,1,1]
        pooled = pooled.flatten(1)# [B,512]
        out = self.fc(pooled)# [B,out_dim]
        return out