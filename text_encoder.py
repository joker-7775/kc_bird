'''文本编码器TextEncoder'''
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", out_dim=512, freeze=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(768, out_dim) # BERT默认输出768维，投影到512

        # 新增：冻结BERT主干
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
            print("✅ BERT主干已冻结，只训练投影层proj")
        else:
            print("✅ BERT主干全部参数参与微调")

    def forward(self, input_ids, attention_mask):
        # input_ids: token编号，[B, L]
        # attention_mask：掩码，1=有效词，0=PAD填充
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = bert_out.last_hidden_state # [B, L,768] 每个token的向量

        # Mean Pooling，屏蔽PAD token
        mask_expand = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_hidden = torch.sum(last_hidden * mask_expand, dim=1)
        sum_mask = torch.clamp(mask_expand.sum(1), min=1e-9)
        mean_pool = sum_hidden / sum_mask # [B,768]

        out = self.proj(mean_pool) # [B,512]
        return out

# 单独封装tokenizer加载，方便外部调用
def load_tokenizer(model_name="bert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_name)
