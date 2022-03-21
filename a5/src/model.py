
"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

import attention


# GPT各项参数设置
class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    additive = False
    synthesizer = False  # 自注意力机制的变体名称

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)  # 初始化layerNorm层
        self.ln2 = nn.LayerNorm(config.n_embd)
        if config.additive:
            self.attn = attention.AdditiveSelfAttention(config)
        elif config.synthesizer:
            self.attn = attention.SynthesizerAttention(config)  # 自注意力机制层采用SynthesizerAttention
        else:
            self.attn = attention.CausalSelfAttention(config)  # 自注意力层采用CausalSelfAttention
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )  # mlp层

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # 这里是先进行LayerNorm，再实现自注意力机制，最后是残差网络中的相加
        x = x + self.mlp(self.ln2(x))  # 同理，先进行LayerNorm，再是前馈层，最后残差相加
        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)  # 词嵌入
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))  # 位置向量嵌入
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        # LayerNorm参数是一个数字，则对矩阵的最后一维进行归一化，且这个数字要跟最后一维的大小相同
        # 如果参数是一个矩阵，则对矩阵整体归一化。并且如果处理的矩阵是三维的，则只对最后两维进行归一化
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)  # apply是用来直接调用函数

        print("number of parameters: {}".format(sum(p.numel() for p in self.parameters())))

    def _init_weights(self, module):  # isinstance是用来判断一个对象是否是一个已知类型
        if isinstance(module, (nn.Linear, nn.Embedding)):  # 如果module是后面元组中的一个，则返回Ture
            module.weight.data.normal_(mean=0.0, std=0.02)  # 初始化权重，服从平均值和标准差的正态分布
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()  # 初始化偏置，全部为0
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)  # 初始化权重，全部为1

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)

        return logits, loss

class CustomLayerNorm(nn.Module):
  pass
