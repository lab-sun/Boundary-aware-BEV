from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

def exists(val):
    return val is not None

def uniq(arr):
    return {el:True for el in arr}.keys()

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out*2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False) # query_dim 320; inner_dim 320
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        # import pdb; pdb.set_trace() # 在context is not None时，x:torch.Size([48, 1024, 320])  context:torch.Size([48, 1024, 320])
        h = self.heads

        q = self.to_q(x) # TODO 检查x和context输入时的维度，是向量还是张量
        context = default(context, x) # 若没有context，则返回x，即做self-attention
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out) # torch.Size([8, 1024, 320])


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
    
    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x # torch.Size([bs, 1024, 64])
        x = self.attn2(self.norm2(x), context=context) + x # torch.Size([bs, 1024, 320])
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        self.x_embed = nn.Conv2d(4, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              #in_channels,
                                              4,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        x_in = x # torch.Size([48, 4, 32, 32])
        x = self.x_embed(x) # torch.Size([48, 320, 32, 32])
        b, c, h, w = x.shape
        x = self.norm(x) # torch.Size([48, 320, 32, 32])
        x = self.proj_in(x) # torch.Size([48, 64, 32, 32])
        x = rearrange(x, 'b c h w -> b (h w) c') # torch.Size([48, 1024, 64])

        if context is not None:
            context = self.x_embed(context)
            context_in = context
            context = self.norm(context) # torch.Size([48, 320, 32, 32])
            context = self.proj_in(context) # torch.Size([48, 64, 32, 32])
            context = rearrange(context, 'b c h w -> b (h w) c') # torch.Size([48, 1024, 64])
        for block in self.transformer_blocks:
            x = block(x, context=context) # torch.Size([48, 1024, 64])
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w) # torch.Size([48, 320, 32, 32])
        x = self.proj_out(x) # torch.Size([48, 4, 32, 32])
        return x + x_in



######################################################################
class CrossTransformer(nn.Module):
    def __init__(self, in_dim):
        super(CrossTransformer, self).__init__()
        self.x_embed = nn.Conv2d(4, in_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # 实验 GPU1：out_channels=4  /  GPU3：out_channels=in_dim
        self.f_conv = nn.Conv2d(in_channels=72, out_channels=4, kernel_size=3, \
                                stride=1, padding=1, bias=True)  # 实验 GPU1
        # self.f_conv = nn.Conv2d(in_channels=72, out_channels=in_dim, kernel_size=3, \
        #                         stride=1, padding=1, bias=True)  # 实验 GPU3
        # self.out_conv = nn.Conv2d(in_channels=in_dim, out_channels=4, kernel_size=3, \
        #                         stride=1, padding=1, bias=True)  # 实验 GPU3

        self.res_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def feature_selection(self, input, dim, index):
        views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)
    
    def forward(self, FV_x, BEV_x):
        # import pdb; pdb.set_trace() 
        # 输入 FV_x.shape:torch.Size([bs, 4, 32, 32]), BEV_x.shape:torch.Size([bs, 4, 32, 32])
        FV_x_embeded = self.x_embed(FV_x)  # torch.Size([bs, 64, 32, 32])
        BEV_x_embeded = self.x_embed(BEV_x)  # torch.Size([bs, 64, 32, 32])
        bs, c, h, w = BEV_x_embeded.shape  # bs, 64, 32, 32
        query = self.query_conv(BEV_x_embeded).view(bs, -1, h*w) # torch.Size([bs, 8, 1024])
        key = self.key_conv(FV_x_embeded).view(bs, -1, h*w).permute(0, 2, 1) # torch.Size([bs, 1024, 8])

        score = torch.bmm(key, query) # torch.Size([bs, 1024, 1024])
        score_star, score_star_arg = torch.max(score, dim=1)
        value = self.value_conv(FV_x_embeded).view(bs, -1, h*w)  # torch.Size([bs, 8, 1024])

        T = self.feature_selection(value, 2, score_star_arg).view(score_star.size(0), -1, h, w)  # torch.Size([bs, 8, 32, 32])
        S = score_star.view(score_star.size(0), 1, h, w)  # torch.Size([bs, 1, 32, 32])

        BEV_res = torch.cat((FV_x_embeded, T), dim=1) # torch.Size([bs, 72, 32, 32])
        BEV_res = self.f_conv(BEV_res)  # 实验 GPU1:torch.Size([bs, 4, 32, 32]) / GPU3:torch.Size([bs, 64, 32, 32])
        BEV_res = BEV_res * S   # 实验 GPU1:torch.Size([bs, 4, 32, 32]) / GPU3:torch.Size([bs, 64, 32, 32])
        # 实验 GPU1:torch.Size([bs, 4, 32, 32])
        output = BEV_x + BEV_res 
        # 实验 GPU3:
        # output = BEV_x_embeded + BEV_res  # torch.Size([bs, 64, 32, 32])
        # output = self.out_conv(output)  # torch.Size([bs, 4, 32, 32])

        return output