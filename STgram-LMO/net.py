"""
modification by dragonhan158,Email:dragonhan158@126.com which is the WaveNet for ASD 2024.8.31
"""
from torch import nn
import torch
import torch.nn.functional as F
import math
from torch.nn import Module, Parameter
from wavenet import WavegramWaveNet, SpecWaveNet
from mambaout import MambaOut

# add
from torch import nn, einsum
from einops import rearrange


# Referenced https://github.com/mk-minchul/AdaFace/blob/master/head.py
class AdaFace(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=32.0, m=0.50):
        super(AdaFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m
        self.eps = 1e-3
        self.h = 0.333
        self.s = s
        # ema prep
        self.t_alpha = 1.0
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)
        print('\n\AdaFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)

    def forward(self, embbedings, label):

        kernel_norm = l2_norm(self.kernel, axis=0)
        cosine = torch.mm(embbedings, kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps)  # for stability

        safe_norms = torch.clip(cosine, min=0.001, max=100)  # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)

        # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        output = cosine * self.s
        return output

# Referenced https://github.com/HuangYG123/CurricularFace/tree/master
def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output
# Referenced https://github.com/HuangYG123/CurricularFace/tree/master
class CurricularFace(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=32.0, m=0.50, reg_coeff=0.008):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)
        self.weight = Parameter(torch.Tensor(out_features * 1, in_features))
        self.reg_coeff = reg_coeff  # Add regularization

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis = 1)
        kernel_norm = l2_norm(self.kernel, axis = 0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        # Add regularization
        reg_loss = self.reg_coeff * torch.sum(torch.square(self.weight))
        output += reg_loss
        return output

# Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf)
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=32.0, m=0.50, sub=1, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.sub = sub
        self.weight = Parameter(torch.Tensor(out_features * sub, in_features))
        nn.init.xavier_uniform_(self.weight)
        # init.kaiming_uniform_()
        # self.weight.data.normal_(std=0.001)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if self.sub > 1:
            cosine = cosine.view(-1, self.out_features, self.sub)
            cosine, _ = torch.max(cosine, dim=2)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device)
        # print(x.device, label.device, one_hot.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output

class TgramNet(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=512):
        super(TgramNet, self).__init__()
        # if "center=True" of stft, padding = win_len / 2
        self.conv_extrctor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
        self.conv_encoder = nn.Sequential(
            *[nn.Sequential(
                # 313(10) , 63(2), 126(4)  [128, 128, 313]
                nn.LayerNorm(313),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(mel_bins, mel_bins, 3, 1, 1, bias=False),
            ) for idx in range(num_layer)])

    def forward(self, x):
        #print("Input shape:", x.shape) 
        out = self.conv_extrctor(x)
        out = self.conv_encoder(out)
        return out
## CA注意力使用
# class STgramMFN(nn.Module):
#     def __init__(self, num_classes,
#                  c_dim=128,
#                  win_len=1024,
#                  hop_len=512,
#                  use_arcface=False, m=0.5, s=30, sub=1,
#                  attention_embed_dim=128, attention_heads=8):
#         super(STgramMFN, self).__init__()
#         #ArcMarginProduct  AdaptiveMarginProduct CurricularFace
#         # self.arcface3 = AdaFace(in_features=128, out_features=num_classes,
#         #                                 m=m, s=s) if use_arcface else use_arcface
#         self.arcface2 = CurricularFace(in_features=128, out_features=num_classes,
#                                         m=m, s=s) if use_arcface else use_arcface
#         self.arcface = ArcMarginProduct(in_features=128, out_features=num_classes,
#                                         m=m, s=s, sub=sub) if use_arcface else use_arcface
#         self.tgramnet = TgramNet(mel_bins=c_dim, win_len=win_len, hop_len=hop_len)
#         # 将TgramNet换为wavenet
#         #self.wavegramWaveNet = WavegramWaveNet()
#         #self.shuffleFaceNet = ShuffleFaceNet()
#         self.mambaOut = MambaOut()
#         self.feq_attention = Feq_Attention(feature_dim=c_dim)
#         #self.self_attention = SelfAttentionLayer(attention_embed_dim, attention_heads)
#
#     def forward(self, x_wav, coeffs, label=None):
#         x_wav, coeffs = x_wav.unsqueeze(1), coeffs.unsqueeze(1)
#         x_t = self.tgramnet(x_wav).unsqueeze(1)
#         #x_t = self.wavenet(x_wav).unsqueeze(1)
#         #x_mel_feq_att = self.feq_attention(coeffs).unsqueeze(1)
#         x_t_feq_att = self.feq_attention(x_t).unsqueeze(1)
#         #x = torch.cat((coeffs, x_t), dim=1)
#         #x = torch.cat((coeffs, x_t_feq_att), dim=1)
#         x = torch.cat((x_t, coeffs, x_t_feq_att), dim=1)
#         #x = torch.cat((x_t, x_mel_feq_att, x_mel_feq_att), dim=1)
#         #out, feature = self.shuffleNetV2_MetaACON(x, label)
#         #out, feature = self.shuffleFaceNet(x, label)
#         out, feature = self.mambaOut(x, label)
#         #out, feature = self.SwinTransformerBlock(x, label)
#         #out, feature = self.swinTransformer(x, label)
#         #out, feature = self.wavegramWaveNet(x, label)
#         #out, feature = self.effNetV2(x, label)
#         if self.arcface:
#             #out = self.arcface(feature, label)
#             #out = (self.arcface(feature, label) + self.arcface2(feature, label)) / 2.0
#             out = self.arcface2(feature, label)
#         return out, feature

## 多头注意力使用

class LinearAttention(nn.Module):
    def __init__(self, dim, *, heads=4, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h),
            (q, k, v),
        )

        q = q * self.scale
        q, k = q.softmax(dim=-1), k.softmax(dim=-2)

        # if exists(mask):
        #     k.masked_fill_(mask, 0.0)

        context = einsum("b n d, b n e -> b d e", q, k)
        out = einsum("b d e, b n d -> b n e", context, v)
        out = rearrange(out, " (b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class STgramMFN(nn.Module):
    def __init__(self, num_classes,
                 c_dim=128,
                 win_len=1024,
                 hop_len=512,
                 use_arcface=False, m=0.5, s=30, sub=1):
        super(STgramMFN, self).__init__()
        self.arcface2 = CurricularFace(in_features=c_dim, out_features=num_classes,
                                        m=m, s=s) if use_arcface else use_arcface
        self.arcface = ArcMarginProduct(in_features=128, out_features=num_classes,
                                        m=m, s=s, sub=sub) if use_arcface else use_arcface
        self.tgramnet = TgramNet(mel_bins=c_dim, win_len=win_len, hop_len=hop_len)
        self.mambaOut = MambaOut()
        self.self_attention = SelfAttentionLayer(c_dim, 8)
        # self.linear_attn = LinearAttention(
        #     dim=313, heads=6, dim_head=64, dropout=0.1
        # )
        #self.crossAttention = CrossAttention(dim=c_dim)

    def forward(self, x_wav, coeffs, label=None):
        x_wav, coeffs = x_wav.unsqueeze(1), coeffs.unsqueeze(1)
        x_t = self.tgramnet(x_wav)
        #x_mel_feq_att = self.feq_attention(coeffs).unsqueeze(1)
        #x_t_feq_att = self.self_attention(x_t).unsqueeze(1)

        x_t_attn, _ = self.self_attention(x_t)
        #x_l_attn = self.linear_attn(x_t)
        #print("x-t-shape",x_t)
        #x_t_attn, _ = self.crossAttention(x_t,x_wav,coeffs)
        #x = torch.cat((coeffs, x_t), dim=1)
        #x = torch.cat((x_t, coeffs, x_t_feq_att), dim=1)
        #x = torch.cat((coeffs, x_t_attn.unsqueeze(1)), dim=1)

        x = torch.cat((x_t.unsqueeze(1), coeffs, x_t_attn.unsqueeze(1)), dim=1)

        #x = torch.cat((x_t.unsqueeze(1), coeffs, x_l_attn.unsqueeze(1)), dim=1)
        #x = torch.cat((x_t, x_mel_feq_att, x_mel_feq_att), dim=1)
        out, feature = self.mambaOut(x, label)
        if self.arcface:
            out = self.arcface2(feature, label)
            #out = (self.arcface(feature, label) + self.arcface2(feature, label)) / 2.0
        return out, feature

## 4个通道 CA和多头一起使用
# class STgramMFN(nn.Module):
#     def __init__(self, num_classes,
#                  c_dim=128,
#                  win_len=1024,
#                  hop_len=512,
#                  use_arcface=False, m=0.5, s=30, sub=1,
#                  attention_embed_dim=128, attention_heads=8):
#         super(STgramMFN, self).__init__()
#         #ArcMarginProduct  AdaptiveMarginProduct CurricularFace
#         # self.arcface3 = AdaFace(in_features=128, out_features=num_classes,
#         #                                 m=m, s=s) if use_arcface else use_arcface
#         self.arcface2 = CurricularFace(in_features=128, out_features=num_classes,
#                                         m=m, s=s) if use_arcface else use_arcface
#         self.arcface = ArcMarginProduct(in_features=128, out_features=num_classes,
#                                         m=m, s=s, sub=sub) if use_arcface else use_arcface
#         self.tgramnet = TgramNet(mel_bins=c_dim, win_len=win_len, hop_len=hop_len)
#         # 将TgramNet换为wavenet
#         #self.wavegramWaveNet = WavegramWaveNet()
#         #self.shuffleFaceNet = ShuffleFaceNet()
#         self.mambaOut = MambaOut()
#         self.feq_attention = Feq_Attention(feature_dim=c_dim)
#         self.self_attention = SelfAttentionLayer(c_dim, 8)
#
#     def forward(self, x_wav, coeffs, label=None):
#         x_wav, coeffs = x_wav.unsqueeze(1), coeffs.unsqueeze(1)
#         x_t = self.tgramnet(x_wav).unsqueeze(1)
#         x_t1 = self.tgramnet(x_wav)
#         #x_mel_feq_att = self.feq_attention(coeffs).unsqueeze(1)
#         x_t_feq_att = self.feq_attention(x_t).unsqueeze(1)
#         x_t_self_att,_ = self.self_attention(x_t1)
#         #x = torch.cat((coeffs, x_t), dim=1)
#         #x = torch.cat((coeffs, x_t_feq_att), dim=1)
#         x = torch.cat((x_t, coeffs, x_t_feq_att, x_t_self_att.unsqueeze(1)), dim=1)
#         #x = torch.cat((x_t, x_mel_feq_att, x_mel_feq_att), dim=1)
#         #out, feature = self.shuffleNetV2_MetaACON(x, label)
#         #out, feature = self.shuffleFaceNet(x, label)
#         out, feature = self.mambaOut(x, label)
#         #out, feature = self.SwinTransformerBlock(x, label)
#         #out, feature = self.swinTransformer(x, label)
#         #out, feature = self.wavegramWaveNet(x, label)
#         #out, feature = self.effNetV2(x, label)
#         if self.arcface:
#             #out = self.arcface(feature, label)
#             #out = (self.arcface(feature, label) + self.arcface2(feature, label)) / 2.0
#             out = self.arcface2(feature, label)
#         return out, feature

class Feq_Attention(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_pool = nn.MaxPool1d(1)
        self.avg_pool = nn.AvgPool1d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, 1, 128, 313)
        x = x.squeeze(1)       # (B, 128, 313)
        x = x.transpose(1, 2)  # (B, 313, 128)
        x1 = self.max_pool(x)  # (B, 313, 1)
        x2 = self.avg_pool(x)  # (B, 313, 1)
        feats = x1 + x2        # (B, 313, 1)
        #feats = (3 * x1 + x2) / 2.0
        feats = feats.repeat(1, 1, 1)    # (B, 313, feature_dim)
        #print("feats shape:", feats.shape)  # [64, 313, 16384]
        #print("x transposed shape:", x.transpose(1, 2).shape)   # [64, 128, 313]
        refined_feats = self.sigmoid(feats).transpose(1, 2) * x.transpose(1, 2)  # (B, 16384, 313)
        return refined_feats


class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.2):
        super(SelfAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        N, C, T = x.shape  # 假设x的形状是(batch_size, channels, time_steps)
        x = x.view(N, T, C).permute(1, 0, 2)  # 重塑为(time_steps, batch_size, channels)并转置为(time_steps, batch, embed_dim)
        # 如果mask不是None，确保它的形状是(T, N)  加入掩码！！！有提升！！
        if mask is not None:
            assert mask.shape == (T, N), "Mask should have shape (time_steps, batch_size)"
        # 应用自注意力
        attention_output, attention_weights = self.attention(x, x, x, key_padding_mask=mask)
        # 投影并添加残差连接
        projected_output = self.proj(attention_output)
        output = self.norm(projected_output + x)
        # 恢复原始形状
        output = output.permute(1, 0, 2).reshape(N, C, T)
        return output, attention_weights

class CrossAttention(nn.Module):
    r""" Cross Attention Module
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 8
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: False.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        attn_head_dim (int, optional): Dimension of attention head.
        out_dim (int, optional): Dimension of output.
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_head_dim=None,
                 out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads   #16
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        assert all_head_dim == dim

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k, v):
        #print("x-t-shape", x.shape)
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = self.k_bias
            v_bias = self.v_bias

        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads,
                      -1).permute(2, 0, 3, 1,
                                  4).squeeze(0)  # (B, N_head, N_q, dim)

        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1,
                                                             4).squeeze(0)

        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1,
                                                             4).squeeze(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, N_head, N_q, N_k)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

if __name__ == '__main__':
    net = STgramMFN(num_classes=2025)
    x_wav = torch.randn((2, 16000*2))

