import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class GeLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return gelu(x)

class DenseASPP1D(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=(1, 2, 3),
                 kernel=3, dropout=0.1, bottleneck_ratio=0.5,
                 use_gap=True):
        super().__init__()
        self.use_gap = use_gap

        self.blocks = nn.ModuleList()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        bn_ch = max(1, int(out_channels * bottleneck_ratio))

        for i, d in enumerate(dilations):
            in_ch = in_channels + i * out_channels
            pad = d * (kernel - 1) // 2
            self.blocks.append(nn.Sequential(
                # 1x1 bottleneck
                nn.Conv1d(in_ch, bn_ch, kernel_size=1, bias=False),
                nn.ReLU(),
                # dilated kx1
                nn.Conv1d(bn_ch, out_channels,
                          kernel_size=kernel, dilation=d, padding=pad, bias=False),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))

        if self.use_gap:
            self.gap_conv_input = nn.Conv1d(in_channels, out_channels, 1, bias=True)
            nn.init.xavier_uniform_(self.gap_conv_input.weight)
            self.gap_conv_concat = nn.Conv1d(in_channels + len(dilations) * out_channels,
                                             out_channels, 1, bias=True)
            nn.init.xavier_uniform_(self.gap_conv_concat.weight)

        concat_ch = in_channels + len(dilations) * out_channels
        if self.use_gap:
            concat_ch += out_channels  # GAP branch가 out_channels를 하나 더 추가
            
        self.project = nn.Conv1d(concat_ch, out_channels, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.project.weight)

        self.ln = nn.LayerNorm(out_channels)

    def forward(self, x):
        B, C, L = x.shape

        feats = [x]
        cur = x
        for blk in self.blocks:
            cur = blk(torch.cat(feats, dim=1))
            feats.append(cur)

        y = torch.cat(feats, dim=1)

        if self.use_gap:
            gap = F.adaptive_avg_pool1d(x, 1)                  
            gap = self.gap_conv_input(gap)                     

            gap = F.interpolate(gap, size=L, mode="linear", align_corners=False)  
            y = torch.cat([y, gap], dim=1) 

        y = self.project(y)                         
        y = self.ln(y.transpose(1, 2)).transpose(1, 2)
        return self.dropout(y)

class ChannelProj1x1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.ReLU(),
        )
        nn.init.xavier_uniform_(self.net[0].weight)

    def forward(self, x): 
        return self.net(x)

class Fusion1D(nn.Module):
    def __init__(self, hidden=256, mode="concat_project", dropout=0.1):
        super().__init__()
        self.mode = mode
        self.dropout = nn.Dropout(dropout)

        if mode == "concat_project":
            self.proj = nn.Conv1d(2*hidden, hidden, kernel_size=1, bias=False)
            nn.init.xavier_uniform_(self.proj.weight)
            self.act = nn.ReLU()

    def forward(self, x1, x2): 
        if self.mode == "sum":
            return x1 + x2
        else: 
            y = torch.cat([x1.transpose(1, 2), x2.transpose(1, 2)], dim=1) 
            y = self.proj(y)                                               
            y = self.act(y)
            y = self.dropout(y)
            return y.transpose(1, 2)                                      

class PocketConvLayer(nn.Module):
    def __init__(self,
                 config,
                 in_ch=1280, mid_ch=512, out_ch=256,    
                 kernel=3,
                 dilations=(1, 2, 3),
                 dropout=0.1,
                 use_residual=True,
                 # DenseASPP 옵션
                 use_gap=True,
                 bottleneck_ratio=0.5):
        super().__init__()
        self.config = config
        self.use_residual = use_residual

        self.proj = ChannelProj1x1(in_ch, mid_ch)
        self.post_proj_dropout = nn.Dropout(0.1)

        self.context = DenseASPP1D(
            in_channels=mid_ch,
            out_channels=out_ch,
            dilations=dilations,
            kernel=kernel,
            dropout=dropout,
            bottleneck_ratio=bottleneck_ratio,
            use_gap=use_gap
        )
        if self.use_residual:
            self.match_skip = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
            nn.init.xavier_uniform_(self.match_skip.weight)

    def forward(self, aa_embeddings):
        x = aa_embeddings.transpose(1, 2)  # (B, in, L)

        if self.use_residual:
            skip = self.match_skip(x)       # (B, out, L)

        x = self.proj(x)                    # (B, mid, L)
        x = self.post_proj_dropout(x)

        x = self.context(x)                 # (B, out, L)

        if self.use_residual:
            x = x + skip

        return x.transpose(1, 2)            # (B, L, out)

class Planet_X(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden = hidden = int(config["architectures"]["hidden_size"])
        self.dropout = nn.Dropout(float(self.config["train"]["dropout"]))

        max_len = int(config["prots"]["max_lengths"])

        self.LayerNorm_raw  = nn.LayerNorm(1280, eps=1e-12)   # total_prots_data 정규화

        self.protein_features = nn.Linear(1280, hidden)

        arch = config["architectures"]

        # DenseASPP 옵션
        dense_use_gap   = bool(arch.get("denseaspp_use_gap", True))
        dense_bn_ratio  = float(arch.get("denseaspp_bottleneck_ratio", 0.5))

        k3_dils         = tuple(arch.get("k3_dilations", (1, 2, 3)))
        k5_dils         = tuple(arch.get("k5_dilations", (2, 4, 6)))
        context_dropout = float(arch.get("context_dropout", 0.3))

        self.branch_k3 = PocketConvLayer(
            config=self.config,
            in_ch=1280, mid_ch=512, out_ch=hidden,
            kernel=3, dilations=k3_dils,
            dropout=context_dropout, use_residual=True,
            use_gap=dense_use_gap, bottleneck_ratio=dense_bn_ratio,
        )
        self.branch_k5 = PocketConvLayer(
            config=self.config,
            in_ch=1280, mid_ch=512, out_ch=hidden,
            kernel=5, dilations=k5_dils,
            dropout=context_dropout, use_residual=True,
            # DenseASPP 옵션
            use_gap=dense_use_gap, bottleneck_ratio=dense_bn_ratio,
        )

        fusion_mode = arch.get("fusion_mode", "concat_project")
        self.fusion = Fusion1D(hidden=hidden, mode=fusion_mode, dropout=0.1)

        clf_in = 2 * hidden
        self.classifier = nn.Sequential(
            GeLU(),
            nn.LayerNorm(clf_in, eps=1e-12),
            self.dropout,
            nn.Linear(clf_in, hidden // 2 if hidden >= 128 else max(64, hidden // 2)),
            GeLU(),
            nn.LayerNorm(hidden // 2 if hidden >= 128 else max(64, hidden // 2), eps=1e-12),
            self.dropout,
            nn.Linear(hidden // 2 if hidden >= 128 else max(64, hidden // 2), max(64, hidden // 4)),
            GeLU(),
            nn.LayerNorm(max(64, hidden // 4), eps=1e-12),
            self.dropout,
            nn.Linear(max(64, hidden // 4), 1),
        )

    @torch.no_grad()
    def _make_ext_mask(self, attention_mask, dtype, device=None):
        m = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=torch.float32)
        return (1.0 - m) * -1e4

    def forward(self, prots, total_prots_data, attention_mask, position_ids, token_type_ids):

        B, L, _ = prots.shape

        feats = prots
        feats_base = feats  # 두 분기에 동일 입력

        prot_feats = self.LayerNorm_raw(total_prots_data)              # (B,L,1280)
        prot_feats = self.protein_features(prot_feats)                 # (B,L,H)
        prot_feats = self.dropout(prot_feats)

        f1 = self.branch_k3(feats_base)   
        f2 = self.branch_k5(feats_base)   
        fused = self.fusion(f1, f2)                                  

        cat = torch.cat([fused, prot_feats], dim=-1)                   
        logits = self.classifier(cat).squeeze(-1)                          
        return fused, logits
