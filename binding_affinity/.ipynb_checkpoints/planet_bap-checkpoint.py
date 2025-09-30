import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def ensure_bool_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Ensure mask is boolean. Returns None if mask is None.
    Expected True=pad, False=valid for key_padding_mask.
    HF attention_mask (1=valid,0=pad) -> convert to True=pad/False=valid
    """
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        return mask
    return (mask == 0)

class DilatedConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, dilation=d, bias=False
        )

    def forward(self, x):
        return self.conv(x)

class DilatedConvBlock(nn.Module):
    """ (B,C_in,L) -> (B,C_out,L) """
    def __init__(self, nIn, nOut):
        super().__init__()
        n = nOut // 4
        n1 = nOut - 3 * n
        self.c1 = nn.Conv1d(nIn, n, 1)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = DilatedConv(n, n1, 3, d=1)
        self.d2 = DilatedConv(n, n, 3, d=2)
        self.d4 = DilatedConv(n, n, 3, d=4)
        self.d8 = DilatedConv(n, n, 3, d=8)
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

    def forward(self, x):
        out = self.c1(x)
        out = self.br1(out)
        d1 = self.d1(out)
        d2 = self.d2(out)
        d4 = self.d4(out)
        d8 = self.d8(out)
        combine = torch.cat([d1, d2, d4, d8], dim=1)
        out = self.br2(combine)
        return out

class LigandEmbedder(nn.Module):
    def __init__(
        self,
        chembert_model: nn.Module,
        global_feat_dim: int = 10,
        target_dim: int = 128,
        g_dim: int = 128,
        freeze_chembert: bool = True,
        positive_weight: bool = True,
    ):
        super().__init__()
        self.chembert = chembert_model
        self.freeze_chembert = freeze_chembert
        self.global_fc = nn.Linear(global_feat_dim, g_dim)
        if positive_weight:
            self.global_weight_logit = nn.Parameter(torch.zeros(g_dim))
            self._pos = True
        else:
            self.global_weight = nn.Parameter(torch.ones(g_dim))
            self._pos = False
        hidden_size = getattr(self.chembert.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("chembert_model.config.hidden_size 없음")
        self.proj = nn.Linear(hidden_size + g_dim, target_dim)
        if self.freeze_chembert:
            for p in self.chembert.parameters():
                p.requires_grad = False
            self.chembert.eval()

    def _scale_global(self, global_expand):
        if self._pos:
            w = F.softplus(self.global_weight_logit) + 1e-6
        else:
            w = self.global_weight
        return global_expand * w

    def forward(self, input_ids, attention_mask, global_feat):
        if self.freeze_chembert:
            with torch.no_grad():
                chem_out = self.chembert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            chem_out = self.chembert(input_ids=input_ids, attention_mask=attention_mask)

        chembert_embed = chem_out.last_hidden_state 
        g = self.global_fc(global_feat)              
        g_expand = g.unsqueeze(1).expand(-1, chembert_embed.size(1), -1)
        g_scaled = self._scale_global(g_expand)
        concat_embed = torch.cat([chembert_embed, g_scaled], dim=-1)
        out = self.proj(concat_embed)               
        return out

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv, kv_mask: Optional[torch.Tensor], need_weights=False):
        B, Lq, _ = q.size()
        B, Lk, _ = kv.size()
        q_proj = self.q_proj(q).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        k_proj = self.k_proj(kv).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v_proj = self.v_proj(kv).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) * self.scale  # (B,h,Lq,Lk)

        if kv_mask is not None:
            kv_mask_exp = kv_mask.unsqueeze(1).unsqueeze(2).clone() 
            all_pad = kv_mask.all(dim=1)
            if all_pad.any():
                kv_mask_exp[all_pad, :, :, -1] = False
            scores = scores.masked_fill(kv_mask_exp, -1e4)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v_proj)  
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.out_proj(out)

        x = self.ln1(q + self.dropout(out))
        x2 = self.ffn(x)
        out = self.ln2(x + self.dropout(x2))
        return (out, attn if need_weights else None)

class ProteinLigandModel(nn.Module):
    def __init__(
        self,
        pt_feat_size=1286,
        pt_embed=256,
        d_model=128,
        chembert_model=None,
        global_feat_dim=10,
        g_dim=128,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        if chembert_model is None:
            raise ValueError("chembert_model must be provided")

        self.pt_linear = nn.Linear(pt_feat_size, pt_embed)
        self.pt_conv = DilatedConvBlock(pt_embed, d_model)
        self.pt_pool = nn.AdaptiveMaxPool1d(1)

        self.lig_embedder = LigandEmbedder(
            chembert_model=chembert_model,
            global_feat_dim=global_feat_dim,
            target_dim=d_model,
            g_dim=g_dim,
            freeze_chembert=True,
        )

        self.cross_pt_to_lig = CrossAttentionBlock(d_model, num_heads, dropout)
        self.cross_lig_to_pt = CrossAttentionBlock(d_model, num_heads, dropout)

        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        pt_seq,
        smi_seq,
        smi_mask,
        lig_global,
        pt_mask: Optional[torch.Tensor] = None,
        return_attn=False,
    ):

        pt = self.pt_linear(pt_seq).transpose(1, 2) 
        pt = self.pt_conv(pt)                     
        pt_seq_feats = pt.transpose(1, 2)             
        pt_query = self.pt_pool(pt).squeeze(-1).unsqueeze(1)  

        lig_feats = self.lig_embedder(smi_seq, smi_mask, lig_global)  
        if smi_mask.dtype != torch.float32:
            smi_mask = smi_mask.float()
        lig_query = (lig_feats * smi_mask.unsqueeze(-1)).sum(1, keepdim=True) / (
            smi_mask.sum(1, keepdim=True).clamp_min(1e-6).unsqueeze(-1)
        )

        lig_kpad = ensure_bool_mask(smi_mask.bool())
        pt_kpad = ensure_bool_mask(pt_mask)

        pt_ref, attn_pl = self.cross_pt_to_lig(pt_query, lig_feats, lig_kpad, need_weights=return_attn)
        lig_ref, attn_lp = self.cross_lig_to_pt(lig_query, pt_seq_feats, pt_kpad, need_weights=return_attn)

        z = torch.cat([pt_ref.squeeze(1), lig_ref.squeeze(1)], dim=1)
        out = self.fc(z)

        if return_attn:
            return out, {"pt_to_lig": attn_pl, "lig_to_pt": attn_lp}
        return out
