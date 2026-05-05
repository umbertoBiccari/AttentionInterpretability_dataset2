import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureTokenizer(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int):
        super().__init__()
        self.value_weight = nn.Parameter(torch.randn(in_dim, emb_dim) * 0.02)
        self.feature_embedding = nn.Parameter(torch.randn(in_dim, emb_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, d]
        x_exp = x.unsqueeze(-1)                           # [B, d, 1]
        tokens = x_exp * self.value_weight.unsqueeze(0)  # [B, d, emb_dim]
        tokens = tokens + self.feature_embedding.unsqueeze(0)
        return tokens

class TransformerEncoderLayerWithAttn(nn.Module):
    """
    Pre-norm transformer encoder block with explicit attention return.
    """
    def __init__(self, emb_dim: int, nhead: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

        self.linear1 = nn.Linear(emb_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, emb_dim)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        self.dropout_act = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, return_attn: bool = False):
        # Pre-norm attention block
        x = self.norm1(src)
        attn_out, attn_weights = self.self_attn(
            x, x, x,
            need_weights=return_attn,
            average_attn_weights=False
        )
        src = src + self.dropout_attn(attn_out)

        # Pre-norm FFN block
        x = self.norm2(src)
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout_act(x)
        x = self.linear2(x)
        src = src + self.dropout_ff(x)

        if return_attn:
            # attn_weights: [B, H, N, N]
            return src, attn_weights
        return src

class TransformerTrunkWithAttn(nn.Module):
    def __init__(
        self,
        in_dim: int,
        emb_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(in_dim, emb_dim)
        self.use_cls_token = use_cls_token

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
            nn.init.normal_(self.cls_token, std=0.02)

        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttn(
                emb_dim=emb_dim,
                nhead=nhead,
                ff_dim=ff_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        # z: [B, d, emb_dim]
        z = self.tokenizer(x)
        B = z.size(0)

        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)   # [B, 1, emb_dim]
            z = torch.cat([cls, z], dim=1)           # [B, 1+d, emb_dim]

        attn_layers = []

        for layer in self.layers:
            if return_attn:
                z, attn = layer(z, return_attn=True)
                attn_layers.append(attn)
            else:
                z = layer(z, return_attn=False)

        z = self.norm(z)

        if self.use_cls_token:
            h = z[:, 0, :]          # CLS representation
        else:
            h = z.mean(dim=1)       # fallback mean pooling

        if return_attn:
            return h, attn_layers
        return h

class TwoHeadTransformerNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        emb_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.trunk = TransformerTrunkWithAttn(
            in_dim=in_dim,
            emb_dim=emb_dim,
            nhead=nhead,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            use_cls_token=use_cls_token,
        )
        self.head_am = nn.Linear(emb_dim, 2)
        self.head_sex = nn.Linear(emb_dim, 1)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if return_attn:
            h, attn_layers = self.trunk(x, return_attn=True)
            am = self.head_am(h)
            s = self.head_sex(h)
            y = torch.cat([am, s], dim=1)
            return y, attn_layers

        h = self.trunk(x, return_attn=False)        
        am = self.head_am(h)
        s = self.head_sex(h)
        return torch.cat([am, s], dim=1)


class SingleOutTransformerNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        emb_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.trunk = TransformerTrunkWithAttn(
            in_dim=in_dim,
            emb_dim=emb_dim,
            nhead=nhead,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            use_cls_token=use_cls_token,
        )
        self.head = nn.Linear(emb_dim, 1)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if return_attn:
            h, attn_layers = self.trunk(x, return_attn=True)
            y = self.head(h)
            return y, attn_layers

        h = self.trunk(x, return_attn=False)
        return self.head(h)