from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.transformer import _get_activation_fn

TOKEN_WHITE_LIST = [
    1,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
    154,
    155,
    156,
    157,
    158,
    159,
    160,
    161,
    162,
    163,
    164,
    165,
    166,
    167,
    168,
    169,
    170,
    171,
    172,
    173,
    174,
    175,
    176,
    177,
    178,
    179,
    180,
    181,
    182,
    183,
    184,
    185,
    186,
    187,
    188,
    189,
    190,
    191,
    192,
    193,
    194,
    195,
    196,
    197,
    198,
    199,
    200,
    201,
    202,
    203,
    204,
    205,
    206,
    207,
    208,
    209,
    210,
    211,
    212,
    213,
    214,
    215,
    216,
    217,
    218,
    219,
    220,
    221,
    222,
    223,
    224,
    225,
    226,
    227,
    228,
    229,
    230,
    231,
    232,
    233,
    234,
    235,
    236,
    237,
    238,
    239,
    240,
    241,
    242,
    243,
    244,
    245,
    246,
    247,
    248,
    249,
    250,
    251,
    252,
    253,
    254,
    255,
    256,
    257,
    258,
    259,
    260,
    261,
    262,
    263,
    264,
    265,
    266,
    267,
    268,
    269,
    270,
    271,
    272,
    273,
    274,
    275,
    276,
    277,
    278,
    279,
    280,
    281,
    282,
    283,
    284,
    285,
    286,
    287,
    288,
    289,
    290,
    291,
    292,
    293,
    294,
    295,
    296,
    297,
    298,
    299,
    300,
    301,
    302,
    303,
    304,
    305,
    306,
    307,
    308,
    309,
    310,
    311,
    312,
    313,
    314,
    315,
    316,
    317,
    318,
    319,
    320,
    321,
    322,
    323,
    324,
    325,
    326,
    327,
    328,
    329,
    330,
    331,
    332,
    333,
    334,
    335,
    336,
    337,
    338,
    339,
    340,
    341,
    342,
    343,
    344,
    345,
    346,
    347,
    348,
    349,
    350,
    351,
    352,
    353,
    354,
    355,
    356,
    357,
    358,
    359,
    360,
    361,
    362,
    363,
    364,
    365,
    366,
    367,
    368,
    369,
    370,
    371,
    372,
    373,
    374,
    375,
    376,
    377,
    378,
    379,
    380,
    381,
    382,
    383,
    384,
    385,
    386,
    387,
    388,
    389,
    390,
    391,
    392,
    393,
    394,
    395,
    396,
    397,
    398,
    399,
    400,
    401,
    402,
    403,
    404,
    405,
    406,
    407,
    408,
    409,
    410,
    411,
    412,
    413,
    414,
    415,
    416,
    417,
    418,
    419,
    420,
    421,
    422,
    423,
    424,
    425,
    426,
    427,
    428,
    429,
    430,
    431,
    432,
    433,
    434,
    435,
    436,
    437,
    438,
    439,
    440,
    441,
    442,
    443,
    444,
    445,
    446,
    447,
    448,
    449,
    450,
    451,
    452,
    453,
    454,
    455,
    456,
    457,
    458,
    459,
    460,
    461,
    462,
    463,
    464,
    465,
    466,
    467,
    468,
    469,
    470,
    471,
    472,
    473,
    474,
    475,
    476,
    477,
    478,
    479,
    480,
    481,
    482,
    483,
    484,
    485,
    486,
    487,
    488,
    489,
    490,
    491,
    492,
    493,
    494,
    495,
    496,
    497,
    498,
    499,
    500,
    501,
    502,
    503,
    504,
    505,
    506,
    507,
    508,
    509,
]


class ImgLinearBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        patch_size: int,
        in_chan: int = 3,
    ) -> None:
        super().__init__()

        self.conv_proj = nn.Conv2d(
            in_chan,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_proj(x)
        x = x.flatten(start_dim=-2).transpose(1, 2)
        return x


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.patch_size = 16
        self.d_model = 768
        self.dropout = 0
        self.activation = "gelu"
        self.norm_first = True
        self.ff_ratio = 4
        self.nhead = 12
        self.max_seq_len = 1024
        self.n_encoder_layer = 12
        encoder_layer = nn.TransformerEncoderLayer(
            self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.ff_ratio * self.d_model,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            norm_first=self.norm_first,
        )
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(self.d_model)
        self.backbone = ImgLinearBackbone(
            d_model=self.d_model, patch_size=self.patch_size
        )
        self.pos_embed = PositionEmbedding(
            max_seq_len=self.max_seq_len, d_model=self.d_model, dropout=self.dropout
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_encoder_layer, enable_nested_tensor=False
        )

    def forward(self, x: Tensor) -> Tensor:
        src_feature = self.backbone(x)
        src_feature = self.pos_embed(src_feature)
        memory = self.encoder(src_feature)
        memory = self.norm(memory)
        return memory


class PositionEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        # assume x is batch first
        if input_pos is None:
            _pos = torch.arange(x.shape[1], device=x.device)
        else:
            _pos = input_pos
        out = self.embedding(_pos)
        return self.dropout(out + x)


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int,
    ) -> None:
        super().__init__()
        assert vocab_size > 0
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    n_layer: int = 4
    n_head: int = 12
    dim: int = 768
    intermediate_size: int = None
    head_dim: int = 64
    activation: str = "gelu"
    norm_first: bool = True

    def __post_init__(self):
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head


class KVCache(nn.Module):
    def __init__(
        self,
        max_batch_size,
        max_seq_length,
        n_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cpu",
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer(
            "k_cache",
            torch.zeros(cache_shape, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(cache_shape, dtype=dtype, device=device),
            persistent=False,
        )

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        # assert input_pos.shape[0] == k_val.shape[2]

        bs = k_val.shape[0]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:bs, :, input_pos] = k_val
        v_out[:bs, :, input_pos] = v_val

        return k_out[:bs], v_out[:bs]


class GPTFastDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.vocab_size = 960
        self.padding_idx = 2
        self.prefix_token_id = 11
        self.eos_id = 1
        self.max_seq_len = 1024
        self.dropout = 0
        self.d_model = 768
        self.nhead = 12
        self.activation = "gelu"
        self.norm_first = True
        self.n_decoder_layer = 4
        config = ModelArgs(
            n_layer=self.n_decoder_layer,
            n_head=self.nhead,
            dim=self.d_model,
            intermediate_size=self.d_model * 4,
            activation=self.activation,
            norm_first=self.norm_first,
        )
        self.config = config
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.token_embed = TokenEmbedding(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            padding_idx=self.padding_idx,
        )
        self.pos_embed = PositionEmbedding(
            max_seq_len=self.max_seq_len, d_model=self.d_model, dropout=self.dropout
        )
        self.generator = nn.Linear(self.d_model, self.vocab_size)
        self.token_white_list = TOKEN_WHITE_LIST
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length, dtype, device):
        for b in self.layers:
            b.multihead_attn.k_cache = None
            b.multihead_attn.v_cache = None

        if (
            self.max_seq_length >= max_seq_length
            and self.max_batch_size >= max_batch_size
        ):
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size

        for b in self.layers:
            b.self_attn.kv_cache = KVCache(
                max_batch_size,
                max_seq_length,
                self.config.n_head,
                head_dim,
                dtype,
                device,
            )
            b.multihead_attn.k_cache = None
            b.multihead_attn.v_cache = None

        self.causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        ).to(device)

    def forward(self, memory: Tensor, tgt: Tensor) -> Tensor:
        input_pos = torch.tensor([tgt.shape[1] - 1], device=tgt.device, dtype=torch.int)
        tgt = tgt[:, -1:]
        tgt_feature = self.pos_embed(self.token_embed(tgt), input_pos=input_pos)
        # tgt = self.decoder(tgt_feature, memory, input_pos)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):
            logits = tgt_feature
            tgt_mask = self.causal_mask[None, None, input_pos]
            for i, layer in enumerate(self.layers):
                logits = layer(logits, memory, input_pos=input_pos, tgt_mask=tgt_mask)
        # return output
        logits = self.generator(logits)[:, -1, :]
        total = set([i for i in range(logits.shape[-1])])
        black_list = list(total.difference(set(self.token_white_list)))
        logits[..., black_list] = -1e9
        probs = F.softmax(logits, dim=-1)
        _, next_tokens = probs.topk(1)
        return next_tokens


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.self_attn = Attention(config)
        self.multihead_attn = CrossAttention(config)

        layer_norm_eps = 1e-5

        d_model = config.dim
        dim_feedforward = config.intermediate_size

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = config.norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = _get_activation_fn(config.activation)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor,
        input_pos: Tensor,
    ) -> Tensor:
        if self.norm_first:
            x = tgt
            x = x + self.self_attn(self.norm1(x), tgt_mask, input_pos)
            x = x + self.multihead_attn(self.norm2(x), memory)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = tgt
            x = self.norm1(x + self.self_attn(x, tgt_mask, input_pos))
            x = self.norm2(x + self.multihead_attn(x, memory))
            x = self.norm3(x + self._ff_block(x))
        return x

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.activation(self.linear1(x)))
        return x


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, 3 * config.dim)
        self.wo = nn.Linear(config.dim, config.dim)

        self.kv_cache: Optional[KVCache] = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.dim = config.dim

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_head * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y


class CrossAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        self.query = nn.Linear(config.dim, config.dim)
        self.key = nn.Linear(config.dim, config.dim)
        self.value = nn.Linear(config.dim, config.dim)
        self.out = nn.Linear(config.dim, config.dim)

        self.k_cache = None
        self.v_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim

    def get_kv(self, xa: torch.Tensor):
        if self.k_cache is not None and self.v_cache is not None:
            return self.k_cache, self.v_cache

        k = self.key(xa)
        v = self.value(xa)

        # Reshape for correct format
        batch_size, source_seq_len, _ = k.shape
        k = k.view(batch_size, source_seq_len, self.n_head, self.head_dim)
        v = v.view(batch_size, source_seq_len, self.n_head, self.head_dim)

        if self.k_cache is None:
            self.k_cache = k
        if self.v_cache is None:
            self.v_cache = v

        return k, v

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
    ):
        q = self.query(x)
        batch_size, target_seq_len, _ = q.shape
        q = q.view(batch_size, target_seq_len, self.n_head, self.head_dim)
        k, v = self.get_kv(xa)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        wv = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            is_causal=False,
        )
        wv = wv.transpose(1, 2).reshape(
            batch_size,
            target_seq_len,
            self.n_head * self.head_dim,
        )

        return self.out(wv)
