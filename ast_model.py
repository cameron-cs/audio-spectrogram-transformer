import torch
import torch.nn as nn
import math


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        # create and register the sinusoidal positional embeddings as a buffer
        self.register_buffer('positional_embedding', self.create_sinusoidal_embeddings(num_patches, embed_dim))

    def create_sinusoidal_embeddings(self, num_patches, embed_dim):
        # compute the positional embeddings using sine and cosine functions
        position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        sinusoidal_embedding = torch.zeros(num_patches, embed_dim)
        sinusoidal_embedding[:, 0::2] = torch.sin(position * div_term)  # sine to even indices
        sinusoidal_embedding[:, 1::2] = torch.cos(position * div_term)  # cosine to odd indices
        return sinusoidal_embedding.unsqueeze(0)  # batch dimension

    def forward(self, x):
        # positional embeddings to the input
        return x + self.positional_embedding[:, :x.size(1), :]


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.embed_dim = embed_dim

        # convolutional projection to generate patch embeddings
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # the convolutional projection and flatten the patches
        x = self.proj(x).flatten(2).transpose(1, 2)  # flatten and transpose for transformer input
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5  # scaling factor for the attention mechanism

        # linear layers to compute queries, keys, and values
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(0.1)  # dropout for attention scores
        self.proj = nn.Linear(embed_dim, embed_dim)  # linear layer to project the output
        self.proj_drop = nn.Dropout(0.1)  # dropout for the projected output

    def forward(self, x):
        B, N, C = x.shape  # batch size, number of patches, embedding dimension
        # compute queries, keys, and values
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # compute scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # combine the attention output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)  # the final linear projection
        x = self.proj_drop(x)  # dropout
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., qkv_bias=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)  # layer normalisation before attention
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(embed_dim)  # layer normalisation before the feedforward network
        # feedforward network (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),  # activation function
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x):
        # attention and add residual connection
        x = x + self.attn(self.norm1(x))
        # feedforward network and add residual connection
        x = x + self.mlp(self.norm2(x))
        return x


# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, mlp_ratio=4., qkv_bias=False):
        super().__init__()
        # stack multiple Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, qkv_bias)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)  # final layer normalisation

    def forward(self, x):
        # pass the input through each layer in the encoder
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)  # normalise the output


# MLP Head for classification
class MLPHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)  # normalise the input
        self.fc = nn.Linear(embed_dim, num_classes)  # final linear layer to produce class scores

    def forward(self, x):
        x = self.norm(x)  # normalise the input
        x = self.fc(x)  # the final linear layer
        return x


class ASTModel(nn.Module):
    def __init__(self,
                 embed_dim=768,
                 depth=6,
                 num_heads=6,
                 label_dim=527,
                 input_fdim=128,
                 input_tdim=1024):
        super(ASTModel, self).__init__()
        # patch embedding layer
        self.patch_embed = PatchEmbed(img_size=input_tdim, patch_size=16, in_chans=1, embed_dim=embed_dim)

        # n of patches after patch embedding
        num_patches = (input_tdim // 16) * (input_fdim // 16)

        # sinusoidal positional embedding
        self.pos_embed = SinusoidalPositionalEmbedding(num_patches + 1, embed_dim)

        # classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=.02)  # initialise the classification token

        # transformer encoder with stacked layers
        self.encoder = TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads)
        # MLP head for final classification
        self.mlp_head = MLPHead(embed_dim=embed_dim, num_classes=label_dim)

    def forward(self, x):
        B = x.shape[0]  # batch size
        x = x.unsqueeze(1)  # add channel dimension (for grayscale spectrograms)
        x = self.patch_embed(x)  # apply patch embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)  # expand the classification token across the batch
        x = torch.cat((cls_tokens, x), dim=1)  # concatenate the classification token with the patch embeddings

        # sinusoidal positional embedding
        x = self.pos_embed(x)

        x = self.encoder(x)  # pass through the transformer encoder
        x = x[:, 0]  # extract the output corresponding to the classification token
        x = self.mlp_head(x)  # pass through the MLP head for classification
        return x


if __name__ == '__main__':
    input_tdim = 100
    ast_mdl = ASTModel(input_tdim=input_tdim)
    test_input = torch.rand([10, input_tdim, 128])
    test_output = ast_mdl(test_input)
    print(test_output.shape)

    input_tdim = 256
    ast_mdl = ASTModel(input_tdim=input_tdim, label_dim=50)
    test_input = torch.rand([10, input_tdim, 128])
    test_output = ast_mdl(test_input)
    print(test_output.shape)
