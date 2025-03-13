import torch.nn as nn
import torch

class ECA(nn.Module):
    def __init__(self, kernel_size=5):
        """
        Input:
        - kernel_size: int, size of the convolution kernel (default is 5).

        Output:
        - Initializes an ECA block for adaptive channel attention.

        Description: Implements an Efficient Channel Attention (ECA) block that enhances the representation 
                     of the input tensor by applying a 1D convolution followed by a sigmoid activation.
        """
        super(ECA, self).__init__()
        self.conv = nn.Conv1d(192, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
    
    def forward(self, inputs):
        """
        Input:
        - inputs: tensor of shape [B, C, T], where B is the batch size, C is the number of channels, 
                  and T is the sequence length.

        Output:
        - Tensor with the same shape as inputs after applying ECA.

        Description: Computes the channel-wise attention using global average pooling, applies the 
                     convolution, and scales the input tensor with the attention weights.
        """
        nn = inputs.mean(dim=-1, keepdim=True)
        nn = self.conv(nn)
        nn = torch.sigmoid(nn)
        return inputs * nn

class CausalDWConv1D(nn.Module):
    def __init__(self, in_channels, kernel_size=17, dilation_rate=1, padding='causal'):
        """
        Input:
        - in_channels: int, number of input channels.
        - kernel_size: int, size of the convolution kernel (default is 17).
        - dilation_rate: int, dilation rate for the convolution (default is 1).
        - padding: str, type of padding to use (default is 'causal').

        Output:
        - Initializes a causal depthwise convolutional layer.

        Description: Implements a causal depthwise convolutional layer to prevent information leakage 
                     from future time steps in sequential data.
        """
        super(CausalDWConv1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, dilation=dilation_rate, padding=0, bias=False)
        self.pad = nn.ConstantPad1d((dilation_rate * (kernel_size - 1), 0), 0)

    def forward(self, x):
        """
        Input:
        - x: tensor of shape [B, C, T], where B is the batch size, C is the number of channels, 
              and T is the sequence length.

        Output:
        - Processed tensor after causal convolution.

        Description: Applies causal padding and depthwise convolution to the input tensor.
        """
        x = self.pad(x)
        x = self.conv(x)
        return x

class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=17, dilation_rate=1, drop_rate=0.2):
        """
        Input:
        - in_channels: int, number of input channels.
        - out_channels: int, number of output channels.
        - kernel_size: int, size of the convolution kernel (default is 17).
        - dilation_rate: int, dilation rate for the convolution (default is 1).
        - drop_rate: float, dropout rate (default is 0.2).

        Output:
        - Initializes a 1D convolutional block.

        Description: Combines convolution, batch normalization, ReLU activation, 
                     Efficient Channel Attention (ECA), and dropout.
        """
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, dilation=dilation_rate)
        self.bn = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(drop_rate)
        self.eca = ECA()

    def forward(self, x):
        """
        Input:
        - x: tensor of shape [B, C, T].

        Output:
        - Processed tensor after passing through the convolutional block.

        Description: Applies convolution, batch normalization, activation, ECA, and dropout to the input tensor.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.eca(x)
        x = self.drop(x)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.2):
        """
        Input:
        - dim: int, the dimensionality of the input tensor.
        - num_heads: int, number of attention heads.
        - dropout: float, dropout rate (default is 0.2).

        Output:
        - Initializes a multi-head self-attention layer.

        Description: Implements a multi-head self-attention mechanism with scaling and dropout.
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Input:
        - x: tensor of shape [B, N, C], where B is the batch size, N is the number of tokens, 
              and C is the dimension.

        Output:
        - Processed tensor after applying multi-head self-attention.

        Description: Computes the attention scores, applies softmax, and performs 
                     the weighted sum of values, followed by a linear projection.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each has shape [B, num_heads, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, expand_ratio=4, dropout=0.2):
        """
        Input:
        - dim: int, the dimensionality of the input tensor.
        - num_heads: int, number of attention heads.
        - expand_ratio: int, ratio for expanding the dimensionality in the feed-forward network (default is 4).
        - dropout: float, dropout rate (default is 0.2).

        Output:
        - Initializes a transformer block.

        Description: Combines multi-head self-attention and a feed-forward network with layer normalization.
        """
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expand_ratio),
            nn.ReLU(),
            nn.Linear(dim * expand_ratio, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Input:
        - x: tensor of shape [B, N, C].

        Output:
        - Processed tensor after passing through the transformer block.

        Description: Applies self-attention and feed-forward layers with residual connections and normalization.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class CNN1DTransformer(nn.Module):
    def __init__(self, num_classes, seq_len=103, dim=192, num_heads=4, dropout=0.2):
        """
        Input:
        - num_classes: int, number of output classes for classification.
        - seq_len: int, length of the input sequences (default is 103).
        - dim: int, dimensionality of the feature space (default is 192).
        - num_heads: int, number of attention heads (default is 4).
        - dropout: float, dropout rate (default is 0.2).

        Output:
        - Initializes a CNN-Transformer model for sequence classification.

        Description: Combines several convolutional blocks followed by a transformer block for 
                     processing sequential data, concluding with a linear classification layer.
        """
        super(CNN1DTransformer, self).__init__()

        self.conv1 = Conv1DBlock(164, dim, kernel_size=17, drop_rate=dropout)
        self.conv2 = Conv1DBlock(dim, dim, kernel_size=17, drop_rate=dropout)
        self.conv3 = Conv1DBlock(dim, dim, kernel_size=17, drop_rate=dropout)

        self.transformer = TransformerBlock(dim=dim, num_heads=num_heads, dropout=dropout)

        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        """
        Input:
        - x: tensor of shape [batch_size, 164, seq_len].

        Output:
        - Tensor of shape [batch_size, num_classes] representing class scores.

        Description: Processes the input through convolutional layers, 
                     transformer block, and a final classification layer.
        """
        # Input shape: [batch_size, 164, seq_len]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Permute to [batch_size, seq_len, dim] for the Transformer
        x = x.permute(0, 2, 1)

        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Final classification layer
        x = self.fc(x)
        return x


