# Audio Spectrogram Transformer (AST) model

## Overview

The Audio Spectrogram Transformer (AST) model is designed for processing audio spectrograms using a transformer-based architecture. This README explains each component of the model, including the Patch Embedding, Sinusoidal Positional Embedding, Multi-head Self Attention, Transformer Encoder, and MLP Head. The AST model is particularly useful for tasks such as audio classification, where the input data is in the form of 2D spectrograms.

## Model architecture

The AST model consists of the following main components:

1. **Patch embedding**: converts the input spectrogram into a sequence of patches.
2. **Sinusoidal positional embedding**: adds positional information to the patch embeddings.
3. **Multi-head self attention**: captures dependencies between patches.
4. **Transformer encoder**: encodes the sequence of patches using multiple layers of self-attention and feed-forward networks.
5. **MLP head**: maps the encoded patches to the desired output classes.

## 1. Patch embedding

The `PatchEmbed` class splits the input spectrogram into non-overlapping patches and projects each patch into an embedding space.

**Input**: A spectrogram with dimensions `[B, C, H, W]`, where:
- `B`: Batch size
- `C`: Number of channels (1 for grayscale spectrograms)
- `H`: Height (frequency dimension)
- `W`: Width (time dimension)

**Output**: A sequence of patches with dimensions `[B, N, D]`, where:
- `N`: Number of patches, which is calculated as `(H // P) * (W // P)` where `P` is the patch size.
- `D`: Embedding dimension.

The patch embedding is achieved by applying a 2D convolution with a kernel size of `P` and a stride of `P`. This operation divides the input spectrogram into smaller patches, each of which is projected into an embedding space.

## 2. Sinusoidal positional embedding


The `SinusoidalPositionalEmbedding` class adds positional information to the patch embeddings. The sinusoidal embedding is deterministic and does not require learning.

### Explanation

Positional embeddings are necessary because the transformer architecture is permutation-invariant, meaning it doesn't inherently know the order of the input sequence. The sinusoidal positional embedding uses sine and cosine functions of different frequencies to create a unique positional encoding for each position in the input sequence.

For even positions, the positional encoding is derived using the sine function, and for odd positions, the cosine function is used. This ensures that each position in the sequence has a unique encoding, which helps the transformer understand the relative positions of the patches.

## 3. Multi-head self attention


The `MultiHeadSelfAttention` class implements the self-attention mechanism with multiple heads. This allows the model to capture relationships between different patches.

### Explanation

Self-attention allows the model to focus on different parts of the input sequence when making decisions. In multi-head self-attention, the input is split into multiple "heads," each of which computes its own attention score. These different attention scores are then combined to form a more comprehensive understanding of the input.

The attention mechanism works by creating three vectors for each input patch: a query vector, a key vector, and a value vector. The attention score for each patch is computed by taking the dot product of its query vector with the key vectors of all other patches, scaling by the square root of the dimension, and applying a softmax function. The final output is a weighted sum of the value vectors, where the weights are the attention scores.

## 4. Transformer encoder

The `TransformerEncoderLayer` class implements a single layer of the transformer encoder, which consists of a multi-head self-attention mechanism followed by a feed-forward neural network.

### Explanation

Each transformer encoder layer consists of two main components:

1. **Multi-head Self Attention**: This captures relationships between the patches.
2. **Feed-Forward Network**: This further processes the output from the attention mechanism.

Both components are followed by layer normalisation and residual connections, which help with the flow of gradients during training.

The `TransformerEncoder` class stacks multiple `TransformerEncoderLayer` instances to form the full encoder. The number of layers (or "depth") is a hyperparameter that you can tune based on the complexity of the task.

## 5. MLP head

The `MLPHead` class is a simple feed-forward network that maps the output of the transformer encoder to the desired output classes.

### Explanation

After processing the input patches through the transformer encoder, the final representation is passed through a multi-layer perceptron (MLP) to produce the output predictions. The MLP consists of a layer normalisation followed by a linear layer that maps the encoded features to the number of output classes.


@inproceedings{gong21b_interspeech,
  author={Yuan Gong and Yu-An Chung and James Glass},
  title={{AST: Audio Spectrogram Transformer}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={571--575},
  doi={10.21437/Interspeech.2021-698}
}

@ARTICLE{gong_psla, 
    author={Gong, Yuan and Chung, Yu-An and Glass, James},  
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},   
    title={PSLA: Improving Audio Tagging with Pretraining, Sampling, Labeling, and Aggregation},   
    year={2021}, 
    doi={10.1109/TASLP.2021.3120633}
}