# Trump Tweet Generator

A a Transformer-based model (similar to GPT) trained on Donald Trump's tweets. The model architecture consists of multi-head self-attention layers and feed-forward neural networks. The dataset used is the [(Better) - Donald Trump Tweets!](https://www.kaggle.com/datasets/kingburrito666/better-donald-trump-tweets/data) from Kaggle.

## Model Architecture

- **Single Head Attention**: Implements scaled dot-product attention with a single attention head.
- **Multi-Head Self Attention**: Stacks multiple attention heads for better performance.
- **Vanilla Neural Network**: Feed-forward network after the attention mechanism.
- **Transformer Block**: A combination of multi-head attention and feed-forward layers.
- **GPT Model**: Stacks multiple transformer blocks, uses token and positional embeddings, and predicts the next character in the sequence.

The model is trained using a context length of 128 characters, 252-dimensional embedding and hidden states, 6 transformer blocks, and 6 attention heads.