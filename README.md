# GPT Model Training from Scratch

A PyTorch implementation of a GPT (Generative Pre-trained Transformer) model trained from scratch using the TinyStories dataset. This project demonstrates how to build, train, and generate text with a transformer-based language model.

## Features

- **Custom GPT Architecture**: Full implementation of transformer blocks with multi-head attention
- **Flexible Training Pipeline**: Configurable model parameters and training settings
- **Text Generation**: Generate coherent text samples after training
- **Dataset Support**: Built-in support for TinyStories dataset with easy dataset switching
- **GPU Acceleration**: CUDA support for faster training

## Model Architecture

The GPT model includes:
- Multi-head self-attention mechanism
- Position and token embeddings
- Layer normalization
- Feed-forward networks with GELU activation
- Causal masking for autoregressive generation

## Requirements

```bash
pip install torch tiktoken datasets matplotlib pandas
```


## Quick Start

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd gpt-training
   ```

2. **Run the Jupyter notebook**
    run all the cells except the \_\_main__ for training

    

3. **Execute cells sequentially** to:
   - Install dependencies
   - Load and preprocess data
   - Train the model
   - Generate text samples

## Configuration

### Model Parameters (GPT_CONFIG_124M)

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size (GPT-2 tokenizer)
    "context_length": 256,  # Maximum sequence length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of transformer layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-key-value bias
}
```

**Key Parameters to Modify:**
- `context_length`: Increase for longer context (more memory usage)
- `emb_dim`: Model size - larger = more capacity but slower training
- `n_layers`: Model depth - more layers = better performance but slower
- `drop_rate`: Regularization - increase if overfitting

### Training Settings (OTHER_SETTINGS)

```python
OTHER_SETTINGS = {
    "learning_rate": 5e-4,  # Learning rate
    "num_epochs": 20,       # Number of training epochs
    "batch_size": 8,        # Batch size
    "weight_decay": 0.1     # Weight decay for regularization
}
```

**Training Parameters to Adjust:**
- `learning_rate`: Lower for stable training, higher for faster convergence
- `batch_size`: Increase if you have more GPU memory
- `num_epochs`: More epochs = better training but risk of overfitting
- `weight_decay`: Regularization strength

## Dataset Configuration

### Using TinyStories (Default)

The notebook uses the TinyStories dataset by default:

```python
df = load_dataset("roneneldan/TinyStories")
df = df['train'].to_pandas(batch_size=50000)
df = df['text'].to_list()
```

### Switching to Custom Dataset

To use your own text dataset, replace the dataset loading section:

#### Option 1: Text File
```python
# Replace the dataset loading with:
with open('your_text_file.txt', 'r', encoding='utf-8') as f:
    text_data = f.read()

# Convert to list format
df = [text_data]
```

#### Option 2: Multiple Text Files
```python
import glob

text_files = glob.glob('path/to/your/texts/*.txt')
df = []
for file in text_files:
    with open(file, 'r', encoding='utf-8') as f:
        df.append(f.read())
```

#### Option 3: HuggingFace Dataset
```python
# For other HuggingFace datasets:
df = load_dataset("dataset_name")
df = df['train']['text']  # Adjust field name as needed
```

### Dataset Preprocessing Parameters

```python
# In create_dataloader_v1 function:
train_loader = create_dataloader_v1(
    text_data[:split_idx],
    batch_size=settings["batch_size"],
    max_length=gpt_config["context_length"],  # Sequence length
    stride=gpt_config["context_length"],      # Overlap between sequences
    drop_last=True,
    shuffle=True,
    num_workers=0
)
```

**Preprocessing Options:**
- `max_length`: Length of each training sequence
- `stride`: Step size for sliding window (smaller = more overlap)
- `train_ratio`: Proportion of data for training (default: 0.90)

## Usage Examples

### Training with Different Model Sizes

**Small Model (Faster Training):**
```python
GPT_CONFIG_SMALL = {
    "vocab_size": 50257,
    "context_length": 128,   # Reduced context
    "emb_dim": 384,          # Smaller embedding
    "n_heads": 6,            # Fewer heads
    "n_layers": 6,           # Fewer layers
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

**Large Model (Better Performance):**
```python
GPT_CONFIG_LARGE = {
    "vocab_size": 50257,
    "context_length": 512,   # Longer context
    "emb_dim": 1024,         # Larger embedding
    "n_heads": 16,           # More heads
    "n_layers": 24,          # More layers
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

### Text Generation

After training, generate text samples:

```python
# Load trained model
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load('model.pth', weights_only=True))
model.to(device)

# Generate text
tokenizer = tiktoken.get_encoding("gpt2")
generate_and_print_sample(model, tokenizer, device, "Your prompt here")
```

## Training Tips

1. **Start Small**: Begin with a smaller model to verify everything works
2. **Monitor Loss**: Training loss should decrease steadily
3. **Validation Loss**: Should follow training loss without large gaps
4. **GPU Memory**: Reduce batch_size if you encounter out-of-memory errors
5. **Checkpointing**: Save model checkpoints during long training runs

## File Structure

```
├── Trained.ipynb          # Main training notebook
├── model.pth             # Saved model weights (after training)
├── loss.pdf              # Training loss plot (generated)
└── README.md             # This file
```

## Troubleshooting

### Common Issues

**Out of Memory Error:**
- Reduce `batch_size` in OTHER_SETTINGS
- Reduce `context_length` in model config
- Use smaller model dimensions

**Slow Training:**
- Ensure GPU is being used (`torch.cuda.is_available()`)
- Increase `batch_size` if memory allows
- Reduce model size for experimentation

**Poor Text Generation:**
- Train for more epochs
- Increase model size
- Use larger/better dataset
- Adjust learning rate

### Hardware Requirements

- **Minimum**: 8GB RAM, any GPU with 4GB+ VRAM
- **Recommended**: 16GB+ RAM, GPU with 8GB+ VRAM
- **Training Time**: 
  - Small model: ~30 minutes on modern GPU
  - Default model: 1-3 hours on modern GPU
  - Large model: 4+ hours on modern GPU

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Based on the GPT architecture from "Attention Is All You Need"
- Uses TinyStories dataset for training
- Inspired by Andrej Karpathy's educational materials
