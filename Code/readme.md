

## Running the code:

To run the code, you can upload the **google_colab_code.ipynb** onto google colab and run using the GPU, below are the running time for the same parameters used for the experiments (CF readme).


- **Val Train:** 1.57
- **Val Loss:** 1.73
- **Training Time:** 10.26
- **Perplexity:** 6.62 +- 1.3

| **GPU/CPU**             | **Training Time (minutes)**                   | **GPU RAM** |
|----------------------------|-----------------------------------|-----------|
| **CPU**                     | Too long                                 |        |
| **A100 GPU**                | 4.49                          | 4.5/40 Gb       |
| **L4 GPU**                  | 7.31                                | 4.2/22.5       |
| **T4 GPU**                  | 14.16                          | 4.4/15 Gb         |

## Code structure for training, validation and test:

### 1. **Data Preparation**
- **Train/Test/Validation Split**: The dataset is split into **80% training**, **10% validation**, and **10% test**.
- **Character Encoding**: Text is encoded into integers using a character-to-integer mapping, and the dataset is tokenized for training.

```
train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1

text = "".join(lines)
chars = sorted(list(set(text)))
vocab_size = len(chars)
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])
```

### 2. **Batching Mechanism**
- **Sliding Window Batching**: Data is divided into overlapping windows of a fixed `block_size` length. 
  - For each window, the input is the text sequence `[i:i+block_size]` and the target is the next sequence `[i+1:i+block_size+1]`.
- Batches are created randomly by sampling a subset of these windows using the `get_sliding_window_batch` function.

```
def get_sliding_window_batch(dataset, step_size):
    data = dataset.data
    windows, targets = [], []
    for i in range(0, len(data) -block_size, step_size):
        windows.append(data[i:i + block_size])
        targets.append(data[i + 1:i + block_size + 1])
    windows = torch.stack(windows)
    targets = torch.stack(targets)
    indices = torch.randperm(len(windows))[:batch_size]
    return windows[indices].to(device), targets[indices].to(device)

```

### 3. **Evaluation**
- **Metric - Perplexity**: 
  - The model is evaluated on the test dataset by computing the average loss for non-overlapping windows of `block_size`.
  - Perplexity is calculated as the exponential of the average loss, representing how well the model predicts the test data (lower perplexity is better).

```
window_losses = []

for start in range(0, len(test_dataset.data) - block_size, block_size):
    xb = test_dataset.data[start:start + block_size].unsqueeze(0).to(device)
    yb = test_dataset.data[start + 1:start + block_size + 1].unsqueeze(0).to(device)

    with torch.no_grad():
        logits, loss = model(xb, yb)
        window_losses.append(loss.item())

avg_test_loss = np.mean(window_losses)
std_test_loss = np.std(window_losses)

perplexity = math.exp(avg_test_loss)
perplexity_std = math.exp(std_test_loss)

print(f"Test Perplexity: {perplexity:.3f} ± {perplexity_std:.3f}")
```

### References for the code (the two references are tutorials)

- [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [ Create a Large Language Model from Scratch with Python – Tutorial ](https://www.youtube.com/watch?v=UU1WVnMk4E8&t=8658s)