import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_attention_weights(attention_weights, tokens, layer=0, head=0,
                           save_path='attention_weights.png'):
    """
    Plot attention weights for visualization
    """
    if len(attention_weights.shape) == 4:  # [batch, heads, seq_len, seq_len]
        attn = attention_weights[0, head].cpu().detach().numpy()
    else:
        attn = attention_weights.cpu().detach().numpy()

    plt.figure(figsize=(10, 8))
    plt.imshow(attn, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Attention Weights - Layer {layer}, Head {head}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')

    if tokens is not None:
        plt.xticks(range(len(tokens)), tokens, rotation=45)
        plt.yticks(range(len(tokens)), tokens)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_text(model, start_string, dataset, length=100, temperature=1.0):
    """
    Generate text using the trained model
    """
    model.eval()
    chars = [ch for ch in start_string]

    with torch.no_grad():
        for _ in range(length):
            # Convert current sequence to tensor
            input_ids = torch.tensor([
                dataset.char_to_idx[ch] for ch in chars[-dataset.seq_len:]
            ], device=model.device).unsqueeze(0)

            # Create mask
            seq_len = input_ids.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=model.device)).unsqueeze(0)

            # Get model prediction
            output = model(input_ids, mask)
            last_logits = output[0, -1, :] / temperature

            # Sample from distribution
            probs = torch.softmax(last_logits, dim=-1)
            next_char_idx = torch.multinomial(probs, 1).item()

            chars.append(dataset.idx_to_char[next_char_idx])

    return ''.join(chars)