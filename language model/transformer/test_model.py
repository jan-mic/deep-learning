import torch

from task2.lstm.dataset import TadeuszDataModule
from task2.transformer.model import TransformerTadeusz, TadeuszTransformerModule


def generate_text(model, datamodule, seed_text, gen_length=300, temperature=0.8, k=5):
    model.eval()
    char2idx = datamodule.char2idx
    idx2char = datamodule.idx2char
    device = next(model.parameters()).device
    input_indices = [char2idx[ch] for ch in seed_text.lower() if ch in char2idx]
    input_tensor = torch.tensor(input_indices, dtype=torch.long, device=device).unsqueeze(0)
    generated_text = seed_text
    with torch.no_grad():
        for _ in range(gen_length):
            logits = model.model(input_tensor)
            last_token_logits = logits[0, -1, :] / temperature
            top_k_values, top_k_indices = torch.topk(last_token_logits, k)
            probs = torch.softmax(top_k_values, dim=-1)
            sample_idx = torch.multinomial(probs, num_samples=1).item()
            next_idx = top_k_indices[sample_idx].item()
            generated_text += idx2char[next_idx]
            next_input = torch.tensor([[next_idx]], dtype=torch.long, device=device)
            input_tensor = torch.cat([input_tensor, next_input], dim=1)
            if input_tensor.size(1) > 150:
                input_tensor = input_tensor[:, 1:]

    return generated_text


def generate_text_greeedy(model, datamodule, seed_text, gen_length=300):
    model.eval()
    char2idx = datamodule.char2idx
    idx2char = datamodule.idx2char
    device = next(model.parameters()).device
    input_indices = [char2idx[ch] for ch in seed_text.lower() if ch in char2idx]
    input_tensor = torch.tensor(input_indices, dtype=torch.long, device=device).unsqueeze(0)
    generated_text = seed_text
    with torch.no_grad():
        for _ in range(gen_length):
            logits = model.model(input_tensor)
            last_token_logits = logits[0, -1, :]
            next_idx = torch.argmax(last_token_logits, dim=-1).item()
            generated_text += idx2char[next_idx]
            next_input = torch.tensor([[next_idx]], dtype=torch.long, device=device)
            input_tensor = torch.cat([input_tensor, next_input], dim=1)
            if input_tensor.size(1) > 150:
                input_tensor = input_tensor[:, 1:]
    return generated_text


if __name__ == "__main__":
    data = TadeuszDataModule()
    data.setup()

    model = TadeuszTransformerModule.load_from_checkpoint(
        checkpoint_path="lightning_logs/memory_csv_logs/version_0/checkpoints/best-loss-epoch=39-val_loss=1.72.ckpt",
        model=TransformerTadeusz(
            vocab_size=data.vocab_size, d_model=128, nhead=4, num_layers=3, dim_feedforward=512, dropout=0.2
        ),
    )

    prompt = input("Wpisz początek Pana Tadeusza: \n\n\r")
    wynik = generate_text(model, data, prompt, temperature=0.5, k=5)
    print("\n--- Wygenerowany tekst ---\n")
    print(wynik)
