import torch
from transformers import PreTrainedTokenizerFast
from model import TransformerTadeusz, TadeuszTransformerModule


def generate_text(model, tokenizer, seed_text, gen_length=150, temperature=0.8, seq_length=150, k=20):
    model.eval()
    device = next(model.parameters()).device

    input_ids = tokenizer.encode(seed_text, add_special_tokens=True, return_tensors="pt").to(device)
    generated_ids = input_ids[0].tolist()

    with torch.no_grad():
        for _ in range(gen_length):
            logits = model(input_ids)

            last_token_logits = logits[0, -1, :] / temperature
            top_k_values, top_k_indices = torch.topk(last_token_logits, k)
            probs = torch.softmax(top_k_values, dim=-1)

            next_id_idx = torch.multinomial(probs, num_samples=1).item()
            next_id = top_k_indices[next_id_idx].item()

            generated_ids.append(next_id)

            if next_id == tokenizer.eos_token_id:
                break

            next_input = torch.tensor([[next_id]], device=device)
            input_ids = torch.cat([input_ids, next_input], dim=1)

            if input_ids.size(1) > seq_length:
                input_ids = input_ids[:, 1:]

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def generate_text_greeedy(model, tokenizer, seed_text, gen_length=150, seq_length=150):
    model.eval()
    device = next(model.parameters()).device

    input_ids = tokenizer.encode(seed_text, add_special_tokens=True, return_tensors="pt").to(device)
    generated_ids = input_ids[0].tolist()

    with torch.no_grad():
        for _ in range(gen_length):
            logits = model(input_ids)

            last_token_logits = logits[0, -1, :]

            next_id = torch.argmax(last_token_logits, dim=-1).item()

            generated_ids.append(next_id)

            if next_id == tokenizer.eos_token_id:
                break

            next_input = torch.tensor([[next_id]], device=device)
            input_ids = torch.cat([input_ids, next_input], dim=1)

            if input_ids.size(1) > seq_length:
                input_ids = input_ids[:, 1:]

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


if __name__ == "__main__":
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="tadeusz_tokenizer.json")

    checkpoint_path = "final_model/memory/version_31/checkpoints/best-loss-epoch=49-val_loss=5.31.ckpt"
    model = TadeuszTransformerModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=TransformerTadeusz(
            vocab_size=len(tokenizer),  d_model=128, nhead=4, num_layers=8, dim_feedforward=512, dropout=0.2
        ),
    )

    prompt = input("Wpisz początek (małymi literami): \n\n\r")
    wynik = generate_text(model.model, tokenizer, prompt, k=50, temperature=0.8)
    print("\n--- Wygenerowany tekst ---\n")
    print(wynik)
