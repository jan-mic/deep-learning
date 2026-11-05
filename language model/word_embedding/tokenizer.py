from tokenizers import Tokenizer, decoders
from tokenizers import normalizers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

if __name__ == "__main__":
    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=6000,
        min_frequency=3,
        show_progress=True,
        special_tokens=["[UNK]"]
    )

    files = [
        "../data/pantadeusz_clean.txt",
        "../data/quovadis_clean.txt",
        "../data/lalka_tom_pierwszy_clean.txt",
        "../data/lalka_tom_drugi_clean.txt",
        "../data/w_pustyni_i_w_puszczy_clean.txt"
    ]

    tokenizer.train(files, trainer)

    tokenizer.save("tadeusz_tokenizer.json")
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tadeusz_tokenizer.json")
    fast_tokenizer.add_special_tokens({'unk_token': '[UNK]'})
