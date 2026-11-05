import re

CHARS_TO_REPLACE = {
    "…": ".",
    '«': '',
    '»': '',
    "\n": ' ',
    "\r": ' ',
    "*": '',
    "/": '',
}

if __name__ == "__main__":
    with open("../data/pantadeusz.txt", "r", encoding="utf-8") as file:
        text = file.read()
        text = re.sub(r'([?!])…', '', text)
        for char, replacement in CHARS_TO_REPLACE.items():
            text = text.replace(char, replacement)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n+', '\n', text)
    with open("../data/pantadeusz_clean.txt", "w", encoding="utf-8") as file:
        file.write(text)

    with open("../data/quovadis.txt", "r", encoding="utf-8") as file:
        text = file.read()
        text = re.sub(r'\n—\s', ' ', text)
        text = re.sub(r'([?!])…', '', text)
        for char, replacement in CHARS_TO_REPLACE.items():
            text = text.replace(char, replacement)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
    with open("../data/quovadis_clean.txt", "w", encoding="utf-8") as file:
        file.write(text)

    with open("../data/lalka_tom_pierwszy.txt", "r", encoding="utf-8") as file:
        text = file.read()

        text = re.sub(r'\n—\s', ' ', text)
        text = re.sub(r'(.)\1{4,}', r' ', text)
        text = re.sub(r'([?!])…', '', text)
        for char, replacement in CHARS_TO_REPLACE.items():
            text = text.replace(char, replacement)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
    with open("../data/lalka_tom_pierwszy_clean.txt", "w", encoding="utf-8") as file:
        file.write(text)

    with open("../data/lalka_tom_drugi.txt", "r", encoding="utf-8") as file:
        text = file.read()
        text = re.sub(r'\n—\s', ' ', text)
        text = re.sub(r'(.)\1{4,}', r' ', text)
        text = re.sub(r'([?!])…', '', text)
        for char, replacement in CHARS_TO_REPLACE.items():
            text = text.replace(char, replacement)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
    with open("../data/lalka_tom_drugi_clean.txt", "w", encoding="utf-8") as file:
        file.write(text)

    with open("../data/w_pustyni_i_w_puszczy.txt", "r", encoding="utf-8") as file:
        text = file.read()
        text = re.sub(r'\n—\s', ' ', text)
        text = re.sub(r'(.)\1{4,}', r' ', text)
        text = re.sub(r'ROZDZIAŁ [IVXLCDM]+', ' ', text)
        text = re.sub(r'([?!])…', '', text)
        for char, replacement in CHARS_TO_REPLACE.items():
            text = text.replace(char, replacement)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
    with open("../data/w_pustyni_i_w_puszczy_clean.txt", "w", encoding="utf-8") as file:
        file.write(text)
