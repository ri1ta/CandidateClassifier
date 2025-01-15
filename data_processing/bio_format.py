'''
Был взят файл .json формата, с вручную размеченными в Label Studio сущностями NER
данный код пореобразовывает этот файл в BIO-разметку: beginning / inside / outside

все для извлечения ключевой информации в столбце "описание вакансии"
'''

import json
import re
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
MAX_SEQ_LEN = 512

# нормализация текста
def normalize_text(text):
    text = text.lower()
    return text

# разделение текста на сегменты
def split_text_into_chunks(text, max_length=MAX_SEQ_LEN):
    sentences = re.split(r'(?<=\.)\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        tokenized_sentence = tokenizer.tokenize(sentence)
        if current_length + len(tokenized_sentence) > max_length:
            chunks.append(current_chunk)
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += len(tokenized_sentence)

    if current_chunk:
        chunks.append(current_chunk)

    return [' '.join(chunk) for chunk in chunks]

# JSON в BIO
def convert_to_bio_with_chunks(json_path, output_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    bio_data = []
    for record in data:
        text = record["data"]["text"]
        entities = record["annotations"][0]["result"] if record["annotations"] else []

        text = normalize_text(text)

        chunks = split_text_into_chunks(text)

        for chunk in chunks:
            tokenized_text = tokenizer.tokenize(chunk)
            token_labels = ["O"] * len(tokenized_text)

            for entity in entities:
                if "value" in entity:
                    start = entity["value"]["start"]
                    end = entity["value"]["end"]
                    label = entity["value"]["labels"][0]
                    char_index = 0

                    i = 0
                    while i < len(tokenized_text):
                        token = tokenized_text[i]
                        token_start = chunk.find(token, char_index)
                        token_end = token_start + len(token)
                        char_index = token_end

                        if token_start >= start and token_end <= end:
                            if token_start == start:
                                token_labels[i] = f"B-{label}"
                            elif token_start > start:
                                token_labels[i] = f"I-{label}"

                        i += 1

            for i in range(1, len(tokenized_text)):
                if tokenized_text[i].startswith('##'):
                    token_labels[i] = token_labels[i - 1]

            for token, label in zip(tokenized_text, token_labels):
                bio_data.append(f"{token}\t{label}")
            bio_data.append("")

    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write("\n".join(bio_data))

    print(f"BIO данные успешно сохранены в файл: {output_path}")

json_path = "/Users/rii_beltz/Downloads/project-5-at-2025-01-10-14-23-dbeec074.json"
output_path = "bio_data_cleaned.tsv"

convert_to_bio_with_chunks(json_path, output_path)
