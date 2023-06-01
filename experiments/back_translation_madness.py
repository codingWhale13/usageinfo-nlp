from transformers import MarianMTModel, MarianTokenizer

config = {"max_new_tokens": 512}
model_to_de_name = "Helsinki-NLP/opus-mt-en-de"
model_to_en_name = "Helsinki-NLP/opus-mt-de-en"

language_en = "en"
language_de = "de"
tokenizer_to_de = MarianTokenizer.from_pretrained(model_to_de_name)
model_to_de = MarianMTModel.from_pretrained(model_to_de_name)

tokenizer_to_en = MarianTokenizer.from_pretrained(model_to_en_name)
model_to_en = MarianMTModel.from_pretrained(model_to_en_name)

texts = ["chilling beer"]


def to_de(input_strings):
    input_tokenized = tokenizer_to_de(
        input_strings, return_tensors="pt", padding=True, truncation=True
    )

    translated_tokens = model_to_de.generate(**input_tokenized, **config)

    return [
        tokenizer_to_de.decode(t, skip_special_tokens=True) for t in translated_tokens
    ]


def to_en(input_strings):
    input_tokenized = tokenizer_to_en(
        input_strings, return_tensors="pt", padding=True, truncation=True
    )

    translated_tokens = model_to_en.generate(**input_tokenized, **config)

    return [
        tokenizer_to_en.decode(t, skip_special_tokens=True) for t in translated_tokens
    ]


print(f"Original text: {texts}")

german = to_de(texts)
print(f"After first translation (EN->DE): {german}")

result = to_en(german)
print(f'After second translation (DE->EN, without ">>[...]"<<): {result}')

german = [f">>{language_en}<< {text}" for text in german]

result = to_en(german)
print(f'After second translation (DE->EN, with ">>[...]<<"): {result}')

print(
    'Result: Not using the ">>[...]<<" tags is the way to go. This way, no weird special symbols like "♪" occur.'
)
