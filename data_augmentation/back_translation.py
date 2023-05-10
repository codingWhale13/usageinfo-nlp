from transformers import MarianMTModel, MarianTokenizer
from data_augmentation.core import TextAugmentation
import torch
import time


class BackTranslationTextAugmentation(TextAugmentation):
    def __init__(self, temporary_language="de", translation_rounds=1) -> None:
        super().__init__()
        self.temporary_language = temporary_language
        self.target_language = "en"

        self.translation_rounds = translation_rounds

        self.first_tokenizer = None
        self.first_model = None
        self.second_tokenizer = None
        self.second_model = None
        self.generation_config = {"max_new_tokens": 512}

    def __load_models(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        start_loading_time = time.time()
        self.first_tokenizer = MarianTokenizer.from_pretrained(self.first_checkpoint)
        self.first_model = MarianMTModel.from_pretrained(self.first_checkpoint).to(
            self.device
        )

        self.second_tokenizer = MarianTokenizer.from_pretrained(self.second_checkpoint)
        self.second_model = MarianMTModel.from_pretrained(self.second_checkpoint).to(
            self.device
        )
        print(f"Loaded models in {time.time() - start_loading_time} seconds")

    def __are_models_loaded(self):
        return (
            self.first_model
            and self.second_model
            and self.first_tokenizer
            and self.second_tokenizer
        )

    def metadata(self) -> dict:
        return {
            "first_checkpoint": self.first_checkpoint,
            "second_checkpoint": self.second_checkpoint,
            "translation_rounds": self.translation_rounds,
            "generation_config": self.generation_config,
        }

    @property
    def first_checkpoint(self):
        return f"Helsinki-NLP/opus-mt-{self.target_language}-{self.temporary_language}"

    @property
    def second_checkpoint(self):
        return f"Helsinki-NLP/opus-mt-{self.temporary_language}-{self.target_language}"

    def __translate_batch(
        self, texts: list[str], model: MarianMTModel, tokenizer, language
    ):
        texts = [f">>{language}<< {text}" for text in texts]
        input = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(
            self.device
        )
        translated_tokens = model.generate(
            **input,
            **self.generation_config,
        )
        translated_texts = [
            tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens
        ]
        return translated_texts

    def max_batch_size(self) -> int:
        return 32

    def augment_batch(self, texts: list[str]):
        if not self.__are_models_loaded():
            self.__load_models()
        for _ in range(self.translation_rounds):
            texts = self.__translate_batch(
                texts, self.first_model, self.first_tokenizer, self.temporary_language
            )
            texts = self.__translate_batch(
                texts, self.second_model, self.second_tokenizer, self.target_language
            )
        return texts
