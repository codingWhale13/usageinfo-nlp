#%%

from scipy.stats import entropy
def split_text_by_space(text: str):
    return text.split(" ")

def text_entropy(texts: list[str], tokenizer=split_text_by_space) -> float:
    vocabulary = {}
    total_word_count = 0
    for text in texts:
        for word in tokenizer(text):
            total_word_count += 1
            try:
                vocabulary[word] += 1
            except KeyError:
                vocabulary[word] = 1
    
    return entropy([word_count/total_word_count for word_count in vocabulary.values()], base=2)

