# %%
from helpers.review_set import ReviewSet

review_set = ReviewSet.from_files(
    "/hpi/fs00/share/fg-demelo/bsc2022-usageinfo/data_labeled/golden_dataset/golden_v2.json"
)

# %%
from active_learning.vectorize_review import vectorize, columns
from helpers.label_selection import AnyLabelSelectionStrategy

strategy = AnyLabelSelectionStrategy()
review_id = "RLWCGIO5HLK9F"
review = review_set.get_review(review_id)

print(review)
vectorize(review, strategy)

# %%
import pandas as pd

review_vectors = [vectorize(review, strategy) for review in review_set]
review_vectors

df = pd.DataFrame(review_vectors, columns=columns())
df

#$$
# %%
%reload_ext autoreload
%autoreload 2
from active_learning.processing import one_hot_encode

one_hot_encode(df, "product_category")

#%%
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
generation_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)

#%%
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

inputs = tokenizer(["Hello, my dog is cute", "Batch two"], return_tensors="pt", padding=True)
outputs = model(**inputs, labels=labels)

#%%
labels, outputs.logits

#%%
import numpy as np
from torch.nn.functional import cross_entropy
softmax = outputs.logits.detach().softmax(dim=1)
cross_entropy(softmax, labels, reduction='none'), softmax

#%%
# Example of target with class probabilities
input = torch.randn(2, 2, 32128, requires_grad=True)
target = torch.randn([[0, 2], [4, 5]])
loss = cross_entropy(input, target, reduction='none')

input, target, loss

#%%
np.amax(softmax.numpy(), axis=1), softmax

#%%
lm_logits = outputs.logits.detach()
from torch.nn import CrossEntropyLoss
loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
lm_logits.view(-1, lm_logits.size(-1)), lm_logits, labels.view(-1)

#%%
labels = torch.tensor([0,1]) # Batch size 1
loss_fct(lm_logits, labels), outputs.logits.detach().softmax(dim=1), lm_logits.shape, labels.shape