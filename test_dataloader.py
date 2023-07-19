#%%
from transformers import (
    T5Tokenizer
)
from helpers.review_set import ReviewSet
from helpers.label_selection import LabelIDSelectionStrategy
tokenizer =  T5Tokenizer.from_pretrained("t5-base", model_max_length=512)
review_set = ReviewSet.from_files("silver-v1.json")

print(tokenizer)
#%%
print(len(review_set))

#%%

dataloader,metadata =  review_set.get_dataloader(
            batch_size=16,
            num_workers=0,
            tokenizer=tokenizer,
            model_max_length=512,
            for_training=False,
        )

#%%
import random

dataloader, _ = review_set.get_dataloader(
    tokenizer=tokenizer, model_max_length=512, for_training=True,stratified_drop_out=True, rng=random.Random(45) ,selection_strategy=LabelIDSelectionStrategy("*"), multiple_usage_options_strategy="default", drop_last=True, shuffle=True, batch_size=16, num_workers=4, pin_memory=True
)

#%%
step = {}
#%%

for i, batch in enumerate(dataloader):
    if i in step:
        print(step[i] == set(batch["review_id"]))
    else:
        step[i] = set(batch["review_id"])
