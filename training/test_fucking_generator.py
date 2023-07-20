from helpers.review_set import ReviewSet
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from transformers.models.t5.modeling_t5 import BaseModelOutput
from training import utils

review_set_name = "hard_reviews.json"
reviews = ReviewSet.from_files(review_set_name).filter(lambda review: review.review_id == "R38BZ67UE744CE", inplace=False)

review = reviews.get_review("R38BZ67UE744CE").get_prompt("active_learning_v1")
#model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to("cuda")
#tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")


(model,
tokenizer,
max_length,
model_name) = utils.initialize_model_tuple({"name": "car-barfs-stupid-155-7", "checkpoint":"best"})

model = model.to("cuda")
input = tokenizer([review], return_tensors="pt", padding="max_length", max_length=512 )
input_ids = input.input_ids.to("cuda")
attention_mask = input.attention_mask.to("cuda")

with torch.inference_mode():
    outputs = model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True
    )
    #print(outputs)

    encoder_outputs = BaseModelOutput(
        outputs.last_hidden_state,

    )
  

    decoder_input_ids = torch.stack([torch.tensor([0], dtype=torch.int32)]).to("cuda")

    for _ in range(10):
        print(decoder_input_ids)
        decoder_outputs = model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=outputs.last_hidden_state,
            encoder_attention_mask=attention_mask
        )

        sequence_outputs = decoder_outputs[0]
        logits = model.lm_head(sequence_outputs)


       
        top_k = torch.topk(logits, k=5,dim=-1)
        next_token = torch.tensor(torch.argmax(logits[:,-1,:], dim=-1)).to("cuda")
        print(top_k)
        decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(-1)], dim=-1)
        #decoder_input_ids = torch.cat([decoder_input_ids, torch.tensor([next_token])], dim=-1).to("cuda")
        print("Next_token:", next_token)


print(model.generate(input_ids=input_ids, attention_mask=attention_mask))
print(decoder_input_ids)