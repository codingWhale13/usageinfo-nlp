# This code does not quite work; I discontinued this approach

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import T5ForConditionalGeneration, T5Tokenizer
from helpers.review_set import ReviewSet
import helpers.label_selection as ls

# Step 0: Define constants
MODEL_MAX_LEN = 512

# Step 1: Load the pre-trained model
pretrained_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", model_max_length=MODEL_MAX_LEN)

# Step 2: Extract the encoder and decoder weights
teacher = pretrained_model.encoder.block[:10]

# Step 3: Define a smaller 2-block encoder
student = nn.Sequential(*teacher[:2])

# Step 4: Get data
rs = ReviewSet.from_files(
    "/home/codingwhale/Documents/Studium/HPI/Materialien/BP/bsc2022-usageinfo/experiments/nils_ba/experiment_4_compression/val.json"
)

# Step 5: Define a data loader to generate labeled data using the teacher
train_dataloader = rs.get_dataloader(
    tokenizer=tokenizer,
    model_max_length=MODEL_MAX_LEN,
    for_training=True,
    selection_strategy=ls.LabelIDSelectionStrategy("chat_gpt_clean"),  # doesn't matter
    prompt_id="original",
    seed=42,
)

# Step 6: Define the knowledge distillation loss function
criterion = nn.MSELoss()

# Step 7: Train the smaller encoder using knowledge distillation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher.to(device)
student.to(device)

optimizer = optim.Adam(student.parameters())

for epoch in range(num_epochs):
    for inputs in train_dataloader:
        inputs = inputs.to(device)

        # Forward pass through the smaller encoder
        smaller_encoder_outputs = smaller_encoder(inputs)

        # Forward pass through the pre-trained 12-block encoder (teacher)
        teacher_outputs = pretrained_model["encoder"](inputs)

        # Compute the knowledge distillation loss
        loss = criterion(smaller_encoder_outputs, teacher_outputs)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item()}")

torch.save(student.state_dict(), "student.pth")
