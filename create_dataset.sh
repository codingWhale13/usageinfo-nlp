#!/bin/bash
python src/helpers/create_dataset.py datasets/paper-experts-train expert* data/training_reviews.json
python json-cli.py sample datasets/paper-experts-train/reviews.json 200 -o datasets/paper-experts-val/reviews.json -s 42 -d

python src/helpers/create_dataset.py datasets/paper-gpt_4-train *gpt_4* data/training_reviews.json
python json-cli.py sample datasets/paper-gpt_4-train/reviews.json 200 -o datasets/paper-gpt_4-val/reviews.json -s 42 -d

python src/helpers/create_dataset.py datasets/paper-chat_gpt-train *chat_gpt* data/training_reviews.json
python json-cli.py sample datasets/paper-chat_gpt-train/reviews.json 200 -o datasets/paper-chat_gpt-val/reviews.json -s 42 -d

python src/helpers/create_dataset.py datasets/paper-vendor_A-train vendor_A* data/training_reviews.json
python json-cli.py sample datasets/paper-vendor_A-train/reviews.json 200 -o datasets/paper-vendor_A-val/reviews.json -s 42 -d

python src/helpers/create_dataset.py datasets/paper-vendor_B-train vendor_B* data/training_reviews.json
python json-cli.py sample datasets/paper-vendor_B-train/reviews.json 200 -o datasets/paper-vendor_B-val/reviews.json -s 42 -d