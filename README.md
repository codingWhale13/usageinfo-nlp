# Learning to Predict Usage Options of Product Reviews with LLM-Generated Labels

This repository provides the paper and code of **[Learning to Predict Usage Options of Product Reviews with LLM-Generated Labels](paper.pdf)**, a study that explores different annotation approaches for solving a complex task.

## Introduction

Annotating large datasets can be challenging. However, crowd-sourcing is often expensive and can lack quality, especially for non-trivial tasks. We propose a method of using LLMs as few-shot learners for annotating data in a complex natural language task where we learn a standalone model to predict usage options for products from customer reviews. Learning a custom model offers individual control over energy efficiency and privacy measures compared to using the LLM directly for the sequence-to-sequence task. We compare this data annotation approach with other traditional methods and demonstrate how LLMs can enable considerable cost savings. We find that the quality of the resulting data exceeds the
level attained by third-party vendor services and that GPT-4-generated labels even reach the level of domain experts.

## Reproducibility

1. Cloning of the repository with

```bash
git clone https://github.com/aiintelligentsystems/learning-with-llm-labels
cd learning-with-llm-labels
```

2. Download the [data](https://s3.amazonaws.com/amazon-reviews-pds/readme.html)
```bash
python src/data_download.py <data-path> data/reviews.json
```

3. Create a conda environment

```bash
conda env create -f environment.yml
conda activate bp
```

4. Annotation
For annotation you have 4 options:
Use our labels (recommended): 

```bash
python json-cli.py merge_labels data/reviews.json prompt_evaluation_set_labels.json data/prompt_evaluation_reviews.json
python json-cli.py merge_labels data/reviews.json training_set_labels.json data/training_reviews.json
python json-cli.py merge_labels data/reviews.json evaluation_set_labels.json data/evaluation_reviews.json
```
<details> <summary>Other label approaches</summary>
Label by hand

Label using the aws vendors:
```bash
python src/sagemaker/create_manifest.py data/reviews.json data/reviews.manifest
```
Label this file using the vendors
```bash
python src/sagemaker/sagemaker_manifest_to_json.py data/reviews.manifest
```

Label using the OpenAI API:
```bash
python src/openai_api/openai_labelling.py data/reviews.json
python src/openai_api/openai_labelling.py data/reviews.json -p 6_shot
python src/openai_api/openai_labelling.py data/reviews.json -p COT_2_shot
python src/openai_api/openai_labelling.py data/reviews.json -p COT_6_shot
```
</details>

5. Dataset creation
Create for each label source a training and validation dataset. If you have our label names you can use create_dataset.sh to create all necessary datasets 
```bash
python src/helpers/create_dataset.py datasets/dataset_name label_ids_to_use data/reviews.json
```

6. Training
Use the configs provided in src/training/trainings_configs to train the models
```bash
python src/htraining/train.py --config <config_path>
```

## Cite this work
