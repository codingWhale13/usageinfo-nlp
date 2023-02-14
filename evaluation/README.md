# Evaluation

This package evaluates labeled data. Use `evaluation.scoring` to calculate scores and `evaluation.plotting` to visualize them.

## Scoring
Possible inputs:
- labels from humans, generated with labelling tool
- labels from GPT models, JSON structure here is "reviews"->"label"->"prompt_name"

Output: A JSON file containing a list of dictionaries, each having the following content:
- **review_id**: string, pointing to the related review
- **references**: list of strings, the ground truth labels
- **predictions**: list of strings, the predicted labels
- **origin**: string, the source of the label (for example, "Davinci\[prompt_v42\]" or "iMerit\[vendor.eu-central-1.f0200b078ee3dcda\]")
- **scores**: a dictionary containing the requested scores

## Plotting
**Input**: A JSON file containing scores as described above

Possible Outputs:
- 2D scatter plot (precision vs. recall)
- 2D KDE plot (precision vs. recall)
- violin plot showing the F1 score
