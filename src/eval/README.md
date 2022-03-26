# Model Evaluation

After training and predicting the models, you must copy the following files
before evaluating the model performance on the validation set(s).

# 1. Ground-Truth Files

Copy files from

```
data/processed/[split]/data.csv
```

and paste them like this.

```
src/eval/ground_truths/[split].csv
```

# 2. Prediction Files

Copy files from

```
data/processed/[split]/predictions/[model_prediction].csv
```

and paste them like this.

```
src/eval/predictions/[split]/[model_prediction].csv
```