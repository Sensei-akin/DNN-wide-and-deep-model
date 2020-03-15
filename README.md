# DNN-wide-and-deep-model

## Content
* [Overview](#overview)
* [Usage](#usage)
* [Experiments](#experiment)


## Overview
Here, we develop a  **Wide and DNN** for general structural data classification tasks, such as CTR prediction, recommend system, etc.
The repository contains code on how to train and export a DNN linear combined classifier model and how to restore or reload model graph to serve
predictions.
This work is inspired by [wide and deep model](https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html)

## Usage 

## Training and exporting model
You can run the code locally as follows:
```
python Wide and Deep Model.py
```
## Restoring Model for serving predictions
```
python retrain_model.py
```
## Experiments
### settings
For simplicity, we use cross features, it is highly dataset dependent.
We only do some basic feature engineering for generalization.
For continuous features, we use standard normalization transform as input,
for category features, we set hash_bucket_size according to its values size,  
and we use embed category features for deep and we used dicretize continuous features for wide features.
