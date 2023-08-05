# 40.220 The Analytics Edge Kaggle Competition

## Table of Contents

1. [Introduction](#Introduction)
2. [Folder Directory](#Folder-Directory)
3. [Multinomial Logit](#Multinomial-Logit)
4. [Random Forest](#Random-Forest)
5. [XGBoost](#XGBoost)
6. [Crystal Ball](#Crystal-Ball)

## Introduction

This readme is intended to be used as a guide on how to run our code for The Analytics Edge Kaggle Competition. Our team would like to express our deepest gratitude to Professor Karthik Natarajan and Teaching Assistant Arjun Kodagehalli Ramachandra for their steadfast support and tutelage throughout this Machine Learning project.

Our team consists of:

- Boey Sze Min, Jeanelle
- Chai Yu Cheng, Raynard
- Joash Tan Jia Le
- Ng Wei Xian

## Folder Directory

- input
  - train1.csv
  - test1.csv

- Multinomial Logit
  - mlogit.nb
  - mlogit.RMD
  - train1_onehot2.csv
  - test1_onehot2.csv
  - usePackages.R

- Random Forest
  - RandomForest.nb
  - RandomForest.RMD
  - train1_preprocessed.csv
  - test1_preprocessed.csv
  - preprocess.RMD

- XGBoost
  - xgboost.nb
  - xgboost.RMD
  - train1_onehot2.csv
  - test1_onehot2.csv
  - usePackages.R

- Crystal Ball
  - postprocess.nb
  - postporcess.RMD
  - Rforest_2001_trees.csv
  - submission_503515.csv
  - submission_M3.csv
  - submission_xgb3.csv
  - testresults_2001for_allvariables.csv
  - testresults_mlogitM3.csv
  - testresults_xgb3.csv
  - logloss_combination.png

- output
  - submission_MNL.csv
  - submission_RF.csv
  - submission_xgb3.csv
  - submission_503515.csv

## Multinomial Logit

The Multinomial Logit R code is inside the "Multinomial Logit" folder. Run the file "mlogit.RMD" inside RStudio. It cleans up the "training1.csv" and "test1.csv" from the "input" folder and outputs two csv files in the "Multinomial Logit" folder to be used for training - "train1_onehot2.csv" and "test1_onehot2".csv. Running this code gives the required csv for submission in the "output" folder called "submission_MNL.csv".

## Random Forest

The Random Forest R code is inside the "Random Forest" folder. Run the file "RandomForest.RMD" inside RStudio. It uses a separate "preprocess.RMD" within the same folder to clean up the "training1.csv" and "test1.csv" inside the "input" folder and outputs two csv files in the "Random Forest" folder to be used for training - "train1_preprocessed" and "test1_preprocessed". Running this code gives the required csv for submission in the "output" folder called "submission_RF.csv".

## XGBoost

The XGBoost R code is inside the "XGBoost" folder. Run the file "XGBoost.RMD" inside RStudio. It uses the same "train1_onehot2.csv" and "test1_onehot2.csv" from Multinomial Logit's preprocessing code. Running this code gives the required csv for submission in the "output" folder called "submission_xgb3.csv".

## Crystal Ball

The "Crystal Ball" R code as explained in our report is present in the "Crystal Ball" folder. Run the file "postprocess.RMD" inside RStudio. It uses the combination of all the various output csv files from each model and does an ensemble to form a final submission that is written in the "output" folder called "submission_503515.csv".
