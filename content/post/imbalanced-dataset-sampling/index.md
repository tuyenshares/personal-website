---
title: Imbalanced dataset sampling
date: 2021-03-30T07:22:58.108Z
draft: false
featured: false
tags:
  - Data
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---
For my very first project in machine learning trying to predict stroke, I discovered the common problem of handling imbalanced dataset and I wanted to share with you some thoughts about it. 

### What is an imbalanced dataset?

Often found in classification problems, imbalanced datasets are datasets that contain a high majority of one class for the target. They are quite common in areas such as medical diagnosis and fraud detection where data about negative cases (not having the disease or any fraud) are more prevalent than the positive cases. 

To be more concrete, let's say that you have a dataset about patients with a certain illness and you want to predict whether a patient will be diagnosed with this illness or not. In most cases, you will have more data about the part of the population who has not been diagnosed with this illness. 

In my case, I had indeed much more people who have not experienced any stroke than people who have been subject to a cerebrovascular accident. The ratio was more than 95% for the negative class.

![output_8_0](https://user-images.githubusercontent.com/19218787/112954641-28656e00-9171-11eb-9796-5ddb4d790652.png)

### Why is it problem?

The challenge with imbalanced datasets is that machine learning techniques will often ignore the minority class in the training phase which will lead to incorrect performance. For example, some classifiers like Logistic Regression and Decision Tree will tend to predict only the majority class when the minority class is treated as noise. 

In my case, metrics before any sampling show an accuracy of 96% for logistic regression but we can see that the F1 score for class 1 is equal to 0. (With imbalanced dataset, the accuracy is not a metric that we can take into an account because it is based on the the larger part of the target. In other words, this model is very accurate predincting when a people is not having a stroke, which is obviously what we don't need...)

<img width="526" alt="metrics_bef_sampling" src="https://user-images.githubusercontent.com/19218787/112955813-6616c680-9172-11eb-96bc-384a6217e96e.png">

### What are the solution?

One of the popular method is about generating synthetic data with the re sampling technique. 

In python, imbalanced-learn is a package that allows re sampling techniques, compatible with scikit learn. This offers the possibility of under-sampling (by reducing the majority class) or over-sampling (by increasing the number of instances in the minority class). 

The approach I used was to oversample the minority class with the SMOTE technique as my minority class would be too small for an effective training if I were to use the undersampling technique. 

```python
from imblearn.over_sampling import SMOTE
sm = SMOTE()
X_oversampled, y_oversampled = sm.fit_resample(X, y)
```

How does it work? 

Basically, the SMOTE method will generate synthetic data through the near-neighbor method. The algorithm will compute the k-nearest neighbors for one given point so that it can generate all the necessary points for the minority class to reach the same level as the majority class. 

<img width="768" alt="SMOTE_knearest" src="https://user-images.githubusercontent.com/19218787/112955717-4ed7d900-9172-11eb-9611-816c4562fa24.png">
[Source](https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/)



This is the result of the dataset after resampling.

![output_131_0](https://user-images.githubusercontent.com/19218787/112955914-7a5ac380-9172-11eb-8cb4-96a3444f91ed.png)

This method allowed me to do a proper training and generate a correct score. 

![output_144_0](https://user-images.githubusercontent.com/19218787/112956518-21d7f600-9173-11eb-8632-8e665165f7c9.png)

Even though I have only used libraries and packages, it was interesting for me to have touched on this technique and see some powerful methods in data science. Being able to generate synthetic data demonstrates its usefulness when there is an intrinsic limitation of the dataset. However, this experience also leads me to wonder about this possibility of generating this kind of 'fake' data. 

### What's the limitation of synthetic data?