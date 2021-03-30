---
title: Some thoughts about handling imbalanced dataset by generating synthetic data in machine learning
date: 2021-03-30T08:50:17.196Z
draft: false
featured: yes
tags:
  - Data
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---


```python
# 
from imblearn.over_sampling import SMOTE
sm = SMOTE()
X_oversampled, y_oversampled = sm.fit_resample(X, y)
```
This is how the dataset looks like after re sampling.

![output_131_0](https://user-images.githubusercontent.com/19218787/112948670-ec2f0f00-916a-11eb-82df-6aec07246c57.png)

