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
```python
# Example of code highlighting
from imblearn.over_sampling import SMOTE
sm = SMOTE()
X_oversampled, y_oversampled = sm.fit_resample(X, y)
``
