```python
#import libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score
```


```python
#load data
df = pd.read_csv("stroke_data.csv")
```

## Exploratory analysis and data preparation


```python
#first look at the dataset
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9046</td>
      <td>Male</td>
      <td>67.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>228.69</td>
      <td>36.6</td>
      <td>formerly smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>51676</td>
      <td>Female</td>
      <td>61.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Rural</td>
      <td>202.21</td>
      <td>NaN</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>31112</td>
      <td>Male</td>
      <td>80.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>105.92</td>
      <td>32.5</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60182</td>
      <td>Female</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>171.23</td>
      <td>34.4</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1665</td>
      <td>Female</td>
      <td>79.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Rural</td>
      <td>174.12</td>
      <td>24.0</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#total number of rows and columns
df.shape
```




    (5110, 12)




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5110 entries, 0 to 5109
    Data columns (total 12 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   id                 5110 non-null   int64  
     1   gender             5110 non-null   object 
     2   age                5110 non-null   float64
     3   hypertension       5110 non-null   int64  
     4   heart_disease      5110 non-null   int64  
     5   ever_married       5110 non-null   object 
     6   work_type          5110 non-null   object 
     7   Residence_type     5110 non-null   object 
     8   avg_glucose_level  5110 non-null   float64
     9   bmi                4909 non-null   float64
     10  smoking_status     5110 non-null   object 
     11  stroke             5110 non-null   int64  
    dtypes: float64(3), int64(4), object(5)
    memory usage: 479.2+ KB


The dataset consists of 10 metrics for a total of 5110 patients. We have demographic data (gender, age, marital status, type of work and residence) as well as health data including hypertension, heart disease, average glucose level, body mass index (BMI), smoking status and whether the patient has experienced a stroke.


```python
#Let's see how the data spread between people having experienced of a stroke or not
#using count nb of strokes vs not stroke
df['stroke'].value_counts()
```




    0    4861
    1     249
    Name: stroke, dtype: int64




```python
#"A picture is worth a thousand words"
ncount = len(df['stroke'])
ax = sns.countplot(x=df['stroke'])
for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text
plt.savefig('stroke_count.png')
```


    
![png](output_8_0.png)
    



```python
#percentage of people having had a stroke in this dataset
len(df[df['stroke'] == 1])/len(df)*100
```




    4.87279843444227



We can see that the dataset is very imbalanced. It is important to keep it in mind when cleaning and training.

### Checking for missing values


```python
df.isna().sum()
```




    id                     0
    gender                 0
    age                    0
    hypertension           0
    heart_disease          0
    ever_married           0
    work_type              0
    Residence_type         0
    avg_glucose_level      0
    bmi                  201
    smoking_status         0
    stroke                 0
    dtype: int64



There are 201 null values in the 'bmi' column.


```python
# dropping the missing values
df.dropna(inplace = True)
```


```python
#let's check 
df.isna().sum()
```




    id                   0
    gender               0
    age                  0
    hypertension         0
    heart_disease        0
    ever_married         0
    work_type            0
    Residence_type       0
    avg_glucose_level    0
    bmi                  0
    smoking_status       0
    stroke               0
    dtype: int64




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 4909 entries, 0 to 5109
    Data columns (total 12 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   id                 4909 non-null   int64  
     1   gender             4909 non-null   object 
     2   age                4909 non-null   float64
     3   hypertension       4909 non-null   int64  
     4   heart_disease      4909 non-null   int64  
     5   ever_married       4909 non-null   object 
     6   work_type          4909 non-null   object 
     7   Residence_type     4909 non-null   object 
     8   avg_glucose_level  4909 non-null   float64
     9   bmi                4909 non-null   float64
     10  smoking_status     4909 non-null   object 
     11  stroke             4909 non-null   int64  
    dtypes: float64(3), int64(4), object(5)
    memory usage: 498.6+ KB


After removing the null values, we have left with 4909 entries. 

### Drop the id column
The ID column was useful to identify the patients but it will not have any impact on the models, so we can drop it.


```python
df.drop(columns=['id'], inplace=True)
```


```python
#checking
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>67.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>228.69</td>
      <td>36.6</td>
      <td>formerly smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>80.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>105.92</td>
      <td>32.5</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Female</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>171.23</td>
      <td>34.4</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Female</td>
      <td>79.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Rural</td>
      <td>174.12</td>
      <td>24.0</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Male</td>
      <td>81.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>186.21</td>
      <td>29.0</td>
      <td>formerly smoked</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Exploring each variable

#### Gender


```python
#Let's see how the data spread between male and female
df['gender'].value_counts()
```




    Female    2897
    Male      2011
    Other        1
    Name: gender, dtype: int64



There is 1 row with "Other", we can drop it. 


```python
#Dropping 'Other' by selecting rows where gender 1= Other
df = df.loc[df["gender"] != 'Other']
```


```python
#Visualise stroke counts gender wise
sns.countplot(x=df["stroke"], hue=df["gender"])
plt.savefig('stroke_gender.png')
```


    
![png](output_26_0.png)
    


#### Age


```python
#Visualise the spread of the mean for the age variable
fig = sns.FacetGrid(data=df, hue="stroke", aspect=4)
fig.map(sns.kdeplot, "age", shade=True)
fig.add_legend()
plt.savefig('stroke_age.png')
```


    
![png](output_28_0.png)
    


It's not suprising to see that there is a higher risk of stroke when the patient get older.

#### Hypertension


```python
#count hypertension
df['hypertension'].value_counts()
```




    0    4457
    1     451
    Name: hypertension, dtype: int64




```python
#Visualise proportion of people having hypertension between the 2 groups
df_hypertension = df.groupby(['hypertension','stroke'])['hypertension'].count()
df_hypertension_total = df.groupby(['hypertension'])['hypertension'].count()
df_hypertension_fig = df_hypertension / df_hypertension_total * 100
df_hypertension_fig = df_hypertension_fig.unstack()
df_hypertension_fig.plot.bar(stacked=True, figsize=(6,6), width=0.5)
plt.savefig('stroke_hypertension.png')
```


    
![png](output_32_0.png)
    


In proportion, there are more people experiencing stroke in the group with hypertension.

#### Heart disease


```python
#count heart disease
df['heart_disease'].value_counts()
```




    0    4665
    1     243
    Name: heart_disease, dtype: int64




```python
#Visualise proportion of people having heart disease between the 2 groups
df_heart = df.groupby(['heart_disease','stroke'])['heart_disease'].count()
df_heart_total = df.groupby(['heart_disease'])['heart_disease'].count()
df_heart_fig = df_heart / df_heart_total * 100
df_heart_fig = df_heart_fig.unstack()
df_heart_fig.plot.bar(stacked=True, figsize=(6,6), width=0.5)
plt.savefig('stroke_heart.png')
```


    
![png](output_36_0.png)
    


Same constatation as with the group having hypertension, there is a larger proportion of people experiencing stroke.

#### Marital status


```python
#count ever_married
df['ever_married'].value_counts()
```




    Yes    3204
    No     1704
    Name: ever_married, dtype: int64




```python
#plotting stacked bar to see the proportion of people having stroke in this group
df_married = df.groupby(['ever_married','stroke'])['ever_married'].count()
df_married_total = df.groupby(['ever_married'])['ever_married'].count()
df_married_fig = df_married / df_married_total * 100
df_married_fig = df_married_fig.unstack()
df_married_fig.plot.bar(stacked=True, figsize=(6,6), width=0.5)
plt.savefig('stroke_married.png')
```


    
![png](output_40_0.png)
    


The larger proportion of people experiencing stroke for this population can be correlated with what we have seen for age.

#### Work type


```python
#count work_type
df['work_type'].value_counts()
```




    Private          2810
    Self-employed     775
    children          671
    Govt_job          630
    Never_worked       22
    Name: work_type, dtype: int64




```python
df_work = df.groupby(['work_type','stroke'])['work_type'].count()
df_work_total = df.groupby(['work_type'])['work_type'].count()
df_work_fig = df_work / df_work_total * 100
df_work_fig = df_work_fig.unstack()
df_work_fig.plot.bar(stacked=True, figsize=(7,7), width=0.75)
plt.savefig('stroke_work.png')
```


    
![png](output_44_0.png)
    


#### Residence type


```python
#count residence_type
df['Residence_type'].value_counts()
```




    Urban    2490
    Rural    2418
    Name: Residence_type, dtype: int64




```python
df_residence = df.groupby(['Residence_type','stroke'])['Residence_type'].count()
df_residence_total = df.groupby(['Residence_type'])['Residence_type'].count()
df_residence_fig = df_residence / df_residence_total * 100
df_residence_fig = df_residence_fig.unstack()
df_residence_fig.plot.bar(stacked=True, figsize=(6,6), width=0.5)
plt.savefig('stroke_residence.png')
```


    
![png](output_47_0.png)
    


Environmental factors can be a risk factor for stroke but there is no difference in this dataset. 

#### Glucose level


```python
#spread avg_glucose_level
fig = sns.FacetGrid(data=df, hue="stroke", aspect=4)
fig.map(sns.kdeplot, "avg_glucose_level", shade=True)
fig.add_legend()
```




    <seaborn.axisgrid.FacetGrid at 0x106bc9d60>




    
![png](output_50_1.png)
    



```python
sns.violinplot(x="stroke", y="avg_glucose_level", data=df)
plt.savefig('stroke_glucose.png')
```


    
![png](output_51_0.png)
    


The distribution of average glucose level between the two classes is almost similar. There are only a slightly difference for the average glucose level above 150 where more people is experiencing stroke.

#### BMI


```python
#spread bmi
fig = sns.FacetGrid(data=df, hue="stroke", aspect=4)
fig.map(sns.kdeplot, "bmi", shade=True)
fig.add_legend()
plt.savefig('stroke_bmi.png')
```


    
![png](output_54_0.png)
    


There is no real difference between the two groups in terms of BMI.

#### Smoking status


```python
#count smoking_status
df['smoking_status'].value_counts()
```




    never smoked       1852
    Unknown            1483
    formerly smoked     836
    smokes              737
    Name: smoking_status, dtype: int64




```python
df_smoking = df.groupby(['smoking_status','stroke'])['smoking_status'].count()
df_smoking_total = df.groupby(['smoking_status'])['smoking_status'].count()
df_smoking_fig = df_smoking / df_smoking_total * 100
df_smoking_fig = df_smoking_fig.unstack()
df_smoking_fig.plot.bar(stacked=True, figsize=(7,7), width=0.5)
plt.savefig('stroke_smoking.png')
```


    
![png](output_58_0.png)
    


The graph confirms that smoking is a risk factor for stroke. 

### Encoding categorical data


```python
from sklearn.preprocessing import LabelEncoder
```


```python
enc=LabelEncoder()
```


```python
#encoding gender variable
df['gender']=enc.fit_transform(df['gender'])
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>67.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>228.69</td>
      <td>36.6</td>
      <td>formerly smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>80.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>105.92</td>
      <td>32.5</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>171.23</td>
      <td>34.4</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>79.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Rural</td>
      <td>174.12</td>
      <td>24.0</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>81.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>186.21</td>
      <td>29.0</td>
      <td>formerly smoked</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#encoding marital status
df['ever_married']=enc.fit_transform(df['ever_married'])
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>67.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Private</td>
      <td>Urban</td>
      <td>228.69</td>
      <td>36.6</td>
      <td>formerly smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>80.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Private</td>
      <td>Rural</td>
      <td>105.92</td>
      <td>32.5</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Private</td>
      <td>Urban</td>
      <td>171.23</td>
      <td>34.4</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>79.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Self-employed</td>
      <td>Rural</td>
      <td>174.12</td>
      <td>24.0</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>81.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Private</td>
      <td>Urban</td>
      <td>186.21</td>
      <td>29.0</td>
      <td>formerly smoked</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Encode variables with more than 2 Classes
df = pd.get_dummies(df, columns= [i for i in df.columns if df[i].dtypes=='object'],drop_first=True)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>stroke</th>
      <th>work_type_Never_worked</th>
      <th>work_type_Private</th>
      <th>work_type_Self-employed</th>
      <th>work_type_children</th>
      <th>Residence_type_Urban</th>
      <th>smoking_status_formerly smoked</th>
      <th>smoking_status_never smoked</th>
      <th>smoking_status_smokes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>67.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>228.69</td>
      <td>36.6</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>80.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>105.92</td>
      <td>32.5</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>171.23</td>
      <td>34.4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>79.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>174.12</td>
      <td>24.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>81.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>186.21</td>
      <td>29.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 4908 entries, 0 to 5109
    Data columns (total 16 columns):
     #   Column                          Non-Null Count  Dtype  
    ---  ------                          --------------  -----  
     0   gender                          4908 non-null   int64  
     1   age                             4908 non-null   float64
     2   hypertension                    4908 non-null   int64  
     3   heart_disease                   4908 non-null   int64  
     4   ever_married                    4908 non-null   int64  
     5   avg_glucose_level               4908 non-null   float64
     6   bmi                             4908 non-null   float64
     7   stroke                          4908 non-null   int64  
     8   work_type_Never_worked          4908 non-null   uint8  
     9   work_type_Private               4908 non-null   uint8  
     10  work_type_Self-employed         4908 non-null   uint8  
     11  work_type_children              4908 non-null   uint8  
     12  Residence_type_Urban            4908 non-null   uint8  
     13  smoking_status_formerly smoked  4908 non-null   uint8  
     14  smoking_status_never smoked     4908 non-null   uint8  
     15  smoking_status_smokes           4908 non-null   uint8  
    dtypes: float64(3), int64(5), uint8(8)
    memory usage: 543.4 KB


We have now 4908 entries for 16 variables and all our data are either in numerical format so that we can perform the training later.


### Further exploratory analysis and visualisation


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>stroke</th>
      <th>work_type_Never_worked</th>
      <th>work_type_Private</th>
      <th>work_type_Self-employed</th>
      <th>work_type_children</th>
      <th>Residence_type_Urban</th>
      <th>smoking_status_formerly smoked</th>
      <th>smoking_status_never smoked</th>
      <th>smoking_status_smokes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4908.000000</td>
      <td>4908.000000</td>
      <td>4908.000000</td>
      <td>4908.000000</td>
      <td>4908.000000</td>
      <td>4908.000000</td>
      <td>4908.00000</td>
      <td>4908.000000</td>
      <td>4908.000000</td>
      <td>4908.000000</td>
      <td>4908.000000</td>
      <td>4908.000000</td>
      <td>4908.000000</td>
      <td>4908.000000</td>
      <td>4908.000000</td>
      <td>4908.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.409739</td>
      <td>42.868810</td>
      <td>0.091891</td>
      <td>0.049511</td>
      <td>0.652812</td>
      <td>105.297402</td>
      <td>28.89456</td>
      <td>0.042584</td>
      <td>0.004482</td>
      <td>0.572535</td>
      <td>0.157905</td>
      <td>0.136716</td>
      <td>0.507335</td>
      <td>0.170334</td>
      <td>0.377343</td>
      <td>0.150163</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.491836</td>
      <td>22.556128</td>
      <td>0.288901</td>
      <td>0.216954</td>
      <td>0.476125</td>
      <td>44.425550</td>
      <td>7.85432</td>
      <td>0.201937</td>
      <td>0.066808</td>
      <td>0.494761</td>
      <td>0.364689</td>
      <td>0.343582</td>
      <td>0.499997</td>
      <td>0.375964</td>
      <td>0.484771</td>
      <td>0.357268</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.080000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>55.120000</td>
      <td>10.30000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>25.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>77.067500</td>
      <td>23.50000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>44.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>91.680000</td>
      <td>28.10000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>60.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>113.495000</td>
      <td>33.10000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>82.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>271.740000</td>
      <td>97.60000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(df)
plt.savefig('stroke_pairplot.png')
```


    
![png](output_73_0.png)
    



```python
df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>stroke</th>
      <th>work_type_Never_worked</th>
      <th>work_type_Private</th>
      <th>work_type_Self-employed</th>
      <th>work_type_children</th>
      <th>Residence_type_Urban</th>
      <th>smoking_status_formerly smoked</th>
      <th>smoking_status_never smoked</th>
      <th>smoking_status_smokes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gender</th>
      <td>1.000000</td>
      <td>-0.030280</td>
      <td>0.021811</td>
      <td>0.082950</td>
      <td>-0.036380</td>
      <td>0.053161</td>
      <td>-0.026164</td>
      <td>0.006904</td>
      <td>0.012316</td>
      <td>-0.038831</td>
      <td>-0.022210</td>
      <td>0.091732</td>
      <td>-0.004351</td>
      <td>0.039078</td>
      <td>-0.093881</td>
      <td>0.011623</td>
    </tr>
    <tr>
      <th>age</th>
      <td>-0.030280</td>
      <td>1.000000</td>
      <td>0.274395</td>
      <td>0.257104</td>
      <td>0.680742</td>
      <td>0.236000</td>
      <td>0.333314</td>
      <td>0.232313</td>
      <td>-0.079399</td>
      <td>0.120169</td>
      <td>0.327306</td>
      <td>-0.635044</td>
      <td>0.010795</td>
      <td>0.242874</td>
      <td>0.124273</td>
      <td>0.076743</td>
    </tr>
    <tr>
      <th>hypertension</th>
      <td>0.021811</td>
      <td>0.274395</td>
      <td>1.000000</td>
      <td>0.115978</td>
      <td>0.162350</td>
      <td>0.180614</td>
      <td>0.167770</td>
      <td>0.142503</td>
      <td>-0.021345</td>
      <td>-0.004581</td>
      <td>0.111770</td>
      <td>-0.126590</td>
      <td>-0.001140</td>
      <td>0.062252</td>
      <td>0.066671</td>
      <td>0.028188</td>
    </tr>
    <tr>
      <th>heart_disease</th>
      <td>0.082950</td>
      <td>0.257104</td>
      <td>0.115978</td>
      <td>1.000000</td>
      <td>0.111203</td>
      <td>0.154577</td>
      <td>0.041322</td>
      <td>0.137929</td>
      <td>-0.015315</td>
      <td>-0.000239</td>
      <td>0.081466</td>
      <td>-0.088092</td>
      <td>-0.002409</td>
      <td>0.071477</td>
      <td>-0.020722</td>
      <td>0.048667</td>
    </tr>
    <tr>
      <th>ever_married</th>
      <td>-0.036380</td>
      <td>0.680742</td>
      <td>0.162350</td>
      <td>0.111203</td>
      <td>1.000000</td>
      <td>0.151657</td>
      <td>0.341553</td>
      <td>0.105051</td>
      <td>-0.092012</td>
      <td>0.157102</td>
      <td>0.191389</td>
      <td>-0.545687</td>
      <td>0.004707</td>
      <td>0.176745</td>
      <td>0.105062</td>
      <td>0.106479</td>
    </tr>
    <tr>
      <th>avg_glucose_level</th>
      <td>0.053161</td>
      <td>0.236000</td>
      <td>0.180614</td>
      <td>0.154577</td>
      <td>0.151657</td>
      <td>1.000000</td>
      <td>0.175672</td>
      <td>0.138984</td>
      <td>-0.013980</td>
      <td>0.009124</td>
      <td>0.069133</td>
      <td>-0.101122</td>
      <td>-0.007441</td>
      <td>0.073907</td>
      <td>0.032225</td>
      <td>0.011055</td>
    </tr>
    <tr>
      <th>bmi</th>
      <td>-0.026164</td>
      <td>0.333314</td>
      <td>0.167770</td>
      <td>0.041322</td>
      <td>0.341553</td>
      <td>0.175672</td>
      <td>1.000000</td>
      <td>0.042341</td>
      <td>-0.028615</td>
      <td>0.208205</td>
      <td>0.072634</td>
      <td>-0.448779</td>
      <td>-0.000293</td>
      <td>0.107463</td>
      <td>0.107847</td>
      <td>0.088261</td>
    </tr>
    <tr>
      <th>stroke</th>
      <td>0.006904</td>
      <td>0.232313</td>
      <td>0.142503</td>
      <td>0.137929</td>
      <td>0.105051</td>
      <td>0.138984</td>
      <td>0.042341</td>
      <td>1.000000</td>
      <td>-0.014152</td>
      <td>0.014972</td>
      <td>0.055338</td>
      <td>-0.080990</td>
      <td>0.005988</td>
      <td>0.057443</td>
      <td>0.010690</td>
      <td>0.021513</td>
    </tr>
    <tr>
      <th>work_type_Never_worked</th>
      <td>0.012316</td>
      <td>-0.079399</td>
      <td>-0.021345</td>
      <td>-0.015315</td>
      <td>-0.092012</td>
      <td>-0.013980</td>
      <td>-0.028615</td>
      <td>-0.014152</td>
      <td>1.000000</td>
      <td>-0.077658</td>
      <td>-0.029057</td>
      <td>-0.026703</td>
      <td>0.023419</td>
      <td>-0.030404</td>
      <td>0.035857</td>
      <td>-0.028206</td>
    </tr>
    <tr>
      <th>work_type_Private</th>
      <td>-0.038831</td>
      <td>0.120169</td>
      <td>-0.004581</td>
      <td>-0.000239</td>
      <td>0.157102</td>
      <td>0.009124</td>
      <td>0.208205</td>
      <td>0.014972</td>
      <td>-0.077658</td>
      <td>1.000000</td>
      <td>-0.501151</td>
      <td>-0.460556</td>
      <td>-0.016980</td>
      <td>0.024498</td>
      <td>0.111023</td>
      <td>0.099199</td>
    </tr>
    <tr>
      <th>work_type_Self-employed</th>
      <td>-0.022210</td>
      <td>0.327306</td>
      <td>0.111770</td>
      <td>0.081466</td>
      <td>0.191389</td>
      <td>0.069133</td>
      <td>0.072634</td>
      <td>0.055338</td>
      <td>-0.029057</td>
      <td>-0.501151</td>
      <td>1.000000</td>
      <td>-0.172326</td>
      <td>0.012087</td>
      <td>0.096598</td>
      <td>0.029462</td>
      <td>-0.003717</td>
    </tr>
    <tr>
      <th>work_type_children</th>
      <td>0.091732</td>
      <td>-0.635044</td>
      <td>-0.126590</td>
      <td>-0.088092</td>
      <td>-0.545687</td>
      <td>-0.101122</td>
      <td>-0.448779</td>
      <td>-0.080990</td>
      <td>-0.026703</td>
      <td>-0.460556</td>
      <td>-0.172326</td>
      <td>1.000000</td>
      <td>-0.002873</td>
      <td>-0.161383</td>
      <td>-0.243725</td>
      <td>-0.163960</td>
    </tr>
    <tr>
      <th>Residence_type_Urban</th>
      <td>-0.004351</td>
      <td>0.010795</td>
      <td>-0.001140</td>
      <td>-0.002409</td>
      <td>0.004707</td>
      <td>-0.007441</td>
      <td>-0.000293</td>
      <td>0.005988</td>
      <td>0.023419</td>
      <td>-0.016980</td>
      <td>0.012087</td>
      <td>-0.002873</td>
      <td>1.000000</td>
      <td>0.006361</td>
      <td>-0.021511</td>
      <td>0.030910</td>
    </tr>
    <tr>
      <th>smoking_status_formerly smoked</th>
      <td>0.039078</td>
      <td>0.242874</td>
      <td>0.062252</td>
      <td>0.071477</td>
      <td>0.176745</td>
      <td>0.073907</td>
      <td>0.107463</td>
      <td>0.057443</td>
      <td>-0.030404</td>
      <td>0.024498</td>
      <td>0.096598</td>
      <td>-0.161383</td>
      <td>0.006361</td>
      <td>1.000000</td>
      <td>-0.352731</td>
      <td>-0.190464</td>
    </tr>
    <tr>
      <th>smoking_status_never smoked</th>
      <td>-0.093881</td>
      <td>0.124273</td>
      <td>0.066671</td>
      <td>-0.020722</td>
      <td>0.105062</td>
      <td>0.032225</td>
      <td>0.107847</td>
      <td>0.010690</td>
      <td>0.035857</td>
      <td>0.111023</td>
      <td>0.029462</td>
      <td>-0.243725</td>
      <td>-0.021511</td>
      <td>-0.352731</td>
      <td>1.000000</td>
      <td>-0.327233</td>
    </tr>
    <tr>
      <th>smoking_status_smokes</th>
      <td>0.011623</td>
      <td>0.076743</td>
      <td>0.028188</td>
      <td>0.048667</td>
      <td>0.106479</td>
      <td>0.011055</td>
      <td>0.088261</td>
      <td>0.021513</td>
      <td>-0.028206</td>
      <td>0.099199</td>
      <td>-0.003717</td>
      <td>-0.163960</td>
      <td>0.030910</td>
      <td>-0.190464</td>
      <td>-0.327233</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



The variables that have the highest correlation score with stroke are: age, heart disease, glucose level and hypertension, which is what we suspected.


```python
plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),annot=True)
plt.savefig('stroke_corr_heat.png')
```


    
![png](output_76_0.png)
    


Nevertheless, the coefficients are very low (between .14 and .2)

## Training

### 1. Set the independent (X) and the dependent variable (y)


```python
y=df['stroke'].ravel()
```


```python
y
```




    array([1, 1, 1, ..., 0, 0, 0])




```python
X=df.drop('stroke', axis=1)
```


```python
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>work_type_Never_worked</th>
      <th>work_type_Private</th>
      <th>work_type_Self-employed</th>
      <th>work_type_children</th>
      <th>Residence_type_Urban</th>
      <th>smoking_status_formerly smoked</th>
      <th>smoking_status_never smoked</th>
      <th>smoking_status_smokes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>67.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>228.69</td>
      <td>36.6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>80.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>105.92</td>
      <td>32.5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>171.23</td>
      <td>34.4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>79.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>174.12</td>
      <td>24.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>81.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>186.21</td>
      <td>29.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5104</th>
      <td>0</td>
      <td>13.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>103.08</td>
      <td>18.6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5106</th>
      <td>0</td>
      <td>81.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>125.20</td>
      <td>40.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5107</th>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>82.99</td>
      <td>30.6</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5108</th>
      <td>1</td>
      <td>51.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>166.29</td>
      <td>25.6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5109</th>
      <td>0</td>
      <td>44.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>85.28</td>
      <td>26.2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>4908 rows × 15 columns</p>
</div>




```python
#Scaling X 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
```


```python
X_scale=scaler.fit_transform(X)
```


```python
X_scale[:5]
```




    array([[1.        , 0.81689453, 0.        , 1.        , 1.        ,
            0.80126489, 0.30126002, 0.        , 1.        , 0.        ,
            0.        , 1.        , 1.        , 0.        , 0.        ],
           [1.        , 0.97558594, 0.        , 1.        , 1.        ,
            0.23451205, 0.25429553, 0.        , 1.        , 0.        ,
            0.        , 0.        , 0.        , 1.        , 0.        ],
           [0.        , 0.59716797, 0.        , 0.        , 1.        ,
            0.53600776, 0.27605956, 0.        , 1.        , 0.        ,
            0.        , 1.        , 0.        , 0.        , 1.        ],
           [0.        , 0.96337891, 1.        , 0.        , 1.        ,
            0.54934909, 0.15693013, 0.        , 0.        , 1.        ,
            0.        , 0.        , 0.        , 1.        , 0.        ],
           [1.        , 0.98779297, 0.        , 0.        , 1.        ,
            0.60516111, 0.21420389, 0.        , 1.        , 0.        ,
            0.        , 1.        , 1.        , 0.        , 0.        ]])



### 2. Split the data into training and testing sets 


```python
X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.3, stratify=y, shuffle=True, random_state=42)
```


```python
X_train
```




    array([[1.        , 0.74365234, 0.        , ..., 0.        , 0.        ,
            1.        ],
           [1.        , 0.82910156, 0.        , ..., 0.        , 1.        ,
            0.        ],
           [0.        , 0.84130859, 0.        , ..., 1.        , 0.        ,
            0.        ],
           ...,
           [0.        , 0.31640625, 0.        , ..., 0.        , 1.        ,
            0.        ],
           [0.        , 0.35302734, 0.        , ..., 0.        , 1.        ,
            0.        ],
           [0.        , 0.68261719, 0.        , ..., 0.        , 0.        ,
            0.        ]])




```python
X_test
```




    array([[0.        , 0.4140625 , 0.        , ..., 0.        , 0.        ,
            1.        ],
           [0.        , 0.36523438, 0.        , ..., 0.        , 0.        ,
            0.        ],
           [1.        , 0.01806641, 0.        , ..., 0.        , 0.        ,
            0.        ],
           ...,
           [1.        , 0.81689453, 0.        , ..., 1.        , 0.        ,
            0.        ],
           [1.        , 0.52392578, 0.        , ..., 0.        , 0.        ,
            1.        ],
           [1.        , 1.        , 0.        , ..., 0.        , 1.        ,
            0.        ]])




```python
y_train
```




    array([0, 0, 0, ..., 0, 0, 0])




```python
plt.hist(y_train)
```




    (array([3289.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
             146.]),
     array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
     <BarContainer object of 10 artists>)




    
![png](output_92_1.png)
    



```python
y_test
```




    array([0, 0, 0, ..., 0, 0, 0])




```python
plt.hist(y_test)
```




    (array([1410.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
              63.]),
     array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
     <BarContainer object of 10 artists>)




    
![png](output_94_1.png)
    


### 3. Creating models

#### Logistic regression


```python
from sklearn.linear_model import LogisticRegression
```


```python
lr=LogisticRegression(random_state=42)
```


```python
lr.fit(X_train, y_train)
```




    LogisticRegression(random_state=42)




```python
y_pred_lr=lr.predict(X_test)
```


```python
accuracy_score(y_test, y_pred_lr)
```




    0.9572301425661914




```python
print(classification_report(y_test,y_pred_lr))
```

                  precision    recall  f1-score   support
    
               0       0.96      1.00      0.98      1410
               1       0.00      0.00      0.00        63
    
        accuracy                           0.96      1473
       macro avg       0.48      0.50      0.49      1473
    weighted avg       0.92      0.96      0.94      1473
    


    /opt/anaconda3/lib/python3.8/site-packages/sklearn/metrics/_classification.py:1245: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /opt/anaconda3/lib/python3.8/site-packages/sklearn/metrics/_classification.py:1245: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /opt/anaconda3/lib/python3.8/site-packages/sklearn/metrics/_classification.py:1245: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))


1. With imbalanced data, the accuracy is not a metric that we can take into account because it is based on the the larger part of the target. In other words, this model is very accurate predincting when a people is not having a stroke, which is obviously what we don't need...
2. The poor result in class 1 of the target is expected because of the imbalanced dataset as well as the limited correlation among the variables. 

#### Random Forest


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
rf=RandomForestClassifier(random_state=42)
```


```python
rf.fit(X_train, y_train)
```




    RandomForestClassifier(random_state=42)




```python
y_pred_rf=rf.predict(X_test)
```


```python
accuracy_score(y_test, y_pred_rf)
```




    0.955193482688391




```python
print(classification_report(y_test,y_pred_rf))
```

                  precision    recall  f1-score   support
    
               0       0.96      1.00      0.98      1410
               1       0.00      0.00      0.00        63
    
        accuracy                           0.96      1473
       macro avg       0.48      0.50      0.49      1473
    weighted avg       0.92      0.96      0.94      1473
    



```python
confusion_matrix(y_test, y_pred_rf)
```




    array([[1407,    3],
           [  63,    0]])



#### Decision Tree


```python
from sklearn.tree import DecisionTreeClassifier
```


```python
dt=DecisionTreeClassifier(random_state=42)
```


```python
dt.fit(X_train, y_train)
```




    DecisionTreeClassifier(random_state=42)




```python
y_pred_dt=dt.predict(X_test)
```


```python
accuracy_score(y_test, y_pred_dt)
```




    0.923285811269518




```python
print(classification_report(y_test,y_pred_dt))
```

                  precision    recall  f1-score   support
    
               0       0.96      0.96      0.96      1410
               1       0.10      0.10      0.10        63
    
        accuracy                           0.92      1473
       macro avg       0.53      0.53      0.53      1473
    weighted avg       0.92      0.92      0.92      1473
    



```python
confusion_matrix(y_test, y_pred_dt)
```




    array([[1354,   56],
           [  57,    6]])



#### KNN


```python
from sklearn.neighbors import KNeighborsClassifier
```


```python
knn=KNeighborsClassifier()
```


```python
knn.fit(X_train, y_train)
```




    KNeighborsClassifier()




```python
y_pred_knn=knn.predict(X_test)
```


```python
accuracy_score(y_test, y_pred_knn)
```




    0.955193482688391




```python
print(classification_report(y_test,y_pred_knn))
```

                  precision    recall  f1-score   support
    
               0       0.96      1.00      0.98      1410
               1       0.20      0.02      0.03        63
    
        accuracy                           0.96      1473
       macro avg       0.58      0.51      0.50      1473
    weighted avg       0.93      0.96      0.94      1473
    



```python
confusion_matrix(y_test, y_pred_knn)
```




    array([[1406,    4],
           [  62,    1]])



### Handling imbalanced data with sampling


```python
#Using over-sampling method
from imblearn.over_sampling import SMOTE
```


```python
sm = SMOTE()
X_oversampled, y_oversampled = sm.fit_resample(X, y)
```


```python
#data after oversampling
sns.countplot(x = y_oversampled, data = df)
plt.savefig('stroke_oversampled.png')
```


    
![png](output_131_0.png)
    



```python
#train again with the new data
X_train, X_test, y_train, y_test = train_test_split(X_oversampled, y_oversampled, test_size = 0.2, random_state = 42)
```


```python
#Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print(confusion_matrix(lr_pred, y_test))
print(classification_report(lr_pred, y_test))
```

    [[809 143]
     [113 815]]
                  precision    recall  f1-score   support
    
               0       0.88      0.85      0.86       952
               1       0.85      0.88      0.86       928
    
        accuracy                           0.86      1880
       macro avg       0.86      0.86      0.86      1880
    weighted avg       0.86      0.86      0.86      1880
    



```python
#Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
print(confusion_matrix(dt_pred, y_test))
print(classification_report(dt_pred, y_test))
```

    [[857  63]
     [ 65 895]]
                  precision    recall  f1-score   support
    
               0       0.93      0.93      0.93       920
               1       0.93      0.93      0.93       960
    
        accuracy                           0.93      1880
       macro avg       0.93      0.93      0.93      1880
    weighted avg       0.93      0.93      0.93      1880
    



```python
#KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print(confusion_matrix(knn_pred, y_test))
print(classification_report(knn_pred, y_test))
```

    [[749  16]
     [173 942]]
                  precision    recall  f1-score   support
    
               0       0.81      0.98      0.89       765
               1       0.98      0.84      0.91      1115
    
        accuracy                           0.90      1880
       macro avg       0.90      0.91      0.90      1880
    weighted avg       0.91      0.90      0.90      1880
    



```python
#Random forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(confusion_matrix(rf_pred, y_test))
print(classification_report(rf_pred, y_test))
```

    [[884  32]
     [ 38 926]]
                  precision    recall  f1-score   support
    
               0       0.96      0.97      0.96       916
               1       0.97      0.96      0.96       964
    
        accuracy                           0.96      1880
       macro avg       0.96      0.96      0.96      1880
    weighted avg       0.96      0.96      0.96      1880
    



```python
conf_mat = confusion_matrix(rf_pred, y_test)
sns.heatmap(conf_mat.T, annot=True, fmt='d', cbar=False,
          xticklabels=['No','Yes'],
          yticklabels=['No','Yes'] )
plt.xlabel('Actuals')
plt.ylabel('Predicted')
plt.savefig('stroke_over_rf_cm.png')
```


    
![png](output_137_0.png)
    



```python
# Creating the feature importances dataframe
feature_importance = np.array(rf.feature_importances_)
feature_names = np.array(X.columns)

feat_imp = pd.DataFrame({'feature_names':feature_names,'feature_importance':feature_importance})
```


```python
plt.figure(figsize=(10,8))
sns.barplot(x=feat_imp['feature_importance'], y=feat_imp['feature_names'])
plt.savefig('stroke_feature_imp.png')
```


    
![png](output_139_0.png)
    



```python
from sklearn import tree

fn = df.columns
cn = ["Yes","No"]

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (40,15))

tree.plot_tree(rf.estimators_[0],
               feature_names = fn, 
               class_names=cn,
               filled = True);
plt.savefig('stroke_over_tree.png')
```


    
![png](output_140_0.png)
    



```python
pred_prob1 = lr.predict_proba(X_test)
pred_prob2 = dt.predict_proba(X_test)
pred_prob3 = knn.predict_proba(X_test)
pred_prob4 = rf.predict_proba(X_test)
```


```python
from sklearn.metrics import roc_curve

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:,1], pos_label=1)
fpr3, tpr3, thresh3 = roc_curve(y_test, pred_prob3[:,1], pos_label=1)
fpr4, tpr4, thresh4 = roc_curve(y_test, pred_prob4[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
```


```python
from sklearn.metrics import roc_auc_score

# auc scores
auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])
auc_score2 = roc_auc_score(y_test, pred_prob2[:,1])
auc_score3 = roc_auc_score(y_test, pred_prob3[:,1])
auc_score4 = roc_auc_score(y_test, pred_prob4[:,1])

print(auc_score1)
print(auc_score2)
print(auc_score3)
print(auc_score4)
```

    0.9447035807607136
    0.9318695402116666
    0.9564841567075297
    0.9930350196314629



```python
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
plt.plot(fpr2, tpr2, linestyle='--',color='green', label='Decision Tree')
plt.plot(fpr3, tpr3, linestyle='--',color='yellow', label='KNN')
plt.plot(fpr4, tpr4, linestyle='--',color='red', label='Random Forest')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show()
fig.savefig('multiple_roc_curve.png')
```


    
![png](output_144_0.png)
    


After sampling, random forest leads to the best results in terms of metrics as we can see with the ROC curve and a F1 score of .96 


```python

```
