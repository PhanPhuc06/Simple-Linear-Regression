# TASK #1: UNDERSTAND THE PROBLEM STATEMENT


```python
![image.png](attachment:image.png)
```

    /bin/bash: -c: line 1: syntax error near unexpected token `attachment:image.png'
    /bin/bash: -c: line 1: `[image.png](attachment:image.png)'


![image.png](attachment:image.png)

# TASK #2: IMPORT LIBRARIES AND DATASETS


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) 
# setting the style of the notebook to be monokai theme  
# this line of code is important to ensure that we are able to see the x and y axes clearly
# If you don't run this code line, you will notice that the xlabel and ylabel on any plot is black on black and it will be hard to see them. 

```


```python
sales_df = pd.read_csv('IceCreamData.csv')
```


```python

```


```python
sales_df.head(2)
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
      <th>Temperature</th>
      <th>Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24.566884</td>
      <td>534.799028</td>
    </tr>
    <tr>
      <th>1</th>
      <td>26.005191</td>
      <td>625.190122</td>
    </tr>
  </tbody>
</table>
</div>




```python
sales_df
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
      <th>Temperature</th>
      <th>Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24.566884</td>
      <td>534.799028</td>
    </tr>
    <tr>
      <th>1</th>
      <td>26.005191</td>
      <td>625.190122</td>
    </tr>
    <tr>
      <th>2</th>
      <td>27.790554</td>
      <td>660.632289</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20.595335</td>
      <td>487.706960</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11.503498</td>
      <td>316.240194</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>22.274899</td>
      <td>524.746364</td>
    </tr>
    <tr>
      <th>496</th>
      <td>32.893092</td>
      <td>755.818399</td>
    </tr>
    <tr>
      <th>497</th>
      <td>12.588157</td>
      <td>306.090719</td>
    </tr>
    <tr>
      <th>498</th>
      <td>22.362402</td>
      <td>566.217304</td>
    </tr>
    <tr>
      <th>499</th>
      <td>28.957736</td>
      <td>655.660388</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 2 columns</p>
</div>




```python
sales_df.tail(8)
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
      <th>Temperature</th>
      <th>Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>492</th>
      <td>23.056214</td>
      <td>552.819351</td>
    </tr>
    <tr>
      <th>493</th>
      <td>14.931506</td>
      <td>377.430928</td>
    </tr>
    <tr>
      <th>494</th>
      <td>25.112066</td>
      <td>571.434257</td>
    </tr>
    <tr>
      <th>495</th>
      <td>22.274899</td>
      <td>524.746364</td>
    </tr>
    <tr>
      <th>496</th>
      <td>32.893092</td>
      <td>755.818399</td>
    </tr>
    <tr>
      <th>497</th>
      <td>12.588157</td>
      <td>306.090719</td>
    </tr>
    <tr>
      <th>498</th>
      <td>22.362402</td>
      <td>566.217304</td>
    </tr>
    <tr>
      <th>499</th>
      <td>28.957736</td>
      <td>655.660388</td>
    </tr>
  </tbody>
</table>
</div>




```python
sales_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 500 entries, 0 to 499
    Data columns (total 2 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   Temperature  500 non-null    float64
     1   Revenue      500 non-null    float64
    dtypes: float64(2)
    memory usage: 7.9 KB


**PRACTICE OPPORTUNITY #1 [OPTIONAL]:**
- **Calculate the average and maximum temperature and revenue using an alternative method**


```python
sales_df.mean()
```




    Temperature     22.232225
    Revenue        521.570777
    dtype: float64



# TASK #3: PERFORM DATA VISUALIZATION


```python
plt.figure(figsize = (13, 7))
sns.scatterplot(x = 'Temperature', y = 'Revenue', data = sales_df)
plt.grid()
```


    
![png](output_14_0.png)
    


plt.figure(figsize = (13, 7))
sns.regplot(x = 'Temperature', y = 'Revenue', data = sales_df)
plt.grid()

# TASK #4: CREATE TESTING AND TRAINING DATASET


```python

```


```python

```


```python
# reshaping the array from (500,) to (500, 1)
X = X.reshape(-1,1)
print(X.shape)

# reshaping the array from (500,) to (500, 1)
y = y.reshape(-1,1)
print(y.shape)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In [6], line 2
          1 # reshaping the array from (500,) to (500, 1)
    ----> 2 X = X.reshape(-1,1)
          3 print(X.shape)
          5 # reshaping the array from (500,) to (500, 1)


    NameError: name 'X' is not defined



```python
from sklearn.model_selection import train_test_split

```


```python

```


```python

```


```python

```


```python

```

**PRACTICE OPPORUNITY #2 [OPTIONAL]:**
- **Change the split ratio to allocate 30% for testing and 70% for training.**
- **Confirm that the train test split process is successful.**


```python

```

# TASK #5: UNDERSTAND THEORY BEHIND SIMPLE LINER REGRESSION 

![image.png](attachment:image.png)

![image.png](attachment:image.png)

![image.png](attachment:image.png)

# TASK #6: TRAIN A SIMPLE LINEAR REGRESSION MODEL IN SCIKIT LEARN


```python

```


```python

```


```python
from sklearn.linear_model import LinearRegression

```


```python
print('Linear Model Coeff(m)', SimpleLinearRegression.coef_)
print('Linear Model Coeff(b)', SimpleLinearRegression.intercept_)
```

**PRACTICE OPPORTUNITY #3 [OPTIONAL]:**
- **Set the fit_intercept attribute to False and retrain the model. What do you notice? comment on the result.**


```python

```

# TASK #7: EVALUATE TRAINED SIMPLE LINEAR REGRESSION MODEL IN SCIKIT LEARN


```python

```


```python
accuracy_LinearRegression = SimpleLinearRegression.score(X_test, y_test)
accuracy_LinearRegression
```


```python

```


```python
# Use the trained model to generate predictions

Temp = np.array([20])
Temp = Temp.reshape(-1,1)

Revenue = SimpleLinearRegression.predict(Temp)
print('Revenue Predictions =', Revenue)

```

**PRACTICE OPPORTUNITY #4 [OPTIONAL]:**
- **Try at least 3 temperature values and record the output**
- **Perform a sanity check and comment on your results!**


```python

```

# EXCELLENT JOB

# PRACTICE OPPORTUNITIES SOLUTION

**PRACTICE OPPORTUNITY #1 SOLUTION:**
- **Calculate the average and maximum temperature and revenue using an alternative method**


```python
sales_df.mean()
```


```python
sales_df.max()
```

**PRACTICE OPPORUNITY #2 SOLUTION:**
- **Change the split ratio to allocate 30% for testing and 70% for training.**
- **Confirm that the train test split process is successful.**


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
```


```python
X_train.shape
```


```python
y_train.shape
```


```python
X_test.shape
```


```python
y_test.shape
```

**PRACTICE OPPORTUNITY #3 SOLUTION:**
- **Set the fit_intercept attribute to False and retrain the model. What do you notice? comment on the result.**


```python
from sklearn.linear_model import LinearRegression

SimpleLinearRegression = LinearRegression(fit_intercept = False)
SimpleLinearRegression.fit(X_train, y_train)
```

**PRACTICE OPPORTUNITY #4 SOLUTION:**
- **Try at least 3 temperature values and record the output**
- **Perform a sanity check and comment on your results!**


```python
# Temp = 5, Revenue = $150
# Temp = 20, Revenue = $474
# Temp = 40, Revenue = $905
```


```python

```
