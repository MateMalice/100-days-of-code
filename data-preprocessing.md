```python
import pandas as pd


# Get the doco for pd.merge
pd.merge?
```


```python
# Initialise the sample data frames
df_1 = pd.DataFrame({'product': ['red shirt', 'red shirt', 'red shirt', 'white dress'],
                     'price': [49.33, 49.33, 32.49, 199.99]})
print('df_1: \n')
df_1
```

    df_1: 
    
    




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
      <th>product</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>red shirt</td>
      <td>49.33</td>
    </tr>
    <tr>
      <th>1</th>
      <td>red shirt</td>
      <td>49.33</td>
    </tr>
    <tr>
      <th>2</th>
      <td>red shirt</td>
      <td>32.49</td>
    </tr>
    <tr>
      <th>3</th>
      <td>white dress</td>
      <td>199.99</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_2 = pd.DataFrame({'product': ['red shirt', 'blue pants', 'white tuxedo', 'white dress'],
                     'in_stock': [True, True, False, False]})

print('df_2: \n')
df_2
```

    df_2: 
    
    




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
      <th>product</th>
      <th>in_stock</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>red shirt</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>blue pants</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>white tuxedo</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>white dress</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Inner merge
df = pd.merge(left=df_1, right=df_2, on='product', how='inner')
print('Merged df:\n')
df
```

    Merged df:
    
    




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
      <th>product</th>
      <th>price</th>
      <th>in_stock</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>red shirt</td>
      <td>49.33</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>red shirt</td>
      <td>49.33</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>red shirt</td>
      <td>32.49</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>white dress</td>
      <td>199.99</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Outer merge
df_outer = pd.merge(left=df_1, right=df_2, on='product', how='outer')
print('Outer merged df_outer:\n')
df_outer
```

    Outer merged df_outer:
    
    




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
      <th>product</th>
      <th>price</th>
      <th>in_stock</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>red shirt</td>
      <td>49.33</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>red shirt</td>
      <td>49.33</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>red shirt</td>
      <td>32.49</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>white dress</td>
      <td>199.99</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>blue pants</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>white tuxedo</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Deduplicate
df = df_outer.drop_duplicates()
df
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
      <th>product</th>
      <th>price</th>
      <th>in_stock</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>red shirt</td>
      <td>49.33</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>red shirt</td>
      <td>32.49</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>white dress</td>
      <td>199.99</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>blue pants</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>white tuxedo</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print duplicates
df.duplicated()
```




    0    False
    2    False
    3    False
    4    False
    5    False
    dtype: bool




```python
# Sum of duplicates
df.duplicated().sum()
```




    0




```python
df[~df['product'].duplicated()]
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
      <th>product</th>
      <th>price</th>
      <th>in_stock</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>red shirt</td>
      <td>49.33</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>white dress</td>
      <td>199.99</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>blue pants</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>white tuxedo</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.drop_duplicates()
```


```python
df_nonans = df.dropna()
```


```python
df_nonans
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
      <th>product</th>
      <th>price</th>
      <th>in_stock</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>red shirt</td>
      <td>49.33</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>red shirt</td>
      <td>32.49</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>white dress</td>
      <td>199.99</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get doco for fillna
df.fillna?
```


```python
df.fillna(value=df.price.mean())
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
      <th>product</th>
      <th>price</th>
      <th>in_stock</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>red shirt</td>
      <td>49.330000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>red shirt</td>
      <td>32.490000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>white dress</td>
      <td>199.990000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>blue pants</td>
      <td>93.936667</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>white tuxedo</td>
      <td>93.936667</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.fillna(method='pad')
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
      <th>product</th>
      <th>price</th>
      <th>in_stock</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>red shirt</td>
      <td>49.33</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>red shirt</td>
      <td>32.49</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>white dress</td>
      <td>199.99</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>blue pants</td>
      <td>199.99</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>white tuxedo</td>
      <td>199.99</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Encode the labels - we would do this to prepare data for training models
import numpy as np
df = df.fillna(value=df.price.mean())
ratings = [ 'low', 'medium', 'high']
np.random.seed(2)
df['rating'] = np.random.choice(ratings, len(df))
df
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
      <th>product</th>
      <th>price</th>
      <th>in_stock</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>red shirt</td>
      <td>49.330000</td>
      <td>True</td>
      <td>low</td>
    </tr>
    <tr>
      <th>2</th>
      <td>red shirt</td>
      <td>32.490000</td>
      <td>True</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>white dress</td>
      <td>199.990000</td>
      <td>False</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>blue pants</td>
      <td>93.936667</td>
      <td>True</td>
      <td>high</td>
    </tr>
    <tr>
      <th>5</th>
      <td>white tuxedo</td>
      <td>93.936667</td>
      <td>False</td>
      <td>high</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Demonstrate use of map
df.in_stock = df.in_stock.map({False: 0, True: 1})
df
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
      <th>product</th>
      <th>price</th>
      <th>in_stock</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>red shirt</td>
      <td>49.330000</td>
      <td>1</td>
      <td>low</td>
    </tr>
    <tr>
      <th>2</th>
      <td>red shirt</td>
      <td>32.490000</td>
      <td>1</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>white dress</td>
      <td>199.990000</td>
      <td>0</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>blue pants</td>
      <td>93.936667</td>
      <td>1</td>
      <td>high</td>
    </tr>
    <tr>
      <th>5</th>
      <td>white tuxedo</td>
      <td>93.936667</td>
      <td>0</td>
      <td>high</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Test scikit-learn's LabelEncoder to map class labels to integers
from sklearn.preprocessing import LabelEncoder
rating_encoder = LabelEncoder()
_df = df.copy()
_df.rating = rating_encoder.fit_transform(df.rating)
_df
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
      <th>product</th>
      <th>price</th>
      <th>in_stock</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>red shirt</td>
      <td>49.330000</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>red shirt</td>
      <td>32.490000</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>white dress</td>
      <td>199.990000</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>blue pants</td>
      <td>93.936667</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>white tuxedo</td>
      <td>93.936667</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Convert back if needed
rating_encoder.inverse_transform(_df.rating)
```




    array(['low', 'medium', 'low', 'high', 'high'], dtype=object)



#### Note: There is a problem because of the ordinal numbering - 0 should be low and 2 should be high. Let's build the dictionary ourselves and do it properly to fix this!


```python
# We use an ordinal map
ordinal_map = {rating: index for index, rating in enumerate(['low', 'medium', 'high'])}
print(ordinal_map)
df.rating = df.rating.map(ordinal_map)
df
```

    {'low': 0, 'medium': 1, 'high': 2}
    




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
      <th>product</th>
      <th>price</th>
      <th>in_stock</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>red shirt</td>
      <td>49.330000</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>red shirt</td>
      <td>32.490000</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>white dress</td>
      <td>199.990000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>blue pants</td>
      <td>93.936667</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>white tuxedo</td>
      <td>93.936667</td>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Convert class labels to numeric values using one-hot encoding
df = pd.get_dummies(df)
df
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
      <th>price</th>
      <th>in_stock</th>
      <th>rating</th>
      <th>product_blue pants</th>
      <th>product_red shirt</th>
      <th>product_white dress</th>
      <th>product_white tuxedo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>49.330000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32.490000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>199.990000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>93.936667</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>93.936667</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Splitting data into training and test sets


```python
features = ['price', 'rating', 'product_blue pants', 'product_red shirt', 'product_white dress', 'product_white tuxedo']
x = df[features].values
target = 'in_stock'
y = df[target].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print('        shape')
print('_____________')
print('x_train', x_train.shape)
print('x_test', x_test.shape)
print('y_train', y_train.shape)
print('y_test', y_test.shape)
```

            shape
    _____________
    x_train (3, 6)
    x_test (2, 6)
    y_train (3,)
    y_test (2,)
    


```python

```
