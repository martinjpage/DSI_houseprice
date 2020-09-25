# House Price Regression Challenge

**Author**: Martin J Page   
**Date**: 25 September 2020    

**Objective**: Build a regression model to that can predict the price of a house in Cape Town, South Africa from basic features scraped from the web.  

**Data**:   
House price data were scraped from property24.com and sahometraders.co.za, using the search criteria
"Cape Town", as well as the property types "House", "Apartment", and "Townhouse". The resulting search
pages were the inputs for the web scaper. The web scarping code for [property24.com](https://raw.githubusercontent.com/martinjpage/DSI_houseprice/master/Web%20Scaping_Property%2024.py) and
[sahometraders.co.za](https://raw.githubusercontent.com/martinjpage/DSI_houseprice/master/Web%20Scraping_SA%20Hometraders.py) are available on the author's [GitHub](https://github.com/martinjpage/DSI_houseprice). The data were then cleaned and
formatted and the two dataframes were merged, with 11970 observations, with this [code](https://raw.githubusercontent.com/martinjpage/DSI_houseprice/master/Data%20Cleaning.py). This yielded 5 numeric
variables: `price` (the target), `bedroom`, `bathroom`, `garage` (which are the number of the respective objects
in the house) and `floorSize` (which the size of the house in square metres); as well as two
categorical variabels: `location` (the suburb where the house is located) and `propertyType`,
either `House`, `Apartment` or `Townhouse` as defined by the property websites.

**Data Processing**   
In this notebook, the formatted data are explored and processed for modelling. The `propertyType` levels "House" and "Townhouse" were combined into one variable called "House". Observations belonging to a `location` with fewer than 80 examples in the dataset were removed. All the numeric data were skewed to the right and transformed using a log transformation. Missing numeric values were imputed using the column mean. Categorical variables were encoded as dummy variables. The processed data was split into a target (`y`) and features (`X`) set and then into train (80%) and test (20%) datasets for modelling.   

**Modelling**    
The below models were explored using cross-validation and scored using RMSE. Select hyperparameters were tuned.   
  1. Linear regression (no tuning)   
  2. Ridge regression (tuning of alpha)   
  3. Lasso regression (tuning of alpha)  
  4. Random forest regression (tuning of max_depth, min_samples_leaf, min_samples_split, n_estimators)   
  
**Evaluation**   
The best tuning parameter for regularised regression with `alpha = 0`. Therefore the standard linear regression and the random forest regression models were compared on the hold-out data set. Random forest produced the lowest RMSE on the test set at R5 554 764 (predictions were exponentiated to be on the orginal scale). This output was judged to be very poor. One issue might be that even after the log transformation, several variables still did not appear approximately normally distributed. However, a potent reason for the poor predictive performance of the models may be that the features do not offer enough discriminatory ability and do not capture some essential driver(s) of house prices: that is, houses with a similar number of bedrooms, bathrooms, etc. are priced very differently.   

**Note**: to use the notebook, start by running the library code block at the end

## Data Loading   



```python
!pwd
```

    /c/Users/Martin/OneDrive/Documents/Repos/DSI/Regression Twist Challenge
    


```python
housePrice = pd.read_csv("HousePrice_Final.csv")
```

**or** load by url


```python
# url = "https://raw.githubusercontent.com/martinjpage/DSI_houseprice/master/HousePrice_Final.csv"
# housePrice = pd.read_csv(url)
```

## Data Exploration
Examine the structure of the data


```python
housePrice.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 11970 entries, 0 to 11969
    Data columns (total 7 columns):
    price           11970 non-null float64
    location        11970 non-null object
    bedroom         11770 non-null float64
    bathroom        11367 non-null float64
    garage          8171 non-null float64
    floorSize       10566 non-null float64
    propertyType    11970 non-null object
    dtypes: float64(5), object(2)
    memory usage: 654.7+ KB
    

Look at the summary statistics of the numerical columns


```python
housePrice.describe()
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
      <th>bedroom</th>
      <th>bathroom</th>
      <th>garage</th>
      <th>floorSize</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>1.197000e+04</td>
      <td>11770.000000</td>
      <td>11367.000000</td>
      <td>8171.000000</td>
      <td>10566.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>6.404230e+06</td>
      <td>2.706415</td>
      <td>2.127210</td>
      <td>2.091237</td>
      <td>229.348931</td>
    </tr>
    <tr>
      <td>std</td>
      <td>9.023614e+06</td>
      <td>1.589975</td>
      <td>1.482776</td>
      <td>1.769116</td>
      <td>232.886255</td>
    </tr>
    <tr>
      <td>min</td>
      <td>1.990000e+05</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>1.000000</td>
      <td>17.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>1.950000e+06</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>68.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>3.250000e+06</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>116.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>6.950000e+06</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>318.750000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>1.725000e+08</td>
      <td>33.000000</td>
      <td>30.000000</td>
      <td>58.000000</td>
      <td>2203.000000</td>
    </tr>
  </tbody>
</table>
</div>



Look at the correlations between the numerical features


```python
plt.matshow(housePrice.corr()); plt.colorbar(); plt.show()
```


![png](output_11_0.png)


Look at the distribution of the numeric variables


```python
housePrice.iloc[:,0].hist(); plt.show()
housePrice.iloc[:,2].hist(); plt.show()
housePrice.iloc[:,3].hist(); plt.show()
housePrice.iloc[:,4].hist(); plt.show()
housePrice.iloc[:,5].hist(); plt.show()
```


![png](output_13_0.png)



![png](output_13_1.png)



![png](output_13_2.png)



![png](output_13_3.png)



![png](output_13_4.png)


Look the the counts of the categorical features


```python
housePrice.iloc[:,1].hist(); plt.show()
housePrice.iloc[:,6].hist(); plt.show()
```


![png](output_15_0.png)



![png](output_15_1.png)


## Data Processing

Combine house and townhouse as one group


```python
housePrice["propertyType"] = housePrice["propertyType"].str.replace("Townhouse", "House")
```

Look at summmary statistics by property type


```python
housePrice.groupby(by = "propertyType").describe().transpose()
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
      <th>propertyType</th>
      <th>Apartment</th>
      <th>House</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="8" valign="top">price</td>
      <td>count</td>
      <td>6.683000e+03</td>
      <td>5.287000e+03</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>4.681921e+06</td>
      <td>8.581305e+06</td>
    </tr>
    <tr>
      <td>std</td>
      <td>6.368804e+06</td>
      <td>1.116271e+07</td>
    </tr>
    <tr>
      <td>min</td>
      <td>2.450000e+05</td>
      <td>1.990000e+05</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>1.700000e+06</td>
      <td>2.595000e+06</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>2.640000e+06</td>
      <td>4.850000e+06</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>4.550000e+06</td>
      <td>9.992500e+06</td>
    </tr>
    <tr>
      <td>max</td>
      <td>7.500000e+07</td>
      <td>1.725000e+08</td>
    </tr>
    <tr>
      <td rowspan="8" valign="top">bedroom</td>
      <td>count</td>
      <td>6.486000e+03</td>
      <td>5.284000e+03</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>1.838961e+00</td>
      <td>3.771196e+00</td>
    </tr>
    <tr>
      <td>std</td>
      <td>8.262319e-01</td>
      <td>1.654062e+00</td>
    </tr>
    <tr>
      <td>min</td>
      <td>5.000000e-01</td>
      <td>5.000000e-01</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>1.000000e+00</td>
      <td>3.000000e+00</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>2.000000e+00</td>
      <td>3.000000e+00</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>2.000000e+00</td>
      <td>4.000000e+00</td>
    </tr>
    <tr>
      <td>max</td>
      <td>1.500000e+01</td>
      <td>3.300000e+01</td>
    </tr>
    <tr>
      <td rowspan="8" valign="top">bathroom</td>
      <td>count</td>
      <td>6.424000e+03</td>
      <td>4.943000e+03</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>1.553004e+00</td>
      <td>2.873457e+00</td>
    </tr>
    <tr>
      <td>std</td>
      <td>8.894789e-01</td>
      <td>1.744338e+00</td>
    </tr>
    <tr>
      <td>min</td>
      <td>5.000000e-01</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>1.000000e+00</td>
      <td>2.000000e+00</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>1.000000e+00</td>
      <td>2.500000e+00</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>2.000000e+00</td>
      <td>3.500000e+00</td>
    </tr>
    <tr>
      <td>max</td>
      <td>2.500000e+01</td>
      <td>3.000000e+01</td>
    </tr>
    <tr>
      <td rowspan="8" valign="top">garage</td>
      <td>count</td>
      <td>3.850000e+03</td>
      <td>4.321000e+03</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>1.393636e+00</td>
      <td>2.712798e+00</td>
    </tr>
    <tr>
      <td>std</td>
      <td>6.952861e-01</td>
      <td>2.160600e+00</td>
    </tr>
    <tr>
      <td>min</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>1.000000e+00</td>
      <td>2.000000e+00</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>1.000000e+00</td>
      <td>2.000000e+00</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>2.000000e+00</td>
      <td>3.000000e+00</td>
    </tr>
    <tr>
      <td>max</td>
      <td>1.500000e+01</td>
      <td>5.800000e+01</td>
    </tr>
    <tr>
      <td rowspan="8" valign="top">floorSize</td>
      <td>count</td>
      <td>6.430000e+03</td>
      <td>4.136000e+03</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>9.998323e+01</td>
      <td>4.304663e+02</td>
    </tr>
    <tr>
      <td>std</td>
      <td>8.315840e+01</td>
      <td>2.476696e+02</td>
    </tr>
    <tr>
      <td>min</td>
      <td>1.700000e+01</td>
      <td>2.600000e+01</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>5.600000e+01</td>
      <td>2.120000e+02</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>7.800000e+01</td>
      <td>4.100000e+02</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>1.100000e+02</td>
      <td>5.970000e+02</td>
    </tr>
    <tr>
      <td>max</td>
      <td>2.203000e+03</td>
      <td>9.980000e+02</td>
    </tr>
  </tbody>
</table>
</div>



The numerical variables appear to have a non-normal distribution. Check for skewness quantiatively
and perform a log transformation


```python
#log transform target (price)
housePrice.iloc[:,0] = np.log1p(housePrice.iloc[:,0])
```


```python
#make a list of the numerical columns
numeric_feats = housePrice.dtypes[housePrice.dtypes != "object"].index
```


```python
numeric_feats
```




    Index(['price', 'bedroom', 'bathroom', 'garage', 'floorSize'], dtype='object')




```python
#check for positive skewness
skewed_feats = housePrice[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75].index
```


```python
skewed_feats
```




    Index(['bedroom', 'bathroom', 'garage', 'floorSize'], dtype='object')




```python
#overwrite skewed columns with log transformation
housePrice[skewed_feats] = np.log1p(housePrice[skewed_feats])
```

    C:\Users\Martin\Anaconda3\lib\site-packages\ipykernel_launcher.py:2: RuntimeWarning: invalid value encountered in log1p
      
    


```python
housePrice.describe()
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
      <th>bedroom</th>
      <th>bathroom</th>
      <th>garage</th>
      <th>floorSize</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>11970.000000</td>
      <td>11770.000000</td>
      <td>11367.000000</td>
      <td>8171.000000</td>
      <td>10566.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>15.165514</td>
      <td>1.234756</td>
      <td>1.062287</td>
      <td>1.035614</td>
      <td>4.990430</td>
    </tr>
    <tr>
      <td>std</td>
      <td>0.935578</td>
      <td>0.382144</td>
      <td>0.372243</td>
      <td>0.392457</td>
      <td>0.931751</td>
    </tr>
    <tr>
      <td>min</td>
      <td>12.201065</td>
      <td>0.405465</td>
      <td>0.405465</td>
      <td>0.693147</td>
      <td>2.890372</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>14.483340</td>
      <td>1.098612</td>
      <td>0.693147</td>
      <td>0.693147</td>
      <td>4.234107</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>14.994166</td>
      <td>1.098612</td>
      <td>1.098612</td>
      <td>1.098612</td>
      <td>4.762174</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>15.754252</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>1.098612</td>
      <td>5.767539</td>
    </tr>
    <tr>
      <td>max</td>
      <td>18.965908</td>
      <td>3.526361</td>
      <td>3.433987</td>
      <td>4.077537</td>
      <td>7.698029</td>
    </tr>
  </tbody>
</table>
</div>



See distributions of log transformed columns


```python
housePrice.iloc[:,0].hist(); plt.show()
housePrice.iloc[:,2].hist(); plt.show()
housePrice.iloc[:,3].hist(); plt.show()
housePrice.iloc[:,4].hist(); plt.show()
housePrice.iloc[:,5].hist(); plt.show()
```


![png](output_30_0.png)



![png](output_30_1.png)



![png](output_30_2.png)



![png](output_30_3.png)



![png](output_30_4.png)


Fill the missing numerical values with the mean of that column


```python
housePrice[numeric_feats] = housePrice[numeric_feats].fillna(housePrice[numeric_feats].mean())
```

Remove rows that belong to a suburb with fewer than 80 houses


```python
#generate a Boolean list for the levels of the location variable that
#have a prevalence of more than 80
location_mask = housePrice["location"].value_counts() > 80
#extract the names of the suburbs that appear more than 80 ties
locations = location_mask.index[location_mask]
#generate a boolean mask to filter the rows based on the location column
bool_loc = []
for loc in housePrice["location"]:
    bool_loc.append(loc in locations)
#remove rows with suburbs with too little data
housePrice = housePrice[bool_loc]
```

Encode the categorical features as dummy variables. Drop first to avoid duplicate information.


```python
housePrice = pd.get_dummies(housePrice, drop_first=True)
```


```python
housePrice.head()
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
      <th>bedroom</th>
      <th>bathroom</th>
      <th>garage</th>
      <th>floorSize</th>
      <th>location_Camps Bay</th>
      <th>location_Cape Town City Centre</th>
      <th>location_Claremont</th>
      <th>location_Claremont Upper</th>
      <th>location_Clifton</th>
      <th>...</th>
      <th>location_Tamboerskloof</th>
      <th>location_Three Anchor Bay</th>
      <th>location_Tokai</th>
      <th>location_Vredehoek</th>
      <th>location_Waterfront</th>
      <th>location_Woodstock</th>
      <th>location_Wynberg</th>
      <th>location_Wynberg Upper</th>
      <th>location_Zonnebloem</th>
      <th>propertyType_House</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>-0.815329</td>
      <td>-0.356278</td>
      <td>-0.991708</td>
      <td>-0.872676</td>
      <td>2.747319e-16</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.417367</td>
      <td>0.396565</td>
      <td>0.870458</td>
      <td>0.160532</td>
      <td>5.869914e-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.875485</td>
      <td>-0.356278</td>
      <td>0.097590</td>
      <td>0.160532</td>
      <td>1.932514e-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>-0.352046</td>
      <td>0.396565</td>
      <td>0.511721</td>
      <td>0.893605</td>
      <td>-5.236406e-02</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.471303</td>
      <td>0.396565</td>
      <td>0.870458</td>
      <td>0.893605</td>
      <td>2.747319e-16</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 43 columns</p>
</div>



## Modelling


Split the data into target (y) and features (X)


```python
X = housePrice.iloc[:,1:]
y = housePrice.iloc[:,0]
```


```python
X.head()
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
      <th>bedroom</th>
      <th>bathroom</th>
      <th>garage</th>
      <th>floorSize</th>
      <th>location_Camps Bay</th>
      <th>location_Cape Town City Centre</th>
      <th>location_Claremont</th>
      <th>location_Claremont Upper</th>
      <th>location_Clifton</th>
      <th>location_Constantia</th>
      <th>...</th>
      <th>location_Tamboerskloof</th>
      <th>location_Three Anchor Bay</th>
      <th>location_Tokai</th>
      <th>location_Vredehoek</th>
      <th>location_Waterfront</th>
      <th>location_Woodstock</th>
      <th>location_Wynberg</th>
      <th>location_Wynberg Upper</th>
      <th>location_Zonnebloem</th>
      <th>propertyType_House</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.098612</td>
      <td>0.693147</td>
      <td>0.693147</td>
      <td>4.990430</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>1.098612</td>
      <td>5.537334</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.098612</td>
      <td>1.098612</td>
      <td>1.098612</td>
      <td>5.170484</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.386294</td>
      <td>1.252763</td>
      <td>1.386294</td>
      <td>4.941642</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>4.990430</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 42 columns</p>
</div>




```python
y.head()
```




    0    14.402742
    1    15.555977
    2    15.984564
    3    14.836162
    4    15.606437
    Name: price, dtype: float64



Partition the data in a training and test set to perform a final evaluation of the models on a hold-out set


```python
#set the seed for all models
SEED = 2509
```


```python
#make train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = SEED)
```

Define a function that will perform cross validation on a model and computes the root mean squared error


```python
def rmse_cv(model):
    rmse = np.sqrt(np.abs(cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 5)))
    return(rmse)
```

### Model 1: Linear Regression


```python
#instantiate linear regression model
linreg = LinearRegression()
#run cross validation function to obtain the RMSE
cv_linreg = rmse_cv(linreg).mean()
print("CV RMSE for Linear Regression", cv_linreg)
```

    CV RMSE for Linear Regression 0.35861158575254304
    

Fit linear regression model directly for test set prediction


```python
linreg.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



### Model 2: Ridge Regression   


```python
#define hyperparameters to test
alphas = np.arange(0.0000, 1, 0.05)
```


```python
#use list comprehension to iterate through the hyperparameters and obtain CV RMSE using the defined RMSE model function
cv_ridge = [rmse_cv(Ridge(alpha = alf)).mean() for alf in alphas]
cv_ridge = pd.Series(cv_ridge, index=alphas)

#plot searched hyperparamters vs RMSE
cv_ridge.plot(title = "Validation"); plt.xlabel("alpha"); plt.ylabel("rmse"); plt.show()
```


![png](output_54_0.png)



```python
print("CV RMSE for Ridge Regression",cv_ridge.min())
```

    CV RMSE for Ridge Regression 0.35861158575254304
    

### Model 3: Lasso Regression   


```python
#use list comprehension to iterate through the hyperparameters and obtain CV RMSE using the define RMSE model function
cv_lasso = [rmse_cv(Lasso(alpha = alf)).mean() for alf in alphas]
cv_lasso = pd.Series(cv_lasso, index=alphas)

#plot searched hyperparamters vs RMSE
cv_lasso.plot(title = "Validation"); plt.xlabel("alpha"); plt.ylabel("rmse"); plt.show()
```

    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:516: UserWarning: With alpha=0, this algorithm does not converge well. You are advised to use the LinearRegression estimator
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:475: UserWarning: Coordinate descent with no regularization may lead to unexpected results and is discouraged.
      positive)
    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:475: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 412.9122707289364, tolerance: 0.528479113156257
      positive)
    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:516: UserWarning: With alpha=0, this algorithm does not converge well. You are advised to use the LinearRegression estimator
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:475: UserWarning: Coordinate descent with no regularization may lead to unexpected results and is discouraged.
      positive)
    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:475: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 414.4303271008563, tolerance: 0.5312259428844291
      positive)
    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:516: UserWarning: With alpha=0, this algorithm does not converge well. You are advised to use the LinearRegression estimator
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:475: UserWarning: Coordinate descent with no regularization may lead to unexpected results and is discouraged.
      positive)
    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:475: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 404.6595287773862, tolerance: 0.5226164930357472
      positive)
    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:516: UserWarning: With alpha=0, this algorithm does not converge well. You are advised to use the LinearRegression estimator
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:475: UserWarning: Coordinate descent with no regularization may lead to unexpected results and is discouraged.
      positive)
    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:475: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 418.66940867664965, tolerance: 0.5332142768494826
      positive)
    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:516: UserWarning: With alpha=0, this algorithm does not converge well. You are advised to use the LinearRegression estimator
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:475: UserWarning: Coordinate descent with no regularization may lead to unexpected results and is discouraged.
      positive)
    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:475: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 420.1681070891912, tolerance: 0.5229226755343028
      positive)
    


![png](output_57_1.png)



```python
print("CV RMSE for Lasso Regression",cv_lasso.min())
```

    CV RMSE for Lasso Regression 0.35861158575254304
    

Fit the lasso model directly to plot the coefficients to see which features are important


```python
model_lasso = LassoCV(alphas = alphas).fit(X_train, y_train)

coef = pd.Series(model_lasso.coef_, index = X_train.columns)
imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])
imp_coef.plot(kind = "barh"); plt.title("Coefficients in the Lasso Model"); plt.show()
```

    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\model_selection\_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
      warnings.warn(CV_WARNING, FutureWarning)
    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:471: UserWarning: Coordinate descent with alpha=0 may lead to unexpected results and is discouraged.
      tol, rng, random, positive)
    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:471: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 339.6958350165528, tolerance: 0.44236523293124286
      tol, rng, random, positive)
    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:471: UserWarning: Coordinate descent with alpha=0 may lead to unexpected results and is discouraged.
      tol, rng, random, positive)
    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:471: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 339.7631607656149, tolerance: 0.43689771244095704
      tol, rng, random, positive)
    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:471: UserWarning: Coordinate descent with alpha=0 may lead to unexpected results and is discouraged.
      tol, rng, random, positive)
    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:471: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 354.0599514130822, tolerance: 0.4398763201105555
      tol, rng, random, positive)
    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:1227: UserWarning: With alpha=0, this algorithm does not converge well. You are advised to use the LinearRegression estimator
      model.fit(X, y)
    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:475: UserWarning: Coordinate descent with no regularization may lead to unexpected results and is discouraged.
      positive)
    C:\Users\Martin\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:475: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 518.4425562268007, tolerance: 0.6596557352370789
      positive)
    


![png](output_60_1.png)


### Model 4: Random Forest Regression


```python
#instantiate mode
rf = RandomForestRegressor(random_state = SEED)

#find best hyperparameters
params_rf = {'max_depth': [3, 8, 15, None],
              'max_features': ['auto'],
              "min_samples_leaf": [1, 5, 10],
              "min_samples_split": [5, 15, 30],
              'n_estimators': [100, 300]}

grid_rf = GridSearchCV(estimator = rf, param_grid = params_rf, cv = 3, verbose = 2, n_jobs = None, scoring = "neg_mean_squared_error")
grid_rf.fit(X_train, y_train)
```

    Fitting 3 folds for each of 72 candidates, totalling 216 fits
    [CV] max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=100 
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    

    [CV]  max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=100 
    

    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.2s remaining:    0.0s
    

    [CV]  max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=300, total=   1.0s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=300, total=   1.0s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=300, total=   1.0s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=300, total=   1.0s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=300, total=   1.0s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=300, total=   1.0s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=300, total=   1.0s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=300, total=   1.0s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=300, total=   1.0s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=300, total=   1.0s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=300, total=   1.0s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=300, total=   1.0s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=300, total=   1.0s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=300, total=   1.0s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=300, total=   1.0s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=300, total=   0.9s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=300, total=   0.9s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=300, total=   1.0s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=300, total=   1.0s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=300, total=   0.9s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=300, total=   1.0s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=300, total=   1.0s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=300, total=   0.9s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=300, total=   0.9s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=100, total=   0.3s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=300, total=   1.0s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=300, total=   0.9s
    [CV] max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=3, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=300, total=   1.0s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=100, total=   0.7s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=100, total=   0.7s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=100, total=   0.7s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=300, total=   2.3s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=300, total=   2.2s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=300, total=   2.2s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=100, total=   0.7s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=100, total=   0.7s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=100, total=   0.7s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=300, total=   2.2s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=300, total=   2.2s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=300, total=   2.2s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=100, total=   0.7s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=100, total=   0.7s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=100, total=   0.7s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=300, total=   2.2s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=300, total=   2.2s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=300, total=   2.2s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=100, total=   0.7s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=100, total=   0.7s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=100, total=   0.8s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=300, total=   2.2s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=300, total=   2.2s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=300, total=   2.2s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=100, total=   0.7s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=100, total=   0.7s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=100, total=   0.8s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=300, total=   2.2s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=300, total=   2.2s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=300, total=   2.2s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=100, total=   0.7s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=100, total=   0.7s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=100, total=   0.7s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=300, total=   2.2s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=300, total=   2.3s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=300, total=   2.2s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=100, total=   0.8s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=100, total=   0.8s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=100, total=   0.8s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=300, total=   2.2s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=300, total=   2.1s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=300, total=   2.2s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=100, total=   0.7s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=100, total=   0.7s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=100, total=   0.7s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=300, total=   2.2s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=300, total=   2.1s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=300, total=   2.1s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=100, total=   0.7s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=100, total=   0.7s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=100, total=   0.7s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=300, total=   2.2s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=300, total=   2.2s
    [CV] max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=8, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=300, total=   2.1s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=100, total=   1.3s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=100, total=   1.2s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=100, total=   1.2s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=300, total=   3.5s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=300, total=   3.6s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=300, total=   3.5s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=100, total=   1.1s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=100, total=   1.1s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=100, total=   1.1s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=300, total=   3.3s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=300, total=   3.3s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=300, total=   3.2s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=100, total=   1.1s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=100, total=   1.0s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=100, total=   1.0s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=300, total=   3.2s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=300, total=   3.1s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=300, total=   3.1s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=100, total=   1.1s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=100, total=   1.1s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=100, total=   1.1s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=300, total=   3.2s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=300, total=   3.2s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=300, total=   3.1s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=100, total=   1.0s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=100, total=   1.1s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=100, total=   1.0s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=300, total=   3.2s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=300, total=   3.2s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=300, total=   3.1s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=100, total=   1.0s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=100, total=   1.0s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=100, total=   1.0s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=300, total=   2.9s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=300, total=   3.0s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=300, total=   3.0s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=100, total=   1.0s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=100, total=   1.0s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=100, total=   1.0s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=300, total=   2.9s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=300, total=   2.9s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=300, total=   2.9s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=100, total=   1.0s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=100, total=   1.0s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=100, total=   1.0s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=300, total=   2.8s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=300, total=   2.9s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=300, total=   2.9s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=100, total=   0.9s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=100, total=   0.9s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=100, total=   0.9s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=300, total=   2.8s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=300, total=   2.8s
    [CV] max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=15, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=300, total=   2.8s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=100, total=   1.6s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=100, total=   1.6s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=100, total=   1.6s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=300, total=   4.7s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=300, total=   4.8s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=5, n_estimators=300, total=   4.7s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=100, total=   1.4s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=100, total=   1.4s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=100, total=   1.4s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=300, total=   4.3s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=300, total=   4.2s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=15, n_estimators=300, total=   4.1s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=100, total=   1.3s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=100, total=   1.3s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=100, total=   1.3s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=300, total=   3.8s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=300, total=   3.9s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=1, min_samples_split=30, n_estimators=300, total=   3.8s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=100, total=   1.2s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=100, total=   1.2s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=100, total=   1.2s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=300, total=   3.6s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=300, total=   3.6s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=5, n_estimators=300, total=   3.6s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=100, total=   1.2s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=100, total=   1.2s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=100, total=   1.2s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=300, total=   3.5s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=300, total=   3.5s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=15, n_estimators=300, total=   3.5s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=100, total=   1.1s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=100, total=   1.1s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=100, total=   1.1s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=300, total=   3.5s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=300, total=   3.4s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=5, min_samples_split=30, n_estimators=300, total=   3.5s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=100, total=   1.0s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=100, total=   1.0s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=100, total=   1.0s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=300, total=   3.1s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=300, total=   3.1s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=5, n_estimators=300, total=   3.1s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=100, total=   1.0s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=100, total=   1.1s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=100, total=   1.0s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=300, total=   3.0s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=300, total=   3.1s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=15, n_estimators=300, total=   3.0s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=100, total=   1.0s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=100, total=   1.0s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=100 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=100, total=   1.0s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=300, total=   2.9s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=300, total=   2.9s
    [CV] max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=300 
    [CV]  max_depth=None, max_features=auto, min_samples_leaf=10, min_samples_split=30, n_estimators=300, total=   2.9s
    

    [Parallel(n_jobs=1)]: Done 216 out of 216 | elapsed:  5.9min finished
    




    GridSearchCV(cv=3, error_score='raise-deprecating',
                 estimator=RandomForestRegressor(bootstrap=True, criterion='mse',
                                                 max_depth=None,
                                                 max_features='auto',
                                                 max_leaf_nodes=None,
                                                 min_impurity_decrease=0.0,
                                                 min_impurity_split=None,
                                                 min_samples_leaf=1,
                                                 min_samples_split=2,
                                                 min_weight_fraction_leaf=0.0,
                                                 n_estimators='warn', n_jobs=None,
                                                 oob_score=False, random_state=2509,
                                                 verbose=0, warm_start=False),
                 iid='warn', n_jobs=None,
                 param_grid={'max_depth': [3, 8, 15, None],
                             'max_features': ['auto'],
                             'min_samples_leaf': [1, 5, 10],
                             'min_samples_split': [5, 15, 30],
                             'n_estimators': [100, 300]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='neg_mean_squared_error', verbose=2)




```python
grid_rf.best_params_
```




    {'max_depth': None,
     'max_features': 'auto',
     'min_samples_leaf': 1,
     'min_samples_split': 5,
     'n_estimators': 300}




```python
print("CV RMSE for Random Forest Regression", np.sqrt(np.abs(grid_rf.best_score_)))
```

    CV RMSE for Random Forest Regression 0.31763582021072895
    

Save best random forest model


```python
best_rf = grid_rf.best_estimator_
```

Plot the important features of the random forest


```python
#create pd.Series of features importances
importances_rf = pd.Series(best_rf.feature_importances_, index = X.columns)

#sort importance_rf
sorted_importances_rf = importances_rf.sort_values()

#make horizontal bar plot
sorted_importances_rf.plot(kind = "barh", color = "lightgreen"); plt.show()
```


![png](output_68_0.png)


## Test set evaluation

Linear regression model prediction


```python
linreg_pred = linreg.predict(X_test)
print("Test RMSE for Linear Regression",MSE(np.expm1(y_test), np.expm1(linreg_pred))**(0.5))
```

    Test RMSE for Linear Regression 6408568.32673033
    

Random forest model prediction


```python
rf_pred = grid_rf.predict(X_test)
print("Test RMSE for Random Forest Regression",MSE(np.expm1(y_test), np.expm1(rf_pred))**(0.5))
```

    Test RMSE for Random Forest Regression 5554763.841721576
    

#### Libraries
Load these first to run the script


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
```


```python

```
