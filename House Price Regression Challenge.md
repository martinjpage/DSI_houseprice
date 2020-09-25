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


```python
housePrice = pd.read_csv("HousePrice_Final.csv")
```

**or** load by url


```python
# url = https://raw.githubusercontent.com/martinjpage/DSI_houseprice/master/HousePrice_Final.csv
# housePrice = pd.read_csv(url)
```

## Data Exploration
Examine the structure of the data


```python
housePrice.info()
```

Look at the summary statistics of the numerical columns


```python
housePrice.describe()
```

Look at the correlations between the numerical features


```python
plt.matshow(housePrice.corr()); plt.colorbar(); plt.show()
```

Look at the distribution of the numeric variables


```python
housePrice.iloc[:,0].hist(); plt.show()
housePrice.iloc[:,2].hist(); plt.show()
housePrice.iloc[:,3].hist(); plt.show()
housePrice.iloc[:,4].hist(); plt.show()
housePrice.iloc[:,5].hist(); plt.show()
```

Look the the counts of the categorical features


```python
housePrice.iloc[:,1].hist(); plt.show()
housePrice.iloc[:,6].hist(); plt.show()
```

## Data Processing

Combine house and townhouse as one group


```python
housePrice["propertyType"] = housePrice["propertyType"].str.replace("Townhouse", "House")
```

Look at summmary statistics by property type


```python
housePrice.groupby(by = "propertyType").describe().transpose()
```

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


```python
#check for positive skewness
skewed_feats = housePrice[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75].index
```


```python
skewed_feats
```


```python
#overwrite skewed columns with log transformation
housePrice[skewed_feats] = np.log1p(housePrice[skewed_feats])
```


```python
housePrice.describe()
```

See distributions of log transformed columns


```python
housePrice.iloc[:,0].hist(); plt.show()
housePrice.iloc[:,2].hist(); plt.show()
housePrice.iloc[:,3].hist(); plt.show()
housePrice.iloc[:,4].hist(); plt.show()
housePrice.iloc[:,5].hist(); plt.show()
```

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

## Modelling


Split the data into target (y) and features (X)


```python
X = housePrice.iloc[:,1:]
y = housePrice.iloc[:,0]
```


```python
X.head()
```


```python
y.head()
```

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

Fit linear regression model directly for test set prediction


```python
linreg.fit(X_train, y_train)
```

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


```python
print("CV RMSE for Ridge Regression",cv_ridge.min())
```

### Model 3: Lasso Regression   


```python
#use list comprehension to iterate through the hyperparameters and obtain CV RMSE using the define RMSE model function
cv_lasso = [rmse_cv(Lasso(alpha = alf)).mean() for alf in alphas]
cv_lasso = pd.Series(cv_lasso, index=alphas)

#plot searched hyperparamters vs RMSE
cv_lasso.plot(title = "Validation"); plt.xlabel("alpha"); plt.ylabel("rmse"); plt.show()
```


```python
print("CV RMSE for Lasso Regression",cv_lasso.min())
```

Fit the lasso model directly to plot the coefficients to see which features are important


```python
model_lasso = LassoCV(alphas = alphas).fit(X_train, y_train)

coef = pd.Series(model_lasso.coef_, index = X_train.columns)
imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])
imp_coef.plot(kind = "barh"); plt.title("Coefficients in the Lasso Model"); plt.show()
```

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


```python
grid_rf.best_params_
```


```python
print("CV RMSE for Random Forest Regression", np.sqrt(np.abs(grid_rf.best_score_)))
```

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

## Test set evaluation

Linear regression model prediction


```python
linreg_pred = linreg.predict(X_test)
print("Test RMSE for Linear Regression",MSE(np.expm1(y_test), np.expm1(linreg_pred))**(0.5))
```

Random forest model prediction


```python
rf_pred = grid_rf.predict(X_test)
print("Test RMSE for Random Forest Regression",MSE(np.expm1(y_test), np.expm1(rf_pred))**(0.5))
```

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
