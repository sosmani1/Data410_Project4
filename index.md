# Advanced Machine Learning Project 4


### Part 1: Here I create my own multiple boosting algortihm and apply it to combinations of different regressors (for example you can boost regressor 1 with regressor 2 a couple of times) on the "Concrete Compressive Strength" dataset. Then I will show what was the combination that achieved the best cross-validated results.

Let's import some useful information to process the project. 

```
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsmr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import matplotlib.pyplot as plt
from matplotlib import pyplot
```
Now I update the algorithms for repeated boosting:

```
# Tricubic Kernel
def Tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)

# Quartic Kernel
def Quartic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,15/16*(1-d**2)**2)

# Epanechnikov Kernel
def Epanechnikov(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,3/4*(1-d**2)) 
```
Defining the kernel local regression model

```

def lw_reg(X, y, xnew, kern, tau, intercept):
    # tau is called bandwidth K((x-x[i])/(2*tau))
    n = len(X) # the number of observations
    yest = np.zeros(n)

    if len(y.shape)==1: # here we make column vectors
      y = y.reshape(-1,1)

    if len(X.shape)==1:
      X = X.reshape(-1,1)
    
    if intercept:
      X1 = np.column_stack([np.ones((len(X),1)),X])
    else:
      X1 = X

    w = np.array([kern((X - X[i])/(2*tau)) for i in range(n)]) # here we compute n vectors of weights

    #Looping through all X-points
    for i in range(n):          
        W = np.diag(w[:,i])
        b = np.transpose(X1).dot(W).dot(y)
        A = np.transpose(X1).dot(W).dot(X1)
        beta, res, rnk, s = lstsq(A, b)
        yest[i] = np.dot(X1[i],beta)
    if X.shape[1]==1:
      f = interp1d(X.flatten(),yest,fill_value='extrapolate')
    else:
      f = LinearNDInterpolator(X, yest)
    output = f(xnew) # the output may have NaN's where the data points from xnew are outside the convex hull of X
    if sum(np.isnan(output))>0:
      g = NearestNDInterpolator(X,y.ravel()) 
      # output[np.isnan(output)] = g(X[np.isnan(output)])
      output[np.isnan(output)] = g(xnew[np.isnan(output)])
    return output
    
def boosted_lwr(X, y, xnew, kern, tau, intercept, model_boosting, nboost):
  Fx = lw_reg(X,y,X,kern,tau,intercept) # we need this for training the Decision Tree
  output = booster(X,y,xnew,kern,tau,model_boosting,nboost)
  return output 
 
def booster(X,y,xnew,kern,tau,model_boosting,nboost):
  Fx = lw_reg(X,y,X,kern,tau,True)
  Fx_new = lw_reg(X,y,xnew,kern,tau,True)
  new_y = y - Fx
  output = Fx
  output_new = Fx_new
  for i in range(nboost):
    model_boosting.fit(X,new_y)
    output += model_boosting.predict(X)
    output_new += model_boosting.predict(xnew)
    new_y = y - output
  return output_new

concrete = pd.read_csv('concrete.csv')

model_boosting = RandomForestRegressor(n_estimators=100,max_depth=3)

train = concrete.drop(['strength'], axis = 1)
train_labels = concrete.strength

```

Now I will use regularization for feature selection

```

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
model = Lasso(alpha=0.1)
model.fit(train,train_labels)
model.coef_

```
array([ 0.11965545,  0.10367264,  0.08783123, -0.1516849 ,  0.28548338,
        0.01770773,  0.01990805,  0.11417497])
        
```

features = np.array(train.columns)
print("All features:")
print(features)
```

All features:
['cement' 'slag' 'ash' 'water' 'superplastic' 'coarseagg' 'fineagg' 'age']

```
X = concrete[['superplastic', 'water']].values
y = concrete['strength'].values
from sklearn import preprocessing
from sklearn import utils

lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y)

scale = StandardScaler()

xscaled = scale.fit_transform(X)

```

### Part 2: I give a background on the LightGBM algorithm and include a write-up that explains the method. Then I apply the method to the "Concrete Compressive Strength" dataset. 

![images](https://user-images.githubusercontent.com/78623027/156957252-c260f757-733d-4b33-96f1-857e31bc90a0.png)


LightGBM: LightGBM is a gradient boosting function using a tree based learning algorithm. It grows the trees vertically, while other tree based algorithms grow the trees horizontally.

As per the official documentation of Lightgbm. The algorithm is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:

* Faster training speed and higher efficiency.
* Lower memory usage.
* Better accuracy.
* Support of parallel, distributed, and GPU learning.
* Capable of handling large-scale data.

On this experiment run by students at MIT the LightGBM is clearly the best result. The experiment was based on data from numerous financial instititions. 


More about this can be found here: [LightGBM Experiments
](!%5BLINK%5D:%20https://github.com/microsoft/LightGBM/blob/master/docs/Experiments.rst#comparison-experiment)


![Screen Shot 2022-03-06 at 9 21 47 PM](https://user-images.githubusercontent.com/78623027/156956906-b05189f3-d700-4402-9012-01e76af99bf0.png)

Futhermore as Dr. Brownlee, from Swinebure University, suggests LightGBM extends the gradient boosting algorithm by adding a type of automatic feature selection as well as focusing on boosting examples with larger gradients. This can result in a dramatic speedup of training and improved predictive performance.

As such, LightGBM has become a de facto algorithm for machine learning competitions when working with tabular data for regression and classification predictive modeling tasks. As such, it owns a share of the blame for the increased popularity and wider adoption of gradient boosting methods in general, along with Extreme Gradient Boosting (XGBoost).

Let's begin the coding!

Let's import the lightgbm package

```
import lightgbm as lgb
```
We want more nested cross-validations

```
mse_blwr = []

mse_lgb = []

for i in [123]:
  kf = KFold(n_splits=10,shuffle=True,random_state=i)
  # this is the Cross-Validation Loop
  for idxtrain, idxtest in kf.split(X):
    xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = X[idxtest]
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)
    dat_train = np.concatenate([xtrain,ytrain.reshape(-1,1)],axis=1)
    dat_test = np.concatenate([xtest,ytest.reshape(-1,1)],axis=1)
    yhat_blwr = boosted_lwr(xtrain,ytrain, xtest,Epanechnikov,0.9,True,model_boosting,2)
    #yhat_blwr = boosted_lwr(xtrain,ytrain,xtest,Tricubic,1,True,model_boosting,2)
    clf = lgb.LGBMClassifier()
    clf.fit(xtrain, ytrain.astype('int'))
    yhat_lgb =clf.predict(xtest)
    mse_blwr.append(mse(ytest,yhat_blwr))
    mse_lgb.append(mse(ytest,yhat_lgb))
print('The Cross-validated Mean Squared Error for Boosted LWR is : '+str(np.mean(mse_blwr)))
print('The Cross-validated Mean Squared Error for LGB is : '+str(np.mean(mse_lgb)))
```

### The Cross-validated Mean Squared Error for Boosted LWR is : 199.14032290938763
### The Cross-validated Mean Squared Error for LGB is : 284.03590233009703

## Repeated Boosting is the best method, the cross validated MSE is lowest.

