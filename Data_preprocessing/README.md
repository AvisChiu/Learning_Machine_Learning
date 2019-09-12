Data-Preprocessing
--
**實現一些簡單的前處理**    
**目的: 熟悉前處理的一些步驟**    
<br/>


**原始資料資料**
```
Country,  Age,   Salary,  Purchased
------------------------------------
France,   44,     72000,     No
Spain,    27,     48000,    Yes
Germany,  30,     54000,     No
Spain,    38,     61000,     No
Germany,  40,       na,     Yes
France,   35,     58000,    Yes
Spain,    na,     52000,     No
France,   48,     79000,    Yes
Germany,  50,     83000,     No
France,   37,     67000,    Yes
```

<br/>

**取各列的資料**
```
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 3].values
```

<br/>

**補全缺失的資料, 這裡使用了 "most_frequent" **
```
imputer = Imputer(missing_values = "NaN", strategy = "most_frequent", axis = 0)
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
```

```
[['France'   44.0 72000.0]
 ['Spain'    27.0 48000.0]
 ['Germany'  30.0 54000.0]
 ['Spain'    38.0 61000.0]
 ['Germany'  40.0 48000.0]
 ['France'   35.0 58000.0]
 ['Spain'    27.0 52000.0]
 ['France'   48.0 79000.0]
 ['Germany'  50.0 83000.0]
 ['France'   37.0 67000.0]]
```

<br/>
  
**基於某個屬性（國家）來設置標籤**

```
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
```

```
[[0 44.0 72000.0]
 [2 27.0 48000.0]
 [1 30.0 54000.0]
 [2 38.0 61000.0]
 [1 40.0 48000.0]
 [0 35.0 58000.0]
 [2 27.0 52000.0]
 [0 48.0 79000.0]
 [1 50.0 83000.0]
 [0 37.0 67000.0]]
```


<br/>

**# 根據剛剛建立的標籤進行編碼
# You may notice that the columns have increased in the data set.    
# The column 'Country' is broken into three columns and column Gender is broken into two columns.     
# Thus, the resulting number of columns in X vector is increased from four to seven.     
# Also, notice that after applying the OneHotEncoding function,     
# the values in the Panda Dataframe are changed to scientific notation.    
**
```
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
```
```
[[1.0e+00 0.0e+00 0.0e+00 4.4e+01 7.2e+04]
 [0.0e+00 0.0e+00 1.0e+00 2.7e+01 4.8e+04]
 [0.0e+00 1.0e+00 0.0e+00 3.0e+01 5.4e+04]
 [0.0e+00 0.0e+00 1.0e+00 3.8e+01 6.1e+04]
 [0.0e+00 1.0e+00 0.0e+00 4.0e+01 4.8e+04]
 [1.0e+00 0.0e+00 0.0e+00 3.5e+01 5.8e+04]
 [0.0e+00 0.0e+00 1.0e+00 2.7e+01 5.2e+04]
 [1.0e+00 0.0e+00 0.0e+00 4.8e+01 7.9e+04]
 [0.0e+00 1.0e+00 0.0e+00 5.0e+01 8.3e+04]
 [1.0e+00 0.0e+00 0.0e+00 3.7e+01 6.7e+04]]
 
 [0 1 0 0 1 1 0 1 0 1]
```


<br/>

**分割訓練集和測試集**
```
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
```

<br/>

**去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本（各列的 mean 為 0 , 方差為 1）**
```
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
```
```
X_train:
[[-1.          2.64575131 -0.77459667  0.4330127  -1.1851228 ]
 [ 1.         -0.37796447 -0.77459667  0.          0.59842834]
 [-1.         -0.37796447  1.29099445 -1.44337567 -1.1851228 ]
 [-1.         -0.37796447  1.29099445 -1.44337567 -0.80963835]
 [ 1.         -0.37796447 -0.77459667  1.58771324  1.72488169]
 [-1.         -0.37796447  1.29099445  0.14433757  0.03520167]
 [ 1.         -0.37796447 -0.77459667  1.01036297  1.0677839 ]
 [ 1.         -0.37796447 -0.77459667 -0.28867513 -0.24641167]]
 
 X_test:
 [[ 0.  0.  0. -1. -1.]
  [ 0.  0.  0.  1.  1.]]
```
