import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

# 取各列的資料
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 3].values


# 補全缺失的資料, 這裡使用了 "most_frequent"
imputer = Imputer(missing_values = "NaN", strategy = "most_frequent", axis = 0)
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])




# 基於某個屬性來設施 標籤
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])




# 根據剛剛建立的標籤進行編碼
# You may notice that the columns have increased in the data set. 
# The column 'Country' is broken into three columns and column Gender is broken into two columns. 
# Thus, the resulting number of columns in X vector is increased from four to seven. 
# Also, notice that after applying the OneHotEncoding function, 
# the values in the Panda Dataframe are changed to scientific notation.
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)


# 分割訓練集和測試集
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)


# 去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# 各列的 mean 為 0 , 方差為 1
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)



