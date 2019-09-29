import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.preprocessing import MinMaxScaler


train=pd.read_csv('E:\\HACKEREARTH HDFC\\DataSet\\Train.csv')
test=pd.read_csv('E:\\HACKEREARTH HDFC\\DataSet\\Test.csv')



y_train=train['Col2']
Col1=test['Col1']

drop_col_array=['Col1','Col2']
train=train.drop(drop_col_array,axis=1)


# from imblearn.over_sampling import SMOTE
# sm = SMOTE()
# X, y = sm.fit_sample(train, y_train)


test=test.drop('Col1',axis=1)




cols_to_remove=train.columns[test.isnull().mean() >0.5]

train=train.drop(cols_to_remove,axis=1)
test=test.drop(cols_to_remove,axis=1)



train.fillna(train.mean(),inplace=True)
test.fillna(test.mean(),inplace=True)




train_col=train.columns
test_col=test.columns



from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
train=sc.fit_transform(train)
test=sc.transform(test)

scaler=MinMaxScaler(feature_range=(0,1))
train=scaler.fit_transform(train)
test=scaler.transform(test)



output=pd.DataFrame(y_train)

sns.countplot('Col2', data = output)



train=pd.DataFrame(train,columns=train_col)
test=pd.DataFrame(test,columns=test_col)


# from imblearn.over_sampling import SMOTE
# sm = SMOTE()
# X, y = sm.fit_sample(train, output)
# # X = pd.DataFrame(X, columns = train.columns)
# # y=pd.DataFrame(y)








#logistic
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(penalty = 'l2', C = 3,random_state = 100,solver='liblinear')
model.fit(train, output)
# make predictions for test data
y_pred = model.predict(test)



#ridge







#random forest
# model = RandomForestClassifier(random_state=42,max_depth= 15 ,min_samples_split= 5 ,n_estimators= 500,n_jobs=-1, min_samples_leaf=2,  criterion='entropy',class_weight= {0: 0.8, 1:1})
# model.fit(train.values,output.values.ravel())
# y_pred=model.predict(test)






pd.DataFrame({'Col1':Col1,'Col2':y_pred}).set_index('Col1').to_csv('E:\\HACKEREARTH HDFC\\DataSet\\sub.csv')


# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

