# Hacathons_Solutions
approach from various ml competition
For the feature engineering following steps i followed
1> As all the variables were numerical and random, I looked into the missing values. 
2> Removed the variables which were missing more than 38%
3> filled the missing values with the mean of the variable
4> For the standardization and normalization i used standardscaler and minmax
5> After filling the missing value,I checked for oversampling ,for that i Used SMOTE
6> Then i used various classfication approach . I was getting better f1 accuracy with logistic regression with solver =lbfgs and C=3
7> I used Xgboost also was not getting good accuracy
8> Then i did the weighted ensemble of bothe the solution to get better accuracy
