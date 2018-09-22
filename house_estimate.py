# import tools
import time
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing
import numpy as np
import matplotlib
matplotlib.use("SVG")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')



#Create dictionary
from collections import defaultdict
d = defaultdict(preprocessing.LabelEncoder)


# import data
trainDataX = pd.read_csv('/data/datasets/House_prices/train.csv')
testDataX = pd.read_csv('/data/datasets/House_prices/test.csv')


# select data we want
trainDataY = trainDataX['SalePrice']
trainDataX = trainDataX.iloc[:,0:80]
testDataX = testDataX.iloc[:,0:80]


#Describe SalePrice
#print(trainDataY.describe())


#Print SalePrice histogram
sns.distplot(trainDataY)

#plt.show(block=True)
plt.savefig('saleprice_histo.svg')


#Coefficient d asymetrie ( skewness)
print("Skewness: %f" % trainDataY.skew())
#Coefficient d acuite ( kurtosis)
print("Kurtosis: %f" % trainDataY.kurt())


#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([trainDataY, trainDataX[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
plt.savefig('grlivarea_saleprice.svg')


#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([trainDataY, trainDataX[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
plt.savefig('totalbsmtsf_saleprice.svg')

#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([trainDataY, trainDataX[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.savefig('overallqual_saleprice.svg')

#box plot yearbuilt/saleprice
var = 'YearBuilt'
data = pd.concat([trainDataY, trainDataX[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
plt.savefig('yearbuilt_saleprice.svg')



exit(0)


for column in trainDataX:
	mean_column = trainDataX[column].mean()
	trainDataX.replace(np.nan, mean_column, inplace=True)

for column in testDataX:
        mean_column = testDataX[column].mean()
        testDataX.replace(np.nan, mean_column, inplace=True)


fitData = pd.concat([trainDataX,testDataX])

###############################################################
### these line is used for cross validation testing         
###############################################################
#X_train, X_test, y_train,y_test = cross_validation.train_test_split(trainDataX, trainDataY, test_size=0.1, random_state=0)
#clf = linear_model.ElasticNet(alpha = 0.1)
#fitted = clf.fit(X_train,y_train)
#y_predicted = clf.predict(X_test)
#print fitted.score(X_test, y_test)
###############################################################

# log data
trainDataY = np.log1p(trainDataY)


# transform and scale data
#Use LabelEncoder to normalize labels

#Replace Missing 
#fitData.replace(np.nan, 'NAN', inplace=True)
#trainDataX.replace(np.nan, 'NAN', inplace=True)
#testDataX.replace(np.nan, 'NAN', inplace=True)

# Encoding the variable
fitData.apply(lambda x: d[x.name].fit(x))

# Inverse the encoded
#fit.apply(lambda x: d[x.name].inverse_transform(x))

# Using the dictionary to label future data
#df.apply(lambda x: d[x.name].transform(x))


trainDataX = trainDataX.apply(lambda x: d[x.name].transform(x))
testDataX = testDataX.apply(lambda x: d[x.name].transform(x))


#Use StandardScaler()
#scaler_test = preprocessing.StandardScaler()
#scaler_test.fit(testDataX)
#scaler_test.transform(testDataX)

#scaler_train = preprocessing.StandardScaler()
#scaler_train.fit(trainDataX)
#scaler_train.transform(trainDataX)


# train on the data
#clf = linear_model.SGDRegressor(loss = "squared_loss", average=True)
clf = linear_model.ElasticNet(alpha=0.00001,l1_ratio=0.1,fit_intercept=True,normalize=False,precompute=False,max_iter=100000,copy_X=True,tol=0.00001,warm_start=False,positive=False,random_state=None,selection='cyclic')
#clf = linear_model.Lasso(alpha = 0.01,max_iter=100000)
fitted = clf.fit( trainDataX, trainDataY)
predicted = clf.predict(testDataX)

# reverse the log
predicted = (np.exp(predicted)).astype(float)

# create file
print('Generating submission file ...')
results = pd.DataFrame({'SalePrice': predicted}, dtype=int)  


print(clf.score(trainDataX,trainDataY))
        
#Writting to csv
results.index += 1461
results.to_csv('regressor.csv', index=True, header=True, index_label='id')  
