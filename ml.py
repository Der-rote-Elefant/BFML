import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas import Series
from matplotlib import pyplot as plt
from Mldata import Mldata
from datetime import datetime
from dateutil.relativedelta import relativedelta


def lgbTrain(trainData,testData,features,Lable):

    lgb_train = lgb.Dataset(trainData[features], trainData[Lable], free_raw_data=False)
    lgb_eval = lgb.Dataset(testData[features], testData[Lable], reference=lgb_train,free_raw_data=False)

    ### 开始训练
    print('设置参数')
    params = {
                'boosting_type': 'gbdt',
                'application': 'regression',
                'metric': 'l2',

                'learning_rate': 0.01,
                'num_leaves':25,
                'max_depth':6,

                'max_bin':10,
                'min_data_in_leaf':20,

                'feature_fraction': 0.6,
                'bagging_fraction': 1,
                'bagging_freq':0,

                'min_split_gain': 0
    }

    print("开始训练")
    gbm = lgb.train(params,                     # 参数字典
                    lgb_train,                  # 训练集
                    num_boost_round=30,       # 迭代次数
                    valid_sets=lgb_eval,        # 验证集
                    early_stopping_rounds=5)   # 早停系数
    return gbm


md = Mldata('R_ALL_10')
startDt = datetime(2016,9,1)
endDt = datetime(2017,4,1)

md.readData(startDt,endDt)

Lable = "Lable_10"
trainData = md.trainData
trainData = trainData[trainData[Lable]!=-9999]
trainData = trainData.drop('Code',axis=1)
trainData = trainData.drop('Date',axis=1)

testData = md.testData
#testData = testData[testData[Lable]!=-9999]

datacolumns = trainData.columns.size - 1
features = list(trainData.columns[0:datacolumns])

gbm = lgbTrain(trainData,testData,features,Lable)

### 线下预测
print ("线下预测")
preds_offline = gbm.predict(testData[features], num_iteration=gbm.best_iteration)
testData['Pred'] = preds_offline

testDt = md.testDT;
testFile = '%d%02d' % (testDt.year,testDt.month)
saveFeatures = ['Code','Date','Pred',Lable]

testData[saveFeatures].to_csv('Result/{0}.csv'.format(testFile),index=False,sep=',')


s1 = Series(testData['Pred'])
s2 = Series(testData[Lable])
corr=s1.corr(s2)

print(corr)


testData.sort_values(by=['Pred'],ascending=[0],inplace=True)
print(testData)
testData = testData.ix[0:50,['Lable_10']]


plt.figure(figsize=(12,6))
lgb.plot_importance(gbm, max_num_features=30)
plt.title("Featurertances")
plt.show()


