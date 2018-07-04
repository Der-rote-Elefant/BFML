import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from matplotlib import pyplot as plt
from Mldata import Mldata
from datetime import datetime
from dateutil.relativedelta import relativedelta

def getProfit(data,lastSumProfit):
    data['Date'] = pd.to_datetime(data['Date']) #将数据类型转换为日期类型

    data.sort_values(by=['Date','Pred'],ascending=[1,0],inplace=True)

    startDt = data.iloc[0,1]
    endDt = data.iloc[-1,1]

    data = data.set_index('Date')



    #print(data.loc['2016-07-21'])
    #data = data[data['Pred']>5]
    #print(data)
    x=[]
    profit=[]
    sumProfit = lastSumProfit
    while startDt<=endDt:
        try:
            dLen = data.loc[startDt].iloc[:,0].size
            if dLen>10:
                dLen = 10
            mySum = data.loc[startDt].ix[0:dLen,[Lable]].sum()
            print(data.loc[startDt].iloc[0:dLen])
            x.append(startDt)
            sumProfit = sumProfit+mySum
            profit.append(sumProfit)
        except:
            pass
        startDt = startDt + relativedelta(days=1)
        #print(x)
    return x,profit,sumProfit

#Lable = 'Lable'
Lable = 'Lable_10'

startDt = datetime(2014,10,1)
endDt = datetime(2018,5,1)
mlData = DataFrame()
x = []
profit = []
SumProfit = 0
while(startDt<endDt):
    file = '%d%02d' % (startDt.year,startDt.month)
    mlData = pd.read_csv("Result2/{0}.csv".format(file))
    mlData = mlData[mlData[Lable]!=-9999]
    x1,y1,s = getProfit(mlData,SumProfit)
    x = x + x1
    profit = profit + y1
    SumProfit = s
    
    print(startDt)
    startDt = startDt + relativedelta(months=1)

plt.plot(x,profit)
plt.show()