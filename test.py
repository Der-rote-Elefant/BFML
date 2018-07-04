import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
'''  
df = pd.DataFrame([[1,2,3],[2,3,4],[2,4,3],[1,3,7]],  
                  index = ['one','two','three','four'],columns = ['A','B','C'])  
print(df) 
#        A  B  C  
# one    1  2  3  
# two    2  3  4  
# three  2  4  3  
# four   1  3  7  
df.sort_values(by=['A','B'],ascending=[1,1],inplace=True)  
print(df)
#        A  B  C  
# two    2  3  4  
# three  2  4  3  
# one    1  2  3  
# four   1  3  7 
'''
test = pd.read_csv("R_ALL_10/201506.csv")
test['Date'] = pd.to_datetime(test['Date']) #将数据类型转换为日期类型

test.sort_values(by=['Date','Lable'],ascending=[1,0],inplace=True)

startDt = test.iloc[0,1]
endDt = test.iloc[-1,1]

test = test.set_index('Date')

print(test[test['Lable']!=61.45])

'''
x=[]
profit=[]
sumProfit =0
while startDt<=endDt:
    try:
        mySum = test.loc[startDt].ix[0:10,['Lable']].sum()
        x.append(startDt)
        sumProfit = sumProfit+mySum
        profit.append(sumProfit)
    except:
        print('')
    startDt = startDt + relativedelta(days=1)
    #print(startDt)
    pass


plt.plot(x,profit)
plt.show()
'''
