import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt

plt.figure()
data = pd.read_csv("trade.txt")
foo = data.values
data = pd.DataFrame(foo, columns=['date', 'code', 'qty', 'price','type','profit']) 
data = data[data['profit']>100000]
print(data[data['profit']>100000])

data.plot(x='date', y='profit')
plt.show()