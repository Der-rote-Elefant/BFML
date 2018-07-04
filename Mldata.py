import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

class Mldata:

	def __init__(self,folder):
		self.folder = folder

	def readData(self,startDt,endDt):

		self.trainData = self.__getTrainData(startDt,endDt)
		#endDt = endDt + relativedelta(months=1)
		self.testDT = endDt
		file = '%d%02d' % (self.testDT.year,self.testDT.month)
		self.testData = self.__getBigCSV(self.folder+'/{0}.csv'.format(file))

	def __getBigCSV(self,file):
	    f = open(file, encoding='utf8')
	    reader = pd.read_csv(f,iterator=True)
	    loop = True
	    chunkSize = 100000
	    chunks = []
	    while loop:
	        try:
	            chunk = reader.get_chunk(chunkSize)
	            chunks.append(chunk)
	        except StopIteration:
	            print(file+"：读取完成！")
	            loop = False
	    df = pd.concat(chunks, ignore_index=True)
	    return df

	def __getTrainData(self,sDt,eDT):
    	# 读取文件原始数据
	    dataframe = []
	    while(sDt<eDT):
	        file = '%d%02d' % (sDt.year,sDt.month)
	        dataframe.append(self.__getBigCSV(self.folder+'/{0}.csv'.format(file)))
	        sDt = sDt + relativedelta(months=1)
	    return pd.concat(dataframe)