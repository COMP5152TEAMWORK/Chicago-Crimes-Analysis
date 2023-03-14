import pandas as pd
import numpy as np


file_2020 = pd.read_csv("../dataset/2020.csv")
df1 = pd.DataFrame(file_2020, columns=['ID', 'Date'])
df1['Date']=df1['Date'].str.split(' ', expand=True).iloc[:,0]
date1,num1=np.unique(df1['Date'],return_counts=True)
dataframe1=pd.DataFrame({'Date':date1,'Num':num1})
#dataframe1.to_csv("./dataset/test_new.csv",index=False)

file_2021=pd.read_csv("../dataset/2021.csv")
df2=pd.DataFrame(file_2021,columns=['ID','Date'])
df2['Date']=df2['Date'].str.split(' ',expand=True).iloc[:,0]
date2,num2=np.unique(df2['Date'],return_counts=True)
dataframe2=pd.DataFrame({'Date':date2,'Num':num2})
dataframe=pd.concat([dataframe1,dataframe2])
dataframe.to_csv("./dataset/trainset.csv",index=False)


file_2022=pd.read_csv("../dataset/2022.csv")
df3=pd.DataFrame(file_2022,columns=['ID','Date'])
df3['Date']=df3['Date'].str.split(' ',expand=True).iloc[:,0]
date3,num3=np.unique(df3['Date'],return_counts=True)
dataframe3=pd.DataFrame({'Date':date3,'Num':num3})
dataframe3.to_csv('./dataset/testset.csv',index=False)