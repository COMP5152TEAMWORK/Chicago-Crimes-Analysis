from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

file_new1 = pd.read_csv("../dataset/trainset.csv", delimiter=',')
df0 = file_new1['Date'].str.split("/", expand=True)
df0.columns = ['Month', 'Day', 'Year']
file_new1['Date'] = df0[['Year', 'Month', 'Day']].apply('-'.join, axis=1)
file_new1['Date'] = pd.to_datetime(file_new1['Date'])

file_new2 = pd.read_csv("../dataset/testset.csv", delimiter=',')
df = file_new2['Date'].str.split("/", expand=True)
df.columns = ['Month', 'Day', 'Year']
file_new2['Date'] = df[['Year', 'Month', 'Day']].apply('-'.join, axis=1)
file_new2['Date'] = pd.to_datetime(file_new2['Date'])

file_new = pd.concat([file_new1, file_new2])
file_new = file_new.reset_index(drop=True)

new_col = ['ds', 'y']
file_new.columns = new_col

holidays = pd.DataFrame({
    'holiday': 'playoff',
    'ds': pd.to_datetime(['2020-01-01', '2020-07-04', '2020-11-26', '2020-12-25', '2021-01-01', '2021-07-04',
                          '2021-11-26', '2021-12-25', '2022-01-01', '2022-07-04',
                          '2022-11-26', '2022-12-25', '2023-01-01', '2023-07-04',
                          '2023-11-26', '2023-12-25'])})

pro = Prophet(holidays=holidays)

# pro.add_country_holidays(country_name='US')
pro.fit(file_new)
pred = pro.predict(file_new)

future = pro.make_future_dataframe(periods=365)
future.tail()
# 预测数据集
forecast = pro.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# 展示预测结果
pro.plot(forecast)
# 预测的成分分析绘图，展示预测中的趋势、周效应和年度效应
pro.plot_components(forecast)
plt.show()
