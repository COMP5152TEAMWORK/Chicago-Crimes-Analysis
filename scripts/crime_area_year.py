from utilities import exec_over
import pandas as pd

def f(df: pd.DataFrame):
    year = df['Year'][0]
    df = df.dropna()
    df = df['Community Area'].value_counts().reset_index()
    df['Year'] = year
    df.columns = ['Community Area', 'Counts', 'Year']
    df = df[(df['Community Area'] >= 1) & (df['Community Area'] <= 77)]
    df['Community Area'] = df['Community Area'].astype('int')

    return df


if __name__ == '__main__':
    exec_over(f, "crime_area_year.csv")