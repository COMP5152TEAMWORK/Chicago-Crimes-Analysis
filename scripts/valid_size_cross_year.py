from utilities import exec_over
import pandas as pd

def f(df: pd.DataFrame):
    year = df['Year'][0]
    n, p = df.shape
    df = df.dropna()
    n_after, _ = df.shape
    n_dropped = n - n_after
    return pd.DataFrame.from_records(data=[(int(year),n,p,n_after, n_dropped)],
                                     columns=['year', 'size', 'dimension', 'size after', 'size dropped'])


if __name__ == '__main__':
    exec_over(f, "size.csv")