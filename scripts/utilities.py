import pandas as pd

DATA_DIR = '../data/'
DATA_GEN_DIR = '../data_gen/'
suffix = '.csv'


def exec_over(func, output_filename: str, years: list[int] = [i for i in range(2010, 2023)]):
    partial_result = list()
    years.sort()
    for y in years:
        df = pd.read_csv('{}{}{}'.format(DATA_DIR, y, suffix))  # read data
        partial_result.append(func(df))
        del df
    df = pd.concat(partial_result, ignore_index=True)
    df.to_csv('{}{}'.format(DATA_GEN_DIR, output_filename), index=False)

