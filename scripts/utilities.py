import pandas as pd

DATA_DIR = '../data/'
DATA_GEN_DIR = '../data_gen/'
suffix = '.csv'


def exec_over(func, years: list[int], output_filename: str):
    partial_result = list()
    years.sort()
    for y in years:
        df = pd.read_csv('{}{}{}'.format(DATA_DIR, y, suffix))
        partial_result.append(func(df))
        del df
    df = pd.concat(partial_result)
    df.save('{}{}'.format(DATA_GEN_DIR, output_filename))
