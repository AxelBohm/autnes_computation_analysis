from functools import wraps

import pandas as pd


def find_cols(df, **kwargs):
    return df.filter(**kwargs).columns.tolist()


def find_column(df, **kwargs):
    columns = find_columns(df, **kwargs)
    if len(columns) != 1:
        raise ValueError(f'Expected one column but found: {columns}')
    return columns[0]


def has_non_trivial_return_value(function):
    @wraps(function)
    def wrapper(*args, **kwds):
        value = function(*args, **kwds)

        if isinstance(value, pd.Series):
            value = value.to_frame()

        is_okay = ~value.isnull().all().all()

        if is_okay:
            return value
        # A warning maybe more appropriate but for demo purpose we can be drastic.
        raise ValueError('Trivial return value encountered.')

    return wrapper