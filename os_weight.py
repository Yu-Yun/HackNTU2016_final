# -*- coding: utf-8 -*-
import pandas as pd
from collections import Counter
import numpy as np
import string
import json


### read in questionaire
df = pd.read_csv('data/questionaire.csv')

### delete the first column
df = df.drop(df.columns[[0]], axis=1)

### read the os
with open('data/os.txt', 'r') as f:
    os = json.load(f)
### put os into lists with 5 items each


def sub_cat():
    ### read in column values
    col_num = len(df.columns.values)
    sub_cat = np.array_split(range(col_num), 19) # 情境
    sub_cat_dict = {}
    for idx, small_df in enumerate(sub_cat):
        sub_cat_idx = sub_cat[idx]
        cols = df[df.columns[sub_cat_idx]]
        sub_cat_l = []
        for sub_idx in range(5):
            col_vals = cols[cols.columns[sub_idx]]
            count_os = Counter(col_vals)

            os1 = count_os[1] * 1.0
            os2 = count_os[2] * 2.0
            os3 = count_os[3] * 3.0
            os4 = count_os[4] * 4.0
            os5 = count_os[5] * 5.0

            subsub_cat = sum([os1, os2, os3, os4, os5])/sum(count_os.values())
            sub_cat_l.append(subsub_cat)
        sub_cat_dict[string.ascii_lowercase[idx]] = list(zip(os[idx],sub_cat_l))
    return sub_cat_dict
