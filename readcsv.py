import pandas as pd
import numpy as np

df = pd.read_csv('/home/chrispi/Desktop/ted_prj/train_set.csv')

# for column in df.columns:
#     print column

print df.describe()

