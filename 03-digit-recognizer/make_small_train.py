
import pandas as pd

train = pd.read_csv('train.csv')
small_train = train[:10000]

small_train.to_csv('train_small_2.csv', index=False)
