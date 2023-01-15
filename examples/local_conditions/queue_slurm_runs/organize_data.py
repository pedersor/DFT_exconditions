import pandas as pd
import glob

all_files = glob.glob("out/*.csv")
all_df = pd.DataFrame()
for file_ in all_files:
  df = pd.read_csv(file_)
  all_df = pd.concat((df, all_df), axis=0)

all_df.to_csv('comprehensive_search.csv', index=False)
