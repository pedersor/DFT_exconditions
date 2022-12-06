import pandas as pd
import subprocess

header = [
    'xc_id',
    'condition',
    'satisfied',
    'violation fraction',
    'r_s range',
    's range',
    'zeta range',
    'alpha range',
    'q range',
]

# combine all csvs into one
tmp_files = 'tmp_files'
proc = subprocess.Popen(
    f'cat {tmp_files}/*.csv > data.csv',
    shell=True,
    stdout=None,
)
proc.wait()

df = pd.read_csv(
    'data.csv',
    na_values='---',
    header=None,
    names=header,
)

df.sort_values(by=['xc_id'], inplace=True)
df['type'] = df['xc_id'].apply(lambda x: x.split('_c_')[0].upper())
df['xc_name'] = df['xc_id'].apply(
    lambda x: x.split('_c_')[-1].replace('_', '-').upper())
df.to_csv('data.csv', index=False)
