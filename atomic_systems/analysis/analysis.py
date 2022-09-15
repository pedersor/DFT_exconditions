from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

HAR_TO_KCAL = 627.5

out_dir = Path('../out/')
nonempirical = ['scan', 'pbe', 'r2scan', 'am05']

errs_df = {'System': []}
for i, csv in enumerate(out_dir.glob('errs_*.csv')):
  xc_df = pd.read_csv(csv)
  if i == 0:
    # all systems + last entry is the MAE
    systems = pd.concat([xc_df['label'], pd.Series(['MAE'])])
    errs_df['System'] = systems

  xc = csv.stem.split('_')[-1].lower()
  errs = xc_df['error'].abs() * HAR_TO_KCAL
  errs = pd.concat([errs, pd.Series([errs.mean()])])
  errs_df[xc] = errs

errs_df = pd.DataFrame.from_dict(errs_df)
errs_df = errs_df.set_index('System')


# sort columns non-empirical -> empirical
def sort_fn(xc):
  return int(xc not in nonempirical)


cols_to_sort = errs_df.columns
sorted_cols = sorted(cols_to_sort, key=sort_fn)
errs_df = errs_df.reindex(sorted_cols, axis=1)

ax = plt.axes()
plot = sns.heatmap(
    errs_df,
    annot=True,
    fmt='.2g',
    cmap="YlGnBu",
    cbar_kws={'label': 'abs. error [kcal/mol]'},
    ax=ax,
)

ax.set_title('Ionization energies error')
fig = plot.get_figure()
fig.savefig('ie_errs.pdf', bbox_inches='tight')
