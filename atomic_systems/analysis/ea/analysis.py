from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

HAR_TO_KCAL = 627.5

out_dir = Path('data/')
sorted_cols = [
    'pbe', 'am05', 'scan', 'r2scan', 'b3lyp', 'sogga11', 'm06', 'b97', 'mn15'
]
sorted_cols = [xc.upper() for xc in sorted_cols]

pretty_conds = {
    'ec_non_positivity': '$E_c$ non-positivity',
    'ec_scaling_check': '$E_c[n_{\gamma}]$ inequalities',
    'tc_non_negativity': '$T_c$ non-negativity',
    'tc_upper_bound': '$T_c$ upper-bound',
    'adiabatic_ec_concavity': '$E^{\lambda}_c[n]$ concavity',
    'lieb_oxford_bound_exc': r'Lieb-Oxford bound on $E_{xc}$',
    'lieb_oxford_bound_uxc': r'Lieb-Oxford bound on $U_{xc}$',
}

# drop nonbound anions
drop_systems = ['He', 'Be', 'N', 'Ne', 'Mg', 'Ar']


def get_ea_err_fig():

  errs_df = pd.read_csv(out_dir / 'ea_errs.csv', index_col=0)
  errs_df = errs_df.rename(columns=str.upper)
  errs_df = errs_df.drop(drop_systems, axis=0)

  maes = []
  for xc in errs_df.columns:
    xc = xc.upper()
    errs_df[xc] = errs_df[xc].abs() * HAR_TO_KCAL
    mae = errs_df[xc].abs().mean()
    maes.append(mae)

  errs_df.loc['MAE'] = maes
  errs_df = errs_df.reindex(sorted_cols, axis=1)
  # signal that calculations were performed on HF densities
  errs_df.columns = ['HF-' + xc for xc in errs_df.columns]

  ax = plt.axes()
  plot = sns.heatmap(
      errs_df,
      annot=True,
      vmax=10,
      fmt='.2g',
      cmap="YlGnBu",
      cbar_kws={'label': 'abs. error [kcal/mol]'},
      ax=ax,
  )

  ax.set_title('Electron affinities error')
  fig = plot.get_figure()
  fig.savefig('ea_errs.pdf', bbox_inches='tight')


def exact_cond_checks_fig():

  # get system names
  errs_df = pd.read_csv(out_dir / 'ea_errs.csv', index_col=0)
  systems = errs_df.index.repeat(2)

  checks_df = {'Exact conds': []}
  for i, csv in enumerate(out_dir.glob('ea_checks_*.csv')):
    xc_df = pd.read_csv(csv)
    xc_df['Systems'] = systems
    xc_df.set_index('Systems', inplace=True)
    xc_df = xc_df.drop(xc_df.columns[0], axis=1)
    xc_df = xc_df.drop(drop_systems, axis=0)
    # True -> 1, False -> 0
    xc_df = xc_df.astype(int)

    if i == 0:
      # get exact conditions tested
      checks_df['Exact conds'] = list(xc_df.columns)
      num_systems = len(xc_df.index)

    cond_totals = []
    for cond in checks_df['Exact conds']:
      tot = xc_df[cond].sum()
      cond_totals.append(tot)

    xc = csv.stem.split('_')[-1].upper()
    checks_df[xc] = cond_totals

  checks_df = pd.DataFrame.from_dict(checks_df)
  checks_df = checks_df.set_index('Exact conds')
  checks_df = checks_df.rename(pretty_conds, axis='index')
  checks_df = checks_df.reindex(sorted_cols, axis=1)
  # signal that calculations were performed on HF densities
  checks_df.columns = ['HF-' + xc for xc in checks_df.columns]

  ax = plt.axes()
  plot = sns.heatmap(
      checks_df,
      annot=True,
      fmt='.2g',
      cmap="YlGnBu",
      cbar_kws={'label': f'no. of systems satisfied (max. {num_systems})'},
      ax=ax,
  )

  ax.set_title('Exact conditions')
  fig = plot.get_figure()
  fig.savefig('ea_exact_cond_checks.pdf', bbox_inches='tight')


if __name__ == '__main__':
  get_ea_err_fig()
