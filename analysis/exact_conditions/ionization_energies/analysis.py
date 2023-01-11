from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

HAR_TO_KCAL = 627.5

out_dir = Path('data/')
sorted_cols = [
    'pbe',
    'am05',
    'scan',
    'b3lyp',
    'mod-b3lyp',
    'b97',
    'm06',
    'm08-hx',
    'sogga11',
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
    'tc_ec_conjecture': r'Conjecture: $T_c \leq -E_c$',
}


def get_ie_err_fig():
  errs_df = {'System': []}
  for i, csv in enumerate(out_dir.glob('*errs_*.csv')):
    xc_df = pd.read_csv(csv)
    if i == 0:
      # all systems + last entry is the MAE
      systems = pd.concat([xc_df['label'], pd.Series(['MAE'])])
      errs_df['System'] = systems

    xc = csv.stem.split('_')[-1].upper()
    errs = xc_df['error'].abs() * HAR_TO_KCAL
    errs = pd.concat([errs, pd.Series([errs.mean()])])
    errs_df[xc] = errs

  errs_df = pd.DataFrame.from_dict(errs_df)
  errs_df = errs_df.set_index('System')
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
  ax.set_ylabel('')
  ax.set_title('Ionization energies error')
  fig = plot.get_figure()
  fig.savefig('ie_errs.pdf', bbox_inches='tight')
  plt.close()


def exact_cond_checks_fig():

  checks_df = {'Exact conds': []}
  for i, csv in enumerate(out_dir.glob('*checks_*.csv')):
    xc_df = pd.read_csv(csv)
    xc_df = xc_df.drop(xc_df.columns[0], axis=1)
    xc_df = xc_df.drop(columns=['tc_non_negativity'])
    # True -> 1, False -> 0
    xc_df = xc_df.astype(int)

    if i == 0:
      # get exact conditions tested
      checks_df['Exact conds'] = list(xc_df.columns)

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

  ax = plt.axes()
  plot = sns.heatmap(
      checks_df,
      annot=True,
      fmt='.2g',
      cmap="YlGnBu",
      cbar_kws={'label': 'no. of systems satisfied (max. 35)'},
      ax=ax,
  )
  ax.set_ylabel('')
  fig = plot.get_figure()
  fig.savefig('ie_exact_cond_checks.pdf', bbox_inches='tight')
  plt.close()


if __name__ == '__main__':
  get_ie_err_fig()
  exact_cond_checks_fig()
