from evaluator import PyscfEvaluator
from dataset import Dataset

xc = 'pbe'
dset = Dataset('ie_atoms.yaml')
evl = PyscfEvaluator(xc, c=',pbe')

for i in range(len(dset)):
  curr_calc = dset[i]
  label = curr_calc["name"].split(' ')[-1]

  sys_checks = evl.exact_cond_checks(curr_calc)

  print(sys_checks)