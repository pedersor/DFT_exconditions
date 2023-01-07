import pytest
import numpy as np

import dft_exconditions.local_condition_checks as cc

test_local_conditions_data = [
    ('gga_c_pbe', 'negativity_check', 0.0),
    ('gga_c_pbe', 'deriv_lower_bd_check', 0.0),
    ('gga_c_pbe', 'deriv_upper_bd_check_1', 0.0),
    ('gga_c_pbe', 'deriv_upper_bd_check_2', 0.013125),
    ('gga_c_pbe', 'second_deriv_check', 0.0),
    ('gga_c_pbe', 'lieb_oxford_bd_check_Uxc', 0.0),
    ('gga_c_pbe', 'lieb_oxford_bd_check_Exc', 0.0),
    ('hyb_gga_xc_b3lyp', 'negativity_check', 0.3560833333333333),
    ('hyb_gga_xc_b3lyp', 'deriv_lower_bd_check', 0.371375),
    ('hyb_gga_xc_b3lyp', 'deriv_upper_bd_check_1', 0.0),
    ('hyb_gga_xc_b3lyp', 'deriv_upper_bd_check_2', 0.20241666666666666),
    ('hyb_gga_xc_b3lyp', 'second_deriv_check', 0.281125),
    ('hyb_gga_xc_b3lyp', 'lieb_oxford_bd_check_Uxc', 0.0),
    ('hyb_gga_xc_b3lyp', 'lieb_oxford_bd_check_Exc', 0.0),
    ('mgga_c_scan', 'negativity_check', 0.0),
    ('mgga_c_scan', 'deriv_lower_bd_check', 0.0),
    ('mgga_c_scan', 'deriv_upper_bd_check_1', 0.0),
    ('mgga_c_scan', 'deriv_upper_bd_check_2', 0.0),
    ('mgga_c_scan', 'second_deriv_check', 0.0),
    ('mgga_c_scan', 'lieb_oxford_bd_check_Uxc', 0.0),
    ('mgga_c_scan', 'lieb_oxford_bd_check_Exc', 0.0),
    ('mgga_c_m06', 'negativity_check', 0.391575),
    ('mgga_c_m06', 'deriv_lower_bd_check', 0.3651),
    ('mgga_c_m06', 'deriv_upper_bd_check_1', 0.3014901764901765),
    ('mgga_c_m06', 'deriv_upper_bd_check_2', 0.3813),
    ('mgga_c_m06', 'second_deriv_check', 0.3765),
    ('mgga_c_m06', 'lieb_oxford_bd_check_Uxc', 0.32535833333333336),
    ('mgga_c_m06', 'lieb_oxford_bd_check_Exc', 0.3351),
]

test_inp = {
    'r_s': np.linspace(0.0001, 2, 1000),
    's': np.linspace(0, 5, 6),
    'zeta': np.linspace(0, 1, 4, endpoint=True),
}


@pytest.mark.parametrize('func_id, condition_string, expected',
                         test_local_conditions_data)
def test_local_conditions(func_id: str, condition_string: str, expected: float):
  """Test local conditions and check with reference values.
  
  Args:
    func_id: Libxc functional ID.
    condition_string: Local condition string to check.
    expected: Expected fraction of tested points violating the condition.
  """

  if 'mgga_' in func_id:
    test_inp['alpha'] = np.linspace(0, 5, 5)

  df = cc.check_condition(
      func_id,
      condition_string,
      test_inp,
      num_blocks=1,
  )

  np.testing.assert_allclose(
      df['percent_violated'].iloc[0],
      expected,
      atol=1e-3,
  )
