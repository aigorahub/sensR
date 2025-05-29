import pytest
import numpy as np
import os # Added for setting R_HOME and R_LIBS_USER

# Explicitly set R_HOME *before* importing rpy2.robjects
# This is a common requirement for rpy2 to correctly initialize R.
if 'R_HOME' not in os.environ:
    os.environ['R_HOME'] = '/usr/lib/R' # Standard R home on Linux

# Explicitly set R_LIBS_USER for rpy2 to find user-installed packages
r_libs_user = os.path.expanduser('~/R/libs')
if 'R_LIBS_USER' not in os.environ: 
    os.environ['R_LIBS_USER'] = r_libs_user
elif r_libs_user not in os.environ['R_LIBS_USER']: 
    os.environ['R_LIBS_USER'] = f"{r_libs_user}:{os.environ['R_LIBS_USER']}"

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.rinterface_lib.sexp import NULLType as RNULLType 
import warnings 

from senspy.models import BetaBinomial 

try:
    ro.r('library(stats)') 
    sensR = importr('sensR')
    rpy2_available = True
    numpy2ri.activate() 
except Exception as e:
    print(f"RPy2 or sensR setup failed: {e}")
    sensR = None
    rpy2_available = False

test_params = [
    ([20], [50], False, None, "standard_single_group"),
    ([10, 20], [25, 50], False, None, "standard_multi_group"),
    ([5], [100], False, None, "standard_low_x_prop"),
    ([95], [100], False, None, "standard_high_x_prop"),
    ([1], [5], False, None, "standard_small_n_low_x"),
    ([4], [5], False, None, "standard_small_n_high_x"),
    ([0], [10], False, None, "standard_x_zero"),
    ([10], [10], False, None, "standard_x_equals_n"),
    ([20], [50], True, 0.5, "corrected_single_pg0.5"),
    ([20], [50], True, 0.25, "corrected_single_pg0.25"),
    ([10, 20], [25, 50], True, 0.5, "corrected_multi_pg0.5"),
    ([0], [10], True, 0.5, "corrected_x_zero_pg0.5"),
    ([10], [10], True, 0.5, "corrected_x_equals_n_pg0.5"),
]

@pytest.mark.skipif(not rpy2_available, reason="RPy2 or sensR not available or R setup issue.")
@pytest.mark.parametrize("x_obs, n_trials, corrected, p_guess_py, test_id", test_params)
def test_beta_binomial_fit_vs_sensR(x_obs, n_trials, corrected, p_guess_py, test_id):
    with localconverter(default_converter + numpy2ri.converter):
        x_r = ro.conversion.py2rpy(np.array(x_obs, dtype=np.int32))
        n_r = ro.conversion.py2rpy(np.array(n_trials, dtype=np.int32))
    
    corrected_r = corrected 
    current_p_guess_r_val = p_guess_py
    if corrected:
        if p_guess_py is None: 
            current_p_guess_r_val = 0.5
        pg_r = ro.FloatVector([current_p_guess_r_val])
    else:
        pg_r = ro.NULL 
    
    if corrected and (p_guess_py == 0.0 or p_guess_py == 1.0):
        pytest.skip(f"Skipping {test_id} for senspy as p_guess={p_guess_py} is currently not supported by senspy's fit.")

    r_model_fit = sensR.betaBin(x_r, n_r, corrected=corrected_r, pg=pg_r)
    r_coeffs = np.array(r_model_fit.rx2('coefficients'))
    r_alpha, r_beta = r_coeffs[0], r_coeffs[1]   
    r_loglik = r_model_fit.rx2('logLik')[0]

    py_model = BetaBinomial() 
    try:
        py_model.fit(np.array(x_obs), np.array(n_trials), corrected=corrected, p_guess=p_guess_py)
    except ValueError as e:
        pytest.fail(f"senspy fit failed for {test_id} with p_guess={p_guess_py}: {e}")
        return
    except Exception as e: 
        pytest.fail(f"senspy fit raised an unexpected error for {test_id}: {e}")
        return

    np.testing.assert_allclose(py_model.alpha, r_alpha, rtol=1e-12, atol=1e-12, err_msg=f"{test_id}: Alpha mismatch")
    np.testing.assert_allclose(py_model.beta, r_beta, rtol=1e-12, atol=1e-12, err_msg=f"{test_id}: Beta mismatch")
    np.testing.assert_allclose(py_model.loglik, r_loglik, rtol=1e-9, atol=1e-9, err_msg=f"{test_id}: LogLik mismatch")

# This test does NOT depend on rpy2_available, it tests senspy's own methods.
def test_beta_binomial_summary_and_confint_basic():
    x_obs_py, n_trials_py = [20, 30], [50, 60]
    corrected_py, p_guess_py_fit = False, None 
    
    py_model = BetaBinomial()
    try:
        py_model.fit(np.array(x_obs_py), np.array(n_trials_py), corrected=corrected_py, p_guess=p_guess_py_fit)
    except Exception as e:
        pytest.fail(f"senspy fit failed during summary/confint test setup: {e}")

    if not py_model.convergence_status:
        warnings.warn(UserWarning(f"Fit did not converge for summary/confint test. Results might be unreliable."))

    summary_str = py_model.summary()
    assert isinstance(summary_str, str), "summary() should return a string."
    assert "BetaBinomial Model Summary" in summary_str
    assert "alpha:" in summary_str
    assert "beta:" in summary_str
    assert "Log-Likelihood:" in summary_str

    try:
        conf_intervals = py_model.confint(parm=['alpha', 'beta'], level=0.95)
        assert isinstance(conf_intervals, dict)
        assert "alpha" in conf_intervals and "beta" in conf_intervals
        assert isinstance(conf_intervals["alpha"], tuple) and isinstance(conf_intervals["beta"], tuple)
        
        alpha_ci_lower, alpha_ci_upper = conf_intervals["alpha"]
        beta_ci_lower, beta_ci_upper = conf_intervals["beta"]

        if not (np.isnan(alpha_ci_lower) or np.isnan(alpha_ci_upper)):
            assert alpha_ci_lower <= alpha_ci_upper + 1e-7
            assert alpha_ci_lower <= py_model.alpha + 1e-7
            assert py_model.alpha <= alpha_ci_upper + 1e-7
        
        if not (np.isnan(beta_ci_lower) or np.isnan(beta_ci_upper)):
            assert beta_ci_lower <= beta_ci_upper + 1e-7
            assert beta_ci_lower <= py_model.beta + 1e-7
            assert py_model.beta <= beta_ci_upper + 1e-7
    except ValueError as e: 
        pytest.fail(f"confint() raised ValueError: {e}")
    except RuntimeError as e: 
        warnings.warn(UserWarning(f"confint() raised RuntimeError: {e}. CIs: {conf_intervals if 'conf_intervals' in locals() else 'not computed'}"))
        if 'conf_intervals' not in locals() or not isinstance(conf_intervals,dict):
             pytest.fail(f"confint() failed structurally before or due to RuntimeError: {e}")
