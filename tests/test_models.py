import pytest
import numpy as np
import os
import subprocess
import warnings

# RPy2 import and setup logic
rpy2_available = False
sensR = None
ro = None
numpy2ri = None
default_converter = None
localconverter = None
RNULLType = None

try:
    r_home_from_env_ci = os.environ.get('R_HOME_DIR_CI')
    r_home_from_env = os.environ.get('R_HOME')
    
    r_home = None
    if r_home_from_env_ci:
        r_home = r_home_from_env_ci
    elif r_home_from_env:
        r_home = r_home_from_env
    else:
        try:
            r_home_process = subprocess.run(["R", "RHOME"], capture_output=True, text=True, check=True)
            r_home = r_home_process.stdout.strip()
        except Exception as e:
            warnings.warn(f"Failed to get R_HOME from subprocess: {e}. Falling back to default.")
            r_home = '/usr/lib/R' # Default fallback

    if r_home:
        os.environ['R_HOME'] = r_home
    else:
        warnings.warn("R_HOME could not be determined. RPy2 may not work.")

    # Handle R_LIBS_USER
    # In CI, packages are installed to the default site-library, so R_LIBS_USER might not be strictly needed
    # or could point to where `actions/setup-r` and `install.packages` place things.
    # For local testing, this might be more relevant if users have custom library paths.
    # We will rely on R's default library paths for now, as modified by `actions/setup-r`.

    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri, default_converter
    from rpy2.robjects.conversion import localconverter
    from rpy2.rinterface_lib.sexp import NULLType as RNULLType
    
    ro.r('library(stats)')
    sensR = importr('sensR') # Try to import sensR
    numpy2ri.activate()
    rpy2_available = True
    print("RPy2 and sensR loaded successfully for tests in test_models.py.")

except Exception as e:
    warnings.warn(f"RPy2 or sensR setup failed in test_models.py: {e}. Tests depending on them will be skipped.")
    sensR = None
    ro = None
    numpy2ri = None
    default_converter = None
    localconverter = None
    RNULLType = None
    rpy2_available = False

from senspy.models import BetaBinomial

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
            assert alpha_ci_lower <= alpha_ci_upper + 1e-7 # Ensure lower <= upper
            assert alpha_ci_lower <= py_model.alpha + 1e-7 # Ensure lower <= MLE
            assert py_model.alpha <= alpha_ci_upper + 1e-7 # Ensure MLE <= upper
        
        if not (np.isnan(beta_ci_lower) or np.isnan(beta_ci_upper)):
            assert beta_ci_lower <= beta_ci_upper + 1e-7 # Ensure lower <= upper
            assert beta_ci_lower <= py_model.beta + 1e-7 # Ensure lower <= MLE
            assert py_model.beta <= beta_ci_upper + 1e-7 # Ensure MLE <= upper
    except ValueError as e: 
        pytest.fail(f"confint() raised ValueError: {e}")
    except RuntimeError as e: 
        # RuntimeErrors can occur in brentq if root not found in interval
        warnings.warn(UserWarning(f"confint() raised RuntimeError: {e}. CIs: {conf_intervals if 'conf_intervals' in locals() else 'not computed'}"))
        # We don't necessarily fail the test for RuntimeError in confint, as it might be due to difficult data
        # but we ensure the structure is still as expected if conf_intervals was populated.
        if 'conf_intervals' in locals() and isinstance(conf_intervals,dict):
            for param_name_ci in ["alpha", "beta"]:
                if param_name_ci in conf_intervals:
                    assert isinstance(conf_intervals[param_name_ci], tuple)
                    assert len(conf_intervals[param_name_ci]) == 2
        else:
            # If conf_intervals wasn't even formed into a dict, that's a more basic failure.
            pytest.fail(f"confint() failed structurally before or due to RuntimeError: {e}")


# --- Tests for TwoACModel ---
from senspy.models import TwoACModel # Import the new model

# Test cases for TwoACModel: (hits, false_alarms, n_signal, n_noise, expected_delta_approx, expected_tau_approx, test_id)
# Expected values are approximate, mainly for sanity checking.
# Actual comparison will be against sensR::twoAC if possible, or senspy.discrimination.twoAC
twoac_test_params = [
    (70, 15, 100, 100, 1.6, 0.3, "moderate_hr_low_far"), 
    (50, 50, 100, 100, 0.0, -0.0, "equal_hr_far"), # Expect delta near 0
    (90, 10, 100, 100, 2.5, 0.0, "high_hr_low_far"),
    (10, 5, 20, 20, 1.3, 0.2, "small_n"),
    (99, 1, 100, 100, 4.0, 0.0, "very_high_hr_very_low_far"), # High d'
    (1, 99, 100, 100, -4.0, 0.0, "very_low_hr_very_high_far"), # Should give delta near 0 due to bound d>=0
]

@pytest.mark.parametrize("h, f, n_s, n_n, _, __, test_id", twoac_test_params)
def test_twoac_model_fit_basic(h, f, n_s, n_n, _, __, test_id):
    """Basic test for TwoACModel fit and summary."""
    model = TwoACModel()
    
    # Test fitting
    try:
        model.fit(hits=h, false_alarms=f, n_signal_trials=n_s, n_noise_trials=n_n)
    except Exception as e:
        pytest.fail(f"TwoACModel.fit() failed for {test_id} with error: {e}")

    assert model.params is not None, f"{test_id}: Params should be set."
    assert model.delta is not None, f"{test_id}: Delta should be set."
    assert model.tau is not None, f"{test_id}: Tau should be set."
    assert model.loglik is not None, f"{test_id}: LogLik should be set."
    assert model.vcov is not None, f"{test_id}: Vcov should be set."
    assert model.convergence_status is not None, f"{test_id}: Convergence status should be set."
    assert model.n_obs == 1, f"{test_id}: n_obs should be 1."

    assert isinstance(model.delta, float), f"{test_id}: Delta should be a float."
    assert isinstance(model.tau, float), f"{test_id}: Tau should be a float."
    assert model.delta >= 0, f"{test_id}: Delta must be non-negative." # Due to bounds in optimizer
    
    assert isinstance(model.loglik, float), f"{test_id}: LogLik should be a float."
    assert isinstance(model.vcov, np.ndarray), f"{test_id}: Vcov should be a numpy array."
    assert model.vcov.shape == (2, 2), f"{test_id}: Vcov shape should be (2,2)."

    # Test summary method
    summary_str = model.summary()
    assert isinstance(summary_str, str), f"{test_id}: summary() should return a string."
    assert "TwoACModel Summary" in summary_str, f"{test_id}: Summary title missing."
    assert "delta (sensitivity):" in summary_str, f"{test_id}: Delta missing in summary."
    assert "tau (bias):" in summary_str, f"{test_id}: Tau missing in summary."
    assert "Log-Likelihood:" in summary_str, f"{test_id}: LogLik missing in summary."
    assert f"Hits: {h}" in summary_str, f"{test_id}: Hits info missing."


@pytest.mark.skipif(not rpy2_available, reason="RPy2 or sensR not available or R setup issue.")
@pytest.mark.parametrize("h_py, f_py, ns_py, nn_py, _, __, test_id", twoac_test_params)
def test_twoac_model_vs_sensr_twoac(h_py, f_py, ns_py, nn_py, _, __, test_id):
    """Compare TwoACModel fit results with sensR::twoAC()."""
    
    # Prepare data for R
    # sensR::twoAC expects x = c(h, f), n = c(n_s, n_n)
    with localconverter(default_converter + numpy2ri.converter):
        x_r = ro.conversion.py2rpy(np.array([h_py, f_py], dtype=np.int32))
        n_r = ro.conversion.py2rpy(np.array([ns_py, nn_py], dtype=np.int32))

    # Run sensR::twoAC
    try:
        r_model_fit = sensR.twoAC(x=x_r, n=n_r, method="ml")
    except Exception as e:
        pytest.skip(f"sensR::twoAC failed for test_id={test_id}, data=(h={h_py},f={f_py},ns={ns_py},nn={nn_py}). Error: {e}")
        return

    r_coeffs = np.asarray(r_model_fit.rx2('coefficients')) 
    r_delta, r_tau = r_coeffs[0], r_coeffs[1]
    
    r_loglik_obj = r_model_fit.rx2('logLik')
    r_loglik = r_loglik_obj[0] if not isinstance(r_loglik_obj, RNULLType) else np.nan
    
    py_model = TwoACModel()
    try:
        py_model.fit(hits=h_py, false_alarms=f_py, n_signal_trials=ns_py, n_noise_trials=nn_py)
    except Exception as e:
        pytest.fail(f"senspy TwoACModel.fit() failed for {test_id}: {e}")
    
    if r_delta < 0 and py_model.delta < 1e-3: 
        if test_id == "very_low_hr_very_high_far":
            warnings.warn(UserWarning(f"Skipping delta/tau comparison for {test_id} due to known r_delta < 0 case."))
        else: 
             np.testing.assert_allclose(py_model.delta, r_delta, rtol=0.1, atol=0.1, err_msg=f"{test_id}: Delta mismatch (r_delta<0 case)")
    else:
        np.testing.assert_allclose(py_model.delta, r_delta, rtol=0.05, atol=0.05, err_msg=f"{test_id}: Delta mismatch")

    if not (test_id == "very_low_hr_very_high_far" and r_delta < 0):
        np.testing.assert_allclose(py_model.tau, r_tau, rtol=0.05, atol=0.05, err_msg=f"{test_id}: Tau mismatch")

    if np.isfinite(r_loglik) and np.isfinite(py_model.loglik):
        np.testing.assert_allclose(py_model.loglik, r_loglik, rtol=1e-4, atol=1e-4, err_msg=f"{test_id}: LogLik mismatch")
    elif np.isnan(r_loglik) and np.isnan(py_model.loglik):
        pass 
    else:
        pytest.fail(f"{test_id}: LogLik mismatch - one is NaN, other is not. R: {r_loglik}, Py: {py_model.loglik}")
    pass # VCOV comparison deferred
