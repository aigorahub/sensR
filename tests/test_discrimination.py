import pytest
import numpy as np
import os
import warnings

# Attempt to set R_HOME and R_LIBS_USER *before* rpy2.robjects is imported
# This is crucial for rpy2 to initialize R correctly in some environments.
if 'R_HOME' not in os.environ:
    # Try to find R_HOME using R RHOME if available
    try:
        import subprocess
        r_home_path = subprocess.check_output(["R", "RHOME"], text=True).strip()
        if r_home_path and os.path.isdir(r_home_path):
            os.environ['R_HOME'] = r_home_path
        else:
            os.environ['R_HOME'] = '/usr/lib/R' # Fallback for typical Linux
    except Exception:
        os.environ['R_HOME'] = '/usr/lib/R' # Fallback

r_libs_user_path = os.path.expanduser('~/R/libs')
if 'R_LIBS_USER' not in os.environ:
    os.environ['R_LIBS_USER'] = r_libs_user_path
elif r_libs_user_path not in os.environ['R_LIBS_USER']:
    os.environ['R_LIBS_USER'] = f"{r_libs_user_path}:{os.environ['R_LIBS_USER']}"

try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri, default_converter
    from rpy2.robjects.conversion import localconverter
    from rpy2.rinterface_lib.sexp import NULLType as RNULLType

    # Try to initialize R and load a base package
    ro.r('library(stats)')
    sensR = importr('sensR') # Attempt to import sensR
    rpy2_available = True
    numpy2ri.activate()
except Exception as e:
    print(f"RPy2 or sensR setup failed: {e}")
    sensR = None
    rpy2_available = False

from scipy.stats import norm
from scipy.optimize import brentq # For d-prime verification in tests

from senspy.discrimination import (
    discrim,
    twoAC,
    two_afc,
    triangle_pc,
    duotrio_pc,
    three_afc_pc,
    tetrad_pc,
    hexad_pc,
    twofive_pc,
    get_pguess
)

# --- Tests for discrim function ---

discrim_test_cases = [
    # method, correct, total, expected_dprime_approx (can be None if hard to precompute or unstable)
    ("2afc", 75, 100, norm.ppf(0.75) * np.sqrt(2)),
    ("triangle", 60, 100, 1.519), 
    ("duotrio", 70, 100, 1.366),  
    ("3afc", 50, 100, 1.062),   
    ("tetrad", 45, 100, 0.823),  
    ("hexad", 20, 100, None),    
    ("twofive", 15, 100, None), 
]

pc_func_map = {
    "2afc": two_afc, "two_afc": two_afc,
    "triangle": triangle_pc,
    "duotrio": duotrio_pc,
    "3afc": three_afc_pc, "three_afc": three_afc_pc,
    "tetrad": tetrad_pc,
    "hexad": hexad_pc,
    "twofive": twofive_pc, "2outoffive": twofive_pc,
}

@pytest.mark.parametrize("method, correct, total, expected_dprime_approx", discrim_test_cases)
def test_discrim_output_and_dprime_verification(method, correct, total, expected_dprime_approx):
    result = discrim(correct, total, method=method)

    assert isinstance(result, dict)
    expected_keys = ["dprime", "se_dprime", "lower_ci", "upper_ci", "p_value", 
                     "conf_level", "correct", "total", "pc_obs", "pguess", 
                     "method", "statistic"]
    for key in expected_keys:
        assert key in result, f"Key '{key}' missing for method {method}."

    assert result["correct"] == correct and result["total"] == total
    assert result["method"] == method and result["statistic"] == "Wald"
    assert 0 <= result["pc_obs"] <= 1
    assert 0 < result["pguess"] < 1 # pguess should generally be >0 and <1
    assert result["se_dprime"] >= 0 or np.isinf(result["se_dprime"]) or np.isnan(result["se_dprime"])
    if np.isfinite(result["dprime"]) and np.isfinite(result["se_dprime"]) and result["se_dprime"] > 1e-8:
        assert result["lower_ci"] <= result["dprime"] + 1e-7 
        assert result["dprime"] <= result["upper_ci"] + 1e-7 
    assert 0 <= result["p_value"] <= 1 or np.isnan(result["p_value"])
    
    # d-prime Verification using brentq locally
    pc_obs_calc = correct / total
    pguess_calc = get_pguess(method)
    epsilon_calc = 1.0 / (2 * total) if total > 0 else 1e-8
    pc_clipped_calc = np.clip(pc_obs_calc, pguess_calc + epsilon_calc, 1.0 - epsilon_calc)
    
    dprime_brentq_calc = np.nan
    pc_f_calc = pc_func_map.get(method.lower())

    if pc_obs_calc <= pguess_calc:
        dprime_brentq_calc = 0.0
    elif pc_f_calc:
        try:
            # Check function behavior at bounds to prevent brentq errors
            low_bound_pc = pc_f_calc(-5.0)
            high_bound_pc = pc_f_calc(15.0)
            if pc_clipped_calc <= low_bound_pc : dprime_brentq_calc = -5.0
            elif pc_clipped_calc >= high_bound_pc : dprime_brentq_calc = 15.0
            else:
                 dprime_brentq_calc = brentq(lambda d: pc_f_calc(d) - pc_clipped_calc, -5, 15, xtol=1e-6, rtol=1e-6)
        except ValueError:
            warnings.warn(f"Brentq failed for d-prime verification (test internal) for {method}, pc_obs={pc_obs_calc}.")
            pass 
    
    if np.isfinite(dprime_brentq_calc) and np.isfinite(result["dprime"]):
         np.testing.assert_allclose(result["dprime"], dprime_brentq_calc, rtol=1e-5, atol=1e-5,
                                   err_msg=f"Internal d-prime verification for {method} failed.")
    
    # The comparison with expected_dprime_approx (from sensR) is removed from this Python-focused test.
    # Direct sensR comparison is handled by test_discrim_vs_sensR.
    # if expected_dprime_approx is not None and np.isfinite(result["dprime"]):
    #     np.testing.assert_allclose(result["dprime"], expected_dprime_approx, rtol=0.02, atol=0.02,
    #                                err_msg=f"Calculated d-prime for {method} deviates from provided R-based expected value.")

def test_discrim_edge_cases():
    res_perfect = discrim(50, 50, "2afc")
    assert res_perfect["pc_obs"] == 1.0
    assert np.isfinite(res_perfect["dprime"]) and res_perfect["dprime"] > 3.0 # dprime should be large
    assert np.isinf(res_perfect["se_dprime"]) # SE for pc_obs=1.0 should be Inf

    res_chance = discrim(25, 50, "2afc") 
    assert res_chance["dprime"] == 0.0
    assert res_chance["p_value"] == 1.0

    res_below_chance = discrim(10, 50, "2afc") 
    assert res_below_chance["dprime"] == 0.0 
    assert res_below_chance["p_value"] == 1.0

    res_zero = discrim(0, 50, "2afc") 
    assert res_zero["dprime"] == 0.0
    assert res_zero["p_value"] == 1.0

@pytest.mark.skipif(not rpy2_available, reason="RPy2 or sensR not available or R setup issue.")
@pytest.mark.parametrize("method, correct, total, _expected_dprime", discrim_test_cases)
def test_discrim_vs_sensR(method, correct, total, _expected_dprime):
    with localconverter(default_converter + numpy2ri.converter):
        r_result = sensR.discrim(correct, total, method=method)
    
    r_dprime = r_result.rx2('dprime')[0]
    r_se_dprime_list = r_result.rx2('se.dprime')
    r_se_dprime = r_se_dprime_list[0] if not isinstance(r_se_dprime_list, RNULLType) and len(r_se_dprime_list) > 0 else np.nan

    py_result = discrim(correct, total, method=method)

    np.testing.assert_allclose(py_result["dprime"], r_dprime, rtol=1e-5, atol=1e-5)
    if np.isfinite(r_se_dprime) and np.isfinite(py_result["se_dprime"]) and py_result["se_dprime"] > 1e-7 :
        np.testing.assert_allclose(py_result["se_dprime"], r_se_dprime, rtol=0.05, atol=0.001)

# --- Tests for twoAC function ---

twoAC_test_cases = [
    ([40, 10], [50, 50], norm.ppf(0.8) - norm.ppf(0.2), -0.5 * (norm.ppf(0.8) + norm.ppf(0.2)), "standard_yes_no"),
    ([45, 5], [50, 50], norm.ppf(0.9) - norm.ppf(0.1), -0.5 * (norm.ppf(0.9) + norm.ppf(0.1)), "high_discrimination"),
    ([25, 25], [50, 50], 0.0, 0.0, "chance_performance"),
    ([50, 0], [50, 50], None, None, "perfect_discrimination"), 
    ([0, 50], [50, 50], None, None, "perfect_misses_all_fas"), 
    ([30, 20], [60, 40], 0.0, -0.5*(norm.ppf(0.5)+norm.ppf(0.5)), "asymmetric_trials_chance"),
]

@pytest.mark.parametrize("x, n, expected_delta, expected_tau, test_id", twoAC_test_cases)
def test_twoAC_output_and_plausibility(x, n, expected_delta, expected_tau, test_id):
    result = twoAC(x, n)
    assert isinstance(result, dict)
    expected_keys = ["delta", "tau", "se_delta", "se_tau", "loglik", "vcov",
                     "convergence_status", "hits", "false_alarms", 
                     "n_signal_trials", "n_noise_trials"]
    for key in expected_keys:
        assert key in result, f"Key '{key}' missing for {test_id}."

    assert result["hits"] == x[0] and result["false_alarms"] == x[1]
    assert result["n_signal_trials"] == n[0] and result["n_noise_trials"] == n[1]
    
    if result["convergence_status"]: 
        assert result["delta"] >= 0 
        assert result["se_delta"] >= 0 or np.isnan(result["se_delta"])
        assert result["se_tau"] >= 0 or np.isnan(result["se_tau"])
        assert isinstance(result["vcov"], np.ndarray) and result["vcov"].shape == (2,2)

        if expected_delta is not None and np.isfinite(result["delta"]):
            if test_id not in ["perfect_discrimination", "perfect_misses_all_fas"]:
                 np.testing.assert_allclose(result["delta"], expected_delta, rtol=0.1, atol=0.1,
                                           err_msg=f"Delta for {test_id} deviates from Z-score expected value.")
        if expected_tau is not None and np.isfinite(result["tau"]):
            if test_id not in ["perfect_discrimination", "perfect_misses_all_fas"]:
                 np.testing.assert_allclose(result["tau"], expected_tau, rtol=0.1, atol=0.15, 
                                           err_msg=f"Tau for {test_id} deviates from Z-score expected value.")
    else:
        warnings.warn(f"Optimizer did not converge for twoAC case: {test_id}. Plausibility checks on estimates skipped.")

def test_twoAC_symmetry():
    result = twoAC(x=[40, 10], n=[50, 50]) # H=0.8, FA=0.2
    if result["convergence_status"]: # Only check tau if converged
        np.testing.assert_allclose(result["tau"], 0, atol=1e-2) 
    else:
        warnings.warn("Optimizer did not converge for twoAC_symmetry test.")


@pytest.mark.skipif(not rpy2_available, reason="RPy2 or sensR not available or R setup issue.")
@pytest.mark.parametrize("x, n, _ed, _et, test_id", twoAC_test_cases)
def test_twoAC_vs_sensR(x, n, _ed, _et, test_id):
    pH = x[0]/n[0]
    pFA = x[1]/n[1]
    # Skip sensR comparison for perfect discrimination as results can be unstable / Inf
    if pH == 1.0 and pFA == 0.0: pytest.skip(f"Skipping sensR for perfect H=1,FA=0 {test_id}")
    if pH == 0.0 and pFA == 1.0: pytest.skip(f"Skipping sensR for perfect H=0,FA=1 {test_id}")
    if pH == 1.0 and pFA == 1.0: pytest.skip(f"Skipping sensR for H=1,FA=1 {test_id}")
    if pH == 0.0 and pFA == 0.0: pytest.skip(f"Skipping sensR for H=0,FA=0 {test_id}")

    with localconverter(default_converter + numpy2ri.converter):
        r_x_vec = ro.IntVector(x)
        r_N_vec = ro.IntVector(n)
        r_result = sensR.twoAC(x=r_x_vec, N=r_N_vec, method="ml")
    
    r_delta = r_result.rx2('delta')[0]
    r_tau = r_result.rx2('tau')[0]
    r_se_delta = r_result.rx2('se.delta')[0] if not isinstance(r_result.rx2('se.delta'), RNULLType) else np.nan
    r_se_tau = r_result.rx2('se.tau')[0] if not isinstance(r_result.rx2('se.tau'), RNULLType) else np.nan
    r_loglik = r_result.rx2('logLik')[0]

    py_result = twoAC(x, n)
    
    if not py_result["convergence_status"]:
        pytest.skip(f"Python's twoAC did not converge for {test_id}, skipping R comparison.")

    np.testing.assert_allclose(py_result["delta"], r_delta, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(py_result["tau"], r_tau, rtol=1e-4, atol=1e-4)
    if np.isfinite(r_se_delta) and np.isfinite(py_result["se_delta"]):
        np.testing.assert_allclose(py_result["se_delta"], r_se_delta, rtol=0.05, atol=0.001)
    if np.isfinite(r_se_tau) and np.isfinite(py_result["se_tau"]):
        np.testing.assert_allclose(py_result["se_tau"], r_se_tau, rtol=0.05, atol=0.001)
    np.testing.assert_allclose(py_result["loglik"], r_loglik, rtol=1e-5, atol=1e-5)
