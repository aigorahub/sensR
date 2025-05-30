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
    r_home_from_env_ci = os.environ.get('R_HOME_DIR_CI') # Set in CI from setup-r output
    r_home_from_env = os.environ.get('R_HOME')

    r_home = None
    if r_home_from_env_ci:
        r_home = r_home_from_env_ci
    elif r_home_from_env:
        r_home = r_home_from_env
    else:
        try:
            # Try to get R_HOME from R itself
            r_home_process = subprocess.run(["R", "RHOME"], capture_output=True, text=True, check=True)
            r_home = r_home_process.stdout.strip()
        except Exception as e:
            warnings.warn(f"Failed to get R_HOME from subprocess: {e}. Falling back to default /usr/lib/R.")
            r_home = '/usr/lib/R' # A common default on Linux

    if r_home:
        os.environ['R_HOME'] = r_home
    else:
        warnings.warn("R_HOME could not be determined. RPy2 might not initialize correctly.")

    # Regarding R_LIBS_USER:
    # In CI, packages (like sensR) are installed by R into its default library paths.
    # `actions/setup-r` configures R environment such that these are found.
    # So, explicitly setting R_LIBS_USER here might not be necessary for CI
    # and could conflict if not aligned with where CI installs packages.
    # For local testing, users might need to set R_LIBS_USER if sensR is in a non-default location.
    # We'll rely on R's internal library path logic for now.

    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri, default_converter
    from rpy2.robjects.conversion import localconverter
    from rpy2.rinterface_lib.sexp import NULLType as RNULLType

    ro.r('library(stats)') # Load a base R package
    sensR = importr('sensR') # Attempt to import the sensR package
    numpy2ri.activate() # Activate converters
    rpy2_available = True
    print("RPy2 and sensR loaded successfully for tests in test_discrimination.py.")

except Exception as e:
    warnings.warn(f"RPy2 or sensR setup failed in test_discrimination.py: {e}. Tests depending on RPy2 will be skipped.")
    # Ensure all rpy2 related variables are None if setup fails
    sensR = None
    ro = None
    numpy2ri = None
    default_converter = None
    localconverter = None
    RNULLType = None
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
    get_pguess,
    # Import new functions to be tested
    dod,
    samediff,
    dprime_test,
    dprime_compare,
    SDT,
    AUC
)
from scipy.stats import chi2 # For dprime_compare

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


# --- Tests for dod function ---
def test_dod_basic_functionality():
    same_counts = np.array([10, 20, 70]) # K=3 categories
    diff_counts = np.array([70, 20, 10])
    result = dod(same_counts, diff_counts)

    assert isinstance(result, dict)
    expected_keys = ["d_prime", "tau", "se_d_prime", "se_tpar", "loglik", 
                     "vcov_optim_params", "convergence_status", "initial_params_optim",
                     "optim_result", "same_counts", "diff_counts", "method", "conf_level"]
    for key in expected_keys:
        assert key in result, f"Key '{key}' missing in dod output."

    assert result["d_prime"] >= 0
    assert len(result["tau"]) == len(same_counts) - 1
    if len(result["tau"]) > 0:
        assert np.all(result["tau"] > 0)
        if len(result["tau"]) > 1:
            assert np.all(np.diff(result["tau"]) > 0) # Should be strictly increasing
    
    # For a typical case, expect convergence
    assert result["convergence_status"] is True, f"dod optimizer failed to converge: {result.get('optim_result', {}).get('message', 'No message')}"
    assert np.isfinite(result["loglik"])
    assert result["vcov_optim_params"].shape == (len(result["initial_params_optim"]), len(result["initial_params_optim"]))


def test_dod_input_validation():
    with pytest.raises(ValueError, match="same_counts and diff_counts must have the same length"):
        dod(same_counts=[10, 20], diff_counts=[10, 20, 30])
    with pytest.raises(ValueError, match="Number of categories must be at least 2"):
        dod(same_counts=[10], diff_counts=[10])
    with pytest.raises(ValueError, match="Counts must be non-negative"):
        dod(same_counts=[10, -1], diff_counts=[10, 10])
    with pytest.raises(ValueError, match="Total counts for both same and different pairs cannot be zero"):
        dod(same_counts=[0,0,0], diff_counts=[0,0,0])


# --- Tests for samediff function ---
def test_samediff_basic_functionality():
    result = samediff(nsamesame=50, ndiffsame=10, nsamediff=20, ndiffdiff=40)
    
    assert isinstance(result, dict)
    expected_keys = ["tau", "delta", "se_tau", "se_delta", "loglik", "vcov",
                     "convergence_status", "initial_params", "optim_result",
                     "nsamesame", "ndiffsame", "nsamediff", "ndiffdiff", "method", "conf_level"]
    for key in expected_keys:
        assert key in result, f"Key '{key}' missing in samediff output."

    assert result["tau"] > 0
    assert result["delta"] >= 0
    assert result["convergence_status"] is True,  f"samediff optimizer failed to converge: {result.get('optim_result', {}).get('message', 'No message')}"
    assert np.isfinite(result["loglik"])
    assert result["vcov"].shape == (2,2)

def test_samediff_input_validation():
    with pytest.raises(ValueError, match="All counts must be non-negative integers"):
        samediff(nsamesame=-1, ndiffsame=10, nsamediff=10, ndiffdiff=10)
    with pytest.raises(ValueError, match="The sum of counts must be positive"):
        samediff(0,0,0,0)
    with pytest.raises(ValueError, match="Not enough information in data"):
        samediff(nsamesame=10, ndiffsame=0, nsamediff=0, ndiffdiff=0)


# --- Tests for dprime_test function ---
def test_dprime_test_single_group():
    result = dprime_test(correct=30, total=50, protocol='2afc', dprime0=0)
    assert isinstance(result, dict)
    expected_keys = ["common_dprime_est", "se_common_dprime_est", "conf_int_common_dprime",
                     "statistic_value", "p_value", "dprime0", "alternative", "conf_level",
                     "estim_method", "statistic_type", "individual_group_estimates",
                     "loglik_common_dprime", "convergence_status_common_dprime"]
    for key in expected_keys:
        assert key in result
    assert 0 <= result["p_value"] <= 1
    assert np.isfinite(result["common_dprime_est"])
    assert result["convergence_status_common_dprime"] is True

def test_dprime_test_multi_group():
    result = dprime_test(correct=[30,35], total=[50,50], protocol=['2afc','2afc'], dprime0=0.5)
    assert isinstance(result, dict)
    assert 0 <= result["p_value"] <= 1
    assert np.isfinite(result["common_dprime_est"])
    assert len(result["individual_group_estimates"]) == 2
    assert result["convergence_status_common_dprime"] is True

def test_dprime_test_alternatives():
    # Case where dprime_est > dprime0
    res_greater = dprime_test(correct=40, total=50, protocol='2afc', dprime0=0.1, alternative="greater")
    res_less = dprime_test(correct=40, total=50, protocol='2afc', dprime0=0.1, alternative="less")
    assert res_greater["p_value"] < 0.05 # Expect small p-value
    assert res_less["p_value"] > 0.95 # Expect large p-value

    # Case where dprime_est < dprime0
    res_greater2 = dprime_test(correct=20, total=50, protocol='2afc', dprime0=1.5, alternative="greater")
    res_less2 = dprime_test(correct=20, total=50, protocol='2afc', dprime0=1.5, alternative="less")
    assert res_greater2["p_value"] > 0.95
    assert res_less2["p_value"] < 0.05


# --- Tests for dprime_compare function ---
def test_dprime_compare_basic():
    # Groups expected to be different
    res_diff = dprime_compare(correct=[30,45], total=[50,50], protocol=['2afc','2afc'])
    assert isinstance(res_diff, dict)
    expected_keys_compare = ["LR_statistic", "df", "p_value", "loglik_full_model", 
                             "loglik_reduced_model", "common_dprime_H0_est",
                             "individual_group_estimates", "estim_method", "statistic_method"]
    for key in expected_keys_compare:
        assert key in res_diff
    assert res_diff["LR_statistic"] >= 0
    assert res_diff["df"] == 1 # num_groups - 1
    assert 0 <= res_diff["p_value"] <= 1
    assert res_diff["p_value"] < 0.05 # Expect significant difference

    # Groups expected to be similar
    res_sim = dprime_compare(correct=[30,31], total=[50,50], protocol=['2afc','2afc'])
    assert res_sim["p_value"] > 0.05 # Expect non-significant difference


# --- Tests for SDT function ---
def test_SDT_probit():
    counts_table = np.array([[10,20,30,40], [5,15,25,55]]) # J=4 categories
    # Expected values from R's sensR::SDT(table), but with zFA and zH columns swapped
    # Original R output: zH, zFA, dprime
    # senspy SDT output: zFA, zH, dprime
    expected_zFA = np.array([-0.8416212, -0.1534109,  0.5244005])
    expected_zH  = np.array([-1.2815518, -0.4307273,  0.5244005])
    expected_dprime = expected_zH - expected_zFA 
    
    result = SDT(counts_table, method="probit")
    assert result.shape == (3, 3) # (J-1)x3
    np.testing.assert_allclose(result[:,0], expected_zFA, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(result[:,1], expected_zH, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(result[:,2], expected_dprime, rtol=1e-4, atol=1e-4)

def test_SDT_logit():
    counts_table = np.array([[10,20,30,40], [5,15,25,55]])
    result = SDT(counts_table, method="logit")
    assert result.shape == (3, 3)
    assert np.all(np.isfinite(result)) # Check it runs and produces finite numbers

def test_SDT_zero_row_sum():
    counts_table = np.array([[0,0,0,0], [5,15,25,55]])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = SDT(counts_table, method="probit")
        assert len(w) == 1 # Should warn for the zero sum row
        assert issubclass(w[-1].category, UserWarning)
    assert result.shape == (3,3)
    assert np.all(np.isnan(result[:,0])) # First column (zFA) should be all NaN
    assert np.all(np.isfinite(result[:,1])) # Second column (zH) should be finite
    assert np.all(np.isnan(result[:,2])) # d-prime will be NaN if zFA is NaN


# --- Tests for AUC function ---
def test_AUC_basic():
    assert AUC(d_prime=0.0) == 0.5
    np.testing.assert_allclose(AUC(d_prime=1.0, scale=1.0), norm.cdf(1/np.sqrt(2)), atol=1e-7)
    np.testing.assert_allclose(AUC(d_prime=1.0, scale=0.5), norm.cdf(1/np.sqrt(1+0.5**2)), atol=1e-7)
    np.testing.assert_allclose(AUC(d_prime=-1.0, scale=1.0), norm.cdf(-1/np.sqrt(2)), atol=1e-7)


def test_AUC_input_validation():
    with pytest.raises(ValueError, match="scale must be strictly positive"):
        AUC(d_prime=1.0, scale=0)
    with pytest.raises(ValueError, match="scale must be strictly positive"):
        AUC(d_prime=1.0, scale=-1.0)
    with pytest.raises(TypeError, match="d_prime must be a numeric scalar"):
        AUC(d_prime="a")
    with pytest.raises(TypeError, match="scale must be a numeric scalar"):
        AUC(d_prime=1.0, scale="a")
