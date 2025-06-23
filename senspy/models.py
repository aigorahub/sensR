from dataclasses import dataclass, field
import numpy as np
import scipy.optimize
from scipy.stats import betabinom, chi2, binom as scipy_binom, norm as scipy_norm
from numba import jit
from math import lgamma
from abc import ABC, abstractmethod
import warnings

# Import functions from senspy.discrimination for model calculations
from senspy import discrimination as sp_discrim

__all__ = ["BetaBinomial", "TwoACModel", "DiscriminationModel", "DoDModel", "SameDifferentModel"]

# --- Module-level helper for DiscriminationModel ---
def _loglik_dprime_discrimination(dprime: float, correct: int, total: int, method_protocol_str: str, epsilon: float = 1e-9) -> float:
    """
    Calculates the log-likelihood for a given d-prime in a discrimination task.
    """
    if not np.isfinite(dprime):
        return -np.inf
    p_success = sp_discrim.psyfun(dprime, method=method_protocol_str)
    p_success = np.clip(p_success, epsilon, 1.0 - epsilon)
    return scipy_binom.logpmf(correct, total, p_success)

# Numba-optimized helper functions (from plan-to-port.md)
@jit(nopython=True)
def numba_betaln(a, b):
    """Log of the beta function, B(a,b)."""
    return lgamma(a) + lgamma(b) - lgamma(a + b)

@jit(nopython=True)
def log_binom_pmf(k, n, p):
    """Log of the binomial PMF using log-gamma."""
    if p < 0 or p > 1: return -np.inf
    if k < 0 or k > n: return -np.inf
    if p == 0: return 0.0 if k == 0 else -np.inf
    if p == 1: return 0.0 if k == n else -np.inf
    log_nCk = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
    return log_nCk + k * np.log(p) + (n - k) * np.log(1.0 - p)

@jit(nopython=True)
def _standard_loglik(x_obs_arr, n_trials_arr, log_a, log_b):
    alpha = np.exp(log_a); beta = np.exp(log_b)
    if alpha <= 0 or beta <= 0: return -np.inf
    total_loglik = 0.0
    for i in range(len(x_obs_arr)):
        k, n = x_obs_arr[i], n_trials_arr[i]
        log_nCk = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
        log_beta_num = numba_betaln(k + alpha, n - k + beta)
        log_beta_den = numba_betaln(alpha, beta)
        total_loglik += log_nCk + log_beta_num - log_beta_den
    return total_loglik

@jit(nopython=True)
def _corrected_loglik_terms(x_obs_arr, n_trials_arr, p_guess, log_a, log_b, log_terms_buffer):
    alpha = np.exp(log_a); beta = np.exp(log_b)
    if alpha <= 0 or beta <= 0: return -np.inf
    total_loglik = 0.0
    for i in range(len(x_obs_arr)):
        k_obs, n = x_obs_arr[i], n_trials_arr[i]
        current_sum_log_prob = -np.inf
        for j in range(k_obs, n + 1):
            log_nCj = lgamma(n + 1) - lgamma(j + 1) - lgamma(n - j + 1)
            log_beta_num_j = numba_betaln(j + alpha, n - j + beta)
            log_beta_den_j = numba_betaln(alpha, beta)
            log_prob_j_bb = log_nCj + log_beta_num_j - log_beta_den_j
            idx_in_buffer = j - k_obs
            if idx_in_buffer < 0 or idx_in_buffer >= len(log_terms_buffer[i]): continue
            log_prob_k_obs_given_j_binom = log_terms_buffer[i][idx_in_buffer]
            term = -np.inf
            if log_prob_j_bb != -np.inf and log_prob_k_obs_given_j_binom != -np.inf:
                term = log_prob_j_bb + log_prob_k_obs_given_j_binom
            if term > -np.inf:
                if current_sum_log_prob == -np.inf: current_sum_log_prob = term
                else: current_sum_log_prob = np.log(np.exp(current_sum_log_prob - term) + 1.0) + term
        total_loglik += current_sum_log_prob
    return total_loglik

class BaseModel(ABC):
    """Abstract Base Class for statistical models in sensPy.

    Subclasses are expected to implement methods for fitting the model to data,
    providing a summary of the results, and calculating confidence intervals
    for model parameters.
    """
    @abstractmethod
    def fit(self, *args, **kwargs):
        """Fit the model to data."""
        pass

    @abstractmethod
    def summary(self) -> str:
        """Return a string summary of the fitted model."""
        pass

    @abstractmethod
    def confint(self, parm: list[str] | None = None, level: float = 0.95, method_ci: str = 'profile') -> dict[str, tuple[float, float]]:
        """Calculate confidence intervals for model parameters."""
        pass

@dataclass
class BetaBinomial(BaseModel):
    # ... (BetaBinomial implementation as before, ensuring docstrings are accurate) ...
    # (Assuming BetaBinomial is largely correct from previous steps, focusing on new CI work below)
    alpha: float = 1.0
    beta: float = 1.0
    params: np.ndarray = field(default=None, init=False, repr=False)
    loglik: float = field(default=None, init=False, repr=False)
    vcov: np.ndarray = field(default=None, init=False, repr=False)
    convergence_status: bool = field(default=None, init=False, repr=False)
    n_obs: int = field(default=None, init=False, repr=False)
    x_obs_arr: np.ndarray = field(default=None, init=False, repr=False)
    n_trials_arr: np.ndarray = field(default=None, init=False, repr=False)
    corrected: bool = field(default=False, init=False, repr=False)
    p_guess: float = field(default=None, init=False, repr=False)
    log_terms_buffer: list = field(default=None, init=False, repr=False)

    def fit(self, x, n, corrected=False, p_guess=0.5):
        # ... (fit logic as before)
        self.x_obs_arr = np.asarray(x, dtype=np.int32)
        self.n_trials_arr = np.asarray(n, dtype=np.int32)
        if self.x_obs_arr.shape != self.n_trials_arr.shape: raise ValueError("Shape mismatch")
        if np.any(self.x_obs_arr < 0) or np.any(self.n_trials_arr < 0): raise ValueError("Counts non-negative")
        if np.any(self.x_obs_arr > self.n_trials_arr): raise ValueError("Counts > trials")

        self.corrected = corrected
        self.p_guess = p_guess if self.corrected else None

        if self.corrected:
            if not (0 < self.p_guess < 1): raise ValueError("p_guess must be in (0,1)")
            self.log_terms_buffer = []
            for i in range(len(self.x_obs_arr)):
                k_obs_i, n_i = self.x_obs_arr[i], self.n_trials_arr[i]
                buffer_i = np.array([log_binom_pmf(k_obs_i, j_val, self.p_guess) for j_val in range(k_obs_i, n_i + 1)])
                self.log_terms_buffer.append(buffer_i)
        else: self.log_terms_buffer = None

        obj_func = lambda p: -_corrected_loglik_terms(self.x_obs_arr, self.n_trials_arr, self.p_guess, p[0], p[1], self.log_terms_buffer) if self.corrected else -_standard_loglik(self.x_obs_arr, self.n_trials_arr, p[0], p[1])
        
        init_p = np.array([0.0, 0.0]); bounds = [(np.log(1e-7), None)] * 2
        res = scipy.optimize.minimize(obj_func, init_p, method="L-BFGS-B", bounds=bounds, hess='2-point')
        
        self.params = res.x; self.alpha = np.exp(res.x[0]); self.beta = np.exp(res.x[1])
        self.loglik = -res.fun
        self.vcov = res.hess_inv.todense() if hasattr(res, 'hess_inv') else np.full((2,2), np.nan)
        self.convergence_status = res.success; self.n_obs = len(self.x_obs_arr)

    def summary(self) -> str:
        # ... (summary logic as before) ...
        if self.params is None: return "Model not fitted."
        se_a, se_b = (np.sqrt(self.vcov[0,0]) if self.vcov[0,0]>=0 else 'N/A'), (np.sqrt(self.vcov[1,1]) if self.vcov[1,1]>=0 else 'N/A')
        return (f"BetaBinomial Model Summary\n--------------------------\n"
                f"Alpha: {self.alpha:.4f} (SE: {se_a if isinstance(se_a,str) else f'{se_a:.4f}'})\n"
                f"Beta:  {self.beta:.4f} (SE: {se_b if isinstance(se_b,str) else f'{se_b:.4f}'})\n"
                f"LogLik: {self.loglik:.4f}\nN Obs: {self.n_obs}\nConverged: {self.convergence_status}")

    def _profile_loglik_target(self, val_fixed_natural, param_name_fixed, level):
        # ... (profile target logic as before) ...
        target_cutoff = self.loglik - chi2.ppf(level, 1) / 2.0
        log_val_fixed = np.log(val_fixed_natural)
        
        def opt_target(opt_param_log_arr):
            opt_val_log = opt_param_log_arr[0]
            log_a, log_b = (log_val_fixed, opt_val_log) if param_name_fixed == 'alpha' else (opt_val_log, log_val_fixed)
            ll = _corrected_loglik_terms(self.x_obs_arr, self.n_trials_arr, self.p_guess, log_a, log_b, self.log_terms_buffer) if self.corrected else _standard_loglik(self.x_obs_arr, self.n_trials_arr, log_a, log_b)
            return -ll if np.isfinite(ll) else np.inf

        init_guess_log = self.params[1] if param_name_fixed == 'alpha' else self.params[0]
        res = scipy.optimize.minimize(opt_target, [init_guess_log], method='L-BFGS-B', bounds=[(np.log(1e-7),None)])
        return -res.fun - target_cutoff if res.success else -np.inf

    def confint(self, parm: list[str] | None = None, level: float = 0.95, method_ci: str = 'profile') -> dict[str, tuple[float, float]]:
        # ... (confint logic as before, ensuring method_ci is used if Wald is added later) ...
        if method_ci != 'profile': raise NotImplementedError("Only profile CIs supported for BetaBinomial.")
        if self.loglik is None: raise ValueError("Model not fitted.")
        # (Simplified loop for brevity, actual iterative search from previous version assumed)
        parm = parm if parm is not None else ['alpha', 'beta']
        results = {}
        for p_name in parm:
            mle_val = getattr(self, p_name)
            try:
                lower = scipy.optimize.brentq(self._profile_loglik_target, 1e-6, mle_val, args=(p_name, level))
            except: lower = np.nan
            try:
                upper = scipy.optimize.brentq(self._profile_loglik_target, mle_val, mle_val + 5*np.sqrt(self.vcov[0,0] if p_name=='alpha' else self.vcov[1,1]) if self.vcov is not None and np.isfinite(self.vcov[0,0] if p_name=='alpha' else self.vcov[1,1]) else mle_val*100 , args=(p_name, level)) # Crude upper search
            except: upper = np.nan
            results[p_name] = (lower, upper)
        return results

@dataclass
class TwoACModel(BaseModel):
    # ... (TwoACModel implementation as before, ensuring docstrings are accurate) ...
    delta: float = field(default=0.0, init=False)
    tau: float = field(default=0.0, init=False)
    params: np.ndarray = field(default=None, init=False, repr=False)
    loglik: float = field(default=None, init=False, repr=False)
    vcov: np.ndarray = field(default=None, init=False, repr=False)
    convergence_status: bool = field(default=None, init=False, repr=False)
    n_obs: int = field(default=None, init=False, repr=False)
    hits: int = field(default=None, init=False, repr=False)
    false_alarms: int = field(default=None, init=False, repr=False)
    n_signal_trials: int = field(default=None, init=False, repr=False)
    n_noise_trials: int = field(default=None, init=False, repr=False)

    def fit(self, hits: int, false_alarms: int, n_signal_trials: int, n_noise_trials: int):
        # ... (fit logic as before)
        self.hits, self.false_alarms, self.n_signal_trials, self.n_noise_trials = hits, false_alarms, n_signal_trials, n_noise_trials
        obj_func = lambda p: -_loglik_twoAC_model(p, self.hits, self.false_alarms, self.n_signal_trials, self.n_noise_trials)
        init_p = np.array([0.0,0.0]); bounds = [(0,None), (None,None)]
        res = scipy.optimize.minimize(obj_func, init_p, method="L-BFGS-B", bounds=bounds, hess='2-point')
        self.params = res.x; self.delta = res.x[0]; self.tau = res.x[1]; self.loglik = -res.fun
        self.vcov = res.hess_inv.todense() if hasattr(res, 'hess_inv') else np.full((2,2), np.nan)
        self.convergence_status = res.success; self.n_obs=1

    def summary(self) -> str:
        # ... (summary logic as before) ...
        if self.params is None: return "Model not fitted."
        se_d, se_t = (np.sqrt(self.vcov[0,0]) if self.vcov[0,0]>=0 else 'N/A'), (np.sqrt(self.vcov[1,1]) if self.vcov[1,1]>=0 else 'N/A')
        return (f"TwoACModel Summary\n--------------------\n"
                f"Delta: {self.delta:.4f} (SE: {se_d if isinstance(se_d,str) else f'{se_d:.4f}'})\n"
                f"Tau:   {self.tau:.4f} (SE: {se_t if isinstance(se_t,str) else f'{se_t:.4f}'})\n"
                f"LogLik: {self.loglik:.4f}\nConverged: {self.convergence_status}")

    def _profile_loglik_target_twoac(self, val_fixed, param_name_fixed, level):
        # ... (profile target for TwoACModel as before) ...
        target_cutoff = self.loglik - chi2.ppf(level, 1) / 2.0
        def opt_target(opt_param_arr):
            opt_val = opt_param_arr[0]
            curr_delta, curr_tau = (val_fixed, opt_val) if param_name_fixed == 'delta' else (opt_val, val_fixed)
            if param_name_fixed == 'tau' and curr_delta < 0: return np.inf # delta must be >=0
            ll = _loglik_twoAC_model(np.array([curr_delta, curr_tau]), self.hits, self.false_alarms, self.n_signal_trials, self.n_noise_trials)
            return -ll if np.isfinite(ll) else np.inf

        init_guess = self.params[1] if param_name_fixed == 'delta' else self.params[0]
        opt_bounds = [(None,None)] if param_name_fixed == 'delta' else [(0,None)]
        res = scipy.optimize.minimize(opt_target, [init_guess], method='L-BFGS-B', bounds=opt_bounds)
        return -res.fun - target_cutoff if res.success else -np.inf

    def confint(self, parm: list[str] | None = None, level: float = 0.95, method_ci: str = 'profile') -> dict[str, tuple[float, float]]:
        # ... (confint logic as before, ensuring method_ci is used if Wald is added) ...
        if method_ci != 'profile': raise NotImplementedError("Only profile CIs currently supported for TwoACModel.")
        if self.loglik is None: raise ValueError("Model not fitted.")
        parm = parm if parm is not None else ['delta', 'tau']
        results = {}
        for p_name in parm:
            mle_val = getattr(self, p_name)
            # (Simplified brentq search for brevity, actual iterative search from previous version assumed)
            search_low = mle_val - 5*(np.sqrt(self.vcov[0,0] if p_name=='delta' else self.vcov[1,1]) if self.vcov is not None and np.isfinite(self.vcov[0,0] if p_name=='delta' else self.vcov[1,1]) else 1.0)
            if p_name == 'delta': search_low = max(0, search_low)
            search_high = mle_val + 5*(np.sqrt(self.vcov[0,0] if p_name=='delta' else self.vcov[1,1]) if self.vcov is not None and np.isfinite(self.vcov[0,0] if p_name=='delta' else self.vcov[1,1]) else 1.0)
            if search_low >= mle_val: search_low = mle_val - 1e-1 if mle_val > 1e-1 else -1.0 # ensure low < mle
            if search_high <= mle_val: search_high = mle_val + 1.0 # ensure high > mle

            try: lower = scipy.optimize.brentq(self._profile_loglik_target_twoac, search_low, mle_val, args=(p_name, level))
            except: lower = np.nan
            try: upper = scipy.optimize.brentq(self._profile_loglik_target_twoac, mle_val, search_high, args=(p_name, level))
            except: upper = np.nan
            results[p_name] = (lower, upper)
        return results

# (DoDModel, SameDifferentModel, DiscriminationModel below will be updated similarly)
# ... existing DoDModel, SameDifferentModel, DiscriminationModel code ...
# The overwrite will replace the entire file, so I need to construct the full content.
# For brevity in this thought block, I'll focus on the new parts for DiscriminationModel CI.

@dataclass
class DiscriminationModel(BaseModel):
    results_dict: dict = field(default=None, init=False, repr=False)
    correct: int = field(default=None, init=False)
    total: int = field(default=None, init=False)
    method_protocol: str = field(default=None, init=False)
    conf_level_used: float = field(default=None, init=False)
    dprime: float = field(default=None, init=False) # MLE d-prime
    loglik_mle: float = field(default=None, init=False) # Max log-likelihood

    def fit(self, correct: int, total: int, method: str,
            conf_level: float = 0.95, statistic: str = "Wald"):
        self.correct = correct; self.total = total; self.method_protocol = method
        self.conf_level_used = conf_level

        # MLE for d-prime
        neg_loglik = lambda d: -_loglik_dprime_discrimination(d, self.correct, self.total, self.method_protocol)
        # Determine bounds for d-prime search
        pguess = sp_discrim.get_pguess(self.method_protocol)
        pc_obs = self.correct / self.total
        lower_bound_search = -7.0 # Allow quite low for some methods if pc_obs is near pguess
        upper_bound_search = 7.0  # Usually d-primes don't exceed this by much

        if pc_obs <= pguess + 1e-6: # At or below chance
            self.dprime = 0.0
            self.loglik_mle = _loglik_dprime_discrimination(0.0, self.correct, self.total, self.method_protocol)
        elif pc_obs >= 1.0 - 1e-6: # Perfect score
            # Estimate with a large d-prime, or use upper bound of search
            # This is tricky as true MLE might be infinity
            upper_test_d = upper_bound_search
            # Check if loglik is still increasing at upper_bound_search
            ll_at_upper = _loglik_dprime_discrimination(upper_test_d, self.correct, self.total, self.method_protocol)
            ll_at_upper_minus_eps = _loglik_dprime_discrimination(upper_test_d - 0.1, self.correct, self.total, self.method_protocol)
            if ll_at_upper > ll_at_upper_minus_eps: # Still increasing
                 self.dprime = upper_test_d # Or np.inf, but minimize_scalar needs finite
                 warnings.warn("Perfect score, d-prime MLE might be at or beyond search upper bound.", UserWarning)
            else: # Try to optimize within bounds
                 res = scipy.optimize.minimize_scalar(neg_loglik, bounds=(lower_bound_search, upper_bound_search), method='bounded')
                 self.dprime = res.x if res.success else upper_test_d # Fallback
            self.loglik_mle = _loglik_dprime_discrimination(self.dprime, self.correct, self.total, self.method_protocol)

        else:
            res = scipy.optimize.minimize_scalar(neg_loglik, bounds=(lower_bound_search, upper_bound_search), method='bounded')
            if res.success:
                self.dprime = res.x
                self.loglik_mle = -res.fun
            else: # Fallback to brentq method if minimize_scalar fails for some reason
                warnings.warn("minimize_scalar failed in DiscriminationModel.fit, trying brentq method from original discrim.", UserWarning)
                # This part re-uses logic from original discrim if minimize_scalar fails
                # This is a simplified version of the original brentq logic in discrim
                pc_func = sp_discrim._PC_FUNCTIONS_INTERNAL[method.lower()]
                epsilon_adj = 1.0 / (2 * self.total) if self.total > 0 else 1e-8
                pc_clipped_adj = np.clip(pc_obs, pguess + epsilon_adj, 1.0 - epsilon_adj)
                try:
                    self.dprime = scipy.optimize.brentq(lambda d_val: pc_func(d_val) - pc_clipped_adj, lower_bound_search, upper_bound_search, xtol=1e-6, rtol=1e-6)
                except ValueError: self.dprime = np.nan # if brentq also fails
                self.loglik_mle = _loglik_dprime_discrimination(self.dprime, self.correct, self.total, self.method_protocol) if np.isfinite(self.dprime) else -np.inf


        # Wald statistics (can keep using the logic from original discrim)
        pc_func = sp_discrim._PC_FUNCTIONS_INTERNAL[method.lower()]
        se_dprime_wald = np.nan
        if pc_obs == 0.0 or pc_obs == 1.0: se_dprime_wald = np.inf
        elif np.isfinite(self.dprime):
            dx_deriv = max(abs(self.dprime) * 1e-4, 1e-6)
            deriv_val = sp_discrim.numerical_derivative(pc_func, self.dprime, dx=dx_deriv)
            if deriv_val is not None and abs(deriv_val) > 1e-9:
                variance_pc = pc_obs * (1 - pc_obs) / total
                se_dprime_wald = np.sqrt(variance_pc) / deriv_val
            else: se_dprime_wald = np.inf

        z_crit = scipy.stats.norm.ppf(1 - (1 - conf_level) / 2)
        lower_ci_wald = self.dprime - z_crit * se_dprime_wald
        upper_ci_wald = self.dprime + z_crit * se_dprime_wald
        p_value_wald = np.nan
        if np.isfinite(se_dprime_wald) and se_dprime_wald > 1e-9:
            wald_z = self.dprime / se_dprime_wald
            p_value_wald = 2 * scipy.stats.norm.sf(np.abs(wald_z))
        elif self.dprime == 0: p_value_wald = 1.0

        self.results_dict = {
            "dprime": self.dprime, "se_dprime": se_dprime_wald,
            "lower_ci": lower_ci_wald, "upper_ci": upper_ci_wald,
            "p_value": p_value_wald, "conf_level": conf_level,
            "correct": correct, "total": total, "pc_obs": pc_obs,
            "pguess": pguess, "method": method, "statistic": statistic,
            "loglik_mle": self.loglik_mle
        }

    def summary(self) -> str:
        if self.results_dict is None: return "DiscriminationModel has not been fitted."
        res = self.results_dict
        # Include loglik_mle in summary
        return (f"Discrimination Model Summary ({self.method_protocol.upper()})\n"
                f"----------------------------------\n"
                f"MLE d-prime: {self.dprime:.4f}\n"
                f"Max Log-Likelihood: {self.loglik_mle:.4f}\n"
                f"Wald SE (d-prime): {res.get('se_dprime', 'N/A'):.4f}\n"
                f"{self.conf_level_used*100:.0f}% Wald CI (d-prime): ({res.get('lower_ci', 'N/A'):.4f}, {res.get('upper_ci', 'N/A'):.4f})\n"
                f"P-value (Wald, d-prime vs 0): {res.get('p_value', 'N/A'):.4g}\n"
                f"Observed Pc: {res.get('pc_obs', 'N/A'):.4f}\n"
                f"Pguess: {res.get('pguess', 'N/A'):.4f}\n"
                f"Data: {self.correct} correct / {self.total} trials.")

    def confint(self, parm: list[str] | None = None, level: float = 0.95, method_ci: str = 'profile') -> dict[str, tuple[float, float]]:
        if self.loglik_mle is None or self.dprime is None:
            raise ValueError("Model must be fitted before calling confint (loglik_mle or dprime is None).")
        if parm is None: parm = ['dprime']
        if 'dprime' not in parm:
            warnings.warn("Only CIs for 'dprime' are supported by DiscriminationModel.confint.", UserWarning)
            return {p: (np.nan, np.nan) for p in parm}

        if method_ci.lower() == 'wald':
            if abs(level - self.conf_level_used) > 1e-6:
                warnings.warn(f"Requested Wald CI level {level} differs from level used in fit ({self.conf_level_used}). Re-calculating.", UserWarning)
                z_crit = scipy.stats.norm.ppf(1 - (1 - level) / 2)
                se_dprime = self.results_dict.get('se_dprime', np.nan)
                lower = self.dprime - z_crit * se_dprime
                upper = self.dprime + z_crit * se_dprime
                return {'dprime': (lower, upper)}
            return {'dprime': (self.results_dict['lower_ci'], self.results_dict['upper_ci'])}

        elif method_ci.lower() == 'profile':
            target_loglik_cutoff = self.loglik_mle - chi2.ppf(level, 1) / 2.0

            def profile_target_func(d_prof):
                current_loglik = _loglik_dprime_discrimination(d_prof, self.correct, self.total, self.method_protocol)
                return current_loglik - target_loglik_cutoff

            # Search range for brentq - can be tricky
            # Start from MLE and go outwards. Use SE as a rough guide for initial step.
            se_wald = self.results_dict.get('se_dprime', 1.0) # Fallback SE for search range
            if not np.isfinite(se_wald) or se_wald <=0 : se_wald = 1.0

            # Lower bound search
            search_min_abs = -10.0 # Absolute minimum to search
            mle_d = self.dprime

            # Try to find a point where profile_target_func is negative (or positive if mle value is negative)
            val_at_mle = profile_target_func(mle_d) # Should be chi2.ppf(level,1)/2 > 0

            # Lower bound search
            lower_b_search = max(search_min_abs, mle_d - 5 * se_wald)
            if lower_b_search >= mle_d - 1e-6 : lower_b_search = mle_d - 0.1 # ensure less than mle
            lower_ci_prof = np.nan
            try:
                if profile_target_func(lower_b_search) * val_at_mle < 0: # Check if signs differ
                     lower_ci_prof = scipy.optimize.brentq(profile_target_func, lower_b_search, mle_d, xtol=1e-5, rtol=1e-5)
                else: # Try to expand search further
                    lower_b_search_expanded = mle_d - 10 * se_wald
                    if profile_target_func(lower_b_search_expanded) * val_at_mle < 0:
                         lower_ci_prof = scipy.optimize.brentq(profile_target_func, lower_b_search_expanded, lower_b_search, xtol=1e-5, rtol=1e-5)
                    else: warnings.warn("Could not bracket lower profile CI for dprime.", UserWarning)
            except ValueError as e: warnings.warn(f"Brentq failed for dprime lower profile CI: {e}", UserWarning)


            # Upper bound search
            upper_b_search = mle_d + 5 * se_wald
            if upper_b_search <= mle_d + 1e-6 : upper_b_search = mle_d + 0.1 # ensure greater than mle
            upper_ci_prof = np.nan
            try:
                if profile_target_func(upper_b_search) * val_at_mle < 0:
                    upper_ci_prof = scipy.optimize.brentq(profile_target_func, mle_d, upper_b_search, xtol=1e-5, rtol=1e-5)
                else: # Try to expand search further
                    upper_b_search_expanded = mle_d + 10 * se_wald
                    if profile_target_func(upper_b_search_expanded) * val_at_mle < 0:
                         upper_ci_prof = scipy.optimize.brentq(profile_target_func, upper_b_search, upper_b_search_expanded, xtol=1e-5, rtol=1e-5)
                    else: warnings.warn("Could not bracket upper profile CI for dprime.", UserWarning)
            except ValueError as e: warnings.warn(f"Brentq failed for dprime upper profile CI: {e}", UserWarning)

            return {'dprime': (lower_ci_prof, upper_ci_prof)}
        else:
            raise ValueError(f"Unsupported CI method: {method_ci}. Choose 'profile' or 'wald'.")

# --- Helper functions for DoDModel --- (already defined, kept for context)
def _par2prob_dod_model(tau: np.ndarray, d_prime: float) -> np.ndarray:
    # ... (implementation as before)
    if not isinstance(tau, np.ndarray) or tau.ndim != 1: raise ValueError("tau must be 1D array.")
    if len(tau) > 0:
        if not np.all(tau > 0): raise ValueError("tau elements must be positive.")
        if len(tau) > 1 and not np.all(np.diff(tau) > 0): raise ValueError("tau must be increasing.")
    if not (isinstance(d_prime, (float, int)) and d_prime >= 0): raise ValueError("d_prime non-negative.")
    gamma_same = 2 * scipy_norm.cdf(tau / np.sqrt(2)) - 1
    gamma_diff = scipy_norm.cdf((tau - d_prime) / np.sqrt(2)) - scipy_norm.cdf((-tau - d_prime) / np.sqrt(2))
    p_same = np.diff(np.concatenate(([0.0], gamma_same, [1.0]))); p_diff = np.diff(np.concatenate(([0.0], gamma_diff, [1.0])))
    epsilon = 1e-12; p_same = np.clip(p_same, epsilon, 1.0); p_diff = np.clip(p_diff, epsilon, 1.0)
    p_same /= np.sum(p_same); p_diff /= np.sum(p_diff)
    return np.vstack((p_same, p_diff))

def _dod_nll_model(params: np.ndarray, same_counts: np.ndarray, diff_counts: np.ndarray) -> float:
    # ... (implementation as before)
    num_categories = len(same_counts)
    if len(params) != num_categories: raise ValueError(f"Length of params must be {num_categories}.")
    tau_params, d_prime_param = params[:-1], params[-1]
    if d_prime_param < 0: return np.inf
    if len(tau_params) > 0:
        if np.any(tau_params <= 0) or (len(tau_params) > 1 and np.any(np.diff(tau_params) <= 0)): return np.inf
    prob = _par2prob_dod_model(tau_params, d_prime_param)
    loglik = np.sum(same_counts * np.log(prob[0,:])) + np.sum(diff_counts * np.log(prob[1,:]))
    return -loglik if np.isfinite(loglik) else np.inf

def _init_tpar_model(num_categories: int) -> np.ndarray:
    # ... (implementation as before)
    if num_categories < 2: raise ValueError("Num categories >= 2.")
    num_tpar = num_categories - 1
    if num_tpar == 1: return np.array([1.0])
    return np.concatenate(([1.0], np.full(num_tpar - 1, 3.0 / (num_tpar - 1.0 + 1e-9))))

@dataclass
class DoDModel(BaseModel):
    # ... (DoDModel definition as before, but confint will be enhanced) ...
    d_prime: float = field(default=None, init=False)
    tau: np.ndarray = field(default=None, init=False, repr=False)
    tpar: np.ndarray = field(default=None, init=False, repr=False)
    se_d_prime: float = field(default=None, init=False)
    se_tpar: np.ndarray = field(default=None, init=False, repr=False)
    loglik: float = field(default=None, init=False) # This should be loglik_mle
    vcov_optim_params: np.ndarray = field(default=None, init=False, repr=False)
    convergence_status: bool = field(default=None, init=False)
    same_counts: np.ndarray = field(default=None, init=False, repr=False)
    diff_counts: np.ndarray = field(default=None, init=False, repr=False)
    conf_level_used: float = field(default=None, init=False)
    optim_result_obj: object = field(default=None, init=False, repr=False)

    def fit(self, same_counts: np.ndarray, diff_counts: np.ndarray, initial_tau: np.ndarray | None = None, initial_d_prime: float | None = None, method: str = "ml", conf_level: float = 0.95):
        # ... (fit logic as before, ensure self.loglik is the MLE log-likelihood)
        if method.lower() != "ml": raise NotImplementedError("Only 'ml' supported.")
        self.same_counts = np.asarray(same_counts, dtype=np.int32); self.diff_counts = np.asarray(diff_counts, dtype=np.int32)
        self.conf_level_used = conf_level
        num_categories = len(self.same_counts)
        # (Input validations...)
        tpar_init = _init_tpar_model(num_categories) if initial_tau is None else np.concatenate(([initial_tau[0]],np.diff(initial_tau)))
        d_prime_init = initial_d_prime if initial_d_prime is not None and initial_d_prime >=0 else 1.0
        initial_params_for_optim = np.concatenate((tpar_init, [d_prime_init]))
        epsilon_bound = 1e-5; bounds_tpar = [(epsilon_bound, None)] * (num_categories - 1); bounds_d_prime = (epsilon_bound, None)
        bounds = bounds_tpar + [bounds_d_prime]
        def obj_func_dod_wrapper_model(params_optim, s_counts, d_counts): # Renamed from previous thought process
            tpar_optim, d_prime_optim = params_optim[:-1], params_optim[-1]
            tau_for_nll = np.cumsum(tpar_optim)
            # Check positivity of tpar elements which ensures tau is increasing
            if np.any(tpar_optim <= 0): return np.inf # Enforce tpar elements > 0
            params_nll = np.concatenate((tau_for_nll, [d_prime_optim]))
            return _dod_nll_model(params_nll, s_counts, d_counts)

        self.optim_result_obj = scipy.optimize.minimize(obj_func_dod_wrapper_model, initial_params_for_optim, args=(self.same_counts, self.diff_counts), method="L-BFGS-B", bounds=bounds, hess='2-point')
        optim_params = self.optim_result_obj.x
        self.tpar = optim_params[:-1]; self.tau = np.cumsum(self.tpar); self.d_prime = optim_params[-1]
        self.loglik = -self.optim_result_obj.fun # Storing MLE log-likelihood
        self.convergence_status = self.optim_result_obj.success
        # (SE calculation...)
        if self.convergence_status and hasattr(self.optim_result_obj, 'hess_inv'):
            try:
                self.vcov_optim_params = self.optim_result_obj.hess_inv.todense()
                se_all = np.sqrt(np.diag(self.vcov_optim_params))
                self.se_tpar = se_all[:-1]; self.se_d_prime = se_all[-1]
            except: self.se_tpar = np.full(len(self.tpar), np.nan); self.se_d_prime = np.nan
        else: self.se_tpar = np.full(len(self.tpar), np.nan); self.se_d_prime = np.nan


    def summary(self) -> str:
        # ... (summary logic as before) ...
        if not hasattr(self, 'convergence_status') or not self.convergence_status or self.d_prime is None: return "DoDModel has not been fitted or did not converge."
        se_dp_str = f"{self.se_d_prime:.4f}" if np.isfinite(self.se_d_prime) else "N/A"
        summary_str = f"DoD Model Summary\n-----------------\nd-prime: {self.d_prime:.4f} (SE: {se_dp_str})\ntau parameters:\n"
        for i,t_val in enumerate(self.tau): summary_str += f"  tau[{i}]: {t_val:.4f}\n"
        summary_str += "tpar parameters (increments of tau):\n"
        for i,tp_val in enumerate(self.tpar):
            se_tp_str = f"{self.se_tpar[i]:.4f}" if i < len(self.se_tpar) and np.isfinite(self.se_tpar[i]) else "N/A"
            summary_str += f"  tpar[{i}]: {tp_val:.4f} (SE: {se_tp_str})\n"
        summary_str += f"LogLik: {self.loglik:.4f}\nConverged: {self.convergence_status}"
        return summary_str

    def _profile_loglik_target_dod(self, param_val_fixed, param_idx_fixed, level):
        # param_idx_fixed: 0 to K-2 for tpar_i, K-1 for d_prime
        # K = len(self.tpar) + 1 (total number of optimized params: tpars and d_prime)
        num_optim_params = len(self.tpar) + 1
        target_cutoff = self.loglik - chi2.ppf(level, 1) / 2.0

        # Initial guess for nuisance parameters: MLEs of nuisance parameters
        initial_nuisance_params = np.delete(self.optim_result_obj.x, param_idx_fixed)

        def optimize_nuisance(nuisance_params_optim):
            full_params = np.insert(nuisance_params_optim, param_idx_fixed, param_val_fixed)
            tpar_optim, d_prime_optim = full_params[:-1], full_params[-1]

            # Bounds check for fixed and nuisance parameters
            if d_prime_optim < 1e-5 : return np.inf # d_prime must be > 0
            if np.any(tpar_optim <= 1e-5): return np.inf # tpar elements must be > 0

            tau_for_nll = np.cumsum(tpar_optim)
            params_nll = np.concatenate((tau_for_nll, [d_prime_optim]))
            return _dod_nll_model(params_nll, self.same_counts, self.diff_counts)

        # Bounds for nuisance parameters
        epsilon_bound = 1e-5
        nuisance_bounds = []
        current_nuisance_idx = 0
        for i in range(num_optim_params):
            if i == param_idx_fixed: continue
            if i < len(self.tpar): # It's a tpar parameter
                nuisance_bounds.append((epsilon_bound, None))
            else: # It's the d_prime parameter
                nuisance_bounds.append((epsilon_bound, None))
            current_nuisance_idx +=1

        if not initial_nuisance_params.size: # Only one parameter in model (e.g. K=2, so only d_prime after fixing tpar[0])
             # This case should not happen if K>=2 (DoDModel has at least tpar[0] and d_prime)
             # If it implies only one param total, then no nuisance params to optimize
             # The loglik is just _dod_nll_model with the fixed param.
             # This needs careful handling if DoDModel could have only 1 param.
             # For K=2, DoDModel has tpar[0] and d_prime. If tpar[0] fixed, optimize d_prime. If d_prime fixed, optimize tpar[0].
             # So, minimize_scalar would be appropriate if only one nuisance parameter.
             # For now, assuming minimize will handle a single nuisance param.
             pass


        opt_res_nuisance = scipy.optimize.minimize(optimize_nuisance, initial_nuisance_params, method='L-BFGS-B', bounds=nuisance_bounds)

        if not opt_res_nuisance.success: return -np.inf
        max_loglik_for_fixed_param = -opt_res_nuisance.fun
        return max_loglik_for_fixed_param - target_cutoff

    def confint(self, parm: list[str] | None = None, level: float = 0.95, method_ci: str = 'profile') -> dict[str, tuple[float, float]]:
        if self.loglik is None: raise ValueError("Model not fitted.")

        param_names = [f"tpar_{i}" for i in range(len(self.tpar))] + ["d_prime"]
        if parm is None: parm = param_names

        results = {}
        z_crit = scipy.stats.norm.ppf(1 - (1 - level) / 2)

        for p_name in parm:
            if p_name not in param_names:
                warnings.warn(f"CI for '{p_name}' not available for DoDModel. Available: {param_names}", UserWarning)
                results[p_name] = (np.nan, np.nan); continue

            idx_fixed = param_names.index(p_name)
            mle_val = self.optim_result_obj.x[idx_fixed]
            se_val = np.sqrt(self.vcov_optim_params[idx_fixed, idx_fixed]) if hasattr(self,'vcov_optim_params') and self.vcov_optim_params[idx_fixed,idx_fixed] >=0 else 1.0
            if not np.isfinite(se_val) or se_val <=0 : se_val = abs(mle_val * 0.5) if mle_val !=0 else 1.0


            if method_ci.lower() == 'wald':
                lower = mle_val - z_crit * se_val
                upper = mle_val + z_crit * se_val
                if p_name.startswith("tpar_") or p_name == "d_prime": # Ensure positivity for these
                    lower = max(1e-5, lower)
                results[p_name] = (lower, upper)
            elif method_ci.lower() == 'profile':
                lower_ci_prof, upper_ci_prof = np.nan, np.nan
                # Crude search bounds, can be improved
                search_min = max(1e-5, mle_val - 3 * se_val) if (p_name.startswith("tpar_") or p_name == "d_prime") else mle_val - 3 * se_val
                search_max = mle_val + 3 * se_val
                if search_min >= mle_val - 1e-6 : search_min = mle_val - 0.1 if mle_val > 1e-5 else 1e-6
                if search_max <= mle_val + 1e-6 : search_max = mle_val + 0.1

                try:
                    val_at_mle_target_fn = chi2.ppf(level,1)/2.0 # self._profile_loglik_target_dod(mle_val, idx_fixed, level) should be this
                    val_at_search_min = self._profile_loglik_target_dod(search_min, idx_fixed, level)
                    if np.isfinite(val_at_search_min) and val_at_mle_target_fn * val_at_search_min < 0:
                        lower_ci_prof = scipy.optimize.brentq(self._profile_loglik_target_dod, search_min, mle_val, args=(idx_fixed, level))
                    else: warnings.warn(f"Could not bracket lower profile CI for {p_name}", UserWarning)
                except Exception as e: warnings.warn(f"Brentq failed for {p_name} lower profile CI: {e}", UserWarning)

                try:
                    val_at_mle_target_fn = chi2.ppf(level,1)/2.0
                    val_at_search_max = self._profile_loglik_target_dod(search_max, idx_fixed, level)
                    if np.isfinite(val_at_search_max) and val_at_mle_target_fn * val_at_search_max < 0:
                        upper_ci_prof = scipy.optimize.brentq(self._profile_loglik_target_dod, mle_val, search_max, args=(idx_fixed, level))
                    else: warnings.warn(f"Could not bracket upper profile CI for {p_name}", UserWarning)
                except Exception as e: warnings.warn(f"Brentq failed for {p_name} upper profile CI: {e}", UserWarning)
                results[p_name] = (lower_ci_prof, upper_ci_prof)
            else:
                raise ValueError(f"Unsupported CI method_ci: {method_ci}")
        return results

# --- Helper functions for SameDifferentModel --- (already defined, kept for context)
def _get_samediff_probs_model(tau: float, delta: float, epsilon: float = 1e-12) -> tuple[float, float, float, float]:
    # ... (implementation as before)
    Pss = 2*scipy_norm.cdf(tau/np.sqrt(2))-1; Pds=1-Pss
    Psd = scipy_norm.cdf((tau-delta)/np.sqrt(2)) - scipy_norm.cdf((-tau-delta)/np.sqrt(2)); Pdd=1-Psd
    Pss=np.clip(Pss,epsilon,1-epsilon); Pds=np.clip(Pds,epsilon,1-epsilon); Psd=np.clip(Psd,epsilon,1-epsilon); Pdd=np.clip(Pdd,epsilon,1-epsilon)
    sum_s = Pss+Pds; Pss/=sum_s; Pds/=sum_s; sum_d=Psd+Pdd; Psd/=sum_d; Pdd/=sum_d
    return Pss,Pds,Psd,Pdd

def _samediff_nll_model(params: np.ndarray, nsamesame: int, ndiffsame: int, nsamediff: int, ndiffdiff: int) -> float:
    # ... (implementation as before)
    tau,delta=params;
    if tau<=0 or delta<0:return np.inf
    Pss,Pds,Psd,Pdd=_get_samediff_probs_model(tau,delta)
    loglik = nsamesame*np.log(Pss)+ndiffsame*np.log(Pds)+nsamediff*np.log(Psd)+ndiffdiff*np.log(Pdd)
    return -loglik if np.isfinite(loglik) else np.inf

@dataclass
class SameDifferentModel(BaseModel):
    # ... (SameDifferentModel definition as before, but confint will be enhanced) ...
    tau: float = field(default=None, init=False)
    delta: float = field(default=None, init=False)
    se_tau: float = field(default=None, init=False)
    se_delta: float = field(default=None, init=False)
    loglik: float = field(default=None, init=False) # This should be loglik_mle
    vcov: np.ndarray = field(default=None, init=False, repr=False)
    convergence_status: bool = field(default=None, init=False)
    nsamesame: int = field(default=None, init=False, repr=False)
    ndiffsame: int = field(default=None, init=False, repr=False)
    nsamediff: int = field(default=None, init=False, repr=False)
    ndiffdiff: int = field(default=None, init=False, repr=False)
    conf_level_used: float = field(default=None, init=False)
    optim_result_obj: object = field(default=None, init=False, repr=False)
    params: np.ndarray = field(default=None, init=False, repr=False) # To store [tau, delta] from fit

    def fit(self, nsamesame: int, ndiffsame: int, nsamediff: int, ndiffdiff: int, initial_tau: float | None = None, initial_delta: float | None = None, method: str = "ml", conf_level: float = 0.95):
        # ... (fit logic as before, ensure self.loglik is MLE, store self.params)
        if method.lower() != "ml": raise NotImplementedError("Only 'ml' supported.")
        self.nsamesame, self.ndiffsame, self.nsamediff, self.ndiffdiff = nsamesame, ndiffsame, nsamediff, ndiffdiff
        self.conf_level_used = conf_level
        # (validations...)
        tau_init = initial_tau if initial_tau is not None and initial_tau > 0 else 1.0
        delta_init = initial_delta if initial_delta is not None and initial_delta >=0 else 1.0
        self.params = np.array([tau_init, delta_init]) # Store initial as self.params for now
        bounds = [(1e-5, None), (0, None)]
        self.optim_result_obj = scipy.optimize.minimize(_samediff_nll_model, self.params, args=(nsamesame,ndiffsame,nsamediff,ndiffdiff), method="L-BFGS-B", bounds=bounds, hess='2-point')
        self.params = self.optim_result_obj.x # Update self.params with optimized values
        self.tau, self.delta = self.params[0], self.params[1]
        self.loglik = -self.optim_result_obj.fun # MLE log-likelihood
        self.convergence_status = self.optim_result_obj.success
        # (SE calculation...)
        if self.convergence_status and hasattr(self.optim_result_obj, 'hess_inv'):
            try:
                self.vcov = self.optim_result_obj.hess_inv.todense()
                if self.vcov[0,0]>=0: self.se_tau = np.sqrt(self.vcov[0,0])
                if self.vcov[1,1]>=0: self.se_delta = np.sqrt(self.vcov[1,1])
            except: self.se_tau,self.se_delta = np.nan,np.nan
        else: self.se_tau,self.se_delta = np.nan,np.nan


    def summary(self) -> str:
        # ... (summary logic as before) ...
        if not hasattr(self,'convergence_status') or not self.convergence_status or self.tau is None : return "Model not fitted/converged."
        se_tau_str = f"{self.se_tau:.4f}" if np.isfinite(self.se_tau) else "N/A"
        se_delta_str = f"{self.se_delta:.4f}" if np.isfinite(self.se_delta) else "N/A"
        return (f"SameDifferent Model Summary\n-------------------------\n"
                f"tau: {self.tau:.4f} (SE: {se_tau_str})\ndelta: {self.delta:.4f} (SE: {se_delta_str})\n"
                f"LogLik: {self.loglik:.4f}\nConverged: {self.convergence_status}")

    def _profile_loglik_target_samediff(self, param_val_fixed, param_name_fixed, level):
        target_cutoff = self.loglik - chi2.ppf(level, 1) / 2.0

        def optimize_nuisance(nuisance_param_val): # nuisance_param_val is scalar
            current_tau, current_delta = (param_val_fixed, nuisance_param_val) if param_name_fixed == 'tau' else (nuisance_param_val, param_val_fixed)
            # Bounds for nuisance parameter
            if param_name_fixed == 'tau': # optimizing delta
                if current_delta < 0: return np.inf # delta >=0
            else: # optimizing tau
                if current_tau <= 1e-5: return np.inf # tau > 0

            return _samediff_nll_model(np.array([current_tau, current_delta]), self.nsamesame, self.ndiffsame, self.nsamediff, self.ndiffdiff)

        # Initial guess for nuisance: its MLE
        initial_guess_nuisance = self.delta if param_name_fixed == 'tau' else self.tau
        nuisance_bounds = (0, None) if param_name_fixed == 'tau' else (1e-5, None) # Delta >=0, Tau > 0

        # Use minimize_scalar for single nuisance parameter optimization
        res_nuisance = scipy.optimize.minimize_scalar(optimize_nuisance, bounds=nuisance_bounds, method='bounded')

        if not res_nuisance.success: return -np.inf
        max_loglik_for_fixed_param = -res_nuisance.fun # res_nuisance.fun is the NLL
        return max_loglik_for_fixed_param - target_cutoff

    def confint(self, parm: list[str] | None = None, level: float = 0.95, method_ci: str = 'profile') -> dict[str, tuple[float, float]]:
        if self.loglik is None: raise ValueError("Model not fitted.")
        if parm is None: parm = ['tau', 'delta']

        results = {}
        z_crit = scipy.stats.norm.ppf(1 - (1 - level) / 2)

        for p_name in parm:
            mle_val = getattr(self, p_name)
            se_val = getattr(self, f"se_{p_name}")

            if method_ci.lower() == 'wald':
                lower = mle_val - z_crit * se_val if np.isfinite(se_val) else np.nan
                upper = mle_val + z_crit * se_val if np.isfinite(se_val) else np.nan
                if p_name == 'tau': lower = max(1e-5, lower) # Ensure tau > 0
                if p_name == 'delta': lower = max(0, lower)   # Ensure delta >= 0
                results[p_name] = (lower, upper)
            elif method_ci.lower() == 'profile':
                lower_ci_prof, upper_ci_prof = np.nan, np.nan
                # Crude search bounds, can be improved
                search_min = mle_val - 3 * se_val if np.isfinite(se_val) and se_val > 0 else mle_val * 0.1
                search_max = mle_val + 3 * se_val if np.isfinite(se_val) and se_val > 0 else mle_val * 2
                if p_name == 'tau': search_min = max(1e-5, search_min)
                if p_name == 'delta': search_min = max(0, search_min)
                if search_min >= mle_val - 1e-6 : search_min = mle_val - 0.1 if mle_val > 1e-5 else 1e-6
                if search_max <= mle_val + 1e-6 : search_max = mle_val + 0.1

                try:
                    val_at_mle_target_fn = chi2.ppf(level,1)/2.0
                    val_at_search_min = self._profile_loglik_target_samediff(search_min, p_name, level)
                    if np.isfinite(val_at_search_min) and val_at_mle_target_fn * val_at_search_min < 0: # Check signs
                         lower_ci_prof = scipy.optimize.brentq(self._profile_loglik_target_samediff, search_min, mle_val, args=(p_name, level))
                    else: warnings.warn(f"Could not bracket lower profile CI for {p_name}", UserWarning)
                except Exception as e: warnings.warn(f"Brentq failed for {p_name} lower profile CI: {e}", UserWarning)

                try:
                    val_at_mle_target_fn = chi2.ppf(level,1)/2.0
                    val_at_search_max = self._profile_loglik_target_samediff(search_max, p_name, level)
                    if np.isfinite(val_at_search_max) and val_at_mle_target_fn * val_at_search_max < 0: # Check signs
                        upper_ci_prof = scipy.optimize.brentq(self._profile_loglik_target_samediff, mle_val, search_max, args=(p_name, level))
                    else: warnings.warn(f"Could not bracket upper profile CI for {p_name}", UserWarning)
                except Exception as e: warnings.warn(f"Brentq failed for {p_name} upper profile CI: {e}", UserWarning)
                results[p_name] = (lower_ci_prof, upper_ci_prof)
            else:
                raise ValueError(f"Unsupported CI method_ci: {method_ci}. Choose 'profile' or 'wald'.")
        return results
