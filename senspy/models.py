from dataclasses import dataclass, field
import numpy as np
import scipy.optimize
from scipy.stats import betabinom, chi2 # chi2 for confint, betabinom for simple pmf/logpmf if needed
from numba import jit
from math import lgamma

__all__ = ["BetaBinomial", "TwoACModel", "DiscriminationModel", "DoDModel", "SameDifferentModel"]

# Numba-optimized helper functions (from plan-to-port.md)
@jit(nopython=True)
def numba_betaln(a, b):
    """Log of the beta function, B(a,b)."""
    return lgamma(a) + lgamma(b) - lgamma(a + b)

@jit(nopython=True)
def log_binom_pmf(k, n, p):
    """Log of the binomial PMF using log-gamma."""
    if p < 0 or p > 1:
        return -np.inf
    if k < 0 or k > n:
        return -np.inf
    if p == 0:
        return 0.0 if k == 0 else -np.inf
    if p == 1:
        return 0.0 if k == n else -np.inf
    
    # Log(nCk) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
    log_nCk = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
    log_pmf = log_nCk + k * np.log(p) + (n - k) * np.log(1 - p)
    return log_pmf

# Numba-optimized log-likelihood functions
@jit(nopython=True)
def _standard_loglik(x_obs_arr, n_trials_arr, log_a, log_b):
    """Standard beta-binomial log-likelihood."""
    alpha = np.exp(log_a)
    beta = np.exp(log_b)
    if alpha <= 0 or beta <= 0: # Ensure parameters are positive
        return -np.inf

    total_loglik = 0.0
    for i in range(len(x_obs_arr)):
        k = x_obs_arr[i]
        n = n_trials_arr[i]
        
        # Log PMF of Beta-Binomial: log(nCk) + log(B(k+a, n-k+b)) - log(B(a,b))
        log_nCk = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
        log_beta_num = numba_betaln(k + alpha, n - k + beta)
        log_beta_den = numba_betaln(alpha, beta)
        
        total_loglik += log_nCk + log_beta_num - log_beta_den
        
    return total_loglik

@jit(nopython=True)
def _corrected_loglik_terms(x_obs_arr, n_trials_arr, p_guess, log_a, log_b, log_terms_buffer):
    """Chance-corrected beta-binomial log-likelihood using precomputed terms."""
    alpha = np.exp(log_a)
    beta = np.exp(log_b)
    if alpha <= 0 or beta <= 0: # Ensure parameters are positive
        return -np.inf

    total_loglik = 0.0
    for i in range(len(x_obs_arr)):
        k_obs = x_obs_arr[i]
        n = n_trials_arr[i]
        
        # log_terms_buffer[j] = log_binom_pmf(k_obs, j, p_guess)
        # This buffer should be pre-calculated based on k_obs and p_guess for j from k_obs to n
        # For this function, it's assumed log_terms_buffer is already correct for the i-th observation
        # The structure of log_terms_buffer needs to be aligned with its usage here.
        # Assuming log_terms_buffer is passed per observation or is globally accessible and correctly indexed.
        # For this example, let's assume log_terms_buffer is specific to current k_obs, n.
        # This part needs careful implementation based on how log_terms_buffer is structured.
        # The example from plan-to-port.md implies log_terms_buffer is specific to each (k_obs, n_val) pair.
        # For simplicity, we'll assume log_terms_buffer is computed appropriately outside
        # or this function is called in a loop that prepares it.
        # Let's re-evaluate the sum: sum_{j=k_obs to n} [ P(j | BB(a,b)) * P(k_obs | j, Binom(p_g)) ]
        # log P(k_obs | j, Binom(p_g)) is log_terms_buffer[j-k_obs] for this k_obs
        
        current_sum_log_prob = -np.inf # log(0)
        for j in range(k_obs, n + 1):
            # Log PMF of Beta-Binomial for j: log(nCj) + log(B(j+a, n-j+b)) - log(B(a,b))
            log_nCj = lgamma(n + 1) - lgamma(j + 1) - lgamma(n - j + 1)
            log_beta_num_j = numba_betaln(j + alpha, n - j + beta)
            log_beta_den_j = numba_betaln(alpha, beta)
            log_prob_j_bb = log_nCj + log_beta_num_j - log_beta_den_j
            
            # Log PMF of Binomial(k_obs | j, p_guess) - this is log_terms_buffer[i][j-k_obs]
            # The plan-to-port.md suggests log_terms_buffer[i][j] = log_binom_pmf(x_obs_arr[i], j, p_guess)
            # This means for a given x_obs_arr[i], we need log_binom_pmf(x_obs_arr[i], j, p_guess)
            # where j is the true number of successes.
            # Let's assume log_terms_buffer[i] is an array for the i-th observation,
            # and its k-th element is log_binom_pmf(x_obs_arr[i], k, p_guess) for k in some range.
            # The definition in plan-to-port was:
            # self.log_terms_buffer[i, j_idx] = log_binom_pmf(self.x_obs_arr[i], j_val, self.p_guess)
            # where j_val ranges from x_obs_arr[i] to n_trials_arr[i]. So j_idx = j_val - x_obs_arr[i].
            
            idx_in_buffer = j - k_obs # This assumes log_terms_buffer[i] starts at j=k_obs
            if idx_in_buffer < 0 or idx_in_buffer >= len(log_terms_buffer[i]):
                # This should not happen if buffer is correctly sized (n - k_obs + 1)
                # For safety, skip if out of bounds, though it indicates an issue.
                continue 
            
            log_prob_k_obs_given_j_binom = log_terms_buffer[i][idx_in_buffer]

            if log_prob_j_bb == -np.inf or log_prob_k_obs_given_j_binom == -np.inf:
                term = -np.inf
            else:
                term = log_prob_j_bb + log_prob_k_obs_given_j_binom
            
            # Log-sum-exp trick for sum_log_prob = log(exp(sum_log_prob) + exp(term))
            if term > -np.inf: # only add if term is valid
                if current_sum_log_prob == -np.inf:
                    current_sum_log_prob = term
                else:
                    current_sum_log_prob = np.log(np.exp(current_sum_log_prob - term) + 1.0) + term
        
        total_loglik += current_sum_log_prob
        
    return total_loglik

from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Base class for statistical models."""
    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def summary(self) -> str:
        pass

    @abstractmethod
    def confint(self, parm: list[str] | None = None, level: float = 0.95, method: str = 'profile') -> dict[str, tuple[float, float]]:
        pass

@dataclass
class BetaBinomial(BaseModel):
    """Beta-binomial model."""
    alpha: float = 1.0  # Default initial value
    beta: float = 1.0   # Default initial value
    n: int = field(default=0, init=False) # Not directly used by fit if x,n are arrays. Placeholder.

    # Attributes to be set by fit()
    params: np.ndarray = field(default=None, init=False, repr=False) # [log_alpha, log_beta]
    loglik: float = field(default=None, init=False, repr=False)
    vcov: np.ndarray = field(default=None, init=False, repr=False)
    convergence_status: bool = field(default=None, init=False, repr=False)
    n_obs: int = field(default=None, init=False, repr=False)
    
    x_obs_arr: np.ndarray = field(default=None, init=False, repr=False)
    n_trials_arr: np.ndarray = field(default=None, init=False, repr=False)
    corrected: bool = field(default=False, init=False, repr=False)
    p_guess: float = field(default=None, init=False, repr=False) # Single float
    log_terms_buffer: list = field(default=None, init=False, repr=False) # List of arrays for corrected

    def fit(self, x, n, corrected=False, p_guess=0.5):
        self.x_obs_arr = np.asarray(x, dtype=np.int32)
        self.n_trials_arr = np.asarray(n, dtype=np.int32)
        if self.x_obs_arr.shape != self.n_trials_arr.shape:
            raise ValueError("Shape mismatch between x_obs_arr and n_trials_arr.")
        if np.any(self.x_obs_arr < 0) or np.any(self.n_trials_arr < 0):
            raise ValueError("Counts (x) and trials (n) must be non-negative.")
        if np.any(self.x_obs_arr > self.n_trials_arr):
            raise ValueError("Counts (x) cannot exceed trials (n).")

        self.corrected = corrected
        self.p_guess = p_guess if self.corrected else None

        if self.corrected:
            if not (0 < self.p_guess < 1):
                # sensR allows p_guess = 0 or 1, but formula may break with log(0)
                # Let's restrict to (0,1) for stability with logs in correction.
                # Or handle p_guess=0 or 1 as special cases (e.g. becomes standard BB or simpler).
                # For now, raise error.
                raise ValueError("p_guess must be between 0 and 1 for corrected model.")
            
            self.log_terms_buffer = []
            for i in range(len(self.x_obs_arr)):
                k_obs_i = self.x_obs_arr[i]
                n_i = self.n_trials_arr[i]
                # Buffer for j from k_obs_i to n_i. Length is n_i - k_obs_i + 1
                buffer_i = np.empty(n_i - k_obs_i + 1, dtype=np.float64)
                for j_idx, j_val in enumerate(range(k_obs_i, n_i + 1)):
                    buffer_i[j_idx] = log_binom_pmf(k_obs_i, j_val, self.p_guess)
                self.log_terms_buffer.append(buffer_i)
        else:
            self.log_terms_buffer = None # Not used

        def objective_func(params_log_scale):
            log_a, log_b = params_log_scale
            if self.corrected:
                neg_loglik = -_corrected_loglik_terms(
                    self.x_obs_arr, self.n_trials_arr, self.p_guess, 
                    log_a, log_b, self.log_terms_buffer
                )
            else:
                neg_loglik = -_standard_loglik(
                    self.x_obs_arr, self.n_trials_arr, log_a, log_b
                )
            return neg_loglik

        initial_params_log_scale = np.array([0.0, 0.0])  # log(a=1), log(b=1)
        # Bounds to keep alpha, beta > 0. log(1e-7) is approx -16.11
        param_bounds_log_scale = [(np.log(1e-7), None), (np.log(1e-7), None)]

        try:
            result = scipy.optimize.minimize(
                objective_func,
                initial_params_log_scale,
                method="L-BFGS-B",
                bounds=param_bounds_log_scale,
                hess='2-point' # Ask for Hessian approximation
            )
            self.params = result.x
            self.alpha = np.exp(self.params[0])
            self.beta = np.exp(self.params[1])
            self.loglik = -result.fun
            # scipy.optimize.minimize returns OptimizeResult.
            # .hess_inv is available for L-BFGS-B, it's an OptimizeResultLBFGSB specific attribute.
            # It's an approximation of the inverse Hessian.
            if hasattr(result, 'hess_inv') and result.hess_inv is not None:
                 # For L-BFGS-B, hess_inv is a sparse matrix object (BFGSUpdate)
                 # To get a dense array for vcov:
                 self.vcov = result.hess_inv.todense()
            else: # Fallback if hess_inv is not available or not computed
                self.vcov = np.full((2, 2), np.nan) # Placeholder if Hessian not available
            self.convergence_status = result.success
            self.n_obs = len(self.x_obs_arr)

        except Exception as e:
            # Broad exception to catch optimization failures
            print(f"Optimization failed: {e}")
            self.params = initial_params_log_scale # Revert to initial if failed
            self.alpha = np.exp(self.params[0])
            self.beta = np.exp(self.params[1])
            self.loglik = -objective_func(self.params) if self.x_obs_arr is not None else None
            self.vcov = np.full((2, 2), np.nan) # Unknown vcov
            self.convergence_status = False
            self.n_obs = len(self.x_obs_arr) if self.x_obs_arr is not None else 0
            # Optionally, re-raise or handle more gracefully
            # raise # Re-raise the exception if preferred

    def mean(self) -> float: # This uses fitted alpha, beta
        if self.alpha is None or self.beta is None:
            return np.nan
        return self.alpha / (self.alpha + self.beta)

    def pmf(self, k: int, n_val: int) -> float: # For a single k, n pair
        if self.alpha is None or self.beta is None:
            raise ValueError("Model has not been fitted yet.")
        return betabinom.pmf(k, n_val, self.alpha, self.beta)

    def logpmf(self, k: int, n_val: int) -> float: # For a single k, n pair
        if self.alpha is None or self.beta is None:
            raise ValueError("Model has not been fitted yet.")
        return betabinom.logpmf(k, n_val, self.alpha, self.beta)

    def summary(self):
        """Prepare and return a summary of the model."""
        if self.params is None: # Not fitted
            return "Model has not been fitted yet."

        se_alpha, se_beta = 'N/A', 'N/A'
        if self.vcov is not None and isinstance(self.vcov, np.ndarray) and self.vcov.shape == (2,2):
            if self.vcov[0, 0] >= 0 : se_alpha = np.sqrt(self.vcov[0, 0]) 
            if self.vcov[1, 1] >= 0 : se_beta = np.sqrt(self.vcov[1, 1])
        
        summary_data = {
            "Coefficients": {"alpha": self.alpha, "beta": self.beta},
            "Standard Errors": {"alpha": se_alpha, "beta": se_beta},
            "Log-Likelihood": self.loglik,
            "Convergence Status": self.convergence_status,
            "N Observations": self.n_obs
        }
        
        output = f"BetaBinomial Model Summary\n"
        output += f"--------------------------\n"
        output += f"Coefficients:\n"
        output += f"  alpha: {summary_data['Coefficients']['alpha']:.4f} (SE: {summary_data['Standard Errors']['alpha'] if isinstance(summary_data['Standard Errors']['alpha'], str) else summary_data['Standard Errors']['alpha']:.4f})\n"
        output += f"  beta:  {summary_data['Coefficients']['beta']:.4f} (SE: {summary_data['Standard Errors']['beta'] if isinstance(summary_data['Standard Errors']['beta'], str) else summary_data['Standard Errors']['beta']:.4f})\n"
        output += f"Log-Likelihood: {summary_data['Log-Likelihood']:.4f}\n"
        output += f"N Observations: {summary_data['N Observations']}\n"
        output += f"Convergence Status: {summary_data['Convergence Status']}\n"
        
        return output

    def _profile_loglik_target(self, param_val_fixed_natural, param_name_fixed, level):
        """Target function for brentq to find profile likelihood CIs."""
        if self.loglik is None:
            raise ValueError("Model must be fitted first (self.loglik is None for profile target).")

        chi2_crit = chi2.ppf(level, 1)
        target_loglik_cutoff = self.loglik - chi2_crit / 2

        try:
            log_param_val_fixed = np.log(param_val_fixed_natural)
        except RuntimeWarning: 
             return -np.inf 

        def optimization_target(opt_param_val_log_arr):
            opt_param_val_log = opt_param_val_log_arr[0]
            if param_name_fixed == 'alpha':
                log_alpha_current, log_beta_current = log_param_val_fixed, opt_param_val_log
            elif param_name_fixed == 'beta':
                log_alpha_current, log_beta_current = opt_param_val_log, log_param_val_fixed
            else:
                raise ValueError(f"Unknown parameter: {param_name_fixed}")

            if self.corrected:
                current_loglik_val = _corrected_loglik_terms(
                    self.x_obs_arr, self.n_trials_arr, self.p_guess,
                    log_alpha_current, log_beta_current, self.log_terms_buffer)
            else:
                current_loglik_val = _standard_loglik(
                    self.x_obs_arr, self.n_trials_arr, log_alpha_current, log_beta_current)
            
            return -current_loglik_val if np.isfinite(current_loglik_val) else np.inf

        # Determine initial guess and bounds for the parameter to be optimized
        # self.params should be [log_alpha_mle, log_beta_mle] from the main fit
        initial_guess_log = self.params[1] if param_name_fixed == 'alpha' else self.params[0]
        # Same bounds as in main fit
        opt_bounds = [(np.log(1e-7), None)] 

        result = scipy.optimize.minimize(
            optimization_target, [initial_guess_log], method='L-BFGS-B', bounds=opt_bounds)

        if not result.success: return -np.inf 
        max_loglik_for_fixed_param = -result.fun 
        return max_loglik_for_fixed_param - target_loglik_cutoff

    def confint(self, parm: list[str] | None = None, level: float = 0.95, method: str = 'profile') -> dict[str, tuple[float, float]]:
        if method != 'profile':
            raise NotImplementedError(f"Method '{method}' not implemented for confidence intervals.")
        if self.loglik is None:
            raise ValueError("Model must be fitted first (self.loglik is None for confint).")
        
        required_attrs_base = ['alpha', 'beta', 'loglik', 'x_obs_arr', 'n_trials_arr', 
                               'params', 'corrected']
        
        # Attributes that must not be None
        required_attrs_not_none = list(required_attrs_base) # Make a copy

        if self.corrected:
            required_attrs_not_none.append('p_guess') # p_guess must be set if corrected
            required_attrs_not_none.append('log_terms_buffer')
        # If not corrected, self.p_guess can be None, so it's not in required_attrs_not_none
        # but it should exist as an attribute.
        
        # Check all required attributes exist
        if not all(hasattr(self, attr) for attr in required_attrs_base + ['p_guess', 'log_terms_buffer']):
            missing_attr = [attr for attr in required_attrs_base + ['p_guess', 'log_terms_buffer'] if not hasattr(self, attr)]
            raise ValueError(f"Model is not properly fitted. Missing attributes: {missing_attr}")

        # Check attributes that must not be None
        if not all(getattr(self, attr) is not None for attr in required_attrs_not_none):
             missing_or_none = [attr for attr in required_attrs_not_none if getattr(self,attr) is None]
             raise ValueError(f"Model is not properly fitted. Required attributes are None: {missing_or_none}")

        if parm is None: parm = ['alpha', 'beta'] 
        results = {}

        for param_name in parm:
            mle_val_natural = getattr(self, param_name)
            val_at_mle = chi2.ppf(level, 1) / 2 # Target function value at MLE
            if val_at_mle <= 0:
                results[param_name] = (np.nan, np.nan); continue

            lower_bound, upper_bound = np.nan, np.nan
            # Search range factors, can be adjusted
            search_min_factor, search_max_factor = 0.01, 100
            search_min_abs = 1e-9

            # Lower bound
            current_search_min = max(search_min_abs, mle_val_natural * search_min_factor)
            for _ in range(5): # Iteratively extend search range if needed
                try:
                    val_at_search_min = self._profile_loglik_target(current_search_min, param_name, level)
                    if np.isfinite(val_at_search_min) and val_at_mle * val_at_search_min < 0:
                        lower_bound = scipy.optimize.brentq(self._profile_loglik_target, current_search_min, mle_val_natural, args=(param_name, level))
                        break
                    elif val_at_search_min == 0: lower_bound = current_search_min; break
                    current_search_min = max(search_min_abs, current_search_min * 0.1) if current_search_min > search_min_abs else current_search_min / 10
                    if current_search_min == 0 and mle_val_natural > 1e-10: current_search_min = search_min_abs/10 # try even smaller if mle is not zero
                except (RuntimeError, ValueError): break # brentq or other optimization error

            # Upper bound
            current_search_max = mle_val_natural * search_max_factor
            if current_search_max <= mle_val_natural : current_search_max = mle_val_natural + 10 # Ensure search_max > mle
            for _ in range(5):
                try:
                    val_at_search_max = self._profile_loglik_target(current_search_max, param_name, level)
                    if np.isfinite(val_at_search_max) and val_at_mle * val_at_search_max < 0:
                        upper_bound = scipy.optimize.brentq(self._profile_loglik_target, mle_val_natural, current_search_max, args=(param_name, level))
                        break
                    elif val_at_search_max == 0: upper_bound = current_search_max; break
                    current_search_max *= 10
                except (RuntimeError, ValueError): break
            
            results[param_name] = (lower_bound, upper_bound)
        return results


# Helper function for TwoACModel log-likelihood
# This is defined at the module level similar to _standard_loglik for BetaBinomial
# It could also be a static method within the class if preferred, but module level is fine.
def _loglik_twoAC_model(params_model: np.ndarray, h: int, f: int, n_signal: int, n_noise: int, epsilon: float = 1e-12) -> float:
    """
    Log-likelihood for the 2AC (Yes/No or Same/Different with bias) model.
    params_model: [delta, tau]
    h: hits
    f: false alarms
    n_signal: number of signal trials
    n_noise: number of noise trials
    epsilon: small value to prevent log(0)
    """
    delta, tau = params_model
    
    # Probabilities using scipy.stats.norm.cdf
    # pHit = P(response="yes" | signal) = Phi(delta/2 - tau)
    # pFA  = P(response="yes" | noise)  = Phi(-delta/2 - tau)
    # Note: Original twoAC from discrimination.py uses norm.cdf(delta / 2 - tau) for pHit
    # and norm.cdf(-delta / 2 - tau) for pFA. This seems to be a common formulation.
    # Let's stick to that.
    
    pHit = scipy.stats.norm.cdf(delta / 2.0 - tau)
    pFA = scipy.stats.norm.cdf(-delta / 2.0 - tau)

    # Clip probabilities to avoid log(0)
    pHit = np.clip(pHit, epsilon, 1.0 - epsilon)
    pFA = np.clip(pFA, epsilon, 1.0 - epsilon)

    # Binomial log-likelihood contributions
    loglik = (
        h * np.log(pHit) +
        (n_signal - h) * np.log(1.0 - pHit) +
        f * np.log(pFA) +
        (n_noise - f) * np.log(1.0 - pFA)
    )
    return loglik


@dataclass
class TwoACModel(BaseModel):
    """
    Thurstonian model for 2-Alternative Choice (Yes/No) with response bias.
    
    Attributes:
        delta (float): Sensitivity parameter.
        tau (float): Bias parameter.
        loglik (float): Log-likelihood of the fitted model.
        vcov (np.ndarray): Variance-covariance matrix of the parameters.
        convergence_status (bool): Whether the optimization converged.
        n_obs (int): Number of observation pairs (typically 1 for this model structure).
        params (np.ndarray): Raw parameter estimates from optimizer [delta, tau].
    """
    delta: float = field(default=0.0, init=False)
    tau: float = field(default=0.0, init=False)
    
    # Attributes to be set by fit()
    params: np.ndarray = field(default=None, init=False, repr=False) # [delta, tau]
    loglik: float = field(default=None, init=False, repr=False)
    vcov: np.ndarray = field(default=None, init=False, repr=False)
    convergence_status: bool = field(default=None, init=False, repr=False)
    n_obs: int = field(default=None, init=False, repr=False) # Number of (h,f,n_s,n_n) sets, usually 1

    # Store input data for reference
    hits: int = field(default=None, init=False, repr=False)
    false_alarms: int = field(default=None, init=False, repr=False)
    n_signal_trials: int = field(default=None, init=False, repr=False)
    n_noise_trials: int = field(default=None, init=False, repr=False)

    def fit(self, hits: int, false_alarms: int, n_signal_trials: int, n_noise_trials: int):
        """
        Fit the TwoACModel to data.

        Args:
            hits (int): Number of hits.
            false_alarms (int): Number of false alarms.
            n_signal_trials (int): Number of signal trials.
            n_noise_trials (int): Number of noise trials.
        """
        if not (0 <= hits <= n_signal_trials and 0 <= false_alarms <= n_noise_trials):
            raise ValueError("Number of hits/false_alarms must be between 0 and respective trial counts.")
        if n_signal_trials <= 0 or n_noise_trials <= 0:
            raise ValueError("Number of signal and noise trials must be positive.")

        self.hits = hits
        self.false_alarms = false_alarms
        self.n_signal_trials = n_signal_trials
        self.n_noise_trials = n_noise_trials
        
        # Objective function for minimization (negative log-likelihood)
        def objective_func_twoac(params_to_opt): # params_to_opt will be [delta, tau]
            return -_loglik_twoAC_model(params_to_opt, self.hits, self.false_alarms, 
                                      self.n_signal_trials, self.n_noise_trials)

        # Initial guesses (can be simple, e.g., 0,0 or derived if needed)
        # sensR uses d' from ignoring bias, and tau=0 as initial.
        # pH_obs = hits / n_signal_trials
        # pFA_obs = false_alarms / n_noise_trials
        # d_init = scipy.stats.norm.ppf(np.clip(pH_obs, 1e-5, 1-1e-5)) - \
        #          scipy.stats.norm.ppf(np.clip(pFA_obs, 1e-5, 1-1e-5))
        # c_init = -0.5 * (scipy.stats.norm.ppf(np.clip(pH_obs, 1e-5, 1-1e-5)) + \
        #                 scipy.stats.norm.ppf(np.clip(pFA_obs, 1e-5, 1-1e-5)))
        # tau_init_from_c = c_init * d_init / np.sqrt(d_init**2 + c_init**2) if (d_init**2+c_init**2)>0 else 0 # This is for a different tau definition
        # The original twoAC function uses delta_init = 0, tau_init = 0. Let's stick to that for consistency.
        initial_params = np.array([0.0, 0.0]) 
        
        # Bounds for parameters: delta >= 0, tau is unbounded (L-BFGS-B handles None for no bound)
        param_bounds = [(0, None),  # delta >= 0
                        (None, None)] # tau is unbounded

        try:
            result = scipy.optimize.minimize(
                objective_func_twoac,
                initial_params,
                method="L-BFGS-B",
                bounds=param_bounds,
                hess='2-point' # Request Hessian approximation for vcov
            )
            
            self.params = result.x
            self.delta = self.params[0]
            self.tau = self.params[1]
            self.loglik = -result.fun 
            self.convergence_status = result.success
            self.n_obs = 1 # Typically, one set of (h, f, n_s, n_n) is fitted at a time.

            if hasattr(result, 'hess_inv') and result.hess_inv is not None:
                try:
                    self.vcov = result.hess_inv.todense()
                except Exception: # Broad catch if .todense() fails or other issue
                    self.vcov = np.full((2, 2), np.nan) # Fallback
            else:
                self.vcov = np.full((2, 2), np.nan) # Fallback if Hessian not available

        except Exception as e:
            # Handle optimization failure
            print(f"Optimization failed: {e}")
            self.params = initial_params # Revert to initial if failed
            self.delta = initial_params[0]
            self.tau = initial_params[1]
            self.loglik = -objective_func_twoac(initial_params)
            self.vcov = np.full((2, 2), np.nan)
            self.convergence_status = False
            self.n_obs = 1 
            # raise # Optionally re-raise

    def summary(self) -> str:
        """Prepare and return a summary of the fitted TwoACModel."""
        if self.params is None: # Not fitted
            return "Model has not been fitted yet."

        se_delta_str, se_tau_str = 'N/A', 'N/A'
        if self.vcov is not None and isinstance(self.vcov, np.ndarray) and self.vcov.shape == (2,2) and not np.all(np.isnan(self.vcov)):
            if self.vcov[0, 0] >= 0:
                se_delta_str = f"{np.sqrt(self.vcov[0, 0]):.4f}"
            if self.vcov[1, 1] >= 0:
                se_tau_str = f"{np.sqrt(self.vcov[1, 1]):.4f}"
        
        summary_str = f"TwoACModel Summary\n"
        summary_str += f"--------------------\n"
        summary_str += f"Coefficients:\n"
        summary_str += f"  delta (sensitivity): {self.delta:.4f} (SE: {se_delta_str})\n"
        summary_str += f"  tau (bias):          {self.tau:.4f} (SE: {se_tau_str})\n"
        summary_str += f"Log-Likelihood: {self.loglik:.4f}\n"
        summary_str += f"N Observations (sets): {self.n_obs}\n" # Number of (h,f,n_s,n_n) sets
        summary_str += f"Convergence Status: {self.convergence_status}\n"
        summary_str += f"Input Data:\n"
        summary_str += f"  Hits: {self.hits}, False Alarms: {self.false_alarms}\n"
        summary_str += f"  Signal Trials: {self.n_signal_trials}, Noise Trials: {self.n_noise_trials}\n"
        
        return summary_str

    def _profile_loglik_target_twoac(self, param_val_fixed_natural, param_name_fixed, level):
        """Target function for brentq to find profile likelihood CIs for TwoACModel."""
        if self.loglik is None:
            raise ValueError("Model must be fitted first (self.loglik is None for profile target).")

        chi2_crit = chi2.ppf(level, 1)
        target_loglik_cutoff = self.loglik - chi2_crit / 2

        # Unlike BetaBinomial, TwoACModel params (delta, tau) are not necessarily log-transformed in optimization
        # delta has a bound (>=0), tau is unbounded.
        # So, we operate on natural scale for fixed param, and optimize other on natural scale.

        def optimization_target_twoac(opt_param_val_arr): # opt_param_val_arr is a single-element array
            opt_param_val = opt_param_val_arr[0]

            if param_name_fixed == 'delta':
                current_delta, current_tau = param_val_fixed_natural, opt_param_val
            elif param_name_fixed == 'tau':
                current_delta, current_tau = opt_param_val, param_val_fixed_natural
            else:
                raise ValueError(f"Unknown parameter: {param_name_fixed}")

            # Ensure delta is non-negative for the optimization call
            if current_delta < 0 and param_name_fixed == 'tau': # i.e. delta is being optimized
                 return np.inf # Invalid region for delta

            current_loglik_val = _loglik_twoAC_model(
                np.array([current_delta, current_tau]),
                self.hits, self.false_alarms, self.n_signal_trials, self.n_noise_trials
            )
            return -current_loglik_val if np.isfinite(current_loglik_val) else np.inf

        # Determine initial guess and bounds for the parameter to be optimized
        # self.params holds [delta_mle, tau_mle]
        if param_name_fixed == 'delta': # Optimize tau
            initial_guess = self.params[1] # tau_mle
            opt_bounds = [(None, None)] # tau is unbounded
        else: # param_name_fixed == 'tau', optimize delta
            initial_guess = self.params[0] # delta_mle
            opt_bounds = [(0, None)] # delta >= 0

        # If the fixed parameter itself is out of bounds (e.g. delta_fixed < 0), return large neg value
        if param_name_fixed == 'delta' and param_val_fixed_natural < 0:
            return -np.inf # Effectively, this point is outside valid parameter space

        result = scipy.optimize.minimize(
            optimization_target_twoac, [initial_guess], method='L-BFGS-B', bounds=opt_bounds)

        if not result.success: return -np.inf
        max_loglik_for_fixed_param = -result.fun
        return max_loglik_for_fixed_param - target_loglik_cutoff

    def confint(self, parm: list[str] | None = None, level: float = 0.95, method: str = 'profile') -> dict[str, tuple[float, float]]:
        if method != 'profile':
            raise NotImplementedError(f"Method '{method}' not implemented for confidence intervals for TwoACModel.")
        if self.loglik is None or self.params is None or self.hits is None: # Check basic fit status
            raise ValueError("Model must be fitted before calculating confidence intervals.")

        required_attrs = ['delta', 'tau', 'loglik', 'params', 'hits', 'false_alarms', 'n_signal_trials', 'n_noise_trials']
        if not all(hasattr(self, attr) and getattr(self, attr) is not None for attr in required_attrs):
             missing_or_none = [attr for attr in required_attrs if not hasattr(self,attr) or getattr(self,attr) is None]
             raise ValueError(f"TwoACModel is not properly fitted. Required attributes are missing or None: {missing_or_none}")

        if parm is None: parm = ['delta', 'tau']
        results = {}

        for param_name in parm:
            if param_name not in ['delta', 'tau']:
                results[param_name] = (np.nan, np.nan)
                warnings.warn(f"Confidence interval requested for unknown parameter '{param_name}' in TwoACModel. Skipping.")
                continue

            mle_val_natural = getattr(self, param_name)

            # Target function should be zero at MLE for a valid CI search
            # self._profile_loglik_target_twoac(mle_val_natural, param_name, level) should be close to 0
            # This is because max_loglik_for_fixed_param at MLE should be self.loglik
            # So, self.loglik - (self.loglik - chi2_crit/2) = chi2_crit/2.
            # This is what brentq expects: f(a)*f(b) < 0.
            # Let's use chi2_crit/2 as the reference for target function value at MLE.
            val_at_mle_target_fn = chi2.ppf(level, 1) / 2.0
            if val_at_mle_target_fn <= 1e-9: # If level is too low or chi2 is zero
                results[param_name] = (np.nan, np.nan)
                continue

            lower_bound, upper_bound = np.nan, np.nan

            # Search range: Use SE if available and finite, else use factors of MLE or absolute values
            se_param = np.nan
            if self.vcov is not None and not np.all(np.isnan(self.vcov)):
                idx = 0 if param_name == 'delta' else 1
                if self.vcov[idx, idx] >= 0:
                    se_param = np.sqrt(self.vcov[idx, idx])

            # Define search start points
            # For lower bound search: from search_min up to mle_val_natural
            # For upper bound search: from mle_val_natural up to search_max

            # Lower CI bound search
            if np.isfinite(se_param) and se_param > 1e-6:
                search_start_lower = mle_val_natural - 3 * se_param
            else:
                search_start_lower = mle_val_natural * 0.1 if mle_val_natural > 1e-3 else mle_val_natural - 2.0
            if param_name == 'delta': # Delta must be >= 0
                search_start_lower = max(0.0, search_start_lower)

            # Ensure search_start_lower < mle_val_natural, otherwise brentq fails
            if search_start_lower >= mle_val_natural - 1e-7 : search_start_lower = mle_val_natural - 0.5
            if param_name == 'delta' and search_start_lower < 0: search_start_lower = 0.0


            # Upper CI bound search
            if np.isfinite(se_param) and se_param > 1e-6:
                search_start_upper = mle_val_natural + 3 * se_param
            else:
                search_start_upper = mle_val_natural * 2.0 if mle_val_natural > 1e-3 else mle_val_natural + 2.0
            # Ensure search_start_upper > mle_val_natural
            if search_start_upper <= mle_val_natural + 1e-7 : search_start_upper = mle_val_natural + 0.5


            # Iteratively try to find bracket for brentq for lower bound
            current_search_min = search_start_lower
            for i in range(10): # Max 10 iterations to find bracket
                try:
                    val_at_search_min = self._profile_loglik_target_twoac(current_search_min, param_name, level)
                    if np.isfinite(val_at_search_min) and val_at_mle_target_fn * val_at_search_min < 0:
                        lower_bound = scipy.optimize.brentq(
                            self._profile_loglik_target_twoac, current_search_min, mle_val_natural,
                            args=(param_name, level), xtol=1e-5, rtol=1e-5
                        )
                        break
                    # If signs are same, expand search range further away from MLE
                    if param_name == 'delta' and current_search_min < 1e-7 : break # Avoid going too low for delta
                    current_search_min = current_search_min - (se_param if np.isfinite(se_param) else abs(mle_val_natural*0.5) + 0.5)
                    if param_name == 'delta': current_search_min = max(0.0, current_search_min) # Respect delta bound
                except (RuntimeError, ValueError) as e:
                    warnings.warn(f"Brentq lower bound search for {param_name} failed: {e}")
                    break

            # Iteratively try to find bracket for brentq for upper bound
            current_search_max = search_start_upper
            for i in range(10):
                try:
                    val_at_search_max = self._profile_loglik_target_twoac(current_search_max, param_name, level)
                    if np.isfinite(val_at_search_max) and val_at_mle_target_fn * val_at_search_max < 0:
                         upper_bound = scipy.optimize.brentq(
                            self._profile_loglik_target_twoac, mle_val_natural, current_search_max,
                            args=(param_name, level), xtol=1e-5, rtol=1e-5
                        )
                         break
                    current_search_max = current_search_max + (se_param if np.isfinite(se_param) else abs(mle_val_natural*0.5) + 0.5)
                except (RuntimeError, ValueError) as e:
                    warnings.warn(f"Brentq upper bound search for {param_name} failed: {e}")
                    break

            results[param_name] = (lower_bound, upper_bound)
        return results

# Ensure scipy.stats is available for _loglik_twoAC_model
import scipy.stats
import warnings # For warnings in confint for TwoACModel

# Import discrim function for DiscriminationModel
# This creates a potential circular import if discrimination.py imports models.py for BaseModel
# To resolve:
# 1. Ensure discrimination.py does not import specific models from models.py if possible,
#    or only imports BaseModel.
# 2. Or, move DiscriminationModel to discrimination.py if it's tightly coupled.
# For now, proceed with the import and address if it becomes an issue at runtime/testing.
from senspy import discrimination as sp_discrim # Use alias to avoid conflict

@dataclass
class DiscriminationModel(BaseModel):
    """
    Model for d-prime estimation from various discrimination tasks.
    This model wraps the functionality of the `senspy.discrimination.discrim` function.
    """
    results_dict: dict = field(default=None, init=False, repr=False)

    # Store key inputs/attributes for summary and confint
    correct: int = field(default=None, init=False)
    total: int = field(default=None, init=False)
    method_protocol: str = field(default=None, init=False) # 'method' from discrim
    conf_level_used: float = field(default=None, init=False)

    def fit(self, correct: int, total: int, method: str,
            conf_level: float = 0.95, statistic: str = "Wald"):
        """
        Fit the discrimination model by calling senspy.discrimination.discrim.

        Args:
            correct (int): Number of correct responses.
            total (int): Total number of trials.
            method (str): Discrimination method (e.g., "2afc", "triangle").
            conf_level (float, optional): Confidence level for the CI. Defaults to 0.95.
            statistic (str, optional): Type of statistic. Defaults to "Wald".
        """
        # Store inputs
        self.correct = correct
        self.total = total
        self.method_protocol = method
        self.conf_level_used = conf_level

        # --- Core logic from discrim function ---
        if statistic.lower() != "wald":
            raise NotImplementedError(f"Statistic '{statistic}' not yet implemented. Only 'Wald' is supported.")

        method_lc = method.lower()

        # Ensure necessary helper functions are accessible
        # These would typically be part of senspy.discrimination or imported there
        # For now, assuming they are available via sp_discrim which imports them or defines them

        # Access pc_funcs and get_pguess through sp_discrim which should have them
        # This requires sp_discrim to be senspy.discrimination module
        pc_funcs = sp_discrim._PC_FUNCTIONS_INTERNAL

        if method_lc not in pc_funcs:
            raise ValueError(f"Unknown method: {method}. Supported methods are: {list(pc_funcs.keys())}")

        pc_func = pc_funcs[method_lc]
        pguess = sp_discrim.get_pguess(method_lc)

        if correct < 0 or total <= 0 or correct > total:
            raise ValueError("Invalid 'correct' or 'total' values.")

        pc_obs = correct / total
        epsilon_lower = 1.0 / (2 * total) if total > 0 else 1e-8
        epsilon_upper = 1.0 / (2 * total) if total > 0 else 1e-8
        pc_clipped = np.clip(pc_obs, pguess + epsilon_lower, 1.0 - epsilon_upper)

        dprime_est = np.nan
        if pc_obs <= pguess:
            dprime_est = 0.0
        elif pc_obs == 1.0 and pguess < 1.0:
             # Simplified: if pc_func(15.0) is already very high, brentq might find it, else use 15 as upper guess
            if pc_func(15.0) < pc_clipped : dprime_est = 15.0
            else:
                try: dprime_est = scipy.optimize.brentq(lambda d: pc_func(d) - pc_clipped, -5.0, 15.0, xtol=1e-6, rtol=1e-6)
                except ValueError: dprime_est = 15.0 # Fallback if brentq fails for pc=1
        else:
            try:
                dprime_est = scipy.optimize.brentq(lambda d: pc_func(d) - pc_clipped, -5.0, 15.0, xtol=1e-6, rtol=1e-6)
            except ValueError: # Fallback if brentq fails
                if pc_func(-5.0) > pc_clipped: dprime_est = -5.0
                elif pc_func(15.0) < pc_clipped: dprime_est = 15.0
                else: dprime_est = np.nan


        se_dprime = np.nan
        if pc_obs == 0.0 or pc_obs == 1.0:
            se_dprime = np.inf
        elif np.isfinite(dprime_est):
            dx_deriv = max(abs(dprime_est) * 1e-4, 1e-6)
            # numerical_derivative is also in sp_discrim
            deriv_val = sp_discrim.numerical_derivative(pc_func, dprime_est, dx=dx_deriv)
            if deriv_val is not None and abs(deriv_val) > 1e-9:
                variance_pc = pc_obs * (1 - pc_obs) / total
                se_dprime = np.sqrt(variance_pc) / deriv_val
            else:
                se_dprime = np.inf

        z_crit = scipy.stats.norm.ppf(1 - (1 - conf_level) / 2)
        lower_ci = dprime_est - z_crit * se_dprime
        upper_ci = dprime_est + z_crit * se_dprime

        p_value = np.nan
        if np.isfinite(se_dprime) and se_dprime > 1e-9:
            wald_z = dprime_est / se_dprime
            p_value = 2 * scipy.stats.norm.sf(np.abs(wald_z))
        elif dprime_est == 0: p_value = 1.0

        self.results_dict = {
            "dprime": dprime_est, "se_dprime": se_dprime, "lower_ci": lower_ci,
            "upper_ci": upper_ci, "p_value": p_value, "conf_level": conf_level,
            "correct": correct, "total": total, "pc_obs": pc_obs,
            "pguess": pguess, "method": method, "statistic": statistic,
        }
        # --- End of core logic ---

        # Optionally, could copy key results to direct attributes for easier access
        # e.g., self.dprime = self.results_dict.get('dprime')
        # For now, keep results primarily in results_dict.

    def summary(self) -> str:
        """
        Return a string summary of the fitted discrimination model.
        """
        if self.results_dict is None:
            return "DiscriminationModel has not been fitted yet."

        res = self.results_dict
        summary_lines = [
            f"Discrimination Model Summary ({self.method_protocol.upper()})",
            f"----------------------------------",
            f"d-prime (Sensitivity): {res.get('dprime', 'N/A'):.4f}",
            f"Standard Error (d-prime): {res.get('se_dprime', 'N/A'):.4f}",
            f"{self.conf_level_used*100:.0f}% Confidence Interval (d-prime): ({res.get('lower_ci', 'N/A'):.4f}, {res.get('upper_ci', 'N/A'):.4f})",
            f"P-value (d-prime vs 0): {res.get('p_value', 'N/A'):.4g}",
            f"Observed Proportion Correct (Pc): {res.get('pc_obs', 'N/A'):.4f}",
            f"Chance Performance (Pguess): {res.get('pguess', 'N/A'):.4f}",
            f"Input Data: {res.get('correct', 'N/A')} correct out of {res.get('total', 'N/A')} trials.",
            f"Method: {res.get('method', 'N/A')}",
            f"Statistic Type: {res.get('statistic', 'N/A')}"
        ]
        return "\n".join(summary_lines)

    def confint(self, parm: list[str] | None = None,
                level: float = 0.95,
                method: str = 'profile') -> dict[str, tuple[float, float]]:
        """
        Return confidence intervals for parameters of the DiscriminationModel.
        Currently, only supports Wald intervals for 'dprime' as computed by fit().
        """
        if self.results_dict is None:
            raise ValueError("Model must be fitted before calling confint.")

        if parm is None:
            parm = ['dprime']

        intervals = {}
        for p_name in parm:
            if p_name == 'dprime':
                # If the requested level is different from the one used in fit(),
                # Wald CIs would need recalculation. For now, assume level matches.
                if abs(level - self.conf_level_used) > 1e-6:
                    warnings.warn(f"Requested CI level {level} differs from level used in fit ({self.conf_level_used}). "
                                  f"Returning stored Wald CI. Re-fit for different level if needed for Wald.", UserWarning)

                # Profile likelihood CIs are not yet implemented for DiscriminationModel.
                if method.lower() == 'profile':
                    warnings.warn("Profile likelihood CIs are not yet implemented for DiscriminationModel. "
                                  "Returning Wald CI for d-prime.", NotImplementedError)

                lower_ci = self.results_dict.get('lower_ci', np.nan)
                upper_ci = self.results_dict.get('upper_ci', np.nan)
                intervals['dprime'] = (lower_ci, upper_ci)
            else:
                warnings.warn(f"Confidence interval for parameter '{p_name}' is not available for DiscriminationModel.", UserWarning)
                intervals[p_name] = (np.nan, np.nan)

        return intervals

# --- Helper functions for DoDModel ---
# These are moved from discrimination.py to be co-located with DoDModel
def _par2prob_dod_model(tau: np.ndarray, d_prime: float) -> np.ndarray:
    """
    Calculate category probabilities for "same" and "different" pairs in a DoD task.
    (Moved from senspy.discrimination.par2prob_dod for DoDModel use)
    """
    if not isinstance(tau, np.ndarray) or tau.ndim != 1:
        raise ValueError("tau must be a 1D NumPy array.")
    if len(tau) > 0: # tau can be empty if num_categories is 2 (K-1 = 1 tpar, 0 actual boundary taus in some formulations)
                     # However, sensR's dod and this implementation expect at least one tau if K=2.
                     # If K=2, tau has 1 element.
        if not np.all(tau > 0):
            raise ValueError("All elements of tau must be positive.")
        if len(tau) > 1 and not np.all(np.diff(tau) > 0): # check diff only if more than one tau
            raise ValueError("Elements of tau must be strictly increasing.")
    if not (isinstance(d_prime, (float, int)) and d_prime >= 0):
        raise ValueError("d_prime must be a non-negative scalar.")

    gamma_same = 2 * scipy.stats.norm.cdf(tau / np.sqrt(2)) - 1
    gamma_diff = scipy.stats.norm.cdf((tau - d_prime) / np.sqrt(2)) - scipy.stats.norm.cdf((-tau - d_prime) / np.sqrt(2))

    p_same = np.diff(np.concatenate(([0.0], gamma_same, [1.0])))
    p_diff = np.diff(np.concatenate(([0.0], gamma_diff, [1.0])))

    epsilon = 1e-12
    p_same = np.clip(p_same, epsilon, 1.0)
    p_diff = np.clip(p_diff, epsilon, 1.0)
    p_same /= np.sum(p_same)
    p_diff /= np.sum(p_diff)

    return np.vstack((p_same, p_diff))

def _dod_nll_model(params: np.ndarray, same_counts: np.ndarray, diff_counts: np.ndarray) -> float:
    """
    Negative log-likelihood for DoD model.
    (Moved from senspy.discrimination.dod_nll for DoDModel use)
    params are [tau_0, ..., tau_{K-2}, d_prime]
    """
    num_categories = len(same_counts)
    if len(params) != num_categories:
        raise ValueError(f"Length of params ({len(params)}) must be {num_categories}.")

    tau_params = params[:-1] # These are the K-1 tau values
    d_prime_param = params[-1]

    if d_prime_param < 0: return np.inf
    if len(tau_params)>0:
        if np.any(tau_params <= 0): return np.inf
        if len(tau_params) > 1 and np.any(np.diff(tau_params) <= 0): return np.inf

    prob = _par2prob_dod_model(tau_params, d_prime_param)

    loglik_same = np.sum(same_counts * np.log(prob[0, :]))
    loglik_diff = np.sum(diff_counts * np.log(prob[1, :]))
    total_loglik = loglik_same + loglik_diff

    return -total_loglik if np.isfinite(total_loglik) else np.inf

def _init_tpar_model(num_categories: int) -> np.ndarray:
    """
    Initial tpar values for DoDModel. (tpar are increments of tau)
    (Moved from senspy.discrimination._init_tpar_sensr for DoDModel use)
    """
    if num_categories < 2: raise ValueError("Num categories must be >= 2.")
    num_tpar = num_categories - 1 # K-1 tpar values for K categories
    if num_tpar == 1: return np.array([1.0])
    # Corrected logic for tpar based on sensR's initTau(ncat = K-1 thresholds)
    # initTau(N) -> c(1, rep(3/(N-1), N-2))
    # tpar_0 = 1.0. Remaining num_tpar-1 elements are 3.0 / (num_tpar - 1)
    return np.concatenate(([1.0], np.full(num_tpar - 1, 3.0 / (num_tpar - 1.0 + 1e-9))))


@dataclass
class DoDModel(BaseModel):
    """Degree of Difference (DoD) model."""
    d_prime: float = field(default=None, init=False)
    tau: np.ndarray = field(default=None, init=False, repr=False) # K-1 boundary parameters
    tpar: np.ndarray = field(default=None, init=False, repr=False) # K-1 optimized tpar (increments of tau)
    se_d_prime: float = field(default=None, init=False)
    se_tpar: np.ndarray = field(default=None, init=False, repr=False)
    loglik: float = field(default=None, init=False)
    vcov_optim_params: np.ndarray = field(default=None, init=False, repr=False) # VCOV of (tpar, d_prime)
    convergence_status: bool = field(default=None, init=False)

    # Store inputs
    same_counts: np.ndarray = field(default=None, init=False, repr=False)
    diff_counts: np.ndarray = field(default=None, init=False, repr=False)
    conf_level_used: float = field(default=None, init=False)
    optim_result_obj: object = field(default=None, init=False, repr=False) # Full scipy result

    def fit(self, same_counts: np.ndarray, diff_counts: np.ndarray,
            initial_tau: np.ndarray | None = None,
            initial_d_prime: float | None = None,
            method: str = "ml", conf_level: float = 0.95):

        if method.lower() != "ml":
            raise NotImplementedError("Only 'ml' (maximum likelihood) method is currently supported for DoDModel.")

        self.same_counts = np.asarray(same_counts, dtype=np.int32)
        self.diff_counts = np.asarray(diff_counts, dtype=np.int32)
        self.conf_level_used = conf_level

        if self.same_counts.ndim != 1 or self.diff_counts.ndim != 1:
            raise ValueError("same_counts and diff_counts must be 1D arrays.")
        if len(self.same_counts) != len(self.diff_counts):
            raise ValueError("same_counts and diff_counts must have the same length.")

        num_categories = len(self.same_counts)
        if num_categories < 2:
            raise ValueError("Number of categories must be at least 2.")
        if np.any(self.same_counts < 0) or np.any(self.diff_counts < 0):
            raise ValueError("Counts must be non-negative.")
        if np.sum(self.same_counts) == 0 and np.sum(self.diff_counts) == 0:
            raise ValueError("Total counts for both same and different pairs cannot be zero.")

        tpar_init = None
        if initial_tau is not None:
            initial_tau_arr = np.asarray(initial_tau, dtype=float)
            if len(initial_tau_arr) != num_categories - 1:
                raise ValueError(f"initial_tau must have length {num_categories - 1}")
            if not (np.all(initial_tau_arr > 0) and (len(initial_tau_arr)==1 or np.all(np.diff(initial_tau_arr) > 0))):
                raise ValueError("initial_tau must be positive and strictly increasing.")
            tpar_init = np.concatenate(([initial_tau_arr[0]], np.diff(initial_tau_arr)))
            if np.any(tpar_init <= 0):
                 warnings.warn("Derived initial_tpar from initial_tau non-positive. Defaulting.", UserWarning)
                 tpar_init = _init_tpar_model(num_categories)
        else:
            tpar_init = _init_tpar_model(num_categories)

        d_prime_init = initial_d_prime if initial_d_prime is not None else 1.0
        if d_prime_init < 0:
            warnings.warn("initial_d_prime negative. Using 0.0.", UserWarning); d_prime_init = 0.0

        initial_params_for_optim = np.concatenate((tpar_init, [d_prime_init]))

        epsilon_bound = 1e-5
        bounds_tpar = [(epsilon_bound, None)] * (num_categories - 1)
        bounds_d_prime = (epsilon_bound, None)
        bounds = bounds_tpar + [bounds_d_prime]

        def obj_func_dod_wrapper_model(params_optim, s_counts, d_counts):
            tpar_optim, d_prime_optim = params_optim[:-1], params_optim[-1]
            tau_for_nll = np.cumsum(tpar_optim)
            params_nll = np.concatenate((tau_for_nll, [d_prime_optim]))
            return _dod_nll_model(params_nll, s_counts, d_counts)

        self.optim_result_obj = scipy.optimize.minimize(
            obj_func_dod_wrapper_model, initial_params_for_optim,
            args=(self.same_counts, self.diff_counts), method="L-BFGS-B",
            bounds=bounds, hess='2-point'
        )

        optim_params = self.optim_result_obj.x
        self.tpar = optim_params[:-1]
        self.tau = np.cumsum(self.tpar)
        self.d_prime = optim_params[-1]
        self.loglik = -self.optim_result_obj.fun
        self.convergence_status = self.optim_result_obj.success

        self.vcov_optim_params = np.full((len(initial_params_for_optim), len(initial_params_for_optim)), np.nan)
        self.se_tpar = np.full(len(self.tpar), np.nan)
        self.se_d_prime = np.nan

        if self.convergence_status and hasattr(self.optim_result_obj, 'hess_inv'):
            try:
                self.vcov_optim_params = self.optim_result_obj.hess_inv.todense()
                se_all = np.sqrt(np.diag(self.vcov_optim_params))
                self.se_tpar = se_all[:-1]
                self.se_d_prime = se_all[-1]
            except Exception as e:
                warnings.warn(f"SE calculation failed for DoDModel: {e}", RuntimeWarning)

    def summary(self) -> str:
        if not self.convergence_status or self.d_prime is None:
            return "DoDModel has not been fitted or did not converge."

        summary_str = f"DoD Model Summary\n-----------------\n"
        summary_str += f"d-prime: {self.d_prime:.4f} (SE: {self.se_d_prime:.4f})\n"
        summary_str += f"tau parameters (boundaries):\n"
        for i, tau_val in enumerate(self.tau):
            # SE for tau requires Delta method from tpar SEs, complex. Report tpar SEs.
            summary_str += f"  tau[{i}]: {tau_val:.4f}\n"
        summary_str += f"tpar parameters (increments of tau):\n"
        for i, tpar_val in enumerate(self.tpar):
            summary_str += f"  tpar[{i}]: {tpar_val:.4f} (SE: {self.se_tpar[i]:.4f})\n"
        summary_str += f"Log-Likelihood: {self.loglik:.4f}\n"
        summary_str += f"Convergence Status: {self.convergence_status}\n"
        return summary_str

    def confint(self, parm: list[str] | None = None, level: float = 0.95, method: str = 'wald') -> dict[str, tuple[float, float]]:
        if self.d_prime is None: raise ValueError("Model not fitted.")
        if method.lower() != 'wald':
            raise NotImplementedError("Only Wald CIs currently implemented for DoDModel.")

        if parm is None: parm = ['d_prime'] # Default to d_prime, tpar CIs are more direct with Wald

        results = {}
        z_crit = scipy.stats.norm.ppf(1 - (1 - level) / 2)

        if 'd_prime' in parm:
            lower = self.d_prime - z_crit * self.se_d_prime if np.isfinite(self.se_d_prime) else np.nan
            upper = self.d_prime + z_crit * self.se_d_prime if np.isfinite(self.se_d_prime) else np.nan
            results['d_prime'] = (lower, upper)

        # For tau or tpar, Wald CIs are for tpar (optimized params)
        # Can list them as tpar_0, tpar_1, ...
        for i in range(len(self.tpar)):
            param_name_tpar = f"tpar_{i}"
            if param_name_tpar in parm:
                se_current_tpar = self.se_tpar[i] if i < len(self.se_tpar) else np.nan
                lower_tpar = self.tpar[i] - z_crit * se_current_tpar if np.isfinite(se_current_tpar) else np.nan
                upper_tpar = self.tpar[i] + z_crit * se_current_tpar if np.isfinite(se_current_tpar) else np.nan
                results[param_name_tpar] = (lower_tpar, upper_tpar)

        # Warn for any requested params not covered
        for p_name_req in parm:
            if p_name_req not in results and not p_name_req.startswith("tpar_"):
                 warnings.warn(f"CI for '{p_name_req}' not available for DoDModel via Wald method. Tau CIs require Delta method or profiling.", UserWarning)
                 results[p_name_req] = (np.nan, np.nan)
        return results

# --- Helper functions for SameDifferentModel ---
def _get_samediff_probs_model(tau: float, delta: float, epsilon: float = 1e-12) -> tuple[float, float, float, float]:
    Pss = 2 * scipy.stats.norm.cdf(tau / np.sqrt(2)) - 1
    Pds = 1 - Pss
    Psd = scipy.stats.norm.cdf((tau - delta) / np.sqrt(2)) - scipy.stats.norm.cdf((-tau - delta) / np.sqrt(2))
    Pdd = 1 - Psd

    Pss = np.clip(Pss, epsilon, 1.0 - epsilon); Pds = np.clip(Pds, epsilon, 1.0 - epsilon)
    Psd = np.clip(Psd, epsilon, 1.0 - epsilon); Pdd = np.clip(Pdd, epsilon, 1.0 - epsilon)

    sum_same = Pss + Pds; Pss /= sum_same; Pds /= sum_same
    sum_diff = Psd + Pdd; Psd /= sum_diff; Pdd /= sum_diff
    return Pss, Pds, Psd, Pdd

def _samediff_nll_model(params: np.ndarray, nsamesame: int, ndiffsame: int, nsamediff: int, ndiffdiff: int) -> float:
    tau, delta = params
    if tau <= 0 or delta < 0: return np.inf
    Pss, Pds, Psd, Pdd = _get_samediff_probs_model(tau, delta)
    loglik = (nsamesame * np.log(Pss) + ndiffsame * np.log(Pds) +
              nsamediff * np.log(Psd) + ndiffdiff * np.log(Pdd))
    return -loglik if np.isfinite(loglik) else np.inf

@dataclass
class SameDifferentModel(BaseModel):
    """Same-Different (SD) Thurstonian model."""
    tau: float = field(default=None, init=False) # Criterion
    delta: float = field(default=None, init=False) # Sensitivity (d-prime like)
    se_tau: float = field(default=None, init=False)
    se_delta: float = field(default=None, init=False)
    loglik: float = field(default=None, init=False)
    vcov: np.ndarray = field(default=None, init=False, repr=False)
    convergence_status: bool = field(default=None, init=False)

    # Store inputs
    nsamesame: int = field(default=None, init=False, repr=False)
    ndiffsame: int = field(default=None, init=False, repr=False)
    nsamediff: int = field(default=None, init=False, repr=False)
    ndiffdiff: int = field(default=None, init=False, repr=False)
    conf_level_used: float = field(default=None, init=False)
    optim_result_obj: object = field(default=None, init=False, repr=False)

    def fit(self, nsamesame: int, ndiffsame: int, nsamediff: int, ndiffdiff: int,
            initial_tau: float | None = None, initial_delta: float | None = None,
            method: str = "ml", conf_level: float = 0.95):
        if method.lower() != "ml":
            raise NotImplementedError("Only 'ml' method supported for SameDifferentModel.")

        self.nsamesame, self.ndiffsame, self.nsamediff, self.ndiffdiff = nsamesame, ndiffsame, nsamediff, ndiffdiff
        self.conf_level_used = conf_level

        counts = np.array([nsamesame, ndiffsame, nsamediff, ndiffdiff])
        if np.any(counts < 0) or not all(isinstance(c, (int, np.integer)) for c in counts):
            raise ValueError("Counts must be non-negative integers.")
        if np.sum(counts) <= 0: raise ValueError("Sum of counts must be positive.")
        if np.sum(counts > 0) < 2: raise ValueError("At least two counts must be non-zero.")

        tau_init = initial_tau if initial_tau is not None and initial_tau > 0 else 1.0
        delta_init = initial_delta if initial_delta is not None and initial_delta >=0 else 1.0
        initial_params = np.array([tau_init, delta_init])

        bounds = [(1e-5, None), (0, None)] # tau > 0, delta >= 0

        self.optim_result_obj = scipy.optimize.minimize(
            _samediff_nll_model, initial_params,
            args=(nsamesame, ndiffsame, nsamediff, ndiffdiff),
            method="L-BFGS-B", bounds=bounds, hess='2-point'
        )

        self.tau, self.delta = self.optim_result_obj.x
        self.loglik = -self.optim_result_obj.fun
        self.convergence_status = self.optim_result_obj.success
        self.vcov = np.full((2,2), np.nan)
        self.se_tau, self.se_delta = np.nan, np.nan

        if self.convergence_status and hasattr(self.optim_result_obj, 'hess_inv'):
            try:
                self.vcov = self.optim_result_obj.hess_inv.todense()
                if self.vcov[0,0] >= 0: self.se_tau = np.sqrt(self.vcov[0,0])
                if self.vcov[1,1] >= 0: self.se_delta = np.sqrt(self.vcov[1,1])
            except Exception as e:
                warnings.warn(f"SE calculation failed for SameDifferentModel: {e}", RuntimeWarning)

    def summary(self) -> str:
        if not self.convergence_status or self.tau is None:
            return "SameDifferentModel has not been fitted or did not converge."

        return (f"SameDifferent Model Summary\n-------------------------\n"
                f"tau (criterion): {self.tau:.4f} (SE: {self.se_tau:.4f})\n"
                f"delta (sensitivity): {self.delta:.4f} (SE: {self.se_delta:.4f})\n"
                f"Log-Likelihood: {self.loglik:.4f}\n"
                f"Convergence Status: {self.convergence_status}\n"
                f"Data: SS={self.nsamesame}, DS={self.ndiffsame}, SD={self.nsamediff}, DD={self.ndiffdiff}")

    def confint(self, parm: list[str] | None = None, level: float = 0.95, method: str = 'wald') -> dict[str, tuple[float, float]]:
        if self.tau is None: raise ValueError("Model not fitted.")
        if method.lower() != 'wald':
            raise NotImplementedError("Only Wald CIs currently implemented for SameDifferentModel.")

        if parm is None: parm = ['tau', 'delta']
        results = {}
        z_crit = scipy.stats.norm.ppf(1 - (1 - level) / 2)

        param_map = {'tau': (self.tau, self.se_tau), 'delta': (self.delta, self.se_delta)}
        for p_name in parm:
            if p_name in param_map:
                mle, se = param_map[p_name]
                lower = mle - z_crit * se if np.isfinite(se) else np.nan
                upper = mle + z_crit * se if np.isfinite(se) else np.nan
                results[p_name] = (lower, upper)
            else:
                warnings.warn(f"CI for '{p_name}' not available for SameDifferentModel.", UserWarning)
                results[p_name] = (np.nan, np.nan)
        return results
