from dataclasses import dataclass, field
import numpy as np
import scipy.optimize
from scipy.stats import betabinom, chi2 # chi2 for confint, betabinom for simple pmf/logpmf if needed
from numba import jit
from math import lgamma

__all__ = ["BetaBinomial", "TwoACModel"]

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


class BaseModel:
    """Base class for statistical models."""
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

# Ensure scipy.stats is available for _loglik_twoAC_model
import scipy.stats
