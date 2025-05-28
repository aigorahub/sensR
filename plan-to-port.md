Final Technical Plan for Porting sensR to Python as sensPyVersion: 1.0 (For Community Feedback)Date: May 27, 20251. Background and ContextThe sensR R package (GitHub, CRAN) is a specialized tool for sensory discrimination analysis, implementing Thurstonian models for protocols such as duotrio, tetrad, triangle, 2-AFC, 3-AFC, A-not A, same-different, 2-AC, degree-of-difference, hexad, and two-out-of-five. It provides functionality for d-prime (δ) estimation, standard errors, sample size and power calculations, profile likelihood confidence intervals, beta-binomial and chance-corrected beta-binomial models, and visualization of results. The package, licensed under GPL (>= 2), relies on R’s statistical ecosystem (e.g., stats, MASS, numDeriv, multcomp) and is widely used in sensory science for analyzing discrimination tests.The goal of this project is to port sensR to Python as sensPy, creating a robust, user-friendly library that integrates with Python’s scientific computing stack (NumPy, SciPy, pandas, matplotlib, Plotly, statsmodels). sensPy will replicate sensR’s core functionality, ensuring numerical fidelity (e.g., |Δ| ≤ 1e-12 for coefficients), performance optimization (via Numba), and modern usability features (e.g., interactive plots, Colab notebooks). The project will be licensed under GPL-2.0-or-later to comply with sensR’s license, and it will be distributed via PyPI with comprehensive documentation and community engagement features.Key Challenges:Replicating R’s statistical precision in Python, accounting for differences in optimization algorithms and BLAS implementations.Optimizing computationally intensive likelihood calculations for large datasets.Ensuring GPL-2.0-or-later compliance while integrating permissive dependencies (e.g., Numba’s BSD 2-Clause, Plotly’s MIT).Providing a seamless transition for sensR users through a migration guide and equivalent API.Target Audience:Sensory scientists transitioning from R to Python.Data scientists and statisticians in food science and related fields.Open-source contributors interested in statistical modeling.2. Project Scope and ObjectivesObjective: Develop sensPy, a Python library that fully replicates sensR’s functionality for Thurstonian models and sensory discrimination, optimized for performance and usability.Scope:Port all sensR functions, including psychometric utilities (psyfun, psyinv, psyderiv, rescale), discrimination protocols, statistical computations, and visualizations.Ensure numerical equivalence to sensR with tolerances: 1e-12 (coefficients), 1e-9 (log-likelihoods), 1e-6 (p-values).Implement performance optimizations using Numba for likelihood calculations.Provide interactive visualizations with Plotly (optional) and static plots with matplotlib.Distribute via PyPI under GPL-2.0-or-later with comprehensive documentation, migration guide, and Colab/Binder notebooks.Deliverables:PyPI package (senspy).Documentation hosted on ReadTheDocs, including API reference, tutorials, migration guide, Legal FAQ, reproducibility note, and CITATION.cff for Zenodo DOI.Unit tests with 95% coverage, validated against sensR using RPy2.Interactive notebooks per protocol with version/SHA reporting.CI/CD pipeline with Linux/Windows support.3. Analysis of sensRSource Repository: https://github.com/aigorahub/sensRKey Features:Psychometric Functions: psyfun (d-prime to proportion correct), psyinv (proportion correct to d-prime), psyderiv (derivative), rescale (conversions between Pc, Pd, d-prime with SEs).Discrimination Protocols: Duotrio, tetrad, triangle, 2-AFC, 3-AFC, A-not A, same-different, 2-AC, degree-of-difference, hexad, two-out-of-five.Statistical Models:D-prime estimation with standard errors (likelihood, score, Wald tests).Beta-binomial and chance-corrected beta-binomial models for overdispersion.Computations:Sample size and power calculations (exact binomial for small n, normal approximation for large n).Profile likelihood confidence intervals with nuisance parameter optimization.Visualization: Confidence intervals, discrimination curves.Dependencies:R: stats (optimization, distributions), MASS (GLM), numDeriv (numerical derivatives), multcomp (multiple comparisons).Python Equivalents:stats → scipy.stats, numpy.MASS → statsmodels.numDeriv → jax (optional), numdifftools.multcomp → statsmodels.stats.multitest.Key Files (from GitHub):R/betaBin.R: Beta-binomial and chance-corrected models.R/discrim.R: D-prime estimation for discrimination tests.R/twoAC.R: 2-AC protocol analysis.R/psyfun.R: Psychometric functions.NEWS: Tracks updates (e.g., power calculations, ANOVA-type tests).Challenges:Numerical precision across BLAS (MKL vs. OpenBLAS).Performance of likelihood calculations for large n.Mapping R’s S3/S4 objects to Python classes.Ensuring GPL compliance with permissive dependencies.4. Technology StackPython Version: 3.8+ for compatibility and modern features.Core Libraries:NumPy: Matrix operations and numerical computations.SciPy: Optimization (scipy.optimize), distributions (scipy.stats), root-finding (brentq).pandas: Data manipulation, R-like data frames.matplotlib: Static plotting.Plotly: Interactive plotting (optional).statsmodels: Statistical modeling, power calculations.jax: Numerical differentiation (optional).numdifftools: Numerical differentiation fallback.numba: Performance optimization for likelihoods.hypothesis: Property-based testing.Development Tools:pytest: Unit testing.Sphinx: Documentation.flake8, black: Code linting/formatting.poetry: Dependency management and packaging.GitHub Actions: CI/CD for Linux/Windows.RPy2: Validation against sensR.Environment Settings:Reproducibility: NUMEXPR_MAX_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, XLA_PYTHON_CLIENT_PREALLOCATE=false.Docker image with pinned OpenBLAS for bit-wise reproducibility.5. Architecture DesignModules:senspy.links: Psychometric functions (psyfun, psyinv, psyderiv, rescale).senspy.models: Statistical models (beta-binomial, discrimination) with BaseModel base class.senspy.power: Sample size and power calculations.senspy.discrimination: Protocol-specific functions (e.g., duotrio, twoAC).senspy.plotting: Visualization (matplotlib, optional Plotly).senspy.utils: Data validation, preprocessing, helpers (has_jax, version).Data Structures:Accept pandas.DataFrame and numpy.ndarray inputs, convert to NumPy internally for performance.Use Python classes for complex models (e.g., BetaBinomial, Discrimination).API Design:Primary: Object-oriented interface via BaseModel with methods fit, summary, confint.summary checks vcov dimensionality before computing SEs, warns if not 2D.Secondary: Thin functional wrappers (e.g., beta_bin(x, n) = BetaBinomial(x, n).fit()).Naming mirrors sensR for familiarity (e.g., discrim, twoAC).Example API Usage:from senspy.models import BetaBinomial
model = BetaBinomial(corrected=True, p_guess=0.5).fit(x=[20, 25], n=[50, 50])
print(model.summary())  # Coefficients, SEs, log-likelihood

from senspy.discrimination import discrim_2afc
result = discrim_2afc(correct=30, total=50)
print(result["d_prime"], result["se"])
6. Implementation PlanThe project is divided into six phases, spanning 34 weeks with a 10% contingency budget. Each phase includes specific tasks, deliverables, and milestones to ensure steady progress and quality.Phase 1: Setup and Detailed Analysis (4 weeks)Objective: Establish project infrastructure and analyze sensR to define scope.Tasks:Clone sensR repository and review source code (R/betaBin.R, R/discrim.R, R/twoAC.R, R/psyfun.R).Create Functionality Inventory (spreadsheet of all sensR functions, inputs, outputs, dependencies) using sensR::lsf.str("package:sensR").Perform Data Structure Mapping (R vectors/lists/data.frames/S3/S4 to Python ndarray/DataFrame/classes).Map R dependencies to Python equivalents (e.g., numDeriv → jax/numdifftools).Consult legal expert on GPL-2.0-or-later obligations; draft Legal FAQ addressing Numba (BSD 2-Clause) and Plotly (MIT) compatibility.Reserve senspy on PyPI and publish placeholder (senspy==0.0.1) with README: “This is a placeholder for senspy; do not depend on this version.”Set up project structure using poetry:sensPy/
├── senspy/
│   ├── __init__.py
│   ├── links.py
│   ├── models.py
│   ├── power.py
│   ├── discrimination.py
│   ├── plotting.py
│   ├── utils.py
├── tests/
├── docs/
├── pyproject.toml
├── README.md
├── CITATION.cff
Configure GitHub Actions for Linux-latest and Windows-latest with pytest.Deliverables:Project repository with GPL-2.0-or-later license.Functionality Inventory and Data Structure Mapping.Legal FAQ draft.PyPI placeholder package.CI configuration for Linux/Windows.Phase 2: Core Functionality and Foundational Utilities (8 weeks)Objective: Implement core statistical models, psychometric functions, and protocol-specific functions with performance optimizations.Tasks:Psychometric Functions:Implement psyfun, psyinv, psyderiv, rescale in links.py using scipy.stats.norm for CDFs.Validate conversions (d-prime ↔ Pc) against sensR.Beta-Binomial Model:Port betaBin from R/betaBin.R with log-space optimization (log(a), log(b)).Implement Numba-compatible likelihoods with numba_betaln and log_binom_pmf.Validate p_guess ∈ (0, 1) to raise ValueError.Discrimination Tests: Implement discrim in discrimination.py for d-prime estimation. Support likelihood, score, and Wald tests using JAX (optional) or numdifftools.2-AC Protocol: Port twoAC with tanh(φ) reparameterization. Use L-BFGS-B with analytic gradients.API Design: Define BaseModel with fit, summary, confint. Implement summary with vcov checks.Utilities: Implement senspy.utils.has_jax() and senspy.version() in __init__.py.Performance: Optimize array layouts (C-contiguous) and use Numba. Verify nopython compatibility.Compiler Milestone:Run pytest -k "beta_bin or discrim" on Linux/Windows with NUMBA_DISABLE_JIT=0, NUMBA_OPT=3 and NUMBA_DISABLE_JIT=1.Benchmark: beta_bin(corrected=False) on 10,000-draw dataset < 0.2s on GitHub runner.Deliverables:links.py, models.py, discrimination.py, utils.py.Unit tests with RPy2 validation (|Δ| ≤ 1e-12 for coefficients).Compiler milestone report.Draft documentation for core functions.Phase 3: Statistical Computations and Visualization (6 weeks)Objective: Implement advanced statistical computations and visualizations.Tasks:Sample Size and Power: Implement exact_power_binom in power.py with signature:from senspy.power import exact_power_binom
from collections import namedtuple
PowerResult = namedtuple("PowerResult", ["power", "n", "alpha", "method"])
power = exact_power_binom(n, p_alt, alpha, method="brentq", tol=1e-4)
Use scipy.optimize.brentq for exact binomial power (small n ≥ 5). Use statsmodels.stats.power for large n approximations.Confidence Intervals: Implement profile likelihood CIs in models.py with nuisance parameter optimization (JAX or numdifftools for gradients).Plotting: Implement plotting functions in plotting.py for CIs and curves. Support matplotlib (static) and Plotly (interactive, optional) with try/except imports. Return raw data for alternative renderers.Deliverables:power.py, plotting.py.Unit tests and plot examples.Updated documentation.Phase 4: Testing and Validation (5 weeks)Objective: Ensure numerical fidelity and robustness through comprehensive testing.Tasks:Write unit tests (pytest) and property-based tests (hypothesis: max_examples=50 in CI, nightly max_examples=500).Validate against sensR (RPy2) with tolerances: Coefficients: |Δ| ≤ 1e-12; Log-likelihoods: |Δ| ≤ 1e-9; P-values: |Δ| ≤ 1e-6.Test edge cases (near-chance, zero-variance, extreme sample sizes, beta_bin with x = n // 2, p_guess = 0.5).Verify vcov output type and standard errors.Document achieved precision per function.Ensure reproducibility (fixed seeds, BLAS settings).Deliverables:Test suite with 95% coverage.Validation report with precision documentation.Performance benchmarks.Phase 5: Documentation and Packaging (3 weeks)Objective: Finalize documentation and package for distribution.Tasks:Write API documentation, tutorials, and “R to Python Migration Guide” (Sphinx).Create Colab/Binder notebooks per protocol, printing senspy.version().Add contribution guidelines and code of conduct.Document: senspy.utils.has_jax(), vcov approximation, reproducibility settings, Docker image.Create CITATION.cff for Zenodo DOI.Add README badges (PyPI, license, tests, docs, Zenodo DOI).Package for PyPI using poetry:[tool.poetry]
name = "senspy"
version = "0.1.0"
description = "Thurstonian Models for Sensory Discrimination in Python"
authors = ["Your Name <your.email@example.com>"] # Placeholder
license = "GPL-2.0-or-later"
readme = "README.md"
homepage = "[https://github.com/YOUR_USERNAME/sensPy](https://github.com/YOUR_USERNAME/sensPy)" # Placeholder
repository = "[https://github.com/YOUR_USERNAME/sensPy](https://github.com/YOUR_USERNAME/sensPy)" # Placeholder
keywords = ["sensory science", "psychophysics", "statistics", "thurstonian models", "d-prime"]

[tool.poetry.dependencies]
python = ">=3.8"
numpy = ">=1.21"
scipy = ">=1.7"
pandas = ">=1.3"
matplotlib = ">=3.4"
statsmodels = ">=0.12"
numba = ">=0.53" # Check Numba's Python version compatibility
numdifftools = ">=0.9"
plotly = {version = ">=5.0", optional = true}
jax = {version = ">=0.4", optional = true}
jaxlib = {version = ">=0.4", optional = true} # Often needed with JAX

[tool.poetry.extras]
interactive = ["plotly"]
derivatives = ["jax", "jaxlib"]

[tool.poetry.group.dev.dependencies] # PEP 621 format for dev dependencies
pytest = ">=6.2"
hypothesis = ">=6.0"
sphinx = ">=4.0" # Or a more recent stable version
rpy2 = ">=3.4" # Check R compatibility
flake8 = ">=3.9"
black = ">=22.0"
# Add other dev tools like sphinx themes, myst-parser etc.

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
Set up GitHub Actions for CI/CD.Deliverables:PyPI package (senspy==0.1.0).ReadTheDocs documentation.Colab/Binder notebooks.CITATION.cff and README badges.CI/CD pipeline.Phase 6: Community Feedback and Iteration (4 weeks)Objective: Release beta, gather feedback, and stabilize for v1.0.0.Tasks:Release beta version and collect feedback (GitHub issues).Fix bugs and implement high-priority features.Optimize standard likelihood loop in beta_bin (backlog item).Consider sandwich estimator for vcov if needed.Gate GA on 95% test coverage and passing profile likelihood CI tests.Deliverables:Stable release (senspy==1.0.0).Community contribution guidelines.7. Risks and MitigationRisk: Numba compilation failures on Windows.Mitigation: Windows CI and compiler milestone.Risk: JAX installation complexity.Mitigation: numdifftools default, JAX optional. Clear documentation on JAX installation.Risk: Numerical noise across BLAS.Mitigation: Document tolerances, provide Docker image for reference environment.Risk: Suboptimal standard likelihood performance.Mitigation: JIT implemented for core loops; further optimization in Phase 6 if needed.Risk: GPL-2.0-or-later compliance issues.Mitigation: Legal consultation and Legal FAQ. Consistent license headers in source files.8. Timeline and ResourcesTotal Duration: 34 weeks (flexible for CI completion).Team:2 Python developers (20 hours/week each).1 statistician (10 hours/week).1 documentation specialist (part-time).Budget: $154,700Developers: $104,000 (2 × 20 hours/week × 26 weeks × $100/hour).Statistician: $31,200 (10 hours/week × 26 weeks × $120/hour).Legal consultation: $5,000.Infrastructure: $500.Contingency (10%): $14,000.9. Revised Beta-Binomial Code ArtifactThis example illustrates the Numba-optimized implementation for the beta-binomial model, including chance correction and careful handling of numerical stability.import numpy as np
from scipy.optimize import minimize
from numba import jit
from numba.np.math import lgamma as _lgamma # Module scope import

@jit(nopython=True)
def numba_betaln(a, b):
    """Numba-compatible betaln using lgamma."""
    return _lgamma(a) + _lgamma(b) - _lgamma(a + b)

@jit(nopython=True)
def log_binom_pmf(k, n_size, p_prob): # n_size is the number of trials for this binomial component
    """Numba-compatible binomial log-PMF."""
    # Handle edge cases for p_prob first
    if p_prob < 0 or p_prob > 1:
        # This case should ideally be caught by p_guess validation earlier
        # but as a safeguard within the numerical function:
        return -np.inf 
    if p_prob == 0:
        return 0.0 if k == 0 else -np.inf
    if p_prob == 1:
        return 0.0 if k == n_size else -np.inf
    
    # Standard case: 0 < p_prob < 1
    # Ensure k is valid for n_size
    if k < 0 or k > n_size:
        return -np.inf # Or handle as error, but -np.inf is common for log-likelihoods

    log_n_choose_k = _lgamma(n_size + 1) - _lgamma(k + 1) - _lgamma(n_size - k + 1)
    return log_n_choose_k + k * np.log(p_prob) + (n_size - k) * np.log(1 - p_prob)

@jit(nopython=True)
def _corrected_loglik_terms(x_obs_arr, n_trials_arr, p_guess, log_a, log_b, log_terms_buffer):
    """
    Calculates sum of log-likelihood terms for chance-corrected beta-binomial.
    x_obs_arr: array of observed successes for each group/observation.
    n_trials_arr: array of total trials for each group/observation.
    p_guess: probability of guessing correctly.
    log_a, log_b: log-transformed beta distribution parameters.
    log_terms_buffer: Pre-allocated buffer for intermediate log terms. Indices > x_obs_arr[i] are ignored.
    """
    a, b = np.exp(log_a), np.exp(log_b)
    total_loglik = 0.0 # Accumulates log-likelihood across all observations

    for i in range(len(x_obs_arr)):
        current_x_obs = int(x_obs_arr[i]) # Number of observed successes for this observation
        current_n_trials = n_trials_arr[i] # Total trials for this observation

        # k_true iterates from 0 to current_x_obs
        # It represents the number of "truly discriminated" items among the observed successes
        for k_true in range(current_x_obs + 1):
            # Binomial part: P(k_true "true" successes | current_x_obs observed successes, p_guess)
            # The number of "trials" for this binomial is current_x_obs
            log_binom_part = log_binom_pmf(k_true, current_x_obs, p_guess) # Corrected: size is current_x_obs
            
            # Beta part: related to the underlying distribution of "true" discrimination
            # Parameters are (a + k_true) and (b + total failures for this observation)
            # Total failures = current_n_trials - current_x_obs
            log_beta_part = numba_betaln(a + k_true, b + (current_n_trials - current_x_obs))
            
            log_terms_buffer[k_true] = log_binom_part + log_beta_part
        
        # Slice the buffer for the current observation's terms
        active_log_terms = log_terms_buffer[:current_x_obs + 1]
        
        # Log-sum-exp for numerical stability: log(sum(exp(terms)))
        max_log_term = np.max(active_log_terms)
        if np.isinf(max_log_term): # Avoid -inf + log(0) if all terms are -inf
             current_obs_loglik = -np.inf
        else:
            sum_exp_terms = np.sum(np.exp(active_log_terms - max_log_term))
            current_obs_loglik = max_log_term + np.log(sum_exp_terms)
        
        # Subtract normalization term B(a,b) for this observation's likelihood component
        # This matches sensR: log( sum(dbinom * beta) / Beta(a,b) )
        # -> log(sum(dbinom * beta)) - log(Beta(a,b))
        total_loglik += current_obs_loglik - numba_betaln(a, b)
        
    return total_loglik

@jit(nopython=True)
def _standard_loglik(x_obs_arr, n_trials_arr, log_a, log_b):
    """Numba-compatible standard beta-binomial log-likelihood."""
    a, b = np.exp(log_a), np.exp(log_b)
    
    total_log_choose = 0.0
    total_log_beta_ratio = 0.0
    
    for i in range(len(x_obs_arr)):
        x_obs = x_obs_arr[i]
        n_trials = n_trials_arr[i]
        
        # log(C(n_trials, x_obs))
        total_log_choose += (_lgamma(n_trials + 1) - 
                             _lgamma(x_obs + 1) - 
                             _lgamma(n_trials - x_obs + 1))
        
        # log( B(a+x_obs, b+n_trials-x_obs) / B(a,b) )
        total_log_beta_ratio += (numba_betaln(a + x_obs, b + (n_trials - x_obs)) - 
                                 numba_betaln(a, b))
                                 
    # Full log-likelihood: sum over observations [ log(C(N,X)) + log(B(a+X,b+N-X)/B(a,b)) ]
    return total_log_choose + total_log_beta_ratio

def beta_bin(x, n, corrected=False, p_guess=0.5):
    """
    Fit a beta-binomial or chance-corrected beta-binomial model.
    Note: vcov is an L-BFGS-B inverse Hessian approximation.
    """
    x_obs_arr, n_trials_arr = np.asarray(x, dtype=np.float64), np.asarray(n, dtype=np.float64) # Ensure float for Numba
    
    if np.any(x_obs_arr < 0) or np.any(n_trials_arr < 0) or np.any(x_obs_arr > n_trials_arr):
        raise ValueError("Input arrays x and n must satisfy 0 <= x <= n.")

    if corrected:
        if not (0 < p_guess < 1): # Strict inequality
            raise ValueError("p_guess must be strictly between 0 and 1 for the corrected model.")
    
    # Pre-allocate buffer for corrected likelihood if used
    # Max value in x_obs_arr determines size
    log_terms_buffer = np.zeros(int(np.max(x_obs_arr)) + 1, dtype=np.float64) if corrected else None

    def objective_func(params): # Scipy minimize wants to minimize, so return -loglik
        log_a, log_b = params
        if corrected:
            # Pass the pre-allocated buffer
            return -_corrected_loglik_terms(x_obs_arr, n_trials_arr, p_guess, 
                                            log_a, log_b, log_terms_buffer)
        else:
            return -_standard_loglik(x_obs_arr, n_trials_arr, log_a, log_b)
    
    # Initial guesses for log_a, log_b (so a=1, b=1)
    initial_params = np.array([0., 0.], dtype=np.float64)
    # Bounds for log_a, log_b (a,b > 1e-6 to avoid log(0))
    param_bounds = [(np.log(1e-6), None), (np.log(1e-6), None)] 
    
    result = minimize(objective_func, initial_params, method="L-BFGS-B", bounds=param_bounds)
    
    # Extract vcov carefully
    vcov_matrix = result.hess_inv # This is an OptimizeResult.hess_inv object
    # For L-BFGS-B, hess_inv is an approximation. To get a dense matrix:
    if isinstance(vcov_matrix, np.ndarray): # Already dense
        pass
    elif hasattr(vcov_matrix, 'todense'): # e.g. sparse matrix
        vcov_matrix = vcov_matrix.todense()
    elif hasattr(vcov_matrix, 'toarray'): # common for sparse
        vcov_matrix = vcov_matrix.toarray()
    # If it's still a LinearOperator (like LbfgsInvHessProduct), more work is needed
    # For simplicity, we assume it can be converted or is already dense.
    # A more robust way to get the dense matrix from LbfgsInvHessProduct:
    # I = np.eye(len(initial_params))
    # vcov_matrix = vcov_matrix.dot(I) # This would make it dense

    return {
        "coefficients": np.exp(result.x), # Return a, b
        "loglik": -result.fun, # Return actual max log-likelihood
        "vcov": vcov_matrix,
        "convergence": result.
