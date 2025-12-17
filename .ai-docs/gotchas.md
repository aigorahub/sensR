# Gotchas and Pitfalls

## R Parity Quirks

- **Approximations**: `sensR` uses specific approximations for protocols like **Hexad** and **2-out-of-5**. `sensPy` replicates these specific math paths rather than generic combinatorial formulas to ensure test parity. Do not "fix" the math unless you verified it matches R.
- **Statistics**: R's `binom.test` and Python's `scipy.stats.binomtest` / `binom.cdf` handle intervals slightly differently. `sensPy` manually implements some exact test logic to match R's specific p-value definitions for difference vs similarity tests.

## Numerical Stability

- **Link Inversion**: `psy_inv` (pc -> d') uses `brentq` root finding. Bounds are set to `_MAX_DPRIME_SEARCH` (20.0). Inputs close to 1.0 pc map to `np.inf`. Inputs at or below chance map to 0.0.
- **Beta-Binomial**: The corrected model (mu=pd) can have numerical instability near boundaries. The implementation uses log-space arithmetic (`_logsumexp`) to handle small gamma values.

## API Behaviors

- **Rescaling**: `rescale()` requires exactly one input (`pc`, `pd`, or `d_prime`). Providing zero or multiple raises an error.
- **Strings vs Enums**: You can pass strings like `"2-AFC"` or `"TwoAFC"`, but internally everything normalizes to the `Protocol` enum. When debugging, check the normalized enum value.
- **Data Shape**: `betabin` expects a 2-column array `[successes, trials]`. This is specific and distinct from `discrim` which takes scalar `correct` and `total`.
