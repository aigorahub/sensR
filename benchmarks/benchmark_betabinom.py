import sys # For sys.path modification
import os # For path joining
import time
import numpy as np
import warnings

# Add the project root directory (one level up from 'benchmarks') to sys.path
# to allow importing senspy when running this script directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Attempt to import from senspy.models
try:
    from senspy.models import BetaBinomial
    # Import JITted functions for potential inspection (though direct inspection is complex here)
    from senspy.models import _standard_loglik, _corrected_loglik_terms, numba_betaln, log_binom_pmf
    senspy_available = True
except ImportError as e:
    print(f"Failed to import from senspy.models: {e}")
    senspy_available = False
    # Define dummy classes/functions if senspy is not available to allow script structure
    class BetaBinomial: pass 
    def _standard_loglik(): pass
    def _corrected_loglik_terms(): pass
    def numba_betaln(): pass
    def log_binom_pmf(): pass

def run_benchmark_betabinom(num_draws=10000, num_repeats=5):
    """
    Runs a performance benchmark for the BetaBinomial.fit() method.
    """
    if not senspy_available:
        print("senspy.models not available. Skipping benchmark.")
        return

    print(f"--- Benchmarking BetaBinomial.fit() ---")
    print(f"Number of data points (draws): {num_draws}")
    print(f"Number of repeats: {num_repeats}\n")

    # Generate synthetic data
    # Ensure x_obs_arr values are less than n_trials_arr elements
    n_val = 100 # Max value for n_trials_arr elements
    x_obs_arr = np.random.randint(0, n_val + 1, size=num_draws) # randint is [low, high)
    n_trials_arr = np.full(num_draws, n_val)

    model = BetaBinomial() # Uses default alpha=1, beta=1

    durations = []
    
    # Qualitative Numba nopython mode check:
    # We expect Numba to compile the JITted functions on their first call (via model.fit).
    # If nopython=True cannot be achieved, Numba typically issues warnings or errors.
    # We will monitor for NumbaPerformanceWarning related to object mode fallback.
    # Printing inspect_types() here is too verbose for a typical benchmark script,
    # but we can note if any Numba warnings appear during the first fit.
    
    print("Running fits (first fit includes JIT compilation time for Numba functions)...")
    for i in range(num_repeats):
        start_time = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", UserWarning) # Catch user warnings too
            warnings.simplefilter("always", RuntimeWarning) # Catch runtime warnings
            # NumbaPerformanceWarning should also be caught if it occurs
            
            model.fit(x_obs_arr, n_trials_arr, corrected=False)

            # Check for NumbaPerformanceWarning specifically if possible
            # NumbaPerformanceWarning is a subclass of PerformanceWarning
            # This check is a bit manual for a benchmark script, relies on string name
            numba_object_mode_warning = False
            for w in caught_warnings:
                if issubclass(w.category, UserWarning) and "NumbaPerformanceWarning" in str(w.message):
                     # This is a heuristic, actual NumbaPerformanceWarning might not directly contain this string
                     # or might be a specific Numba warning class if Numba is imported directly.
                     # Numba's warnings are typically issued via Python's warnings system.
                    if "object mode" in str(w.message).lower() or \
                       "cannot be compiled in nopython mode" in str(w.message).lower():
                        numba_object_mode_warning = True
                        print(f"Repeat {i+1}: Numba Performance Warning (potential object mode): {w.message}")
                elif issubclass(w.category, RuntimeWarning): # e.g. from scipy.optimize
                     print(f"Repeat {i+1}: RuntimeWarning: {w.message}")


        end_time = time.perf_counter()
        duration = end_time - start_time
        durations.append(duration)
        print(f"Repeat {i+1}/{num_repeats}: Fit time = {duration:.4f} seconds. Converged: {model.convergence_status}")
        if numba_object_mode_warning and i==0: # Only need to confirm once
             print("    -> Numba JIT function(s) may have fallen back to object mode or had performance issues.")
        elif i==0: # First run after JIT
            # If no NumbaPerformanceWarning, it's a good sign for nopython=True.
            # We can also inspect the types after the first run if needed, but that's more for debugging.
            # Example: print(_standard_loglik.inspect_types())
            print("    -> First run complete. JIT compilation should be done.")
            print("    -> Qualitative Numba nopython mode check: Absence of NumbaPerformanceWarning related to object mode fallback is a good sign.")
            # Check if any of the key JIT functions failed nopython mode (if they have dispatchers)
            # This is an indirect check, as we don't call them directly here with inspect_types
            # but rely on warnings or errors if nopython=True was not met during compilation.
            # If `senspy.models` was imported, these are the jitted functions.
            # Their compilation happens when `model.fit` calls them.
            # Numba raises NumbaError if nopython=True fails to compile.
            # NumbaPerformanceWarning if it falls back to object mode from nopython=True (if allowed by config).


    avg_time = np.mean(durations)
    min_time = np.min(durations)
    max_time = np.max(durations)

    print("\n--- Benchmark Summary ---")
    print(f"Data points per fit: {num_draws}")
    print(f"Number of fit repetitions: {num_repeats}")
    print(f"Individual fit times (seconds): {[float(f'{d:.4f}') for d in durations]}")
    print(f"Average fit time: {avg_time:.4f} seconds")
    print(f"Min fit time: {min_time:.4f} seconds")
    print(f"Max fit time: {max_time:.4f} seconds")

    target_time = 0.2
    if avg_time <= target_time:
        print(f"\nPerformance target ({target_time:.2f}s) MET for {num_draws} draws (avg: {avg_time:.4f}s).")
    else:
        print(f"\nPerformance target ({target_time:.2f}s) NOT MET for {num_draws} draws (avg: {avg_time:.4f}s).")

if __name__ == "__main__":
    # Parameters from plan-to-port.md: 10,000 draws, target < 0.2s
    # Running multiple repeats to get a sense of variability.
    run_benchmark_betabinom(num_draws=10000, num_repeats=5)
    print("\n--- Numba JIT Function Signatures (Illustrative) ---")
    if senspy_available:
        # This will print the signature Numba decided on after the first compilation.
        # It doesn't directly confirm nopython mode here, but if nopython=True failed,
        # an error would have occurred earlier or a warning about object mode.
        # For functions defined with @jit(nopython=True), if compilation proceeds
        # without error, it means nopython mode was successful.
        try:
            print(f"numba_betaln: {numba_betaln.signatures if hasattr(numba_betaln, 'signatures') else 'Not a Numba JIT function or no signatures yet'}")
            print(f"log_binom_pmf: {log_binom_pmf.signatures if hasattr(log_binom_pmf, 'signatures') else 'Not a Numba JIT function or no signatures yet'}")
            print(f"_standard_loglik: {_standard_loglik.signatures if hasattr(_standard_loglik, 'signatures') else 'Not a Numba JIT function or no signatures yet'}")
            print(f"_corrected_loglik_terms: {_corrected_loglik_terms.signatures if hasattr(_corrected_loglik_terms, 'signatures') else 'Not a Numba JIT function or no signatures yet'}")
            print("If the above show compiled signatures, it implies successful JIT compilation.")
            print("For functions defined with @jit(nopython=True), successful compilation means nopython mode was achieved.")
        except Exception as e:
            print(f"Could not inspect Numba function signatures: {e}")
    else:
        print("senspy.models not available, cannot inspect Numba functions.")
