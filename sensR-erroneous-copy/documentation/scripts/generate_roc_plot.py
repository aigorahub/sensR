import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import os

# Ensure the output directory exists
output_dir = "documentation/src/static/images"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "illustrative_roc_curves.png")

# Generate ROC curve data
false_alarm_rates = np.linspace(0, 1, 100)
d_primes = [0, 1, 2] # Example d-prime values

plt.figure(figsize=(6, 6))

# Plot chance line
plt.plot([0, 1], [0, 1], 'k--', label='Chance (d\'=0)')

for d_prime in d_primes:
    if d_prime == 0: # Already plotted as chance line
        continue
    # For a simple SDT model with equal variances:
    # z(FA) = norm.ppf(FA_rate)
    # z(Hit) = norm.ppf(Hit_rate)
    # d_prime = z(Hit) - z(FA)  =>  z(Hit) = z(FA) + d_prime
    # Hit_rate = norm.cdf(norm.ppf(FA_rate) + d_prime)
    
    # However, it's more direct to plot parametrically if you consider criteria
    # Or, to plot P(Hit) vs P(FA) for given dprime:
    # P(Hit) = norm.cdf(c + dprime/2)
    # P(FA) = norm.cdf(c - dprime/2) (if dprime is mean diff / common_sd)
    # Or, more commonly, using criteria:
    # Let criteria c vary from -inf to +inf
    # FA_rate = 1 - norm.cdf(c) = norm.cdf(-c)  => -c = norm.ppf(FA_rate) => c = -norm.ppf(FA_rate)
    # Hit_rate = 1 - norm.cdf(c - d_prime) = norm.cdf(-(c - d_prime)) = norm.cdf(d_prime - c)
    
    # Use a range of z(FA) values directly
    z_fa_values = norm.ppf(false_alarm_rates)
    # Clip z_fa_values to avoid inf at FA=0 and FA=1 if not handled by ppf
    z_fa_values = np.clip(z_fa_values, -5, 5) # Practical limits for z-scores
    
    hit_rates = norm.cdf(z_fa_values + d_prime)
    
    # Filter out NaNs that might arise from norm.ppf(0) or norm.ppf(1) before clipping
    valid_indices = ~np.isnan(false_alarm_rates) & ~np.isnan(hit_rates)
    
    plt.plot(false_alarm_rates[valid_indices], hit_rates[valid_indices], label=f'd\'={d_prime}')

plt.xlabel("False Alarm Rate (1 - Specificity)")
plt.ylabel("Hit Rate (Sensitivity)")
plt.title("Illustrative ROC Curves")
plt.legend(loc="lower right")
plt.grid(True, linestyle=':', alpha=0.7)
plt.axis('square') # Ensure a square plot for ROC
plt.xlim([0, 1])
plt.ylim([0, 1])

try:
    plt.savefig(output_path)
    print(f"Successfully saved ROC plot to {output_path}")
except Exception as e:
    print(f"Error saving ROC plot: {e}")

plt.close()
