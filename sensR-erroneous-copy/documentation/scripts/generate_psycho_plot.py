import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add the project root directory to sys.path to allow importing senspy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from senspy.links import psyfun
except ImportError:
    print("Error: senspy.links.psyfun could not be imported. Ensure senspy is installed or path is correct.")
    # Define a fallback if senspy is not available, to allow script to run partially for structure check
    from scipy.stats import norm
    def psyfun(dprime, method="2afc"): # Fallback to 2afc only
        if method.lower() == "2afc":
            return norm.cdf(dprime / np.sqrt(2.0))
        else:
            # For other methods, we can't easily replicate without the full senspy.discrimination
            # This fallback will only correctly plot for 2AFC.
            print(f"Warning: Fallback psyfun used, only '2afc' method is accurately plotted.")
            if method.lower() != "2afc": return np.full_like(dprime, np.nan) # Return NaN for other methods
            return norm.cdf(dprime / np.sqrt(2.0))


# Ensure the output directory exists
output_dir = "documentation/src/static/images"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "illustrative_psychometric_function_2afc.png")

# Generate psychometric function data for 2AFC
d_primes = np.linspace(0, 3, 100)
pc_values_2afc = psyfun(d_primes, method="2afc") # Use the imported or fallback psyfun

plt.figure(figsize=(7, 5))
plt.plot(d_primes, pc_values_2afc, label='2AFC Psychometric Function')

plt.xlabel("d-prime (d')")
plt.ylabel("Proportion Correct (Pc)")
plt.title("Illustrative Psychometric Function (2AFC)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.ylim([0.45, 1.05]) # Start y-axis near 0.5 (chance for 2AFC)
plt.xlim([0, 3])

# Add a horizontal line for chance performance (pguess for 2AFC is 0.5)
plt.axhline(y=0.5, color='grey', linestyle='--', label='Chance (Pc=0.5)')
# Re-call legend to include the axhline label if not automatically picked up (depends on matplotlib version)
handles, labels = plt.gca().get_legend_handles_labels()
# Check if 'Chance (Pc=0.5)' is already there to avoid duplicate labels
if not any('Chance (Pc=0.5)' in label for label in labels):
    # Find a way to add it or just rely on the visual cue if this gets complex
    pass # Simpler to just let the line be there without explicit legend entry if it's tricky

plt.legend() # Ensure legend is displayed

try:
    plt.savefig(output_path)
    print(f"Successfully saved psychometric function plot to {output_path}")
except Exception as e:
    print(f"Error saving psychometric function plot: {e}")

plt.close()
