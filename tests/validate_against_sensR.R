# Validation script to generate reference values from sensR
# Run this in R with sensR installed: Rscript validate_against_sensR.R

library(sensR)

cat("=== Double Link Functions Validation ===\n\n")

# Double 2-AFC
cat("# double_twoAFC\n")
d_values <- c(0, 0.5, 1, 2, 3)
for (d in d_values) {
    p <- psyfun(d, method = "twoAFC", double = TRUE)
    cat(sprintf("d=%.1f: p=%.8f\n", d, p))
}

# Double duo-trio
cat("\n# double_duotrio\n")
for (d in d_values) {
    p <- psyfun(d, method = "duotrio", double = TRUE)
    cat(sprintf("d=%.1f: p=%.8f\n", d, p))
}

# Double triangle
cat("\n# double_triangle\n")
for (d in d_values) {
    p <- psyfun(d, method = "triangle", double = TRUE)
    cat(sprintf("d=%.1f: p=%.8f\n", d, p))
}

# Double 3-AFC
cat("\n# double_threeAFC\n")
for (d in d_values) {
    p <- psyfun(d, method = "threeAFC", double = TRUE)
    cat(sprintf("d=%.1f: p=%.8f\n", d, p))
}

# Double tetrad
cat("\n# double_tetrad\n")
for (d in d_values) {
    p <- psyfun(d, method = "tetrad", double = TRUE)
    cat(sprintf("d=%.1f: p=%.8f\n", d, p))
}

cat("\n=== Simulation Functions Validation ===\n\n")

# discrimSim
cat("# discrimSim (set.seed(1))\n")
set.seed(1)
result <- discrimSim(sample.size = 10, replicates = 3, d.prime = 2,
                     method = "triangle", sd.indiv = 1)
cat("Result:", paste(result, collapse = ", "), "\n")

# samediffSim
cat("\n# samediffSim (set.seed(1))\n")
set.seed(1)
result <- samediffSim(n = 5, tau = 1, delta = 1, Ns = 10, Nd = 10)
print(result)

cat("\n=== Protocol Power Validation ===\n\n")

# twoACpwr
cat("# twoACpwr (exact)\n")
result <- twoACpwr(tau = 0.5, d.prime = 0.7, size = 50, tol = 0)
print(result)

cat("\n# twoACpwr (with tolerance)\n")
result <- twoACpwr(tau = 0.5, d.prime = 0.7, size = 50, tol = 1e-5)
print(result)

# samediffPwr
cat("\n# samediffPwr (set.seed(42))\n")
set.seed(42)
result <- samediffPwr(n = 100, tau = 1, delta = 2, Ns = 10, Nd = 10)
cat("Power:", result, "\n")

cat("\n=== Done ===\n")
