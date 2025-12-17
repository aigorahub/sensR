# Generate golden test data from sensR for Python validation
# Run with: Rscript generate_golden_data.R

library(sensR)
library(jsonlite)

cat("Generating golden data from sensR...\n")

# Helper to get guessing probability for a method
get_pguess <- function(method) {
  switch(tolower(method),
    "duotrio" = 0.5,
    "triangle" = 1/3,
    "twoafc" = 0.5,
    "threeafc" = 1/3,
    "tetrad" = 1/3,
    "hexad" = 0.1,
    "twofive" = 0.1,
    "twofivef" = 0.4
  )
}

# =============================================================================
# Link Functions (psyfun, psyinv, psyderiv)
# =============================================================================

methods <- c("duotrio", "triangle", "twoAFC", "threeAFC",
             "tetrad", "hexad", "twofive", "twofiveF")

d_prime_values <- c(0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)

links_data <- list(
  psy_fun = list(),
  psy_inv = list(),
  psy_deriv = list()
)

for (method in methods) {
  method_lower <- tolower(method)

  # psyfun: d-prime -> pc
  pc_values <- sapply(d_prime_values, function(d) psyfun(d, method = method))
  links_data$psy_fun[[method_lower]] <- list(
    d_prime = d_prime_values,
    expected_pc = as.numeric(pc_values)
  )

  # psyinv: pc -> d-prime (use valid pc values above chance)
  p_guess <- get_pguess(method)
  pc_for_inv <- seq(p_guess + 0.05, 0.95, by = 0.1)
  d_prime_from_pc <- sapply(pc_for_inv, function(p) psyinv(p, method = method))
  links_data$psy_inv[[method_lower]] <- list(
    pc = as.numeric(pc_for_inv),
    expected_d_prime = as.numeric(d_prime_from_pc)
  )

  # psyderiv: derivative at various d-prime values
  d_for_deriv <- c(0.5, 1.0, 1.5, 2.0, 2.5)
  deriv_values <- sapply(d_for_deriv, function(d) psyderiv(d, method = method))
  links_data$psy_deriv[[method_lower]] <- list(
    d_prime = d_for_deriv,
    expected_deriv = as.numeric(deriv_values)
  )

  cat(sprintf("  %s: done\n", method))
}

# =============================================================================
# discrim() function
# =============================================================================

discrim_data <- list()

# Test cases: various correct/total combinations
test_cases <- list(
  list(correct = 80, total = 100, method = "triangle"),
  list(correct = 60, total = 100, method = "triangle"),
  list(correct = 40, total = 100, method = "triangle"),
  list(correct = 75, total = 100, method = "twoAFC"),
  list(correct = 60, total = 100, method = "twoAFC"),
  list(correct = 70, total = 100, method = "duotrio"),
  list(correct = 50, total = 100, method = "threeAFC"),
  list(correct = 65, total = 100, method = "tetrad")
)

discrim_results <- list()

for (i in seq_along(test_cases)) {
  tc <- test_cases[[i]]

  # Run discrim with exact statistic
  res_exact <- discrim(correct = tc$correct, total = tc$total,
                       method = tc$method, statistic = "exact")

  # Run discrim with other statistics
  res_wald <- discrim(correct = tc$correct, total = tc$total,
                      method = tc$method, statistic = "Wald")
  res_likelihood <- discrim(correct = tc$correct, total = tc$total,
                            method = tc$method, statistic = "likelihood")
  res_score <- discrim(correct = tc$correct, total = tc$total,
                       method = tc$method, statistic = "score")

  # Extract coefficients
  coef_exact <- res_exact$coefficients

  result <- list(
    input = list(
      correct = tc$correct,
      total = tc$total,
      method = tolower(tc$method)
    ),
    estimates = list(
      pc = coef_exact[1, 1],
      pd = coef_exact[2, 1],
      d_prime = coef_exact[3, 1]
    ),
    std_errors = list(
      pc = coef_exact[1, 2],
      pd = coef_exact[2, 2],
      d_prime = coef_exact[3, 2]
    ),
    confidence_intervals = list(
      pc = c(coef_exact[1, 3], coef_exact[1, 4]),
      pd = c(coef_exact[2, 3], coef_exact[2, 4]),
      d_prime = c(coef_exact[3, 3], coef_exact[3, 4])
    ),
    p_values = list(
      exact = res_exact$p.value,
      wald = res_wald$p.value,
      likelihood = res_likelihood$p.value,
      score = res_score$p.value
    )
  )

  discrim_results[[i]] <- result
  cat(sprintf("  discrim case %d (%d/%d %s): done\n",
              i, tc$correct, tc$total, tc$method))
}

discrim_data$test_cases <- discrim_results

# =============================================================================
# rescale() function
# =============================================================================

rescale_data <- list()

rescale_cases <- list(
  list(pc = 0.6, method = "triangle"),
  list(pc = 0.8, method = "triangle"),
  list(pc = 0.7, method = "twoAFC"),
  list(pd = 0.5, method = "triangle"),
  list(d.prime = 1.5, method = "triangle"),
  list(d.prime = 2.0, method = "twoAFC")
)

rescale_results <- list()

for (i in seq_along(rescale_cases)) {
  rc <- rescale_cases[[i]]

  if (!is.null(rc$pc)) {
    res <- rescale(pc = rc$pc, method = rc$method)
    input_type <- "pc"
    input_value <- rc$pc
  } else if (!is.null(rc$pd)) {
    res <- rescale(pd = rc$pd, method = rc$method)
    input_type <- "pd"
    input_value <- rc$pd
  } else {
    res <- rescale(d.prime = rc$d.prime, method = rc$method)
    input_type <- "d_prime"
    input_value <- rc$d.prime
  }

  coef <- res$coefficients

  rescale_results[[i]] <- list(
    input = list(
      type = input_type,
      value = input_value,
      method = tolower(rc$method)
    ),
    output = list(
      pc = coef$pc,
      pd = coef$pd,
      d_prime = coef$d.prime
    )
  )

  cat(sprintf("  rescale case %d: done\n", i))
}

rescale_data$test_cases <- rescale_results

# =============================================================================
# Power functions (discrimPwr, d.primePwr)
# =============================================================================

cat("Generating power data...\n")

power_data <- list()

# discrimPwr test cases
discrim_pwr_cases <- list(
  list(pdA = 0.3, sample.size = 100, pGuess = 1/3, test = "difference"),
  list(pdA = 0.4, sample.size = 50, pGuess = 1/2, test = "difference"),
  list(pdA = 0.25, sample.size = 200, pGuess = 1/3, test = "difference"),
  list(pdA = 0.1, sample.size = 100, pGuess = 1/3, pd0 = 0.3, test = "similarity")
)

discrim_pwr_results <- list()
for (i in seq_along(discrim_pwr_cases)) {
  tc <- discrim_pwr_cases[[i]]
  pd0 <- if (is.null(tc$pd0)) 0 else tc$pd0

  pwr_exact <- discrimPwr(pdA = tc$pdA, pd0 = pd0, sample.size = tc$sample.size,
                          pGuess = tc$pGuess, test = tc$test, statistic = "exact")
  pwr_normal <- discrimPwr(pdA = tc$pdA, pd0 = pd0, sample.size = tc$sample.size,
                           pGuess = tc$pGuess, test = tc$test, statistic = "normal")
  pwr_cont <- discrimPwr(pdA = tc$pdA, pd0 = pd0, sample.size = tc$sample.size,
                         pGuess = tc$pGuess, test = tc$test, statistic = "cont.normal")

  discrim_pwr_results[[i]] <- list(
    input = list(
      pd_a = tc$pdA,
      pd_0 = pd0,
      sample_size = tc$sample.size,
      p_guess = tc$pGuess,
      test = tc$test
    ),
    power = list(
      exact = as.numeric(pwr_exact),
      normal = as.numeric(pwr_normal),
      cont_normal = as.numeric(pwr_cont)
    )
  )
  cat(sprintf("  discrimPwr case %d: done\n", i))
}
power_data$discrim_power <- discrim_pwr_results

# d.primePwr test cases
dprime_pwr_cases <- list(
  list(d.primeA = 1.5, sample.size = 100, method = "triangle", test = "difference"),
  list(d.primeA = 1.0, sample.size = 50, method = "twoAFC", test = "difference"),
  list(d.primeA = 2.0, sample.size = 80, method = "threeAFC", test = "difference"),
  list(d.primeA = 1.0, sample.size = 100, method = "duotrio", test = "difference")
)

dprime_pwr_results <- list()
for (i in seq_along(dprime_pwr_cases)) {
  tc <- dprime_pwr_cases[[i]]

  pwr_exact <- d.primePwr(d.primeA = tc$d.primeA, sample.size = tc$sample.size,
                          method = tc$method, test = tc$test, statistic = "exact")
  pwr_normal <- d.primePwr(d.primeA = tc$d.primeA, sample.size = tc$sample.size,
                           method = tc$method, test = tc$test, statistic = "normal")

  dprime_pwr_results[[i]] <- list(
    input = list(
      d_prime_a = tc$d.primeA,
      d_prime_0 = 0,
      sample_size = tc$sample.size,
      method = tolower(tc$method),
      test = tc$test
    ),
    power = list(
      exact = as.numeric(pwr_exact),
      normal = as.numeric(pwr_normal)
    )
  )
  cat(sprintf("  d.primePwr case %d: done\n", i))
}
power_data$dprime_power <- dprime_pwr_results

# =============================================================================
# Sample size functions (discrimSS, d.primeSS)
# =============================================================================

cat("Generating sample size data...\n")

sample_size_data <- list()

# discrimSS test cases
discrim_ss_cases <- list(
  list(pdA = 0.3, pGuess = 1/3, target.power = 0.9, test = "difference"),
  list(pdA = 0.4, pGuess = 1/2, target.power = 0.8, test = "difference"),
  list(pdA = 0.25, pGuess = 1/3, target.power = 0.9, test = "difference")
)

discrim_ss_results <- list()
for (i in seq_along(discrim_ss_cases)) {
  tc <- discrim_ss_cases[[i]]

  ss_exact <- discrimSS(pdA = tc$pdA, pGuess = tc$pGuess,
                        target.power = tc$target.power, test = tc$test,
                        statistic = "exact")
  ss_normal <- discrimSS(pdA = tc$pdA, pGuess = tc$pGuess,
                         target.power = tc$target.power, test = tc$test,
                         statistic = "normal")
  ss_cont <- discrimSS(pdA = tc$pdA, pGuess = tc$pGuess,
                       target.power = tc$target.power, test = tc$test,
                       statistic = "cont.normal")

  discrim_ss_results[[i]] <- list(
    input = list(
      pd_a = tc$pdA,
      pd_0 = 0,
      target_power = tc$target.power,
      p_guess = tc$pGuess,
      test = tc$test
    ),
    sample_size = list(
      exact = as.integer(ss_exact),
      normal = as.integer(ss_normal),
      cont_normal = as.integer(ss_cont)
    )
  )
  cat(sprintf("  discrimSS case %d: done\n", i))
}
sample_size_data$discrim_sample_size <- discrim_ss_results

# d.primeSS test cases
dprime_ss_cases <- list(
  list(d.primeA = 1.5, method = "triangle", target.power = 0.9, test = "difference"),
  list(d.primeA = 1.0, method = "twoAFC", target.power = 0.8, test = "difference"),
  list(d.primeA = 1.0, method = "triangle", target.power = 0.9, test = "difference")
)

dprime_ss_results <- list()
for (i in seq_along(dprime_ss_cases)) {
  tc <- dprime_ss_cases[[i]]

  ss_exact <- d.primeSS(d.primeA = tc$d.primeA, method = tc$method,
                        target.power = tc$target.power, test = tc$test,
                        statistic = "exact")
  ss_normal <- d.primeSS(d.primeA = tc$d.primeA, method = tc$method,
                         target.power = tc$target.power, test = tc$test,
                         statistic = "normal")

  dprime_ss_results[[i]] <- list(
    input = list(
      d_prime_a = tc$d.primeA,
      d_prime_0 = 0,
      target_power = tc$target.power,
      method = tolower(tc$method),
      test = tc$test
    ),
    sample_size = list(
      exact = as.integer(ss_exact),
      normal = as.integer(ss_normal)
    )
  )
  cat(sprintf("  d.primeSS case %d: done\n", i))
}
sample_size_data$dprime_sample_size <- dprime_ss_results

# =============================================================================
# Beta-binomial models (betabin)
# =============================================================================

cat("Generating beta-binomial data...\n")

betabin_data <- list()

# Test data from sensR documentation
x <- c(3, 2, 6, 8, 3, 4, 6, 0, 9, 9, 0, 2, 1, 2, 8, 9, 5, 7)
n <- c(10, 9, 8, 9, 8, 6, 9, 10, 10, 10, 9, 9, 10, 10, 10, 10, 9, 10)
dat <- data.frame(x, n)

# Chance-corrected duotrio
bb_corr_duotrio <- betabin(dat, method = "duotrio", corrected = TRUE)
summ_corr_duotrio <- summary(bb_corr_duotrio)

betabin_data$corrected_duotrio <- list(
  input = list(
    x = x,
    n = n,
    method = "duotrio",
    corrected = TRUE
  ),
  coefficients = list(
    mu = as.numeric(coef(bb_corr_duotrio)["mu"]),
    gamma = as.numeric(coef(bb_corr_duotrio)["gamma"])
  ),
  log_likelihood = as.numeric(logLik(bb_corr_duotrio)),
  summary = list(
    pc = summ_corr_duotrio$coefficients["pc", "Estimate"],
    pd = summ_corr_duotrio$coefficients["pd", "Estimate"],
    d_prime = summ_corr_duotrio$coefficients["d-prime", "Estimate"],
    se_mu = summ_corr_duotrio$coefficients["mu", "Std. Error"],
    se_gamma = summ_corr_duotrio$coefficients["gamma", "Std. Error"]
  ),
  lr_overdispersion = list(
    g2 = summ_corr_duotrio$LR.OD,
    p_value = summ_corr_duotrio$p.value.OD
  ),
  lr_association = list(
    g2 = summ_corr_duotrio$LR.null,
    p_value = summ_corr_duotrio$p.value.null
  )
)
cat("  betabin corrected duotrio: done\n")

# Un-corrected duotrio
bb_uncorr_duotrio <- betabin(dat, method = "duotrio", corrected = FALSE)
summ_uncorr_duotrio <- summary(bb_uncorr_duotrio)

betabin_data$uncorrected_duotrio <- list(
  input = list(
    x = x,
    n = n,
    method = "duotrio",
    corrected = FALSE
  ),
  coefficients = list(
    mu = as.numeric(coef(bb_uncorr_duotrio)["mu"]),
    gamma = as.numeric(coef(bb_uncorr_duotrio)["gamma"])
  ),
  log_likelihood = as.numeric(logLik(bb_uncorr_duotrio))
)
cat("  betabin uncorrected duotrio: done\n")

# Corrected triangle
bb_corr_triangle <- betabin(dat, method = "triangle", corrected = TRUE)
summ_corr_triangle <- summary(bb_corr_triangle)

betabin_data$corrected_triangle <- list(
  input = list(
    x = x,
    n = n,
    method = "triangle",
    corrected = TRUE
  ),
  coefficients = list(
    mu = as.numeric(coef(bb_corr_triangle)["mu"]),
    gamma = as.numeric(coef(bb_corr_triangle)["gamma"])
  ),
  log_likelihood = as.numeric(logLik(bb_corr_triangle)),
  summary = list(
    pc = summ_corr_triangle$coefficients["pc", "Estimate"],
    pd = summ_corr_triangle$coefficients["pd", "Estimate"],
    d_prime = summ_corr_triangle$coefficients["d-prime", "Estimate"]
  )
)
cat("  betabin corrected triangle: done\n")

# =============================================================================
# 2-AC models (twoAC)
# =============================================================================

cat("Generating 2-AC data...\n")

twoac_data <- list()

# Simple case from tests: [2, 2, 6]
res_simple <- twoAC(c(2, 2, 6))
vcov_simple <- vcov(res_simple)
se_simple <- sqrt(diag(vcov_simple))
# Access coefficients matrix directly
coef_simple <- res_simple$coefficients

twoac_data$simple_case <- list(
  input = list(
    data = c(2, 2, 6)
  ),
  tau = as.numeric(coef_simple["tau", "Estimate"]),
  d_prime = as.numeric(coef_simple["d.prime", "Estimate"]),
  se_tau = as.numeric(coef_simple["tau", "Std. Error"]),
  se_d_prime = as.numeric(coef_simple["d.prime", "Std. Error"]),
  log_likelihood = as.numeric(logLik(res_simple)),
  vcov = as.matrix(vcov_simple)
)
cat("  twoAC simple case: done\n")

# Larger sample case with likelihood test
res_large <- twoAC(c(15, 15, 20), statistic = "likelihood")
coef_large <- res_large$coefficients
ci_large <- confint(res_large)

twoac_data$large_case <- list(
  input = list(
    data = c(15, 15, 20),
    statistic = "likelihood"
  ),
  tau = as.numeric(coef_large["tau", "Estimate"]),
  d_prime = as.numeric(coef_large["d.prime", "Estimate"]),
  se_tau = as.numeric(coef_large["tau", "Std. Error"]),
  se_d_prime = as.numeric(coef_large["d.prime", "Std. Error"]),
  p_value = as.numeric(res_large$p.value),
  confint_dprime = as.numeric(ci_large["d.prime", ])
)
cat("  twoAC large case: done\n")

# Wald statistic case
res_wald <- twoAC(c(15, 15, 20), statistic = "Wald")
coef_wald <- res_wald$coefficients
ci_wald <- confint(res_wald)

twoac_data$wald_case <- list(
  input = list(
    data = c(15, 15, 20),
    statistic = "Wald"
  ),
  tau = as.numeric(coef_wald["tau", "Estimate"]),
  d_prime = as.numeric(coef_wald["d.prime", "Estimate"]),
  p_value = as.numeric(res_wald$p.value),
  confint_dprime = as.numeric(ci_wald["d.prime", ])
)
cat("  twoAC wald case: done\n")

# Negative d-prime case (prefer A > prefer B)
res_neg <- twoAC(c(6, 2, 2))
coef_neg <- res_neg$coefficients

twoac_data$negative_dprime <- list(
  input = list(
    data = c(6, 2, 2)
  ),
  tau = as.numeric(coef_neg["tau", "Estimate"]),
  d_prime = as.numeric(coef_neg["d.prime", "Estimate"])
)
cat("  twoAC negative d-prime: done\n")

# Non-zero d_prime_0 for similarity testing
res_sim <- twoAC(c(15, 15, 20), d.prime0 = 0.5, alternative = "less")
coef_sim <- res_sim$coefficients

twoac_data$similarity_test <- list(
  input = list(
    data = c(15, 15, 20),
    d_prime_0 = 0.5,
    alternative = "less"
  ),
  tau = as.numeric(coef_sim["tau", "Estimate"]),
  d_prime = as.numeric(coef_sim["d.prime", "Estimate"]),
  p_value = as.numeric(res_sim$p.value)
)
cat("  twoAC similarity test: done\n")

# =============================================================================
# Same-Different models (samediff)
# =============================================================================

cat("Generating Same-Different data...\n")

samediff_data <- list()

# Simple case from sensR documentation: samediff(8, 5, 4, 9)
res_simple <- samediff(8, 5, 4, 9)
vcov_simple <- vcov(res_simple)
se_simple <- sqrt(diag(vcov_simple))

samediff_data$simple_case <- list(
  input = list(
    ss = 8,
    ds = 5,
    sd = 4,
    dd = 9
  ),
  tau = as.numeric(coef(res_simple)["tau"]),
  delta = as.numeric(coef(res_simple)["delta"]),
  se_tau = as.numeric(se_simple["tau"]),
  se_delta = as.numeric(se_simple["delta"]),
  log_likelihood = as.numeric(logLik(res_simple)),
  vcov = as.matrix(vcov_simple)
)
cat("  samediff simple case: done\n")

# Boundary case: sd = 0 -> delta = Inf
res_sd_zero <- samediff(8, 5, 0, 9)
vcov_sd_zero <- vcov(res_sd_zero)

samediff_data$boundary_sd_zero <- list(
  input = list(
    ss = 8,
    ds = 5,
    sd = 0,
    dd = 9
  ),
  tau = as.numeric(coef(res_sd_zero)["tau"]),
  delta = as.numeric(coef(res_sd_zero)["delta"]),
  log_likelihood = as.numeric(logLik(res_sd_zero))
)
cat("  samediff boundary sd=0: done\n")

# Larger sample case
res_large <- samediff(80, 50, 40, 90)
vcov_large <- vcov(res_large)
se_large <- sqrt(diag(vcov_large))

samediff_data$large_case <- list(
  input = list(
    ss = 80,
    ds = 50,
    sd = 40,
    dd = 90
  ),
  tau = as.numeric(coef(res_large)["tau"]),
  delta = as.numeric(coef(res_large)["delta"]),
  se_tau = as.numeric(se_large["tau"]),
  se_delta = as.numeric(se_large["delta"]),
  log_likelihood = as.numeric(logLik(res_large))
)
cat("  samediff large case: done\n")

# =============================================================================
# Save to JSON
# =============================================================================

output <- list(
  metadata = list(
    generated_by = "sensR",
    sensR_version = as.character(packageVersion("sensR")),
    R_version = R.version.string,
    generated_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  ),
  links = links_data,
  discrim = discrim_data,
  rescale = rescale_data,
  power = power_data,
  sample_size = sample_size_data,
  betabin = betabin_data,
  twoac = twoac_data,
  samediff = samediff_data
)

output_path <- "golden_sensr.json"
write_json(output, output_path, pretty = TRUE, auto_unbox = TRUE, digits = 10)

cat(sprintf("\nGolden data written to: %s\n", output_path))
cat("Done!\n")
