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
  rescale = rescale_data
)

output_path <- "golden_sensr.json"
write_json(output, output_path, pretty = TRUE, auto_unbox = TRUE)

cat(sprintf("\nGolden data written to: %s\n", output_path))
cat("Done!\n")
