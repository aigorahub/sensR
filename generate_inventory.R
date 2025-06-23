# Ensure sensR is loaded. If installing from local source in a fresh session:
print(getwd());

# Setup user library path
user_lib_path <- Sys.getenv("R_LIBS_USER")
if (user_lib_path == "") {
  # R_LIBS_USER is not set, use a default path in the user's home directory
  user_lib_path <- file.path(Sys.getenv("HOME"), "R", "library")
}

if (!dir.exists(user_lib_path)) {
  dir.create(user_lib_path, recursive = TRUE, showWarnings = FALSE)
  print(paste("Created user library directory:", user_lib_path))
}
.libPaths(c(user_lib_path, .libPaths()))
print(paste("Using .libPaths():", paste(.libPaths(), collapse=", ")))


if (!requireNamespace("sensR", quietly = TRUE)) {
  print("sensR not found, attempting to install from local source...");
  # Needed for dependencies of sensR that might not be on older R versions by default
  install.packages(c("MASS", "numDeriv", "multcomp", "mvtnorm", "survival", "TH.data", "xtable", "Matrix", "sandwich", "codetools", "zoo", "estimability"), repos="https://cloud.r-project.org/", Ncpus=4);
  # Assuming sensR source is in the working directory (e.g. /app/sensR)
  # The R working directory for Rscript run from /app should be /app
  install.packages("sensR", repos=NULL, type="source", INSTALL_opts="--no-lock");
} else {
  print("sensR already installed or found in lib paths.");
}
library(sensR);
print("sensR library loaded successfully.");

# Get function listing
funcs <- lsf.str("package:sensR")

# Create a data frame
inventory_df <- data.frame(
  function_name = as.character(funcs),
  file = rep("sensR/R/", length(funcs)), # Placeholder, actual file might vary
  description = rep("", length(funcs)),  # Placeholder
  ported_to_senspy = rep("No", length(funcs)), # Placeholder
  senspy_equivalent = rep("", length(funcs)), # Placeholder
  notes = rep("", length(funcs)) # Placeholder
)

# Write to CSV
write.csv(inventory_df, "functionality_inventory_new.csv", row.names = FALSE)
print("New inventory CSV created: functionality_inventory_new.csv");
