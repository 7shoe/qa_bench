#install.packages("SSLASSO")
library(SSLASSO)

# Define the adaptive SSLASSO function
adaptive_SSLASSO <- function(X, y) {
    # Adaptive SSLASSO with unknown variance
    adaptive_ssLASSO <- SSLASSO(X, y, variance = "unknown")
    
    # no of simulations
    n_sim = ncol(adaptive_ssLASSO$intercept)
    
    # intercept
    alpha_0 <- adaptive_ssLASSO$intercept[n_sim]
    
    # var coefficients
    alpha   <- as.numeric(adaptive_ssLASSO$beta[,n_sim])
    
    # alpha
    #alpha <- c(alpha_0, alpha)
    
    return(alpha)
}

# Check if command-line arguments are provided
if (length(commandArgs(TRUE)) >= 2) {
  # Convert the command-line arguments to matrices and vectors
  X <- as.matrix(read.table(commandArgs(TRUE)[1]))
  y <- as.vector(read.table(commandArgs(TRUE)[2]))
  
  # convert 
  y <- as.numeric(unlist(y))
  
  # Call the adaptive_SSLASSO function with the provided inputs
  adaptive_SSLASSO(X, y)
} else {
  cat("Usage: Rscript adaptive_sslasso_script.R <X_file> <y_file>\n")
}