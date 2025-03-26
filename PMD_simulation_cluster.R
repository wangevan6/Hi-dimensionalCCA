# Install required packages if not already installed
if (!requireNamespace("optparse", quietly = TRUE)) {
  install.packages("optparse")
}

if (!requireNamespace("PMA", quietly = TRUE)) {
  install.packages("PMA")
}
# Load required libraries
library(PMA)
library(optparse)
#library(openxlsx)
options(warn = -1)


record_to_txt <- function(output_path, method, size, model, seed, cpu_time, clock_time, errA, errB) {
  filename <- sprintf("%s_model_%d_seed_%d_size_%s.txt", method, model, seed, size)
  file_path <- file.path(output_path, filename)
  
  # Write the information to a text file
  fileConn <- file(file_path)
  writeLines(c(
    sprintf("Method: %s", method),
    sprintf("Size: %s", size),
    sprintf("Seed: %d", seed),
    sprintf("CPU Time: %.4f", cpu_time),
    sprintf("Clock Time: %.4f", clock_time),
    sprintf("Error A: %.6f", errA),
    sprintf("Error B: %.6f", errB)
  ), fileConn)
  close(fileConn)
  
  # Print the output in a readable format
  sink(file = "combined_output.txt", append = TRUE)
  output_string <- sprintf("%-10s %-6s %-6d %-6d %-10.4f %-10.4f %-10.6f %-10.6f",
                           method, size, model, seed, cpu_time, clock_time, errA, errB)
  print(output_string)
  sink()
}




# Function to load data
load_data <- function(model, seed, input_path, size=NULL) {
  x_train_path <- sprintf('%s/X_train_model_%d_%s_seed_%d.csv', input_path, model, size, seed)
  y_train_path <- sprintf('%s/Y_train_model_%d_%s_seed_%d.csv', input_path, model, size, seed)
  x_tune_path <- sprintf('%s/X_tune_model_%d_%s_seed_%d.csv', input_path, model, size, seed)
  y_tune_path <- sprintf('%s/Y_tune_model_%d_%s_seed_%d.csv', input_path, model, size, seed)
  
  x_train <- as.matrix(read.csv(x_train_path))
  y_train <- as.matrix(read.csv(y_train_path))
  x_tune <- as.matrix(read.csv(x_tune_path))
  y_tune <- as.matrix(read.csv(y_tune_path))
  
  list(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune)
}

# Function to construct covariance matrices
construct_covariance_matrices <- function(model, p, q) {  
  construct_AR_covariance_matrix <- function(size, rho) {
    covariance_matrix <- matrix(0, nrow=size, ncol=size)
    for (i in 1:size) {
      for (j in 1:size) {
        covariance_matrix[i, j] <- rho ^ abs(i - j)
      }
    }
    covariance_matrix
  }
  
  construct_sparse_precision_matrix <- function(size) {
    omega <- matrix(0, nrow=size, ncol=size)
    for (i in 1:size) {
      for (j in 1:size) {
        if (i == j) {
          omega[i, j] <- 1
        } else if (abs(i - j) == 1) {
          omega[i, j] <- 0.5
        } else if (abs(i - j) == 2) {
          omega[i, j] <- 0.4
        }
      }
    }
    omega
  }
  
  construct_CS_covariance_matrix <- function(size, rho) {
    covariance_matrix <- matrix(rho, nrow=size, ncol=size)
    diag(covariance_matrix) <- 1
    covariance_matrix
  }
  
  if (model == 1) {
    sigma_X <- diag(p)
    sigma_Y <- diag(q)
  } else if (model == 0) {
    sigma_X <- construct_AR_covariance_matrix(p, 0.6)
    sigma_Y <- construct_AR_covariance_matrix(p, 0.6)
  } else if (model == 2) {
    sigma_X <- construct_AR_covariance_matrix(p, 0.3)
    sigma_Y <- diag(q)
  } else if (model == 3) {
    sigma_X <- construct_AR_covariance_matrix(p, 0.8)
    sigma_Y <- sigma_X
  } else if (model == 4) {
    omega <- construct_sparse_precision_matrix(p)
    sigma_X <- solve(omega)
    lambda <- diag(sqrt(diag(sigma_X)))
    sigma_X <- lambda %*% sigma_X %*% lambda
    sigma_Y <- sigma_X
  } else if (model == 5) {
    sigma_X <- diag(p)
    sigma_Y <- diag(q)
  } else if (model == 6) {
    sigma_X <- construct_AR_covariance_matrix(p, 0.5)
    sigma_Y <- sigma_X
  } else if (model == 7) {
    sigma_X <- construct_AR_covariance_matrix(p, 0.5)
    sigma_Y <- sigma_X
  } else if (model == 8) {
    sigma_X <- construct_CS_covariance_matrix(p, 0.5)
    sigma_Y <- sigma_X
  } else {
    stop("Invalid model number. Choose a model number between 1 and 8.")
  }
  
  list(sigma_X = sigma_X, sigma_Y = sigma_Y)
}

# Function to create true canonical directions
create_true_canonical_directions <- function(model, Normalization_Factor, size=NULL) {
  if (model == 0) {
    A_true <- matrix(0, nrow=200, ncol=1)
    B_true <- matrix(0, nrow=200, ncol=1)
    indices <- 1:8
    values_1 <- rep(1, 8)
    A_true[indices, 1] <- values_1
    B_true[indices, 1] <- values_1
  } else if (model %in% 1:4) {
    A_true <- matrix(0, nrow=300, ncol=2)
    B_true <- matrix(0, nrow=300, ncol=2)
    indices <- c(1, 6, 11, 16, 21)
    values_1 <- c(-2, -1, -1, 2, 2)
    values_2 <- c(0, 0, 0, 1, 1)
    A_true[indices, 1] <- values_1
    A_true[indices, 2] <- values_2
    B_true[indices, 1] <- values_1
    B_true[indices, 2] <- values_2
  } else if (model == 5) {
    p <- switch(size,
                '10H' = 1000,
                '12H' = 1200,
                '15H' = 1500,
                '20H' = 2000,
                stop("Invalid size parameter"))
    A_true <- matrix(0, nrow=p, ncol=1)
    B_true <- matrix(0, nrow=p, ncol=1)
    indices <- 1:4
    values <- rep(1, 4)
    A_true[indices, 1] <- values
    B_true[indices, 1] <- values
  } else if (model == 6) {
    p <- switch(size,
                '10H' = 1000,
                '12H' = 1200,
                '15H' = 1500,
                '20H' = 2000,
                stop("Invalid size parameter"))
    A_true <- matrix(0, nrow=p, ncol=1)
    B_true <- matrix(0, nrow=p, ncol=1)
    indices <- 1:8
    values <- rep(1, 8)
    A_true[indices, 1] <- values
    B_true[indices, 1] <- values
  } else if (model %in% 7:8) {
    p <- switch(size,
                '10H' = 1000,
                '12H' = 1200,
                '15H' = 1500,
                '20H' = 2000,
                stop("Invalid size parameter"))
    A_true <- matrix(0, nrow=p, ncol=2)
    B_true <- matrix(0, nrow=p, ncol=2)
    indices_1 <- 1:4
    indices_2 <- 51:54
    values_1 <- rep(1, 4)
    values_2 <- rep(1, 4)
    A_true[indices_1, 1] <- values_1
    A_true[indices_2, 2] <- values_2
    B_true[indices_1, 1] <- values_1
    B_true[indices_2, 2] <- values_2
  } else {
    stop("Invalid model number")
  }
  
  for (i in 1:ncol(A_true)) {
    A_true[, i] <- A_true[, i] / sqrt(t(A_true[, i]) %*% Normalization_Factor %*% A_true[, i])
    B_true[, i] <- B_true[, i] / sqrt(t(B_true[, i]) %*% Normalization_Factor %*% B_true[, i])
  }
  
  list(A_true = A_true, B_true = B_true)
}

# Function to record results to Excel
#record_to_excel <- function(sheet, method, size, model, seed, cpu_time, clock_time, errA, errB) {
#  data_to_write <- data.frame(Method = method, Size = size, Model = model, Seed = seed, CPU_Time = cpu_time, Clock_Time = clock_time, ErrA = errA, ErrB = errB)
#  writeData(sheet, sheet = "Results", x = data_to_write, startRow = nrow(read.xlsx(sheet, sheet = "Results")) + 1, colNames = FALSE, rowNames = FALSE)
#}


run_PMD <- function(model, seed, input_path, output_path, size_key) {

    sizes <- c('3H', '10H', '12H', '15H', '20H')
    if (model %in% c(0, 1, 2, 3, 4)) {
      size <- sizes[size_key]
    } else if (model %in% c(5, 6, 7, 8)) {
      size <- sizes[size_key + 1]
    } else {
      stop("Invalid model number")
    }
    cat(sprintf("Processing model %d, seed %d\n", model, seed))
    data <- load_data(model, seed, input_path, size)
    x_train <- data$x_train
    y_train <- data$y_train
    x_tune <- data$x_tune
    y_tune <- data$y_tune
    
    cat("Data loaded successfully\n")
    
    # Center the data
    x_train <- sweep(x_train, 2, colMeans(x_train))
    y_train <- sweep(y_train, 2, colMeans(y_train))
    
    # Combine the data for PMD
    combined_train <- cbind(x_train, y_train)
    combined_tune <- cbind(x_tune, y_tune)
    
    # Cross-validation to find optimal sumabs
    start_cpu <- proc.time()
    start_clock <- Sys.time()
    #sink(tempfile())
    
    cv_out <- PMA::CCA.permute(x = x_tune, z = y_tune, typex = "standard", typez = "standard", nperms = 25, standardize = TRUE)
    #sink()  # Stop redirecting output
    
    cat("Cross-validation completed\n")
    
    # Set the number of pairs based on the model number
    if (model %in% c(1, 2, 3, 4)) {
      n_pairs <- 2
    } else if (model %in% c(0, 5, 6, 7, 8)) {
      n_pairs <- 1
    } else {
      stop("Invalid model number")
    }
    # Fit the PMD model using optimal sumabs
    pmd_out <- PMA::CCA(x = x_train, z = y_train, typex = "standard", typez = "standard", K = n_pairs, penaltyx = cv_out$bestpenaltyx, penaltyz = cv_out$bestpenaltyz)
    cat("PMD fitting completed\n")
    
    # Extract the estimated components
    alpha_hat <- pmd_out$u
    beta_hat <- pmd_out$v
    
    end_cpu <- proc.time()
    end_clock <- Sys.time()
    
    # Calculate CPU and clock time
    cpu_time <- (end_cpu - start_cpu)["elapsed"]
    clock_time <- as.numeric(difftime(end_clock, start_clock, units = "secs"))
    
    # Save the estimated components
    #write.csv(alpha_hat, file = sprintf('%s/Alpha_hat_model_%d_%s_seed_%d.csv', output_path, model, size, seed ), row.names = FALSE)
    #write.csv(beta_hat,  file = sprintf(' %s/Beta_hat_model_%d_%s_seed_%d.csv', output_path, model, size, seed), row.names = FALSE)
    write.csv(alpha_hat, file = sprintf('%s/Alpha_hat_model_%d_seed_%d.csv', output_path, model, seed), row.names = FALSE)
    write.csv(beta_hat, file = sprintf('%s/Beta_hat_model_%d_seed_%d.csv', output_path, model, seed), row.names = FALSE)
    
    
    cat("Components saved\n")
    
    
    # Placeholder for error calculations
    errA <- NA
    errB <- NA
    
    # Set dimensions based on the model
    p <- ncol(x_train)
    q <- ncol(y_train)
    n <- row(x_train)
    
    # Construct covariance matrices
    sigma_x <- construct_covariance_matrices(model, p, q)$sigma_X
    sigma_y <- construct_covariance_matrices(model, p, q)$sigma_Y
    
    # Create true canonical directions
    true_directions <- create_true_canonical_directions(model, sigma_y, size)
    theta <- true_directions$A_true
    eta <- true_directions$B_true
    
    errA <- sum((alpha_hat %*% solve(t(alpha_hat) %*% alpha_hat) %*% t(alpha_hat) - theta %*% solve(t(theta) %*% theta) %*% t(theta))^2)^0.5
    errB <- sum((beta_hat %*% solve(t(beta_hat) %*% beta_hat) %*% t(beta_hat) - eta %*% solve(t(eta) %*% eta) %*% t(eta))^2)^0.5
    
    print(paste("Error A (errA) calculated:", errA, "- Error B (errB) calculated:", errB))
    
    # Record results in Excel
    # record_to_excel(sheet, "PMD", size, model, seed, cpu_time, clock_time, errA, errB)
    record_to_txt(output_path, "PMD", size, model, seed, cpu_time, clock_time, errA, errB)
}


# Main function to parse command-line arguments and initiate the process
    # Corrected main function to parse command-line arguments and initiate the process
    main <- function() {
      option_list <- list(
        make_option(c("--a"), type="integer", default=NULL, help="Value for parameter a (model)", metavar="integer"),
        make_option(c("--b"), type="integer", default=NULL, help="Value for parameter b (seed)", metavar="integer"),
        make_option(c("--c"), type="integer", default=NULL, help="Value for parameter c (size index)", metavar="integer")
      )
      
      opt_parser <- OptionParser(option_list=option_list)
      opt <- parse_args(opt_parser)
      
      if (is.null(opt$a) || is.null(opt$b) || is.null(opt$c)) {
        print_help(opt_parser)
        stop("All three parameters (a, b, c) must be supplied.", call.=FALSE)
      }
      
      # Define input and output directories
       input_path <- "/Users/oujakusui/Desktop/SCCA_python_r/Simulations/DATA/Data_Model_1-8_large"
       output_path <- "/Users/oujakusui/Desktop/SCCA_python_r/Cluster_Files/PMD/Results"
      # input_path <- '/scratch/rw3496/SCCA_data/Data_Model_1-8_large'
      # output_path <- '/scratch/rw3496/PMD/Results'
      
      # Run the PMD process
      run_PMD(opt$a, opt$b,  input_path, output_path, opt$c  )
    }
    
    # Run the main function
    main()


