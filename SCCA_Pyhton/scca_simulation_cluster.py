import os
import time
import warnings
import argparse

import numpy as np
import pandas as pd
from numpy.linalg import svd, norm, solve
from scipy.linalg import svd
from scipy.stats import pearsonr, zscore
from sklearn.preprocessing import scale, StandardScaler
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from openpyxl import Workbook, load_workbook
warnings.filterwarnings('ignore', category=UserWarning)


def record_to_txt(output_dir, method, size, model, seed, cpu_time, clock_time, errA, errB):
    filename = f"{method}_model_{model}_seed_{seed}_size_{size}.txt"
    file_path = os.path.join(output_dir, filename)
    with open(file_path, "w") as file:
        file.write(f"Method: {method}\n")
        file.write(f"Size: {size}\n")
        file.write(f"Model: {model}\n")
        file.write(f"Seed: {seed}\n")
        file.write(f"CPU Time: {cpu_time:.4f}\n")
        file.write(f"Clock Time: {clock_time:.4f}\n")
        file.write(f"Error A: {errA:.6f}\n")
        file.write(f"Error B: {errB:.6f}\n")

    # Print the output in a readable format
    output_string = (
        f"{method:<10} {size:<6} {model:<6} {seed:<6} "
        f"{cpu_time:<10.4f} {clock_time:<10.4f} {errA:<10.6f} {errB:<10.6f}"
    )
    print(output_string)



def sweep(matrix, margin, stats, operation='/'):
    """
    Python equivalent of R's sweep function without relying on broadcasting.
    
    Parameters:
    - matrix: 2D NumPy array to be "swept".
    - margin: 0 for row-wise operation, 1 for column-wise operation.
    - stats: 1D NumPy array containing values to be used for the operation.
    - operation: The operation to perform; defaults to '/' (division).
    
    Returns:
    - The "swept" matrix after applying the operation.
    """
    result_matrix = np.empty_like(matrix)
    
    if margin == 1:  # Column-wise operation
        for i in range(matrix.shape[1]):  # Iterate over columns
            if operation == '/':
                result_matrix[:, i] = matrix[:, i] / stats[i]
            elif operation == '*':
                result_matrix[:, i] = matrix[:, i] * stats[i]
            # Add more operations as needed
    elif margin == 0:  # Row-wise operation
        for i in range(matrix.shape[0]):  # Iterate over rows
            if operation == '/':
                result_matrix[i, :] = matrix[i, :] / stats[i]
            elif operation == '*':
                result_matrix[i, :] = matrix[i, :] * stats[i] 
            # Add more operations as needed
    else:
        raise ValueError("Unsupported margin. Use 0 for row-wise or 1 for column-wise operations.")
    return result_matrix

def find_Omega(sigma_YX_hat, npairs, alpha=None, beta=None, y=None, x=None):
    n = y.shape[0]  # Equivalent to nrow in R
    if npairs > 1:
        rho = alpha.T @ sigma_YX_hat @ beta
        omega = np.eye(n) - y @ alpha @ rho @ beta.T @ x.T / n
    else:
        omega = np.eye(n)
    return omega

def init0(sigma_YX_hat, sigma_X_hat, sigma_Y_hat, init_method, npairs, n, d=None):
    
    '''
     The function init0 finds the initial value when no canonical pairs have been obtained. If init.method="sparse", 
     only one pair of initial value will be returned. For other options of init.method, the number of pairs of initial
     values can be specified with the argument npairs.
     
     sigma.YX.hat: Estimated cross-covariance matrix between Y and X.
     sigma.X.hat: Estimated covariance matrix of X.
     sigma.Y.hat: Estimated covariance matrix of Y.
     init.method: A string indicating which initialization method to use.
     npairs: Number of initial pairs needed.
     n: Sample size (used in sparse initialization).
     d: A threshold parameter used in sparse initialization (optional).
     
    '''
    q, p = sigma_Y_hat.shape[1], sigma_X_hat.shape[1]

    if init_method == 'svd':
        u, _, v = np.linalg.svd(sigma_YX_hat, full_matrices=False)
        alpha_init = u[:, :npairs]
        beta_init = v.T[:, :npairs]
    
    
    if init_method == 'uniform':
        alpha_init = np.ones((q, npairs))
        beta_init = np.ones((p, npairs))
    
    if init_method == 'random':
        alpha_init = np.random.normal(size=(q, npairs))
        beta_init = np.random.normal(size=(p, npairs))

    if init_method == 'sparse':
        if d is None:
            d = int(np.sqrt(n))
        thresh = np.sort(np.abs(sigma_YX_hat).ravel())[::-1][d - 1]
        row_max = np.max(np.abs(sigma_YX_hat), axis=1)
        col_max = np.max(np.abs(sigma_YX_hat), axis=0)
        
        selected_rows = np.where(row_max > thresh)[0]
        selected_cols = np.where(col_max > thresh)[0]
        reduced_sigma_YX_hat = sigma_YX_hat[np.ix_(selected_rows, selected_cols)]
        
        u, _, v = np.linalg.svd(reduced_sigma_YX_hat, full_matrices=False)
        alpha1_init = np.zeros(q)
        beta1_init = np.zeros(p)
        alpha1_init[selected_rows] = u[:, 0]
        beta1_init[selected_cols] = v.T[:, 0]

        alpha_init = np.zeros((q, 1))
        beta_init = np.zeros((p, 1))
        alpha_init[:, 0] = alpha1_init
        beta_init[:, 0] = beta1_init

    # Scaling alpha_init and beta_init
    alpha_scale = np.diag(alpha_init.T @ sigma_Y_hat @ alpha_init)
    alpha_init = sweep(alpha_init, margin=1, stats=np.sqrt(alpha_scale), operation='/')
    #alpha_init = alpha_init / np.sqrt(alpha_scale)[:, np.newaxis]
    beta_scale = np.diag(beta_init.T @ sigma_X_hat @ beta_init)
    beta_init = sweep(beta_init, margin=1, stats=np.sqrt(beta_scale), operation='/')
    #beta_init = beta_init / np.sqrt(beta_scale)[:, np.newaxis]
    return {'alpha_init': alpha_init, 'beta_init': beta_init}

def init1(sigma_YX_hat, sigma_X_hat, sigma_Y_hat, init_method, npairs, npairs0, alpha_current, beta_current, n, eps=1e-4, d=None):

    #print("Noticing intering initialization for init1 !!! Noticing intering initialization for init1 !!! Noticing intering initialization for init1 !!!")
    p = sigma_X_hat.shape[1]
    q = sigma_Y_hat.shape[1]
    alpha_init = np.zeros((q, 1))
    beta_init = np.zeros((p, 1))
    alpha_current = np.array(alpha_current)
    beta_current = np.array(beta_current)

    npairs0 = alpha_current.shape[1]

    if init_method == "uniform":
        alpha_init = np.ones((q, 1))
        beta_init = np.ones((p, 1))

    if init_method == "random":
        alpha_init = np.random.normal(size=(q * npairs)).reshape(q, 1)
        beta_init = np.random.normal(size=(p * npairs)).reshape(p, 1)

    if init_method == "svd":
        U, s, Vt = np.linalg.svd(sigma_YX_hat, full_matrices=False)
        alpha_init = U[:, npairs0][:, np.newaxis]  # np.newaxis to keep it 2D
        beta_init = Vt.T[:, npairs0][:, np.newaxis]

    if init_method == "sparse":
        # Identify non-zero indices
        id_nz_alpha = np.where(np.sum(np.abs(alpha_current), axis=1) > eps)[0]
        id_nz_beta = np.where(np.sum(np.abs(beta_current), axis=1) > eps)[0]

        # Compute rho.tmp and sigma.YX.tmp
        #print(f'the shape of alpha_current is {alpha_current.shape}')
        #print(f'the shape of sigma_YX_hat is {sigma_YX_hat.shape}')
        #print(f'the shape of beta_current is {beta_current .shape}')
        rho_tmp = alpha_current.T @ sigma_YX_hat @ beta_current
        sigma_YX_tmp = sigma_YX_hat - sigma_Y_hat @ alpha_current @ rho_tmp @ beta_current.T @ sigma_X_hat

        # Default d if missing
        if d is None:
            d = int(np.sqrt(n))

        # Thresholding
        thresh = np.sort(np.abs(sigma_YX_tmp).ravel())[-d]
        row_max = np.max(np.abs(sigma_YX_tmp), axis=1)
        col_max = np.max(np.abs(sigma_YX_tmp), axis=0)

        id_row = np.unique(np.concatenate((id_nz_alpha, np.where(row_max > thresh)[0])))
        id_col = np.unique(np.concatenate((id_nz_beta, np.where(col_max > thresh)[0])))

        # Perform SVD on the submatrix
        sigma_tmp = sigma_YX_tmp[np.ix_(id_row, id_col)]
        U, s, Vt = np.linalg.svd(sigma_tmp, full_matrices=False)

        # Update alpha.init and beta.init
        alpha_init[id_row] = U[:, 0][:, np.newaxis]
        beta_init[id_col] = Vt.T[:, 0][:, np.newaxis]

        # Normalization (Python equivalent of R's sweep operation)
        alpha_scale = np.sqrt((alpha_init.T @ sigma_Y_hat @ alpha_init).item())
        alpha_init /= alpha_scale
        beta_scale = np.sqrt((beta_init.T @ sigma_X_hat @ beta_init).item())
        beta_init /= beta_scale

    return {'alpha_init': alpha_init, 'beta_init': beta_init}

#Old Fashion Way

def SCCA_solution(x, y, x_Omega, y_Omega, alpha0, beta0, standardize, lambda_alpha, lambda_beta, niter=100, eps=1e-4, column_index=0):
    n = x.shape[0]
    q = y.shape[1]
    p = x.shape[1]
    
    if standardize:
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        x = scaler_x.fit_transform(x)
        y = scaler_y.fit_transform(y)
     
    # Select the appropriate column from alpha0 and beta0
    if alpha0.ndim > 1 and alpha0.shape[1] > 1:
        alpha0 = alpha0[:, column_index]
    if beta0.ndim > 1 and beta0.shape[1] > 1:
        beta0 = beta0[:, column_index]
        
    for i in range(niter):
        #beta0 = beta0[:, :1] 
        x0 = x_Omega @ beta0
        lasso_alpha = Lasso(alpha=lambda_alpha, fit_intercept=False)
        #print(f"Type of lambda_alpha: {type(lambda_alpha)}")
        #print(f"Type of lambda_beta: {type(lambda_beta)}")
        
        # Ensure lambda_alpha and lambda_beta are floats
        if isinstance(lambda_alpha, list):
            lambda_alpha = lambda_alpha[0]
        if isinstance(lambda_beta, list):
            lambda_beta = lambda_beta[0]
        lasso_alpha.fit(y, x0)
        alpha1 = lasso_alpha.coef_
        
        # Debugging prints after Lasso fit
        #print(f"alpha1 shape after Lasso fit: {alpha1.shape}")
        if np.sum(np.abs(alpha1)) < eps:
            alpha0 = np.zeros(q)
            break
        
        # Create the mask
        alpha1_mask = np.abs(alpha1) > eps
        #print(f"Shape of alpha1_mask: {alpha1_mask.shape}")
        
        # Conditional statement to handle different shapes of alpha1
        if alpha1.ndim > 1 and alpha1.shape[1] > 1:
            print(f"Unexpected number of columns in alpha1: {alpha1.shape[1]}")
            print(f"The dimension of alpha1 is: {alpha1.shape}")
            # Handle the case where alpha1 has more than one column
            # For debugging purposes, we can select the first column
            alpha1_mask = np.abs(alpha1[:, 0]) > eps
            alpha1_scale = y[:, alpha1_mask] @ alpha1[:, 0][alpha1_mask]
        else:
            alpha1_scale = y[:, alpha1_mask] @ alpha1[alpha1_mask]
            
            
        alpha1 /= np.sqrt(alpha1_scale.T @ alpha1_scale / (n - 1))
        
        y0 = y_Omega @ alpha1
        lasso_beta = Lasso(alpha=lambda_beta, fit_intercept=False)
        lasso_beta.fit(x, y0)
        beta1 = lasso_beta.coef_
        
        if np.sum(np.abs(beta1)) < eps:
            beta0 = np.zeros(p)
            break
        
        beta1_scale = x[:, np.abs(beta1) > eps] @ beta1[np.abs(beta1) > eps]
        beta1 /= np.sqrt(beta1_scale.T @ beta1_scale / (n - 1))
        
        if np.sum(np.abs(alpha1 - alpha0)) < eps and np.sum(np.abs(beta1 - beta0)) < eps:
            break
        
        alpha0 = alpha1
        beta0 = beta1
    
    return {"alpha": alpha0, "beta": beta0, "niter": i+1}

def SCCA(x, y, lambda_alpha, lambda_beta, alpha_init=None, beta_init=None, niter=1000, npairs=1, init_method="sparse", alpha_current=None, beta_current=None, standardize=True, eps=1e-4):
    p = x.shape[1]
    q = y.shape[1]
    n = x.shape[0]
    
    #print(f'the value of p  is {p}')
    x = scale(x, with_mean=True, with_std=standardize)
    y = scale(y, with_mean=True, with_std=standardize)

    #sigma_YX_hat = np.cov(y, x, rowvar=False)
    sigma_YX_hat =  np.cov(y.T, x.T)[:q, q:] 
    sigma_X_hat = np.cov(x, rowvar=False)
    sigma_Y_hat = np.cov(y, rowvar=False)
    #print(f'the shape of sigma_YX_hat is {sigma_YX_hat.shape}')
    alpha = np.zeros((q, npairs))
    beta = np.zeros((p, npairs))
    rho = np.zeros((npairs, npairs))

    if isinstance(init_method, list):
        init_method = init_method[0]

    if alpha_current is not None:
        npairs0 = alpha_current.shape[1]
        alpha[:, :npairs0] = alpha_current
        beta[:,  :npairs0] = beta_current
    else:
        npairs0 = 0

    if alpha_init is None:
        if alpha_current is None:
            obj_init = init0(sigma_YX_hat, sigma_X_hat, sigma_Y_hat, init_method=init_method, npairs=npairs, n=n)
            alpha_init = obj_init['alpha_init']
            beta_init = obj_init['beta_init']
        else:
            alpha_current = np.asmatrix(alpha_current)
            beta_current  = np.asmatrix(beta_current)
            obj_init = init1(sigma_YX_hat=sigma_YX_hat, sigma_X_hat=sigma_X_hat, sigma_Y_hat=sigma_Y_hat,
                             init_method=init_method, npairs=npairs, npairs0=npairs0, alpha_current=alpha_current,
                             beta_current=beta_current, n=n, eps=eps)
            alpha_init =  obj_init['alpha_init']
            beta_init  =  obj_init['beta_init']
            alpha_current = np.array(alpha_current)
            beta_current  = np.array(beta_current)

    n_iter_converge = np.zeros(npairs - npairs0)

    for ipairs in range(npairs0, npairs):
        #print(f"Processing pair {ipairs+1} of {npairs}")  
        alpha_init = np.array(alpha_init)
        beta_init  = np.array(beta_init)

        omega = find_Omega(sigma_YX_hat, ipairs, alpha=alpha[:, :ipairs], beta=beta[:, :ipairs], y=y, x=x)

        x_tmp = omega.dot(x)
        y_tmp = omega.T.dot(y)
        #print(f"the value of lambda_alpha is {lambda_alpha}")
        #lambda_alpha0 = lambda_alpha[ipairs - npairs0]
        #lambda_beta0 = lambda_beta[ipairs - npairs0]
        try:
            lambda_alpha0 = lambda_alpha[ipairs - npairs0]
        except IndexError:  # Caught if lambda_alpha is not subscriptable
            lambda_alpha0 = lambda_alpha
        try:
            lambda_beta0 = lambda_alpha[ipairs - npairs0]
        except IndexError:  # Caught if lambda_alpha is not subscriptable
            lambda_beta0 = lambda_alpha

        alpha0 = alpha_init
        beta0 = beta_init

        # Placeholder for SCCA_solution function
        obj = SCCA_solution(x=x, y=y, x_Omega=x_tmp, y_Omega=y_tmp, alpha0=alpha0, beta0=beta0, lambda_alpha=lambda_alpha0,
                            lambda_beta=lambda_beta0, niter=niter,eps=eps, standardize = False, column_index=ipairs)

        alpha[:, ipairs] = obj['alpha'].flatten()
        beta[:, ipairs] = obj['beta'].flatten()
        n_iter_converge[ipairs - npairs0] = obj['niter']

        if ipairs < (npairs - 1) and init_method == "sparse":
            # Placeholder for init1 function call
            #print(f'the shape of sigma_YX_hat is {sigma_YX_hat.shape}')
            obj_init = init1(sigma_YX_hat, sigma_X_hat, sigma_Y_hat, init_method=init_method,
                             npairs=npairs, npairs0=ipairs, alpha_current=alpha[:, :ipairs+1], beta_current=beta[:, :ipairs+1],n=n )
            alpha_init = obj_init['alpha_init']
            beta_init = obj_init['beta_init']

    return {
        'alpha': alpha,
        'beta': beta,
        'alpha_init': alpha_init,
        'beta_init': beta_init,
        'n_iter_converge': n_iter_converge
    }



def load_data(model, seed, input_dir, size):
#    print(f"The paramter input are model {model}, seed {seed},size {size} ")
    if model in range(0, 5):
        x_train_path = os.path.join(input_dir, f'X_train_model_{model}_{size}_seed_{seed}.csv')
        y_train_path = os.path.join(input_dir, f'Y_train_model_{model}_{size}_seed_{seed}.csv')
        x_tune_path = os.path.join(input_dir,  f'X_tune_model_{model}_{size}_seed_{seed}.csv')
        y_tune_path = os.path.join(input_dir,  f'Y_tune_model_{model}_{size}_seed_{seed}.csv')
    else:
        x_train_path = os.path.join(input_dir, f'X_train_model_{model}_{size}_seed_{seed}.csv')
        y_train_path = os.path.join(input_dir, f'Y_train_model_{model}_{size}_seed_{seed}.csv')
        x_tune_path = os.path.join(input_dir, f'X_tune_model_{model}_{size}_seed_{seed}.csv')
        y_tune_path = os.path.join(input_dir, f'Y_tune_model_{model}_{size}_seed_{seed}.csv')
    
    X_train = pd.read_csv(x_train_path, header=0).values
    Y_train = pd.read_csv(y_train_path, header=0).values
    X_tune = pd.read_csv(x_tune_path, header=0).values
    Y_tune = pd.read_csv(y_tune_path, header=0).values



    return X_train, Y_train, X_tune, Y_tune

def construct_covariance_matrices(model, p, q):
    
    def construct_AR_covariance_matrix(size, rho):
        covariance_matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                covariance_matrix[i,  j] = rho ** abs(i - j)
        return covariance_matrix

    def construct_sparse_precision_matrix(size):
        omega = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if i == j:
                    omega[i, j] = 1
                elif abs(i - j) == 1:
                    omega[i, j] = 0.5
                elif abs(i - j) == 2:
                    omega[i, j] = 0.4
        return omega
    
    def construct_CS_covariance_matrix(size, rho):
        covariance_matrix = np.full((size, size), rho)
        np.fill_diagonal(covariance_matrix, 1)
        return covariance_matrix

    if model == 1:
        sigma_X = np.eye(p)
        sigma_Y = np.eye(q)
    elif model == 0:
        sigma_X = construct_AR_covariance_matrix(p, 0.6)
        sigma_Y = construct_AR_covariance_matrix(p, 0.6)
    elif model == 2:
        sigma_X = construct_AR_covariance_matrix(p, 0.3)
        sigma_Y = np.eye(q)
    elif model == 3:
        sigma_X = construct_AR_covariance_matrix(p, 0.8)
        sigma_Y = sigma_X
    elif model == 4:
        omega = construct_sparse_precision_matrix(p)
        sigma_X = np.linalg.inv(omega)
        lambda_ = np.diag(np.sqrt(np.diag(sigma_X)))
        sigma_X = lambda_ @ sigma_X @ lambda_
        sigma_Y = sigma_X
    elif model == 5:
        sigma_X = np.eye(p)
        sigma_Y = np.eye(q)
    elif model == 6:
        sigma_X = construct_AR_covariance_matrix(p, 0.5)
        sigma_Y = sigma_X
    elif model == 7:
        sigma_X = construct_AR_covariance_matrix(p, 0.5)
        sigma_Y = sigma_X
    elif model == 8:
        sigma_X = construct_CS_covariance_matrix(p, 0.5)
        sigma_Y = sigma_X
    else:
        raise ValueError("Invalid model number. Choose a model number between 1 and 8.")

    return sigma_X, sigma_Y

def create_true_canonical_directions(model,  Normalization_Factor, size=None):
    if model in [0]:
        A_true = np.zeros((200, 1))
        B_true = np.zeros((200, 1))

        indices = [0,1,2,3,4,5,6,7]
        values_1 = [1,1,1,1,1,1,1,1]

        A_true[indices, 0] = values_1
        B_true[indices, 0] = values_1
        # Normalize the directions

    
    elif model in [1, 2, 3, 4]:
        A_true = np.zeros((300, 2))
        B_true = np.zeros((300, 2))

        indices = [0, 5, 10, 15, 20]
        values_1 = [-2, -1, -1, 2, 2]
        values_2 = [0, 0, 0, 1, 1]

        A_true[indices, 0] = values_1
        A_true[indices, 1] = values_2
        B_true[indices, 0] = values_1
        B_true[indices, 1] = values_2
        # Normalize the directions


    elif model == 5:
        if size == '10H':
            p = 1000
        elif size == '12H':
            p = 1200
        elif size == '15H':
            p = 1500  
        elif size == '20H':
            p = 2000     
        else:
            raise ValueError("Invalid size parameter")
        
        A_true = np.zeros((p, 1))
        B_true = np.zeros((p, 1))
        indices = [0, 1, 2, 3]
        values = [1, 1, 1, 1]
        A_true[indices, 0] = values
        B_true[indices, 0] = values


    elif model == 6:
        if size == '10H':
            p = 1000
        elif size == '12H':
            p = 1200
        elif size == '15H':
            p = 1500
        elif size == '20H':
            p = 2000 
        else:
            raise ValueError("Invalid size parameter")
        
        A_true = np.zeros((p, 1))
        B_true = np.zeros((p, 1))
        indices = [0, 1, 2, 3, 4, 5, 6, 7]
        values = [1, 1, 1, 1, 1, 1, 1, 1]
        A_true[indices, 0] = values
        B_true[indices, 0] = values


    elif model in [7, 8]:
        if size == '10H':
            p = 1000
        elif size == '12H':
            p = 1200
        elif size == '15H':
            p = 1500
        elif size == '20H':
            p = 2000 
        else:
            raise ValueError("Invalid size parameter")
        
        A_true = np.zeros((p, 2))
        B_true = np.zeros((p, 2))
        indices_1 = [0, 1, 2, 3]
        indices_2 = [50, 51, 52, 53] 
        values_1 = [1, 1, 1, 1]
        values_2 = [1, 1, 1, 1]
        A_true[indices_1, 0] = values_1
        A_true[indices_2, 1] = values_2
        B_true[indices_1, 0] = values_1
        B_true[indices_2, 1] = values_2

    # Normalizing columns of theta and eta
    for i in range(A_true.shape[1]):
        A_true[:, i] /= np.sqrt(A_true[:, i].T @ Normalization_Factor @ A_true[:, i])
        B_true[:, i] /= np.sqrt(B_true[:, i].T @ Normalization_Factor @ B_true[:, i])
    return A_true, B_true

def run_scca_experiment(model, seed, input_dir, size_key, output_dir):
    sizes = ['3H', '10H', '12H', '15H', '20H']
    
    if model in [0, 1, 2, 3, 4]:
        size = sizes[size_key - 1]
    elif model in [ 5, 6, 7, 8]:
        size = sizes[size_key]
    else:
        raise ValueError("Invalid model number")
        
    # Load training and tuning data
    x_train, y_train, x_tune, y_tune = load_data(model, seed, input_dir, size)
    #print(f'we are currently under the model {model} and seed {seed}')
    
    # Set the dimensions
    p = x_train.shape[1]
    q = y_train.shape[1]
    n = x_train.shape[0]

    #dynamic adjust parameter "npair"
    # Determine the value of n_pairs based on the model
    if model in [ 1, 2, 3, 4]:
        n_pairs = 2
    elif model in [0, 5, 6, 7, 8]:
        n_pairs = 1
    else:
        raise ValueError("Invalid model number")

    # Compute covariance matrices
    sigma_Y_hat = np.cov(y_train, rowvar=False)
    sigma_X_hat = np.cov(x_train, rowvar=False)
    sigma_YX_hat = np.cov(y_train, x_train, rowvar=False)[:y_train.shape[1], x_train.shape[1]:]

    # Define lambda values for regularization and initialization methods
    lambdas = np.arange(1, 41) / 100
    init_options = ["sparse", "svd", "sparse", "svd"]

    # Initialize the matrix to store correlation results for each lambda and init method
    rho_lambda = np.zeros((len(lambdas), len(init_options)))

    start_cpu = time.process_time()
    start_clock = time.time()
    
    # Loop through each initialization method
    for i_init, init_opt in enumerate(init_options):
        # Initial estimation of alpha and beta
        init_results = init0(sigma_YX_hat, sigma_X_hat, sigma_Y_hat, init_opt, n_pairs, n)
        alpha_init = init_results["alpha_init"]
        beta_init = init_results["beta_init"]

        # Loop through each lambda value
        for i_lambda, lambda_val in enumerate(lambdas):
            lambda_alpha = lambda_val
            lambda_beta = lambda_val

            # Perform SCCA with current lambda and init method
            result = SCCA(x_train, y_train, niter=100, lambda_alpha=lambda_alpha, lambda_beta=lambda_beta, npairs=n_pairs, alpha_init=alpha_init, beta_init=beta_init, init_method=init_opt, standardize=True)
            alpha_hat = result['alpha']
            beta_hat = result['beta']
            #print(f"The current i_lambda is {i_lambda}")
            # Compute the projections for the tuning data
            x_beta = x_tune @ beta_hat[:, 0]
            y_alpha = y_tune @ alpha_hat[:, 0]

            # Check for constant projections
            if np.all(x_beta == x_beta[0]):
                rho = 0  # Handle constant input case
            elif np.all(y_alpha == y_alpha[0]):
                rho = 0  # Handle constant input case
            else:
                # Compute the correlation between the projections
                corr_matrix = np.corrcoef(x_beta, y_alpha)
                rho = corr_matrix[0, 1] ** 2

            # Store the correlation result
            rho_lambda[i_lambda, i_init] = rho

            # Break the loop if the correlation is too small
            if rho_lambda[i_lambda, i_init] < 1e-6:
                break

    # Find the best lambda and initialization method
    cpu_time = time.process_time() - start_cpu
    clock_time = time.time() - start_clock
    
    id_best = np.unravel_index(rho_lambda.argmax(), rho_lambda.shape)
    best_lambda = lambdas[id_best[0]]
    best_init = init_options[id_best[1]]

    # Perform SCCA with the best lambda and initialization method
    best_result = SCCA(x_train, y_train, best_lambda, best_lambda, niter=1000, npairs=n_pairs, init_method=best_init)
    alpha_hat = best_result['alpha']
    beta_hat = best_result['beta']
   
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths for saving the results
    if size:
        alpha_path = os.path.join(output_dir, f'method_model_{model}_{size}_seed_{seed}_A_est.csv')
        beta_path = os.path.join(output_dir, f'method_model_{model}_{size}_seed_{seed}_B_est.csv')
    else:
        alpha_path = os.path.join(output_dir, f'method_model_{model}_seed_{seed}_A_est.csv')
        beta_path = os.path.join(output_dir, f'method_model_{model}_seed_{seed}_B_est.csv')
    
    # Save the estimated alpha and beta matrices
    pd.DataFrame(alpha_hat).to_csv(alpha_path, index=False)
    pd.DataFrame(beta_hat).to_csv(beta_path, index=False)
    
    # Generate the true canonical directions for the given model
    SigmaX, SigmaY = construct_covariance_matrices(model, p, q)
    theta, eta = create_true_canonical_directions(model, SigmaX, size)
    
    # Calculate norm differences between the estimated and true canonical directions
    alpha_norm_diff= np.sqrt(np.sum((alpha_hat @ np.linalg.inv(alpha_hat.T @ alpha_hat) @ alpha_hat.T - theta @ np.linalg.inv(theta.T @ theta) @ theta.T) ** 2))
    beta_norm_diff= np.sqrt(np.sum((beta_hat @ np.linalg.inv(beta_hat.T @ beta_hat) @ beta_hat.T - eta @ np.linalg.inv(eta.T @ eta) @ eta.T) ** 2))

   # Call the new function to record the results in a .txt file
    record_to_txt(output_dir, 'SCCA_python', size, model, seed, cpu_time, clock_time, alpha_norm_diff, beta_norm_diff)

    return alpha_norm_diff, beta_norm_diff ,cpu_time, clock_time



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Run SCCA experiment with specified parameters.')
#     parser.add_argument('--a', type=int, required=True, help='Value for parameter a (model)')
#     parser.add_argument('--b', type=int, required=True, help='Value for parameter b (seed)')
#     parser.add_argument('--c', type=int, required=True, help='Value for parameter c (size index)')
    
#     args = parser.parse_args()

#     input_dir = '/scratch/rw3496/SCCA_data/Data_Model_1-8_large'
#     output_dir = '/scratch/rw3496/SCCA_data/scca_out'
    
#     run_scca_experiment(model=args.a, seed=args.b, input_dir=input_dir, size_key = args.c, output_dir = output_dir)


if __name__ == "__main__":
    # Comment out the original parser code
    # parser = argparse.ArgumentParser(description='Run SCCA experiment with specified parameters.')
    # parser.add_argument('--a', type=int, required=True, help='Value for parameter a (model)')
    # parser.add_argument('--b', type=int, required=True, help='Value for parameter b (seed)')
    # parser.add_argument('--c', type=int, required=True, help='Value for parameter c (size index)')
    # args = parser.parse_args()
    input_dir = '/Users/oujakusui/Desktop/SCCA_python_r/Simulations/DATA/Data_Model_1-8_large'
    output_dir = '/Users/oujakusui/Desktop/SCCA_python_r/Simulations/DATA/Data_Model_1-8_small/result'

    # Define a simple class or object to hold the arguments
    class Args:
        a = 0  # Model (e.g., 1)
        b = 58  # Seed (e.g., 42)
        c = 1   # Size index (e.g., 1)

    args = Args()

    input_dir = ''
    output_dir = ''
    
    run_scca_experiment(model=args.a, seed=args.b, input_dir=input_dir, size_key=args.c, output_dir=output_dir)


