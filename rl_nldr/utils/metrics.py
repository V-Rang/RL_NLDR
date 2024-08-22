import numpy as np
from rl_nldr.utils.utils import compute_reconstr_error, apply_selected_funcs

def training_errors(S, S_ref, U_trunc, S_hat, V_bar, S_nl):
    S_org = S - S_ref
    S_norm = np.linalg.norm(S - S_ref, 'fro')

    # linear error.    
    S_reconstr_linear = U_trunc@S_hat
    linear_error = np.linalg.norm(S_org - S_reconstr_linear, 'fro')/S_norm
    
    # non-linear error.
    S_reconstr_nl = S_reconstr_linear + V_bar@S_nl
    nonlinear_error = np.linalg.norm(S_org - S_reconstr_nl, 'fro')/S_norm

    return linear_error, nonlinear_error

def testing_errors(S, best_sample):
    k = S.shape[1]
    S_ref = np.zeros(shape = (S.shape[0],k))
    for i in range(k):
        S_ref[:,i] = S[:,0]
    
    S_org = S - S_ref
    S_norm = np.linalg.norm(S - S_ref, 'fro')
    
    U_trunc = best_sample['U_trunc']
    library_functions = best_sample['library_functions']
   
    V_bar = best_sample['V_bar']
    selection_arr =  best_sample['selection_arr']
    
    S_hat = U_trunc.T @ (S - S_ref)
         
    # linear error.
    S_reconstr_linear = U_trunc@S_hat
    linear_error = np.linalg.norm(S_org - S_reconstr_linear, 'fro')/S_norm
    
    #non-linear error.
    #pass S_hat through the selection_arr, using library_functions:
    S_mod = apply_selected_funcs(S_hat, library_functions, selection_arr[:len(library_functions)])
    S_nl = S_mod[selection_arr[len(library_functions):].astype('bool')]

    nonlinear_error = np.linalg.norm(S_org - S_nl, 'fro')/S_norm
    
    return linear_error, nonlinear_error
    
    




    
