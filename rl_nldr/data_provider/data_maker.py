import numpy as np

def create_S_hat(S: np.array ,trunc_dim:int) -> tuple[np.array, np.array, np.array]:
    '''
    Given matrix 'S' and SVD truncation dimension 'trunc_dim' 
    return arrays U_trunc, S_ref and S_hat in the expression:
    svd(S -S_ref) ~ U_trunc@S_hat 
    '''
    k = S.shape[1]
    S_ref = np.zeros(shape = (S.shape[0],k))
    for i in range(k):
        S_ref[:,i] = S[:,0]
    U,_,_ = np.linalg.svd(S - S_ref)
    U_trunc = U[:,:trunc_dim]
    S_hat = U_trunc.T @ (S - S_ref)
    return U_trunc,S_ref,S_hat


def apply_selected_funcs(S_hat, lib_funcs, sel_arr):
    '''
    Select functions from 'lib_funcs' based on indices in 'sel_arr' that have 1. Pass S_hat
    through those functions to get S_mod.
    '''
    selected_funcs = [eval(f'lambda _: {func}') for func, sel in zip(lib_funcs, sel_arr) if sel == 1]
    results = [func(S_hat) for func in selected_funcs]
    return np.concatenate(results,axis=0)
