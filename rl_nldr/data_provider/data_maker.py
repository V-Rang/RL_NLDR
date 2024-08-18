import numpy as np
from .data_loader import create_input

def data_maker(setting, flag):
    '''
    Input: dict of parameters, flag for train/test data.
    
    Output: Torch dataset and dataloader.
    '''
    assert flag in ['train','test']
    
    data = np.load(setting['data_path'])
    train_cut = int(0.8*data.shape[1])
    data_train = data[:,:train_cut]
    data_test = data[:,train_cut:]
    
    if(flag == 'train'): 
        data_set, data_loader = create_input(len(setting['library_functions']),
                        setting['num_library_functions_select'],
                        setting['selection_length'],
                        setting['sub_selection_length'],
                        setting['trunc_dim'],
                        setting['num_samples_total'],
                        setting['num_samples_each_batch'])
        
        return data_train, data_set, data_loader 
    
    elif(flag == 'test'):
        return data_test, None, None


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
    U,_,_ = np.linalg.svd(S - S_ref,full_matrices=False)
    U_trunc = U[:,:trunc_dim]
    S_hat = U_trunc.T @ (S - S_ref)
    return U_trunc,S_ref,S_hat