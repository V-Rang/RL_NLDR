import numpy as np
import random
import torch
import torch.nn as nn

def random_selection_arr_maker(k: int,l: int) -> np.array:
    '''
    Input:
    k -> Length of array needed.
    l -> Number of 1s in the array.
    Output:
    selection_arr: Array of length 'k' with 'l' 1s at randomly chosen positions.
    '''
    selection_arr = np.zeros(shape=(k,),dtype=int)
    indices_chosen = random.sample(range(len(selection_arr)), l  )
    
    for i in indices_chosen:
        selection_arr[i] =  1

    return selection_arr


import itertools

def output_grad_computation(network: nn.modules ,x: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
    '''
    Input:
    network -> neural network.
    x -> a Torch Tensor of 1s and 0s; input to network.
    
    Output:
    output of network for input x, grad computation using policy gradient. 
    '''
    def compute_gradients(network, input_tensor):
        output = network(input_tensor)
        output.backward()  
        gradients = {param_name: param.grad.clone() for param_name, param in network.named_parameters()}
        network.zero_grad()  
        return gradients

    accumulated_gradients = {param_name: torch.zeros_like(param) for param_name, param in network.named_parameters()}

    for combination in combinations:
        # combination = combination.unsqueeze(0) 
        gradients = compute_gradients(network, combination)
        for name, grad in gradients.items():
            accumulated_gradients[name] += grad

        # if(combination == x):
            
    # test_inp  = torch.tensor([0,1,0])
    combinations = list(set(itertools.permutations(x.tolist())))
    combinations = [torch.tensor(p) for p in combinations]

    # for combination in combinations:
    #     output = network(combination)
            

    

