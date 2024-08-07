import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self,library_len, selection_length, num_networks_2, num_networks_3):
        self.library_len = library_len
        self.selection_length = selection_length
        self.num_networks_2 = num_networks_2
        self.num_networks_3 = num_networks_3
        
        self.net1 = nn.Sequential(
            nn.Linear(self.library_len,4),
            nn.Tanh(),
            nn.Linear(4,1),
        )
        
        self.net2 = nn.Sequential(
            nn.Linear(self.selection_length+1+1,4),
            nn.Tanh(),
            nn.Linear(4,1),
        )

        self.net3 = nn.Sequential(
            nn.Linear(self.selection_length+1,4),
            nn.Tanh(),
            nn.Linear(4,1),
        )


    def forward(self,x):        
        '''
        forward pass for a single sample:
        Input: torch Tensor of all selection arrays for that sample.
        Output: product of probabilites.
        '''
        
        lib_selection_arr = x[:self.library_len]
        
        #output, grad from network of type 1.
        prob, grad_1 = output_grad_computation(lib_selection_arr)
        
        new_ind = len(lib_selection_arr)
        for i in range(self.num_networks_2):
            selection_arr = x[new_ind : new_ind + self.selection_length+1]
            prob, grad_2 = output_grad_computation(torch.cat(selection_arr, prob))
            new_ind += self.selection_length+1
        
        for i in range(self.num_networks_3):
            selection_arr = x[new_ind : new_ind + self.selection_length]
            prob, grad_3 = output_grad_computation(torch.cat(selection_arr, prob))
            new_ind += self.selection_length
        
        
        
                
        
        
        
        
        
        
        # create a separate function that for any [101...], computes 
        # log(e^{y_{2}1}/(summation()))
        
        