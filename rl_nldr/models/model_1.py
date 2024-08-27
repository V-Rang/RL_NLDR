import torch
import torch.nn as nn
import numpy as np
import itertools
from typing import Tuple
                    
class Model(nn.Module):
    def __init__(self,library_len, num_library_fns, org_dim, selection_length):
        super(Model, self).__init__()
        
        self.library_len = library_len
        self.num_library_fns = num_library_fns
        self.selection_length = selection_length
        
        self.num_networks_2 = (org_dim*self.num_library_fns)%selection_length
        self.num_networks_3 = (org_dim*self.num_library_fns) // selection_length - self.num_networks_2
                
        self.net1 = nn.Sequential(
            nn.Linear(self.library_len,4),
            nn.Tanh(),
            nn.Linear(4,1),
        )
        
        self.net2 = nn.Sequential(
            nn.Linear(self.selection_length+1+1,4), #self.selection_length + 1 (array) + 1 (probability from previous)
            nn.Tanh(),
            nn.Linear(4,1),
        )

        self.net3 = nn.Sequential(
            nn.Linear(self.selection_length+1,4),
            nn.Tanh(),
            nn.Linear(4,1),
        )
                    
    def _compute_lib_output_grad(self, library_selection_arr : torch.Tensor) -> dict:        
        '''
        For a given selection array of library functions, compute the 
        outputs and gradients of all combinations of the input selection array.
        
        Output: dict, key -> combination
                      value -> dict with 
                               key -> ['output', 'gradient']
                               value -> output and grad values.
        
        output -> e^{yi}/(sum(e^{yj}))
        gradient -> (d{yi}/dx) - (1/sum(e^{yj}))* ( sum(e^{yj}d{yj}/dx  )  )
        '''
        
        choices = int(library_selection_arr.sum().item())
        indices = [1 for i in range(choices)] + [0 for i in range(len(library_selection_arr) - choices)]
        combinations = list(set(itertools.permutations(indices)))
        
        output_grad_lib_array = {}
        output_sum = 0. #sum(e^y)
        grad_sum = [torch.zeros_like(param) for param in self.net1.parameters()] #sum(e^y.(dy/dx))
        
        for comb in combinations:
            inp = torch.tensor(comb,dtype = torch.float32, requires_grad=True)
            out = self.net1(inp)
            output_sum += np.exp(out.detach().numpy())
            grads = torch.autograd.grad(outputs=out, inputs=self.net1.parameters(), create_graph=True)
            for zp,gp in zip(grad_sum, grads):
                zp += out * gp
            
            output_grad_lib_array[comb] = {'output':np.exp(out.detach().numpy()[0]), 'gradients':grads}
                    
        for val in output_grad_lib_array.values():
            val['output'] /= output_sum[0]
            for zp, gp in zip(val['gradients'], grad_sum):
                zp -= (1/output_sum[0]) * gp
        
        return output_grad_lib_array
        

    def _compute_selarr_output_grad(self, inp_arr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Given a torch Tensor [probabilty, selection array], compute the probability (output) and the gradient
        wrt to the network parameters.
        '''
        
        # choice of network (self.net2 or self.net3) can be determined by length of selection_arr beyond the first 
        # element (probability from the previous step).
        networks = {
            'self.net2' : self.net2,
            'self.net3': self.net3
        }

        assert(len(inp_arr[1:]) == self.selection_length or len(inp_arr[1:]) == self.selection_length+1)
        
        if(len(inp_arr[1:]) == self.selection_length+1):
            network = networks['self.net2']
        else:
            network = networks['self.net3']

        # print(inp_arr)        
        probability = [inp_arr[0]]
        selection_arr = inp_arr[1:]

        choices = int(selection_arr.sum().item())
        indices = [1 for i in range(choices)] + [0 for i in range(len(selection_arr) - choices)]
        combinations = list(set(itertools.permutations(indices)))
        
        output_grad_array = {}
        output_sum = 0. #sum(e^y)
        grad_sum = [torch.zeros_like(param) for param in network.parameters()] #sum(e^y.(dy/dx))
        
        for comb in combinations:
            inp = torch.cat((torch.tensor(probability), torch.tensor(comb,dtype = torch.float32, requires_grad=True)))
            out = network(inp)
            output_sum += np.exp(out.detach().numpy())
            grads = torch.autograd.grad(outputs=out, inputs=network.parameters(), create_graph=True)
            for zp,gp in zip(grad_sum, grads):
                zp += out * gp
        
            output_grad_array[comb] = {'output':np.exp(out.detach().numpy()[0]), 'gradients':grads}
        
        inp_arr_key = tuple(selection_arr.to(torch.int).tolist())
        
        output_grad_array[inp_arr_key]['output'] /= output_sum[0]
        
        for zp, gp in zip(output_grad_array[inp_arr_key]['gradients'], grad_sum ):
            zp -= (1/output_sum[0]) * gp

        return output_grad_array[inp_arr_key]['output'], output_grad_array[inp_arr_key]['gradients']


    def forward(self, x):        
        '''
        forward pass for a single sample post computation of probability from library selection array.
        
        Input: x -> Dict:
                    probability_lib_selection : library selection probability (list with 1 value.),
                    selection_arr: selection array for S_mod (torch.tensor) 
        
        Output: Dict:
                    sample_probability_array: probability array for the given sample in x (incl. library selection).,
                    grads_network_2: gradients for network 2 (self.net2),
                    grads_network_3: gradients for network 3 (self.net3),
        '''
        
        probability_arr = x['probability_lib_selection']
        selection_arr =  x['selection_arr']
        grads_network_2, grads_network_3 = None, None
                
        if(self.num_networks_2 != 0):
            grads_network_2 = [torch.zeros_like(param) for param in self.net2.parameters()]

        if(self.num_networks_3 != 0):        
            grads_network_3 = [torch.zeros_like(param) for param in self.net3.parameters()]

        ind = 0
        for i in range(self.num_networks_2):
            inp = torch.cat((torch.tensor(probability_arr[-1:]), selection_arr[ind:ind+self.selection_length+1]))
            out, grad = self._compute_selarr_output_grad(inp)
            probability_arr.append(out)
            for zp, gp  in zip(grads_network_2, grad):
                zp += gp                
            ind += self.selection_length + 1
        
        for i in range(self.num_networks_3):
            inp = torch.cat((torch.tensor(probability_arr[-1:]), selection_arr[ind:ind+self.selection_length]))
            out, grad = self._compute_selarr_output_grad(inp)
            probability_arr.append(out)
            for zp, gp  in zip(grads_network_3, grad):
                zp += gp                
            ind += self.selection_length
        
        return {'sample_probability_array': probability_arr,
                'grads_network_2': grads_network_2,
                'grads_network_3': grads_network_3,
                }      