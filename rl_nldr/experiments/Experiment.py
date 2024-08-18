from rl_nldr.models import model_1
from rl_nldr.data_provider.data_maker import data_maker, create_S_hat
import numpy as np
import torch
from rl_nldr.utils.utils import compute_reconstr_error

class Experiment():
    def __init__(self,setting) -> None:
        self.setting = setting
        self.model_dict = {
                'model_1':model_1
            }

    def _get_data(self, flag):
        data, data_set, data_loader = data_maker(self.setting, flag)  
        return data,data_set,data_loader

    #for one sample:
    def train(self):
        train_data, _, train_dataloader = self._get_data('train')
        
        #creating S_hat:
        U_trunc, S_ref, S_hat = create_S_hat(train_data, self.setting['trunc_dim']) 

        model = self.model_dict[self.setting['model']].Model(
            len(self.setting['library_functions']),
            self.setting['num_library_functions_select'],
            S_hat.shape[0],
            self.setting['selection_length']
        )

        best_samples = [] #best sample in each epoch
        
        for epoch in range(self.setting['num_epochs']):        
            # in each epoch, for batch with highest total reward, preserve sample with
            # highest sample reward. 
            best_batch_reward = -np.inf
            
            for _, batch in enumerate(train_dataloader):
                best_sample_reward = -np.inf
                batch_reward = 0.
                
                batch_grads_network_1 = [torch.zeros_like(param) for param in model.net1.parameters()]
                
                if(model.num_networks_2 != 0):
                    batch_grads_network_2 = [torch.zeros_like(param) for param in model.net2.parameters()]
                
                if(model.num_networks_3 != 0):
                    batch_grads_network_3 = [torch.zeros_like(param) for param in model.net3.parameters()]
                
                # return dictionary with network outputs and grad for each possible sample.
                lib_selection_output_grad = model._compute_lib_output_grad(batch[0][:len(self.setting['library_functions'])])
                
                # print(lib_selection_output_grad.keys()) 
                
                # 1. sum up all sample rewards to compute batch reward.
                # 2. preserve the best sample in batch - if batch has highest average reward, preserve the best sample.
                # to preserve: sample selection arr, 
                
                # training for each sample within batch:                                                
                for sample in batch:
                    
                    probabilty_lib_selection = lib_selection_output_grad[tuple(sample[:len(self.setting['library_functions'])].to(torch.int).tolist() )]['output']
                    sample_grads_lib_selection = lib_selection_output_grad[tuple(sample[:len(self.setting['library_functions'])].to(torch.int).tolist() )]['gradients']
                    
                    model_inp = {'probability_lib_selection': [probabilty_lib_selection],
                                 'selection_arr': sample[len(self.setting['library_functions']):]}
                    # # print(lib_network_output_sample)
                    
                    # #send a dictionary to the model.
                    model_out = model(model_inp)
        
                    sample_probability_array = model_out['sample_probability_array']
                    sample_grads_network_2 = model_out['grads_network_2']
                    sample_grads_network_3 = model_out['grads_network_3']
                    
                    for bg, sg in zip(batch_grads_network_1, sample_grads_lib_selection):
                        bg += sg 
                    
                    if(model.num_networks_2 != 0):        
                        for bg, sg in zip(batch_grads_network_2, sample_grads_network_2):
                            bg += sg 
                    
                    if(model.num_networks_3 != 0):                            
                        for bg, sg in zip(batch_grads_network_3, sample_grads_network_3):
                            bg += sg 
                        
                    # #error calc:
                    sample_V_bar, sample_reconstr_error = compute_reconstr_error(train_data, S_ref, U_trunc, S_hat, self.setting['library_functions'], sample.detach().numpy())
                    sample_reward =  - np.prod(sample_probability_array) * (sample_reconstr_error**2)
                    batch_reward += sample_reward
                    
                    # at end of given sample
                    if sample_reward > best_sample_reward:
                        best_sample_reward = sample_reward
                        
                        best_sample = {
                            'selection_arr': sample,
                            'probability_arr': sample_probability_array,
                            'reward': sample_reward,
                            'sample_V_bar': sample_V_bar
                        }        
                
                # at end of a given batch:
                if batch_reward > best_batch_reward:
                    best_batch_reward = batch_reward
                    best_sample_epoch = best_sample
            
                # updating gradients of the networks at end of each batch:
        
                for param, grad in zip(model.net1.parameters(), batch_grads_network_1):
                    param.data += self.setting['learning_rate']* (1/self.setting['num_samples_each_batch'])* grad

                if(model.num_networks_2 != 0):
                    for param, grad in zip(model.net2.parameters(), batch_grads_network_2):
                        param.data += self.setting['learning_rate']* (1/self.setting['num_samples_each_batch'])* grad

                if(model.num_networks_3 != 0):
                    for param, grad in zip(model.net3.parameters(), batch_grads_network_3):
                        param.data += self.setting['learning_rate']* (1/self.setting['num_samples_each_batch'])* grad

                # print(batch_grads_network_1)
            # for param in model.net1.parameters():
            #     print(param)
            # print(batch_grads_network_1)
            # for param, grad in zip(model.net1.parameters(), batch_grads_network_1):
            #     param.data += (1/self.setting['num_samples_each_batch'])* grad
        
            # for param in model.net1.parameters():
            #     print(param)
            # print('\n')
            # for param in model.net1.parameters():
            #     print(param.data)
                    
                # batch_grads_network_1 *= (1/self.setting['num_samples_each_batch'])
                # batch_grads_network_2 *= (1/self.setting['num_samples_each_batch'])
                # batch_grads_network_2 *= (1/self.setting['num_samples_each_batch'])
                
                # # batch_grads_network_1 = [torch.zeros_like(param) for param in model.net1.parameters()]
                # for param,grads in zip(model.net1.parameters(), batch_grads_network_1):
                #     params += self.setting['learning_rate']*batch_grads_network_1
                

                
            
            # at end of each epoch:
            best_samples.append(best_sample_epoch)                    
                
         
                
        #at end of all runs
        np.save('best_samples',np.array(best_samples))
        
            
            
        
        


    #train global
    # def train(self):
    #     train_data, train_dataset, train_dataloader = self._get_data('train')
        
    #     #creating S_hat:
    #     U_trunc, S_ref, S_hat = create_S_hat(train_data, self.setting['trunc_dim']) 

    #     curr_best_batch_reward = -np.inf
    #     for epoch in range(self.setting['num_epochs']):        
    #         for _, batch in enumerate(train_dataloader):
    #             curr_batch_reward, curr_batch_best_sample = batch_train(
    #                 train_data,
    #                 S_ref,
    #                 S_hat,
    #                 U_trunc,
    #                 batch,
    #                 self.setting['library_functions'],
    #                 self.setting['selection_length'],
    #                 curr_best_batch_reward
    #             )
                
    #             if curr_batch_reward > curr_best_batch_reward:
    #                 curr_best_batch_reward = curr_batch_reward
    #                 curr_best_sample = curr_batch_best_sample
                
    #             #update_weights_biases at end of each iteration.
                
                
    # #train batch
    # def batch_train(S_train, S_train_ref, S_hat, U_trunc, batch,
    #                 library_functions, selection_length,curr_best_batch_reward):
        
    #     curr_batch_reward = 0.
    #     best_sample_reward = -np.inf
        
    #     for _, sample in enumerate(batch):
    #         sample_probability_array  = model(S_hat,
    #                                           sample,
    #                                           library_functions,
    #                                           selection_length)
            
    #         sample_V_bar, sample_reconstr_err = compute_reconstruction_error(S_train,
    #                                                                          S_train_ref,
    #                                                                          U_trunc,
    #                                                                          S_hat,
    #                                                                          sample)
            
    #         #this should be a one-liner
    #         sample_reward = compute_reward(sample_probability_array, sample_reconstr_err)
    #         if(sample_reward > best_sample_reward):
    #             best_sample_reward = sample_reward
                
    #             best_sample = {'selection_array': sample,
    #                            'probability_array': sample_probability_array,
    #                            'V_bar': sample_V_bar,
    #                            'reconstruction_error': sample_reconstr_err}
            
    #         curr_batch_reward += sample_reward
            
    #     if(curr_batch_reward > curr_best_batch_reward):
    #         return curr_batch_reward, best_sample         
        
    #     return -np.inf, None
    

    #train sample
    
        
        # for epoch in range(self.setting['num_epochs']):
        #     current_best_batch_reward = -np.inf
        #     current_batch_reward = 0.
            
        #     for i, sample in enumerate(train_dataloader):
                
        #         outputs = model(S_hat, 
        #                         sample,
        #                         self.setting['library_functions'],
        #                         self.setting['selection_length'])

                
                
                # if i== 1:
                #     break

                # for a given batch:
                # have to preserve each sample selection arrays, prob values
                # 
                #outputs: [Probability values], S_reconstr

                
                # current_batch_reward += current_sample_reward
                # # at end of each set of 'm' samples
                # if(i !=0 and i % self.setting['num_samples_each_batch'] == 0):
                    
                #     #update current best batch reward
                #     if(current_batch_reward > current_best_batch_reward):
                #         current_best_batch_reward = current_batch_reward

                #         #preserve best sample in current batch:
                #         cur_best_sample = {
                #             'probability_values': _,
                #             'selection_array': sample[0],
                #             'S_nonlinear':S_nonlinear
                #         }
                    
                    #update gradients
                    
                    
                    
                    
                

                # outputs = model(S_hat, 
                #                 sample[0],
                #                 self.setting['library_functions'],
                #                 self.setting['selection_length'],
                #                 self.setting['sub_selection_length'])



                # # print(S_hat, '\n',sample[0])
                # S_mod = apply_selected_funcs(S_hat,
                #                             self.setting['library_functions'],
                #                             sample[0][:len(self.setting['library_functions'])]
                #                             )
                # # outputs: S_reconstructed, array of probability values for each
                # # selection array.
                # outputs = model(S_mod,
                #                 sample,
                #                 self.setting['selection_length'],
                #                 self.setting['sub_selection_length'])
                
                #update grads at end of "num_samples_per_batch"
                
                # print(sample)
                # for each sample, 
            
            
        
        
    
    