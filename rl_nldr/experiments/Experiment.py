from rl_nldr.models import model_1
from rl_nldr.data_provider.data_maker import data_maker, create_S_hat
import numpy as np
import torch
from rl_nldr.utils.utils import compute_reconstr_error, visual
from tqdm import tqdm
from rl_nldr.utils.metrics import reconstruction_error

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
        S_train, _, train_dataloader = self._get_data('train')
        S_test, _, _ = self._get_data('test') 
        
        #creating S_hat:
        U_trunc, S_train_ref, S_train_hat = create_S_hat(S_train, self.setting['trunc_dim']) 

        model = self.model_dict[self.setting['model']].Model(
            len(self.setting['library_functions']),
            self.setting['num_library_functions_select'],
            S_train_hat.shape[0],
            self.setting['selection_length']
        )

        best_samples = [] #best sample in each epoch
        batch_reward_list = [] #preserving rewards of all batches for plot.
                
        for epoch in tqdm(range(self.setting['num_epochs'])):        
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
                        
                    # error calc:
                    sample_V_bar, sample_reconstr_error = compute_reconstr_error(S_train, S_train_ref, U_trunc, S_train_hat, self.setting['library_functions'], sample.detach().numpy())
                    sample_reward = - np.prod(sample_probability_array) * (sample_reconstr_error**2)
                    batch_reward += sample_reward
                    
                    # at end of given sample
                    if sample_reward > best_sample_reward:
                        best_sample_reward = sample_reward
                        
                        best_sample = {
                            'selection_arr': sample,
                            'probability_arr': sample_probability_array,
                            'reward': sample_reward,
                            'sample_V_bar': sample_V_bar,
                            'U_trunc': U_trunc,
                            'library_functions': self.setting['library_functions'],
                            'trunc_dim': self.setting['trunc_dim'],
                            'reconstr_err': sample_reconstr_error
                        }        
                
                # at end of a given batch:
                batch_reward_list.append(batch_reward)
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
            
            # at end of each epoch:
            best_samples.append(best_sample_epoch)                    
                
        # at end of all runs
        np.save(f'{self.setting["best_samples_each_batch_file"]}',np.array(best_samples))
        
        np.save(f'{self.setting["rewards_each_batch_file"]}',np.array(batch_reward_list))
        visual(batch_reward_list)
        
        # reconstruction errors for best sample choice:
        best_sample = max(best_samples, key=lambda x: x['reward'])
        
        # linear and non-linear reconstruction errors for best sample:
        training_reconstruction_errors = reconstruction_error(S_train, best_sample)        
        testing_reconstruction_errors = reconstruction_error(S_test, best_sample)
        
        np.save(f'{self.setting["best_sample_reconstruction_train_file"]}', training_reconstruction_errors)
        np.save(f'{self.setting["best_sample_reconstruction_test_file"]}', testing_reconstruction_errors)
        
        

        # error computation:
        # 1. training - linear and nl. approx.
        
        # train_errors = training_errors(train_data, S_ref, )
        # 2. training - nl approx. 
        