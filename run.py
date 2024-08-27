# %% importing modules
import numpy as np
from rl_nldr.experiments.Experiment import Experiment
from rl_nldr.models.model_1 import Model
import torch
from rl_nldr.utils.utils import compute_reconstr_error
from rl_nldr.utils.compare import compare_corresp_selection, compare_all_selection
from rl_nldr.utils.metrics import reconstruction_error
#*******************User Inputs******************************
from sklearn.datasets import fetch_lfw_people
import sys, os

# random_dataset
# S = np.random.random((5,15))

# image_dataset
# lfw_people = fetch_lfw_people(min_faces_per_person = 70, resize = 0.4)
# image_count,image_height,image_width = lfw_people.images.shape[0], lfw_people.images.shape[1], lfw_people.images.shape[2]
# S = lfw_people.images.reshape(image_count, image_width * image_height).T #(1850,1288) : 1288 images of dimension 1850 each

dataset_path = './datasets/'

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

data_file = 'random_dataset.npy'

input_params = {
    'is_training': 1,
    'trunc_dim':3,
    'library_functions':["np.sin(_)","(_)","(_)**2","(_)**3","(_)**4"],
    'num_library_functions_select':2,
    'num_epochs':100,
    'num_samples_total':12,
    'num_samples_each_batch':4,
    'selection_length':3, 
    'sub_selection_length':1,
    'chosen_model':'model_1',
    'data_path': dataset_path + data_file,
    'model':'model_1',
    'learning_rate': 1e-3,
    'rewards_each_batch_file':'rewards_batch.npy', # rewards for each batch   
    'best_samples_each_batch_file': 'best_samples_batch.npy', # best sample of each batch.
    'best_sample_reconstruction_train_file': 'training_reconstruction_errors.npy', # best sample (selection array and linear, non-linear error) for training data. 
    'best_sample_reconstruction_test_file': 'testing_reconstruction_errors.npy', # best sample (selection array and linear, non-linear error) for testing data.
    'best_manual_sample_same_selection_file': 'manual_selection_same_array.npy', # best sample (selection array and error) for same number of library functions, selection elements. 
    'best_manual_sample_overall_file': 'manual_selection_overall.npy' # best sample (selection array and error) for arbitrary number of library functions, selection elements.
}

#***********************************************************
exp = Experiment(input_params)

if input_params['is_training'] == 1:
    exp.train()

# best sample in each batch:
best_samples = np.load(input_params['best_samples_each_batch_file'], allow_pickle=True)

# best sample overall:
best_sample = max(best_samples, key=lambda x: x['reward'])

# linear and non-linear reconstruction errors for training and testing data for best sample:
training_reconstruction_errors = np.load(f'{input_params["best_sample_reconstruction_train_file"]}', allow_pickle=True).item()
testing_reconstruction_errors = np.load(f'{input_params["best_sample_reconstruction_test_file"]}', allow_pickle=True).item()

print('Training reconstruction errors for best sample (Linear and non-linear):', 
      training_reconstruction_errors['linear_error'],
      training_reconstruction_errors['nonlinear_error'])

print('Testing reconstruction errors for best sample (Linear and non-linear):', 
      testing_reconstruction_errors['linear_error'],
      testing_reconstruction_errors['nonlinear_error'])

#best selection arr and error
print('Best manual selection array and error:',
      best_sample['selection_arr'],
      best_sample['reconstr_err'])

# manual selection by iterating through all possible SIMILAR choices of library array and
# selection arrays. 
best_selection_corresp = compare_corresp_selection(exp, best_samples, input_params)
print('Best manual selection array and error corresponding to same number of library and selection elements:',
      best_selection_corresp['selection_arr'],
      best_selection_corresp['error'])

# manual selection by iterating through all possible choices of library array and
# selection arrays. 
best_selection_overall = compare_all_selection(exp, best_samples, input_params)
print('Best manual selection array and error corresponding to arbitrary number of library and selection elements:',
      best_selection_overall['selection_arr'],
      best_selection_overall['error'])
