# %% importing modules
from sklearn.datasets import fetch_lfw_people
from utils import create_S_hat, random_selection_arr_maker, apply_selected_funcs
import numpy as np
from  data_provider.data_loader import SelectionDataset, create_input
from data_provider.data_maker import create_S_hat, apply_selected_funcs
from torch.utils.data import Dataset, DataLoader, TensorDataset

from models.model_1 import model

input_params = {
    'trunc_dim':10,
    'library_functions':["np.sin(_)","(_)","(_)**2","(_)**3","(_)**4"],
    'num_library_functions_select':2,
    'num_epochs':10,
    'num_samples':100,
    'num_samples_each_batch':10,
    'selection_length':8,
    'sub_selection_length':4,
}

library = np.array(input_params['library_functions'])

lfw_people = fetch_lfw_people(min_faces_per_person = 70, resize = 0.4)
image_count,image_height,image_width = lfw_people.images.shape[0], lfw_people.images.shape[1], lfw_people.images.shape[2]
S = lfw_people.images.reshape(image_count, image_width * image_height).T #(1850,1288) : 1288 images of dimension 1850 each

train_cut = int(0.8*S.shape[1])
S_train = S[:,:train_cut]
S_test = S[:,train_cut:]

U_trunc,S_ref,S_hat = create_S_hat(S_train,input_params['trunc_dim'])
sel_array = random_selection_arr_maker(len(library),input_params['num_library_functions_select'])
S_mod = apply_selected_funcs(S_hat, library, sel_array)

test_dataset = create_input(num_library_functions=len(input_params['library_functions']),
                            num_library_functions_select=input_params['num_library_functions_select'],
                            selection_length=input_params['selection_length'],
                            sub_selection_length=input_params['sub_selection_length'],
                            trunc_svd_shape=S_hat.shape[0],
                            num_samples_create=input_params['num_samples'])

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)




#code for 1 sample -> put it in a separate function for cleaner code:
# Input: S_mod, selection_length, sub_selection_length, 3 types of networks already created. 
# Output: All selection arrays (library, network choices), all output probabilities, reconstructed error.

#create model class that takes in the above inputs and returns the outputs:


# creating input of 1s and 0s for the library and all selection arrays based on the following info:
# 1. len(library) 
# 2. number of library functions to select
# 3. S_mod.shape[0]

