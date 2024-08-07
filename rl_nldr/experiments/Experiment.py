from rl_nldr.models import model_1

class Experiment():
    def __init__(self,setting) -> None:
        self.trunc_dim = setting['trunc_dim']
        self.library_functions = setting['library_functions']
        self.num_library_functions_select = setting['num_library_functions_select']
        self.num_epochs = setting['num_epochs']
        self.num_samples = setting['num_samples']
        self.num_samples_each_batch = setting['num_samples_each_batch']
        self.selection_length = setting['selection_length']
        self.sub_selection_length = setting['sub_selection_length']
        
        if setting['is_training'] == 1:
            self.training_data = setting['training_data'] 
            self.model_dict = {
                'model_1':model_1
            }
            self.model = self.model_dict[setting['chosen_model']].Model(
                len(self.library_functions), 
                self.training_data.shape[0],
                self.selection_length)
        
        if setting.get('testing_data') is not None: self.testing_data = setting['testing_data']


    # def train()    
    
    