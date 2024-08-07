from sklearn.datasets import fetch_lfw_people

input_params = {
    'library_functions':["np.sin(_)","(_)","(_)**2","(_)**3","(_)**4"],
    'num_library_functions_select':2,
    'trunc_dim':10,
    'selection_length':8,
    'sub_selection_length':4,
    
    'num_epochs':10,
    'num_samples':100,
    'num_samples_each_batch':10
}


lfw_people = fetch_lfw_people(min_faces_per_person = 70, resize = 0.4)
image_count,image_height,image_width = lfw_people.images.shape[0], lfw_people.images.shape[1], lfw_people.images.shape[2]
S = lfw_people.images.reshape(image_count, image_width * image_height).T #(1850,1288) : 1288 images of dimension 1850 each

train_cut = int(0.8*S.shape[1])
S_train = S[:,:train_cut]
S_test = S[:,train_cut:]
