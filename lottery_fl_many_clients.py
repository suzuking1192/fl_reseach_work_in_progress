import copy
from utils import load_data_for_clients,load_val_data,initialize_mask_list,multiply_mask,client_update_lottery_fl,masked_fedavg,client_model_initialization_single_fl,print_avg_personalized_weights_each_layer,print_correlation_between_label_similarity_and_network_similarity,print_correlation_between_label_similarity_and_pruned_model_divergence,print_avg_10_percent_personalized_weights_each_layer,calculate_accuracy,client_model_initialization_lenet5,load_weights
from tensorflow import keras
import tensorflow as tf
import random

def lottery_fl_many_clients(initial_weights,dataset_name,n_client,n_sampling_each_round,n_class,n_neurons,client_model_initialization,dataset_id,n_layer,n_conv_layer,epoch_per_round,n_round,opt,pruned_rate_each_round,pruned_rate_target,accuracy_threshold,batch_size):

    # Initilization
    
    client_models = {}
    client_train = {}
    client_test = {}
    client_val = {}
    accuracy_dict ={}
    
    global_model = client_model_initialization(1,n_class,n_neurons)[0][0]
    
    pruned_rate_dict = {}
    
    weights_dict = {}
    all_pruned = False
    
    
    client_models= client_model_initialization(n_client,n_class,n_neurons)
    client_train,client_test = load_data_for_clients(dataset_name,n_client,dataset_id=dataset_id)
    client_val = load_val_data(dataset_name,dataset_id,n_client)

    
    # Initializa weights and pruned rate
        
    weights_list = []
        
    pruned_rate_list = []
        
    for i in range(n_client):
        weights_list.append([])
        pruned_rate_list.append(0)
        client_models[i][0].set_weights(initial_weights)

    # Initialize mask dict
    binary_mask_list = initialize_mask_list(n_client,n_layer,n_conv_layer,initial_weights)


    # Start training
    for iteration in range(n_round):
        print("iteration =",iteration)
        accuracy_list = []

        # random sampling of clients
        sampled_c_idxs = random.sample(range(n_client),n_sampling_each_round)

        # Create temporary loist for aggregation
        tmp_weights_list = []
        tmp_binary_mask_list = []

        for c_idx in sampled_c_idxs:

            # Update weights

            if iteration > 0:
                weights = global_model.get_weights()
            else:
                weights = copy.deepcopy(initial_weights)
                

            weights = multiply_mask(weights,binary_mask_list[c_idx],n_conv_layer)
            model = client_model_initialization(1,n_class,n_neurons)[0][0]
            model.set_weights(weights)

            # Client update
            

            updated_weights,accuracy,mask,pruned_rate = client_update_lottery_fl(opt,binary_mask_list[c_idx],weights,initial_weights,model,client_train[c_idx],client_val[c_idx],client_test[c_idx],accuracy_threshold,pruned_rate_list[c_idx],pruned_rate_each_round,pruned_rate_target,n_layer,epoch_per_round,n_conv_layer,batch_size=32,retrain=True)
            weights_list[c_idx] = updated_weights
            
            binary_mask_list[c_idx] = mask
            pruned_rate_list[c_idx] = pruned_rate
            accuracy_list.append(accuracy)

            tmp_weights_list.append(updated_weights)
            tmp_binary_mask_list.append(mask)
        
        # Average accuracy
        print("average accuracy = ", sum(accuracy_list)/len(accuracy_list))

        


        # Update global model
        global_weights = masked_fedavg(tmp_weights_list,tmp_binary_mask_list,n_conv_layer)
        global_model.set_weights(global_weights)

    # final accuracy
    accuracy_list = []
    for c_idx in range(n_client):
        model = client_model_initialization(1,n_class,n_neurons)[0][0]
        model.set_weights(weights_list[c_idx])
        accuracy = calculate_accuracy(model,client_test[c_idx][0],client_test[c_idx][1])
        accuracy_list.append(accuracy)
    print("final average accuracy = ", sum(accuracy_list)/len(accuracy_list))

    
    print_avg_10_percent_personalized_weights_each_layer(binary_mask_list,n_conv_layer)
    print_correlation_between_label_similarity_and_network_similarity(client_train,binary_mask_list,n_conv_layer)

    
dataset_name = "cifar10_20"
n_client = 10
n_class = 10
n_neurons = 32
client_model_initialization = client_model_initialization_single_fl
dataset_id = 0
n_layer = 2
n_conv_layer = 0
epoch_per_round = 10
n_round = 100
opt = opt = keras.optimizers.SGD(learning_rate=0.01,momentum=0.5)
pruned_rate_each_round = 0.2
pruned_rate_target = 0.7
accuracy_threshold = 0.5
batch_size = 32
n_sampling_each_round = 10
initial_weights = load_weights("single_fl",0)

lottery_fl_many_clients(initial_weights,dataset_name,n_client,n_sampling_each_round,n_class,n_neurons,client_model_initialization,dataset_id,n_layer,n_conv_layer,epoch_per_round,n_round,opt,pruned_rate_each_round,pruned_rate_target,accuracy_threshold,batch_size)