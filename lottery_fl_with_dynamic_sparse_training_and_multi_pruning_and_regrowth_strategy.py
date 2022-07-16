import copy
from mimetypes import init
from utils import load_weights,load_data_for_clients,load_val_data,initialize_mask_list,multiply_mask,client_update_lottery_fl,masked_fedavg,client_model_initialization_single_fl,model_growing,fill_zero_weights,calculate_accuracy,global_pruning,calculate_affinity_based_on_network,regrowth_based_on_affinity,print_avg_personalized_weights_each_layer,print_correlation_between_label_similarity_and_network_similarity,calculate_affinity_based_on_weight_divergence_of_locally_trained_models
from tensorflow import keras
import tensorflow as tf


def lottery_fl_with_dynamic_sparse_training_many_clients(initial_weights,dataset_name,n_client,n_class,n_neurons,client_model_initialization,dataset_id,n_layer,n_conv_layer,epoch_per_round,n_round,opt,pruned_rate_each_round,pruned_rate_target,accuracy_threshold,batch_size,delta_r,initial_mask_adjustment_rate,lambda_value,similar_network_percent):

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

    # Create locally trained models
    locally_trained_weights_list = []
    for c_idx in range(n_client):
        model = client_model_initialization(1,n_class,n_neurons)[0][0]
        model.set_weights(initial_weights)
            
        model.compile(optimizer=opt,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
        model.fit(client_train[c_idx][0],client_train[c_idx][1],epochs=epoch_per_round,verbose=0,batch_size = batch_size)
        locally_trained_weights_list.append(model.get_weights())

    # Start training
    for iteration in range(n_round):
        print("iteration =",iteration)
        accuracy_list = []
        updated_weights_list_with_pruning_status = []
        for c_idx in range(n_client):

            # Update weights

            if iteration > 0:
                weights = global_model.get_weights()
            else:
                weights = copy.deepcopy(initial_weights)
                
            # Regrow masks
            next_prune_rate = pruned_rate_each_round
            if (pruned_rate_list[c_idx] >= pruned_rate_target) and(iteration%delta_r == 1):
                mask_readjustment_rate = initial_mask_adjustment_rate*(1/(iteration/delta_r))
                #mask_readjustment_rate = initial_mask_adjustment_rate
                #print("mask_regrowth_rate = ",mask_readjustment_rate)
                # Create affinity matrix
                affinty_list = calculate_affinity_based_on_weight_divergence_of_locally_trained_models(locally_trained_weights_list,client_train,c_idx,n_layer)

                # Regrowth based on affinity matrix
                binary_mask_list[c_idx] = regrowth_based_on_affinity(binary_mask_list[c_idx],binary_mask_list,affinty_list,mask_readjustment_rate/2,similar_network_percent)

                # Regrowth randomly
                binary_mask_list[c_idx],pruned_rate_list[c_idx],next_prune_rate = model_growing(binary_mask_list[c_idx],mask_readjustment_rate/2,n_conv_layer)

                

            

            weights = multiply_mask(weights,binary_mask_list[c_idx],n_conv_layer)
            model = client_model_initialization(1,n_class,n_neurons)[0][0]
            model.set_weights(weights)

            
            # Local training
                
            model.compile(optimizer=opt,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
            model.fit(client_train[c_idx][0],client_train[c_idx][1],epochs=epoch_per_round,verbose=0,batch_size = batch_size)
            
            accuracy = calculate_accuracy(model,client_val[c_idx][0],client_val[c_idx][1])

            prune = False
            if (accuracy > accuracy_threshold) and (pruned_rate_list[c_idx] < pruned_rate_target):
                prune = True
            
            weights_list[c_idx] = model.get_weights()
            updated_weights_list_with_pruning_status.append((model.get_weights(),prune))
            

        # Update global model

        ## Global pruning
                
        binary_mask_list,pruned_rate_list = global_pruning(updated_weights_list_with_pruning_status,binary_mask_list,lambda_value,pruned_rate_each_round,n_conv_layer,pruned_rate_list,pruned_rate_target)

        
        
        for c_idx in range(n_client):
                
            weights = multiply_mask(weights_list[c_idx],binary_mask_list[c_idx],n_conv_layer)
            model = client_model_initialization(1,n_class,n_neurons)[0][0]
            model.set_weights(weights)
            accuracy = calculate_accuracy(model,client_test[c_idx][0],client_test[c_idx][1])
            accuracy_list.append(accuracy)
            #print("pruned Rate = ",pruned_rate_list[c_idx])
            #print("accuracy lottery_fl = ",accuracy)

        
        # Average accuracy
        print("average accuracy = ", sum(accuracy_list)/len(accuracy_list))

        # Update global model
        global_weights = masked_fedavg(weights_list,binary_mask_list,n_conv_layer)
        global_weights = fill_zero_weights(global_weights,n_layer,n_conv_layer)
        global_model.set_weights(global_weights)

    print_avg_personalized_weights_each_layer(binary_mask_list)
    print_correlation_between_label_similarity_and_network_similarity(client_train,binary_mask_list)


dataset_name = "cifar10_20"
n_client = 10
n_class = 10
n_neurons = 32
client_model_initialization = client_model_initialization_single_fl
dataset_id = 2
n_layer = 2
n_conv_layer = 0
epoch_per_round = 10
n_round = 400
opt = opt = keras.optimizers.SGD(learning_rate=0.01,momentum=0.5)
pruned_rate_each_round = 0.2
pruned_rate_target = 0.7
accuracy_threshold = 0.5
batch_size = 32
initial_mask_adjustment_rate = 0.2
delta_r = 1
lambda_value= 5
similar_network_percent = 0.3
initial_weights = load_weights("single_fl",0)
lottery_fl_with_dynamic_sparse_training_many_clients(initial_weights,dataset_name,n_client,n_class,n_neurons,client_model_initialization,dataset_id,n_layer,n_conv_layer,epoch_per_round,n_round,opt,pruned_rate_each_round,pruned_rate_target,accuracy_threshold,batch_size,delta_r,initial_mask_adjustment_rate,lambda_value,similar_network_percent)