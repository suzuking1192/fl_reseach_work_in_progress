
from tensorflow.keras import datasets, layers, models
import numpy as np
from tensorflow.keras import initializers
from tensorflow import keras
import pandas as pd
import tensorflow as tf
import copy
import random
import statistics
from scipy.stats import pearsonr

def resnet9(n_class):
    input_img = keras.Input(shape=(32,32,3))
    
    encoder_1 = models.Sequential()
    layer_input = layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3))
    encoder_1.add(layer_input)
    
    
    layer_input = layers.Conv2D(128, (5, 5),strides=(2,2), activation='relu')
    encoder_1.add(layer_input)
    #encoder_2.add(layers.MaxPooling2D((2, 2)))
    
    
    layer_input = layers.Conv2D(128, (3, 3), activation='relu')
    encoder_1.add(layer_input)
    layer_input = layers.Conv2D(128, (3, 3), activation='relu')
    encoder_1.add(layer_input)
    layer_input = layers.Conv2D(256, (3, 3), activation='relu')
    encoder_1.add(layer_input)
    encoder_1.add(layers.MaxPooling2D((2, 2)))
    layer_input = layers.Conv2D(256, (3, 3), activation='relu')
    encoder_1.add(layer_input)
    layer_input = layers.Conv2D(256, (3, 3), activation='relu')
    encoder_1.add(layer_input)
    layer_input = layers.Conv2D(256, (3, 3), activation='relu')
    encoder_1.add(layer_input)
    encoder_1.add(layers.AveragePooling2D((2, 2)))
    encoder_1.add(layers.Flatten())
    
    classification_layer = models.Sequential()
    classification_layer.add(layers.Dense(n_class,activation="softmax"))
    
    output_1 = encoder_1(input_img)
    output = classification_layer(output_1)
    
    
    final_model = keras.Model(input_img,output)
    
    return final_model,encoder_1

def client_model_initialization_resnet9(n_client,n_class,n_neurons):
    client_models = {}
    
    for i in range(n_client):
        client_models[i] = {}
        client_models[i][0],client_models[i][1] = resnet9(n_class)
        
    return client_models


def lenet5(num_classes=10):
    # Define the input placeholder as a tensor with shape input_shape.
    input_shape = (32,32,3)
    X_input = layers.Input(input_shape)

    X = layers.Conv2D(filters=20, kernel_size=5, padding='same',
                      activation='relu',kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.001, seed=0))(X_input)
    X = layers.MaxPool2D()(X)
    X = layers.Conv2D(filters=50, kernel_size=5, padding='same',
                      activation='relu',kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.001, seed=0))(X)
    X = layers.MaxPool2D()(X)
    X = layers.Flatten()(X)
    X = layers.Dense(500, activation='relu',kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.001, seed=0))(X)
    X = layers.Dense(num_classes, activation='softmax')(X)

    model = keras.Model(X_input, X)

    return model
def client_model_initialization_lenet5(n_client,n_class,n_neurons):
    client_models = {}
    
    for i in range(n_client):
        client_models[i] = {}
        client_models[i][0] = lenet5(n_class)
        client_models[i][1] = "encoder"
        
    return client_models

def client_model_initialization_lenet5_no_enc(n_client,n_class,n_neurons):
    client_models = {}
    
    for i in range(n_client):
        
        client_models[i] = lenet5(n_class)
        
    return client_models

def cifar100_single_fl_model(n_class,n_neurons,initialize=False):
    input_img = keras.Input(shape=(32,32,3))
    encoder_1 = models.Sequential()
    encoder_1.add(layers.Flatten())
    if initialize == False:
        encoder_1.add(layers.Dense(n_neurons, activation='relu',kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.001, seed=0)))
        #encoder_1.add(layers.Dense(n_neurons, activation='relu'))
    else:
        encoder_1.add(layers.Dense(n_neurons, activation='sigmoid',kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=0),bias_initializer=initializers.Zeros(),))
    #encoder_1.add(BatchNormalization())
    
    classification_layer = models.Sequential()
    classification_layer.add(layers.Dense(n_class,activation='softmax',kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.001, seed=0)))
    #classification_layer.add(layers.Dense(n_class,activation='softmax'))
    output_1 = encoder_1(input_img)
    output = classification_layer(output_1)
    
    final_model = keras.Model(input_img,output)
    
    return final_model,encoder_1,classification_layer

def client_model_initialization_single_fl(n_client,n_class,n_neurons):
    client_models = {}
    
    for i in range(n_client):
        client_models[i] = {}
        client_models[i][0],_,_ = cifar100_single_fl_model(n_class,n_neurons)
        
    return client_models


def load_data_for_clients(dataset_name,n_client,dataset_id):
    client_train = {}
    client_test = {}
    
    client_train,client_test = load_data(dataset_name,dataset_id,n_client)
    return client_train,client_test
    
def calculate_accuracy(model,test_x,test_y):
    y_pred = model.predict(test_x)
    y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
    m = tf.keras.metrics.Accuracy()
    m.update_state(y_pred,test_y)
    accuracy = m.result().numpy()
    
    return accuracy


def load_data(dataset_name,dataset_id,n_clients):
    train = {}
    test = {}
    
    for i in range(n_clients):
        train[i] = {}
        test[i] = {}
        
        filename_train_x = "data/" + str(dataset_name)  +  "/dataset_id=" + str(dataset_id) + "train_x" + "client_id=" + str(i) +".npy"
        train[i][0] = np.load(filename_train_x)
        
        filename_train_y = "data/" + str(dataset_name)  +  "/dataset_id=" + str(dataset_id) + "train_y" + "client_id=" + str(i) +".npy"
        train[i][1] = np.load(filename_train_y)
        
        filename_test_x = "data/" + str(dataset_name)  +  "/dataset_id=" + str(dataset_id) + "test_x" + "client_id=" + str(i) +".npy"
        test[i][0] = np.load(filename_test_x)
        
        filename_test_y = "data/" + str(dataset_name)  +  "/dataset_id=" + str(dataset_id) + "test_y" + "client_id=" + str(i) +".npy"
        test[i][1] = np.load(filename_test_y)
    
    return train, test


def load_val_data(dataset_name,dataset_id,n_clients):
    val = {}
    
    for i in range(n_clients):
        val[i] = {}
        
        
        filename_val_x = "data/" + str(dataset_name)  +  "/dataset_id=" + str(dataset_id) + "val_x" + "client_id=" + str(i) +".npy"
        val[i][0] = np.load(filename_val_x)
        
        filename_val_y = "data/" + str(dataset_name)  +  "/dataset_id=" + str(dataset_id) + "val_y" + "client_id=" + str(i) +".npy"
        val[i][1] = np.load(filename_val_y)
    
    return val


def initialize_mask_list(n_client,n_layer,n_conv_layer,initial_weights):
    binary_mask_list = []
    
    
    for i in range(n_client):
        
        binary_mask_list.append([])
        
        for j in range(n_layer):
            if j >= n_conv_layer:
                # fully connected layers
                n_neurons_layer = initial_weights[j*2].shape[0]
                n_output = initial_weights[j*2].shape[1]
                binary_mask_list[i].append([])

                for k in range(n_neurons_layer):
                    binary_mask_list[i][j].append([])
                    for l in range(n_output):
                        binary_mask_list[i][j][k].append(1)
            else:
                # conv layers
                conv_layer_weights = initial_weights[j*2]
                n_dim_0 = conv_layer_weights.shape[0]
                n_dim_1 = conv_layer_weights[0].shape[0]
                n_dim_2 = conv_layer_weights[0][0].shape[0]
                n_dim_3 = conv_layer_weights[0][0][0].shape[0]
                binary_mask_list[i].append([])

                for k in range(n_dim_0):
                    binary_mask_list[i][j].append([])
                    for l in range(n_dim_1):
                        binary_mask_list[i][j][k].append([])
                        for m in range(n_dim_2):
                            binary_mask_list[i][j][k][l].append([])
                            for n in range(n_dim_3):
                                binary_mask_list[i][j][k][l][m].append(1)

    return binary_mask_list

def neural_weights_addition(weights1,weights2):
    length = len(weights1)
    
    for i in range(length):
        weights1[i] = weights1[i] + weights2[i]
        
    return weights1

def multiply_mask(weights,mask,n_conv_layer):
    n_weight_layer = int(len(weights)/2)
    for l in range(n_weight_layer):
        if l >= n_conv_layer: # fully connected layers
            weights[l*2] = mask[l] * weights[l*2]
        else:
            n_dim_0 = len(mask[l])
            n_dim_1 = len(mask[l][0])
            n_dim_2 = len(mask[l][0][0])
            for i in range(n_dim_0):
                for j in range(n_dim_1):
                    for k in range(n_dim_2):
                        weights[l*2][i][j][k] = mask[l][i][j][k] * weights[l*2][i][j][k]
        
    return weights



def weights_pruning(model,pruned_rate,pruned_rate_each_round,n_layer,mask,n_conv_layer,pruned_rate_target):
    print("weight pruning start")
    pruning_percent = 1 -  (1 - pruned_rate) * (1-pruned_rate_each_round)
    
    all_weights = {}

    for layer_no in range(n_layer):  
        
        if layer_no >= n_conv_layer:
            layer_weights = (pd.DataFrame(model.get_weights()[layer_no*2]).stack()).to_dict() 
            layer_weights = { (layer_no, k[0], k[1]): v for k, v in layer_weights.items() }
            all_weights.update(layer_weights)
        else:
            # conv_layer
            layer_weights = model.get_weights()[layer_no*2]
            n_dim_0 = layer_weights.shape[0]
            n_dim_1 = layer_weights[0].shape[0]
            for i in range(n_dim_0):
                for j in range(n_dim_1):
                    layer_weights = (pd.DataFrame(model.get_weights()[layer_no*2][i][j]).stack()).to_dict() 
                    layer_weights = { (layer_no,i,j, k[0], k[1]): v for k, v in layer_weights.items() }
                    all_weights.update(layer_weights)

    
   
    all_weights_sorted = {k: v for k, v in sorted(all_weights.items(), key=lambda item: abs(item[1]))}

    
    new_weights = copy.deepcopy(model.get_weights())

    if pruning_percent > pruned_rate_target:
        pruning_percent = pruned_rate_target

    prune_fraction = pruning_percent
    total_no_weights = len(all_weights_sorted) 
    number_of_weights_to_be_pruned = int(prune_fraction*total_no_weights)
    weights_to_be_pruned = {k: all_weights_sorted[k] for k in list(all_weights_sorted)[ :  number_of_weights_to_be_pruned]}     

    for k, v in weights_to_be_pruned.items():
        if k[0] >= n_conv_layer: # fully connected layer
            new_weights[k[0]*2][k[1]][k[2]] = 0
            mask[k[0]][k[1]][k[2]] = 0
        else:
            new_weights[k[0]*2][k[1]][k[2]][k[3]][k[4]] = 0
            mask[k[0]][k[1]][k[2]][k[3]][k[4]] = 0


    model.set_weights(new_weights)


    return model,mask,pruning_percent
    

def client_update_lottery_fl(opt,mask,weights,initial_weights,model,client_train,client_val,client_test,accuracy_threshold,pruned_rate,pruned_rate_each_round,pruned_rate_target,n_layer,epoch_per_round,n_conv_layer,batch_size=32,retrain=True):
    
    
    #print("pruned_rate = ",pruned_rate)
    
    model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.fit(client_train[0],client_train[1],epochs=epoch_per_round,verbose=0,batch_size = batch_size)
    
    accuracy = calculate_accuracy(model,client_val[0],client_val[1])
    

    if (accuracy > accuracy_threshold) and (pruned_rate < pruned_rate_target):
        
        # pruning
        model,mask,pruned_rate = weights_pruning(model,pruned_rate,pruned_rate_each_round,n_layer,mask,n_conv_layer,pruned_rate_target)
        

        if retrain == True:
            weights = multiply_mask(initial_weights,mask,n_conv_layer)
            model.set_weights(weights)
            
            model.compile(optimizer=opt,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
            model.fit(client_train[0],client_train[1],epochs=epoch_per_round,verbose=0,batch_size = batch_size)
            weights = model.get_weights()

        
    accuracy = calculate_accuracy(model,client_test[0],client_test[1])
    weights = model.get_weights()
    
    
    return weights, accuracy, mask,pruned_rate
        

def divide_weights_by_mask(weights,sum_mask,n_layer,n_client,n_conv_layer):
    # This function is to avoid dividing by zero
    for l in range(n_layer):
        if l >= n_conv_layer: # fully connected layers
            input_size = len(weights[l*2])
            output_size = len(weights[l*2][0])
            #print(input_size,output_size)
            

            for i in range(input_size):
                for o in range(output_size):
                    if sum_mask[l][i][o] == 0:
                        weights[l*2][i][o] = 0
                    else:
                        weights[l*2][i][o] = weights[l*2][i][o]/sum_mask[l][i][o]
            
            weights[l*2+1] =    weights[l*2+1]/(n_client ) 
            
        else: # conv layers
            n_dim_0 = len(weights[l*2])
            n_dim_1 = len(weights[l*2][0])
            n_dim_2 = len(weights[l*2][0][0])
            n_dim_3 = len(weights[l*2][0][0][0])
            for i in range(n_dim_0):
                for j in range(n_dim_1):
                    for k in range(n_dim_2):
                        for m in range(n_dim_3):
                            if sum_mask[l][i][j][k][m] == 0:
                                weights[l*2][i][j][k][m] = 0
                            else:
                                weights[l*2][i][j][k][m] = weights[l*2][i][j][k][m] /sum_mask[l][i][j][k][m]
            # bias
            weights[l*2+1] =    weights[l*2+1]/(n_client ) 

    return weights

def masked_fedavg(weights_list,mask_list,n_conv_layer):
    n_layer = int(len(weights_list[0])/2)
    n_client = len(weights_list)
    # Sum up the weights ans masks

    for c_id in range(n_client):
        
        if c_id == 0:
            global_weights = copy.deepcopy(weights_list[c_id])
            sum_binary_mask = copy.deepcopy(np.array(mask_list[c_id]))
            for l in range(n_layer):
                sum_binary_mask[l] = np.array(sum_binary_mask[l])
        else:
            global_weights = neural_weights_addition(global_weights,copy.deepcopy(weights_list[c_id]))
            addition_mask = copy.deepcopy(np.array(mask_list[c_id]))
            for l in range(n_layer):
                addition_mask[l] = np.array(addition_mask[l])
                
            sum_binary_mask = sum_binary_mask + addition_mask
        
            
    
    global_weights = divide_weights_by_mask(global_weights,sum_binary_mask,n_layer,n_client,n_conv_layer)    
    
    return global_weights

def fill_zero_weights(weights,n_layer,n_conv_layer):
    weights_list = []
    for l in range(n_layer):
        if l < n_conv_layer:
            pass # add later
        else:
            n_neurons_layer = weights[l*2].shape[0]
            n_output = weights[l*2].shape[1]

            for i in range(n_neurons_layer):
                for o in range(n_output):
                    if weights[l*2][i][o] == 0:
                        #weights[l*2][i][o] = np.random.normal(0, 0.001, 1)[0]
                        pass
                    else:
                        weights_list.append(weights[l*2][i][o])
    res = statistics.pstdev(weights_list)
    # Printing result
    #print("Standard deviation of weights is : " + str(res))

    for l in range(n_layer):
        if l < n_conv_layer:
            pass # add later
        else:
            n_neurons_layer = weights[l*2].shape[0]
            n_output = weights[l*2].shape[1]

            for i in range(n_neurons_layer):
                for o in range(n_output):
                    if weights[l*2][i][o] == 0:
                        weights[l*2][i][o] = np.random.normal(0, res, 1)[0]
                    

    return weights

def model_growing(mask,mask_adjustment_rate,n_conv_layer):
    updated_mask = copy.deepcopy(mask)
    # Count total number of pruned weights
    n_layer = len(mask)

    counter = 0
    total_count = 0
    for l in range(n_layer):
        if l < n_conv_layer:
            pass # add later
        else:
            # FC layer
            input_size = len(mask[l])
            output_size = len(mask[l][0])

            for i in range(input_size):
                for o in range(output_size):
                    total_count += 1
                    if mask[l][i][o] == 0:
                        counter += 1
                    


    # randomly select necessary number of indexes
    idx_list = random.sample(range(counter-1), int(counter*mask_adjustment_rate))

    

    # Assign 1 to those mask
    idx_counter = 0
    for l in range(n_layer):
        if l < n_conv_layer:
            pass # add later
        else:
            # FC layer
            input_size = len(mask[l])
            output_size = len(mask[l][0])

            for i in range(input_size):
                for o in range(output_size):
                    if mask[l][i][o] == 0:
                        if idx_counter in idx_list:
                            updated_mask[l][i][o] = 1
                            

                        idx_counter += 1

    updated_pruned_rate = (counter - len(idx_list))/total_count

    next_prune_rate = len(idx_list)/(total_count - counter + len(idx_list))

    return updated_mask,updated_pruned_rate,next_prune_rate

def global_pruning(weights_list_with_pruning_status,mask_list,lambda_value,pruned_rate_each_round,n_conv_layer,pruned_rate_list,pruned_target):
    updated_mask_list = []
    updated_pruned_rate_list = []

    n_client = len(mask_list)
    n_layer = len(mask_list[0])
    for c_id in range(n_client):
        weights = copy.deepcopy(weights_list_with_pruning_status[c_id][0])
        mask = copy.deepcopy(mask_list[c_id])
        
        if weights_list_with_pruning_status[c_id][1] == False:
            updated_mask_list.append(mask)
            updated_pruned_rate_list.append(pruned_rate_list[c_id])
        else:

            # Normalize weights and 1/standard deviation loss
            weights_list = []
            std_weights_list = []
            std_weights_list_multi_dim = []
            unpruned = 0
            num_weights = 0

            for l in range(n_layer):
                std_weights_list_multi_dim.append([])
                if l < n_conv_layer:
                    pass # add later
                else:
                    input_size = len(mask[l])
                    output_size = len(mask[l][0])

                    for i in range(input_size):
                        std_weights_list_multi_dim[l].append([])
                        for o in range(output_size):
                            
                            weights_list.append(weights[l*2][i][o])

                            num_weights += 1
                            if mask[l][i][o] != 0:
                                unpruned +=1

                            # Calculate std
                            if mask[l][i][o] == 0:
                                std_weights_list.append(0)
                                std_weights_list_multi_dim[l][i].append(0)
                            else:

                                weight_list_in_one_position = []
                                weight_list_in_one_position_without_ref_weight = []
                                for c_idx in range(n_client):
                                    if mask_list[c_idx][l][i][o] != 0:
                                        if c_id == c_idx:
                                            weight_list_in_one_position.append(weights_list_with_pruning_status[c_idx][0][l*2][i][o])
                                        else:
                                            weight_list_in_one_position_without_ref_weight.append(weights_list_with_pruning_status[c_idx][0][l*2][i][o])
                                            weight_list_in_one_position.append(weights_list_with_pruning_status[c_idx][0][l*2][i][o])
                                if (len(weight_list_in_one_position) == 0) or (len(weight_list_in_one_position_without_ref_weight) == 0):
                                    std_diff = 0
                                else:
                                    std_diff = abs(statistics.pstdev(weight_list_in_one_position) - statistics.pstdev(weight_list_in_one_position_without_ref_weight))
                                if std_diff == 0:
                                    std_weights_list.append(std_diff)
                                    std_weights_list_multi_dim[l][i].append(std_diff)
                                else:
                                    std_weights_list.append(std_diff)
                                    std_weights_list_multi_dim[l][i].append(std_diff)
            
            sum_weights = sum(weights_list)
            sum_std_weights = sum(std_weights_list)
            # print("sum_std_weights =",sum_std_weights)
            # print("statistics of values------------------------")
            # print("mean of weights= ", statistics.mean(weights_list))
            # print("mean of std = ",statistics.mean(std_weights_list))
            # print("maximum of weights =",max(weights_list))
            # print("maximum of std=",max(std_weights_list))
            # print("std of weights =",statistics.stdev(weights_list))
            # print("std of std=",statistics.stdev(std_weights_list))

            # Add values and sort them
            all_weights = {}
            counter = 0
            for l in range(n_layer):
                if l < n_conv_layer:
                    pass # add later
                else:
                    input_size = len(mask[l])
                    output_size = len(mask[l][0])

                    for i in range(input_size):
                        for o in range(output_size):
                            
                            try:
                                # if (i==0) and (o == 0):
                                    # print("weights[l*2][i][o]",abs(weights[l*2][i][o]/sum_weights))
                                    # print("std_weights_list_multi_dim[l][i][o]",std_weights_list_multi_dim[l][i][o]/sum_std_weights)
                            
                                value = abs(weights[l*2][i][o]/sum_weights) - lambda_value * std_weights_list_multi_dim[l][i][o]/sum_std_weights
                            except:
                                if (i==0) and (o == 0):
                                    print("division by zero")
                                value = abs(weights[l*2][i][o]/sum_weights)
                            all_weights[(l,i,o)] = value
                            counter += 1

            all_weights_sorted = {k: v for k, v in sorted(all_weights.items(), key=lambda item: abs(item[1]))}
            # Prune weights


            pruning_percent = 1 -  (unpruned/num_weights) * (1-pruned_rate_each_round)
            if pruning_percent > pruned_target:
                pruning_percent = pruned_target
            # print("unpruned/num_weights",unpruned/num_weights)
            # print("pruning_percent =",pruning_percent)

            number_of_weights_to_be_pruned = int(pruning_percent*num_weights)
            # print("number_of_weights_to_be_pruned= ",number_of_weights_to_be_pruned)
            weights_to_be_pruned = {k: all_weights_sorted[k] for k in list(all_weights_sorted)[ :  number_of_weights_to_be_pruned]}     

            
            for k, v in weights_to_be_pruned.items():
                if k[0] >= n_conv_layer: # fully connected layer
                    
                    mask[k[0]][k[1]][k[2]] = 0
                    
                else:
                    
                    mask[k[0]][k[1]][k[2]][k[3]][k[4]] = 0
            
            updated_mask_list.append(mask)
            updated_pruned_rate_list.append(pruning_percent)
            mask_list[c_id] = copy.deepcopy(mask)

            


    return updated_mask_list,updated_pruned_rate_list


def calculate_affinity_based_on_network(binary_mask_target,binary_mask_list_all,n_conv_layer=0):
    affinity_list = []
    n_client = len(binary_mask_list_all)
    n_layer = len(binary_mask_target)

    for c_idx in range(n_client):
        total_num = 0
        overlap = 0

        for l in range(n_layer):
            if l >= n_conv_layer:
                input_size = len(binary_mask_target[l])
                output_size = len(binary_mask_target[l][0])

                for i in range(input_size):
                    for o in range(output_size):
                        if binary_mask_target[l][i][o] == 1:
                            total_num += 1
                            if binary_mask_list_all[c_idx][l][i][o] == 1:
                                overlap += 1
            else:
                # conv layer
                n_dim_0 = len(binary_mask_target[l])
                n_dim_1 = len(binary_mask_target[l][0])
                n_dim_2 = len(binary_mask_target[l][0][0])
                n_dim_3 = len(binary_mask_target[l][0][0][0])
                for i in range(n_dim_0):
                    for j in range(n_dim_1):
                        for k in range(n_dim_2):
                            for m in range(n_dim_3):
                                if binary_mask_target[l][i][j][k][m] == 1:
                                    total_num += 1
                                    if binary_mask_list_all[c_idx][l][i][j][k][m] == 1:
                                        overlap += 1

        affinity_list.append(overlap/total_num)


    return affinity_list

def calculate_affinity_based_on_weight_divergence_of_locally_trained_models(locally_trained_weights_list,client_train_list,target_idx,n_layer):
    n_client = len(client_train_list)
    affinity_based_on_weight_divergence_list = []

    weights_target = locally_trained_weights_list[target_idx]

    for c_idx in range(n_client):
        weight_divergence = 0 

        for l in range(n_layer):
            input_size = len(weights_target[l*2])
            output_size = len(weights_target[l*2][0])

            for o in range(output_size):
                for i in range(input_size):
                
                    weight_divergence += (weights_target[l*2][i][o] - locally_trained_weights_list[c_idx][l*2][i][o])**2
                weight_divergence += (weights_target[l*2+1][o] - locally_trained_weights_list[c_idx][l*2+1][o])**2

        affinity_based_on_weight_divergence_list.append(- weight_divergence)

    return affinity_based_on_weight_divergence_list

def regrowth_based_on_affinity(binary_mask,binary_mask_list_all,affinty_list,mask_readjustment_rate,similar_network_percent):
    mask = copy.deepcopy(binary_mask)
    n_client = len(binary_mask_list_all)
    n_layer = len(binary_mask)
    # Select similar clients' ids
    selected_client_num = int(n_client * similar_network_percent)

    selected_client_idx = np.argsort(affinty_list)[-selected_client_num:]


    counter = 0
    overlap = 0 
    for l in range(n_layer):
        # if l < n_conv_layer:
        #     pass # add later
        # else:
            # FC layer
            input_size = len(mask[l])
            output_size = len(mask[l][0])

            for i in range(input_size):
                for o in range(output_size):
                    
                    if mask[l][i][o] == 0:
                        counter += 1
                        overlap_status = False
                        for c_idx in selected_client_idx:
                            if binary_mask_list_all[c_idx][l][i][o] == 1:
                                overlap_status = True
                        if overlap_status == True:
                            overlap += 1

                    
    # randomly select necessary number of indexes
    if int(counter*mask_readjustment_rate) > overlap-1:
        idx_list = range(overlap)
    else:
        idx_list = random.sample(range(overlap-1), int(counter*mask_readjustment_rate))

    

    # Assign 1 to those mask
    idx_counter = 0
    for l in range(n_layer):
        # if l < n_conv_layer:
        #     pass # add later
        # else:
            # FC layer
            input_size = len(mask[l])
            output_size = len(mask[l][0])

            for i in range(input_size):
                for o in range(output_size):
                    if mask[l][i][o] == 0:
                        overlap_status = False
                        for c_idx in selected_client_idx:
                            if binary_mask_list_all[c_idx][l][i][o] == 1:
                                overlap_status = True
                        if overlap_status == True:
                            if idx_counter in idx_list:
                                mask[l][i][o] = 1
                            

                            idx_counter += 1

    return mask

def create_selected_client_idx_list(affinity_mat,parameter_to_multiply_avg):

    # calculate average of affinity measure
    affinity_measure_list = []
    n_client = len(affinity_mat)

    for c_idx in range(n_client):
        for compared_idx in range(n_client):
            if c_idx != compared_idx:
                affinity_measure_list.append(affinity_mat[c_idx][compared_idx])
    
    avg_affinity_measure = sum(affinity_measure_list)/len(affinity_measure_list)
    adjusted_threshold = avg_affinity_measure * parameter_to_multiply_avg

    


    # select similar clients
    selected_idxs = []
    for c_idx in range(n_client):
        selected_idxs.append([])
        for compared_idx in range(n_client):
            if c_idx != compared_idx:
                if affinity_mat[c_idx][compared_idx] > adjusted_threshold: # affinity measure is negative valuse because it is multiples by -1 with weight divergence
                    selected_idxs[c_idx].append(compared_idx)

    return selected_idxs

def regrowth_based_on_affinity_c_idxs(binary_mask,binary_mask_list_all,selected_client_idx,mask_readjustment_rate):
    mask = copy.deepcopy(binary_mask)
    
    n_layer = len(binary_mask)

    counter = 0
    overlap = 0 
    for l in range(n_layer):
        # if l < n_conv_layer:
        #     pass # add later
        # else:
            # FC layer
            input_size = len(mask[l])
            output_size = len(mask[l][0])

            for i in range(input_size):
                for o in range(output_size):
                    
                    if mask[l][i][o] == 0:
                        counter += 1
                        overlap_status = False
                        for c_idx in selected_client_idx:
                            if binary_mask_list_all[c_idx][l][i][o] == 1:
                                overlap_status = True
                        if overlap_status == True:
                            overlap += 1

                    
    # randomly select necessary number of indexes
    if int(counter*mask_readjustment_rate) > overlap-1:
        idx_list = range(overlap)
    else:
        idx_list = random.sample(range(overlap-1), int(counter*mask_readjustment_rate))

    

    # Assign 1 to those mask
    idx_counter = 0
    for l in range(n_layer):
        # if l < n_conv_layer:
        #     pass # add later
        # else:
            # FC layer
            input_size = len(mask[l])
            output_size = len(mask[l][0])

            for i in range(input_size):
                for o in range(output_size):
                    if mask[l][i][o] == 0:
                        overlap_status = False
                        for c_idx in selected_client_idx:
                            if binary_mask_list_all[c_idx][l][i][o] == 1:
                                overlap_status = True
                        if overlap_status == True:
                            if idx_counter in idx_list:
                                mask[l][i][o] = 1
                            

                            idx_counter += 1

    return mask


def print_avg_personalized_weights_each_layer(mask_list):
    n_client = len(mask_list)
    n_layer = len(mask_list[0])

    personalized_percentage_list = []

    for l in range(n_layer):
        personalized_percentage_list.append([])

    for c_idx in range(n_client):
        for l in range(n_layer):
            input_size = len(mask_list[c_idx][l])
            output_size = len(mask_list[c_idx][l][0])

            total_unpruned_weights_num = 0
            personalized_num = 0

            for i in range(input_size):
                for o in range(output_size):
                    if mask_list[c_idx][l][i][o] == 1:
                        total_unpruned_weights_num += 1
                        
                        overlap_num = 0
                        for ref_c_idx in range(n_client):
                            if mask_list[ref_c_idx][l][i][o] == 1:
                                overlap_num += 1
                        if overlap_num == 1:
                            
                            personalized_num += 1
            personalized_percentage_list[l].append(personalized_num/total_unpruned_weights_num)
    
    for l in range(n_layer):
        print("layer idx =",l)
        print("average percentage of personalized weights = ", sum(personalized_percentage_list[l])/n_client)

def print_avg_10_percent_personalized_weights_each_layer(mask_list,n_conv_layer=0):
    n_client = len(mask_list)
    n_layer = len(mask_list[0])

    personalized_percentage_list = []

    for l in range(n_layer):
        personalized_percentage_list.append([])

    for c_idx in range(n_client):
        for l in range(n_layer):
            if l >= n_conv_layer:
                input_size = len(mask_list[c_idx][l])
                output_size = len(mask_list[c_idx][l][0])

                total_unpruned_weights_num = 0
                personalized_num = 0

                for i in range(input_size):
                    for o in range(output_size):
                        if mask_list[c_idx][l][i][o] == 1:
                            total_unpruned_weights_num += 1
                            
                            overlap_num = 0
                            for ref_c_idx in range(n_client):
                                if mask_list[ref_c_idx][l][i][o] == 1:
                                    overlap_num += 1
                            if overlap_num <= 0.1 *(n_client):
                                
                                personalized_num += 1
            else:
                total_unpruned_weights_num = 0
                personalized_num = 0
                # conv layer
                n_dim_0 = len(mask_list[c_idx][l])
                n_dim_1 = len(mask_list[c_idx][l][0])
                n_dim_2 = len(mask_list[c_idx][l][0][0])
                n_dim_3 = len(mask_list[c_idx][l][0][0][0])
                for i in range(n_dim_0):
                    for j in range(n_dim_1):
                        for k in range(n_dim_2):
                            for m in range(n_dim_3):
                                if mask_list[c_idx][l][i][j][k][m] == 1:
                                    total_unpruned_weights_num += 1
                                    
                                    overlap_num = 0
                                    for ref_c_idx in range(n_client):
                                        if mask_list[ref_c_idx][l][i][j][k][m] == 1:
                                            overlap_num += 1
                                    if overlap_num <= 0.1 *(n_client):
                                        
                                        personalized_num += 1


            personalized_percentage_list[l].append(personalized_num/total_unpruned_weights_num)
    
    for l in range(n_layer):
        print("layer idx =",l)
        print("average percentage of personalized weights = ", sum(personalized_percentage_list[l])/n_client)


def print_correlation_between_label_similarity_and_network_similarity(client_train_list,mask_list,n_conv_layer=0):
    n_client = len(mask_list)

    label_similarity_list = []
    network_similarity_list = []

    for c_idx in range(n_client):
        affinity_list = calculate_affinity_based_on_network(mask_list[c_idx],mask_list,n_conv_layer)
        #print("affinity_list = ",affinity_list)
        for ref_c_idx in range(n_client):
            if c_idx == ref_c_idx:
                pass
            else:
                label_1 = np.unique(client_train_list[c_idx][1])
                label_2 = np.unique(client_train_list[ref_c_idx][1])

                label_similarity = len(set(label_1)&set(label_2))
                label_similarity_list.append(label_similarity)

                network_similarity_list.append(affinity_list[ref_c_idx])

    corr, _ = pearsonr(label_similarity_list, network_similarity_list)

    print("correlation between label similarity and network similarity = ",corr)

def calculate_weight_divergence_all(weights_target,weights_list):
    weight_divergence_list = []
    n_client = len(weights_list)
    n_layer = int(len(weights_target)/2)

    for c_idx in range(n_client):
        weight_divergence = 0 

        for l in range(n_layer):
            input_size = len(weights_target[l*2])
            output_size = len(weights_target[l*2][0])

            for o in range(output_size):
                for i in range(input_size):
                
                    weight_divergence += (weights_target[l*2][i][o] - weights_list[c_idx][l*2][i][o])**2
                weight_divergence += (weights_target[l*2+1][o] - weights_list[c_idx][l*2+1][o])**2

        weight_divergence_list.append(weight_divergence)


    return weight_divergence_list

def print_correlation_between_label_similarity_and_pruned_model_divergence(client_train_list,weights_list):
    n_client = len(weights_list)

    label_similarity_list = []
    weights_divergence_list = []

    for c_idx in range(n_client):
        weight_divergence_list = calculate_weight_divergence_all(weights_list[c_idx],weights_list)
        
        for ref_c_idx in range(n_client):
            if c_idx == ref_c_idx:
                pass
            else:
                label_1 = np.unique(client_train_list[c_idx][1])
                label_2 = np.unique(client_train_list[ref_c_idx][1])

                label_similarity = len(set(label_1)&set(label_2))
                label_similarity_list.append(label_similarity)

                weights_divergence_list.append(weight_divergence_list[ref_c_idx])

    corr, _ = pearsonr(label_similarity_list, weights_divergence_list)

    print("correlation between label similarity and weight divergence = ",corr)


def load_weights(model_name,model_id):

    filename = "data/"+ str(model_name) +"/" + str(model_id)

    model = tf.keras.models.load_model(filename)

    weights = model.get_weights()

    return weights