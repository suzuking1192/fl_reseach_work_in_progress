
import numpy as np
import random
import copy
import statistics
import torch

def calculate_affinity_based_on_weight_divergence_of_locally_trained_models(locally_trained_weights_list,target_idx,n_conv_layer):
    n_client = len(locally_trained_weights_list)
    n_layer = int(len(locally_trained_weights_list[0])/2)
    affinity_based_on_weight_divergence_list = []

    weights_target = locally_trained_weights_list[target_idx]

    for c_idx in range(n_client):
        weight_divergence = 0 

        for l in range(n_layer):
            if l >= n_conv_layer:
                input_size = len(weights_target[l*2])
                output_size = len(weights_target[l*2][0])

                for o in range(output_size):
                    for i in range(input_size):
                    
                        weight_divergence += (weights_target[l*2][i][o] - locally_trained_weights_list[c_idx][l*2][i][o])**2
                
                # Bias
                bias_input_size = len(weights_target[l*2+1])
                for o in range(bias_input_size):
                    weight_divergence += (weights_target[l*2+1][o] - locally_trained_weights_list[c_idx][l*2+1][o])**2
            else:
                # conv layer
                n_dim_0 = len(weights_target[l*2])
                n_dim_1 = len(weights_target[l*2][0])
                n_dim_2 = len(weights_target[l*2][0][0])
                n_dim_3 = len(weights_target[l*2][0][0][0])
                for i in range(n_dim_0):
                    for j in range(n_dim_1):
                        for k in range(n_dim_2):
                            for m in range(n_dim_3):
                                

                                weight_divergence += (weights_target[l*2][i][j][k][m] - locally_trained_weights_list[c_idx][l*2][i][j][k][m])**2

                for o in range(n_dim_3):
                
                    weight_divergence += (weights_target[l*2+1][o] - locally_trained_weights_list[c_idx][l*2+1][o])**2


        affinity_based_on_weight_divergence_list.append(- weight_divergence)

    return affinity_based_on_weight_divergence_list


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

def create_selected_idx_mat(clients,n_conv_layer,parameter_to_multiply_avg):
    n_client = len(clients)
    # local training
    locally_trained_weights_list = []
    for c_idx in range(n_client):
        clients[c_idx].fake_train()

        weights_list = []
        for tensor in clients[c_idx].get_fake_state_dict().items():
            
            weights_list.append(tensor[1])

        
        locally_trained_weights_list.append(weights_list)

    # create affinity matrix
    affinity_mat = []
    for c_idx in range(n_client):
        affinty_list = calculate_affinity_based_on_weight_divergence_of_locally_trained_models(locally_trained_weights_list,c_idx,n_conv_layer)
        affinity_mat.append(affinty_list)

    # Create selected idx list
    selected_idx_mat = create_selected_client_idx_list(affinity_mat,parameter_to_multiply_avg)

    return selected_idx_mat

def regrowth_based_on_affinity_c_idxs(binary_mask,binary_mask_list_all,selected_client_idx,mask_readjustment_rate,n_conv_layer):
    mask = copy.deepcopy(binary_mask)
    
    n_layer = len(binary_mask)

    counter = 0
    overlap = 0 
    for l in range(n_layer):
        if l < n_conv_layer:
            # conv layer
            n_dim_0 = len(mask[l])
            n_dim_1 = len(mask[l][0])
            n_dim_2 = len(mask[l][0][0])
            n_dim_3 = len(mask[l][0][0][0])
            for i in range(n_dim_0):
                for j in range(n_dim_1):
                    for k in range(n_dim_2):
                        for m in range(n_dim_3):
                            if mask[l][i][j][k][m] == 0:
                                counter += 1
                                overlap_status = False
                                for c_idx in selected_client_idx:
                                    if binary_mask_list_all[c_idx][l][i][j][k][m] == 1:
                                        overlap_status = True
                                if overlap_status == True:
                                    overlap += 1
        else:
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
        if l < n_conv_layer:
            # conv layer
            n_dim_0 = len(mask[l])
            n_dim_1 = len(mask[l][0])
            n_dim_2 = len(mask[l][0][0])
            n_dim_3 = len(mask[l][0][0][0])
            for i in range(n_dim_0):
                for j in range(n_dim_1):
                    for k in range(n_dim_2):
                        for m in range(n_dim_3):
                            if mask[l][i][j][k][m] == 0:
                                overlap_status = False
                                for c_idx in selected_client_idx:
                                    if binary_mask_list_all[c_idx][l][i][j][k][m] == 1:
                                        overlap_status = True
                                if overlap_status == True:
                                    if idx_counter in idx_list:
                                        mask[l][i][j][k][m] = 1
                                    

                                    idx_counter += 1

        else:
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

def model_growing(mask,mask_adjustment_rate,n_conv_layer):
    updated_mask = copy.deepcopy(mask)
    # Count total number of pruned weights
    n_layer = len(mask)

    counter = 0
    total_count = 0
    for l in range(n_layer):
        if l < n_conv_layer:
            # conv layer
            n_dim_0 = len(mask[l])
            n_dim_1 = len(mask[l][0])
            n_dim_2 = len(mask[l][0][0])
            n_dim_3 = len(mask[l][0][0][0])
            for i in range(n_dim_0):
                for j in range(n_dim_1):
                    for k in range(n_dim_2):
                        for m in range(n_dim_3):
                            total_count += 1
                            if mask[l][i][j][k][m] == 0:
                                counter += 1
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
            # conv layer
            n_dim_0 = len(mask[l])
            n_dim_1 = len(mask[l][0])
            n_dim_2 = len(mask[l][0][0])
            n_dim_3 = len(mask[l][0][0][0])
            for i in range(n_dim_0):
                for j in range(n_dim_1):
                    for k in range(n_dim_2):
                        for m in range(n_dim_3):
                            if mask[l][i][j][k][m] == 0:
                                if idx_counter in idx_list:
                                    updated_mask[l][i][j][k][m] = 1
                                    

                                idx_counter += 1
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


def global_multi_criteria_pruning(weights_list_with_pruning_status,mask_list,lambda_value,pruned_rate_each_round,n_conv_layer,pruned_rate_list,pruned_target):
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

            # Normalize weights and standard deviation loss
            weights_list = []
            std_weights_list = []
            std_weights_list_multi_dim = []
            unpruned = 0
            num_weights = 0

            for l in range(n_layer):
                std_weights_list_multi_dim.append([])
                if l < n_conv_layer:
                    # conv layer
                    n_dim_0 = len(mask[l])
                    n_dim_1 = len(mask[l][0])
                    n_dim_2 = len(mask[l][0][0])
                    n_dim_3 = len(mask[l][0][0][0])
                    for i in range(n_dim_0):
                        std_weights_list_multi_dim[l].append([])
                        for j in range(n_dim_1):
                            std_weights_list_multi_dim[l][i].append([])
                            for k in range(n_dim_2):
                                std_weights_list_multi_dim[l][i][j].append([])
                                for m in range(n_dim_3):
                                    weights_list.append(weights[l*2][i][j][k][m])
                                    num_weights += 1

                                    if mask[l][i][j][k][m] != 0 :
                                        unpruned +=1
                                    
                                    # Calculate std
                                    if mask[l][i][j][k][m] == 0:
                                        std_weights_list.append(0)
                                        std_weights_list_multi_dim[l][i][j][k].append(0)
                                    else:
                                        weight_list_in_one_position = []
                                        weight_list_in_one_position_without_ref_weight = []
                                        for c_idx in range(n_client):
                                            if mask_list[c_idx][l][i][j][k][m] != 0:
                                                if c_id == c_idx:
                                                    weight_list_in_one_position.append(weights_list_with_pruning_status[c_idx][0][l*2][i][j][k][m].item())
                                                else:
                                                    weight_list_in_one_position_without_ref_weight.append(weights_list_with_pruning_status[c_idx][0][l*2][i][j][k][m].item())
                                                    weight_list_in_one_position.append(weights_list_with_pruning_status[c_idx][0][l*2][i][j][k][m].item())
                                        if (len(weight_list_in_one_position) == 0) or (len(weight_list_in_one_position_without_ref_weight) == 0):
                                            std_diff = 0
                                        else:
                                            std_diff = abs(statistics.pstdev(weight_list_in_one_position) - statistics.pstdev(weight_list_in_one_position_without_ref_weight))
                                        if std_diff == 0:
                                            std_weights_list.append(std_diff)
                                            std_weights_list_multi_dim[l][i][j][k].append(std_diff)
                                        else:
                                            std_weights_list.append(std_diff)
                                            std_weights_list_multi_dim[l][i][j][k].append(std_diff)


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
                                            weight_list_in_one_position.append(weights_list_with_pruning_status[c_idx][0][l*2][i][o].item())
                                        else:
                                            weight_list_in_one_position_without_ref_weight.append(weights_list_with_pruning_status[c_idx][0][l*2][i][o].item())
                                            weight_list_in_one_position.append(weights_list_with_pruning_status[c_idx][0][l*2][i][o].item())
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
                    # conv layer
                    n_dim_0 = len(mask[l])
                    n_dim_1 = len(mask[l][0])
                    n_dim_2 = len(mask[l][0][0])
                    n_dim_3 = len(mask[l][0][0][0])
                    for i in range(n_dim_0):
                        for j in range(n_dim_1):
                            for k in range(n_dim_2):
                                for m in range(n_dim_3):
                                    try:
                                        value = abs(weights[l*2][i][j][k][m]/sum_weights) - lambda_value * std_weights_list_multi_dim[l][i][j][k][m]/sum_std_weights
                                    except:
                                        value = abs(weights[l*2][i][j][k][m]/sum_weights)
                                    all_weights[(l,i,j,k,m)] = value
                                    counter += 1
             
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


            pruning_percent = 1 -  (unpruned/num_weights) * (1-pruned_rate_each_round/100)
            if pruning_percent > pruned_target/100:
                pruning_percent = pruned_target/100
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

def fill_zero_weights(state_dict,n_conv_layer):
    weights_list = []
    n_layer = int(len(state_dict)/2)

    weights = []
    for tensor in state_dict.items():
        weights.append(tensor[1])

    for l in range(n_layer):
        if l < n_conv_layer:
            # conv layer
            n_dim_0 = len(weights[l*2])
            n_dim_1 = len(weights[l*2][0])
            n_dim_2 = len(weights[l*2][0][0])
            n_dim_3 = len(weights[l*2][0][0][0])
            for i in range(n_dim_0):
                for j in range(n_dim_1):
                    for k in range(n_dim_2):
                        for m in range(n_dim_3):
                            if weights[l*2][i][j][k][m] == 0:
                                #weights[l*2][i][o] = np.random.normal(0, 0.001, 1)[0]
                                pass
                            else:
                                weights_list.append(weights[l*2][i][j][k][m].item())
        else:
            n_neurons_layer = weights[l*2].shape[0]
            n_output = weights[l*2].shape[1]

            for i in range(n_neurons_layer):
                for o in range(n_output):
                    if weights[l*2][i][o] == 0:
                        #weights[l*2][i][o] = np.random.normal(0, 0.001, 1)[0]
                        pass
                    else:
                        weights_list.append(weights[l*2][i][o].item())
    res = statistics.pstdev(weights_list)
    # Printing result
    #print("Standard deviation of weights is : " + str(res))

    for tensor in state_dict.items():
        if "weight" in tensor[0]:
            weight_list = []
            try:
                n_dim_0 = len(tensor[1])
                n_dim_1 = len(tensor[1][0])
                n_dim_2 = len(tensor[1][0][0])
                n_dim_3 = len(tensor[1][0][0][0])
                for i in range(n_dim_0):
                    weight_list.append([])
                    for j in range(n_dim_1):
                        weight_list[i].append([])
                        for k in range(n_dim_2):
                            weight_list[i][j].append([])
                            for m in range(n_dim_3):
                                if tensor[1][i][j][k][m] == 0:
                                    weight_list[i][j][k].append(np.random.normal(0, res, 1)[0])
                                else:
                                    weight_list[i][j][k].append(tensor[1][i][j][k][m].item())
                weight_array = np.array(weight_list)
                weight_tensor = torch.from_numpy(weight_array)
                state_dict[str(tensor[0])] = weight_tensor
                
            except:
                n_dim_0 = len(tensor[1])
                n_dim_1 = len(tensor[1][0])
                for i in range(n_dim_0):
                    weight_list.append([])
                    for j in range(n_dim_1):
                        
                        if tensor[1][i][j] == 0:
                            weight_list[i].append(np.random.normal(0, res, 1)[0])
                        else:
                            weight_list[i].append(tensor[1][i][j].item())
                weight_array = np.array(weight_list)
                weight_tensor = torch.from_numpy(weight_array)
                state_dict[str(tensor[0])] = weight_tensor
    


    return state_dict