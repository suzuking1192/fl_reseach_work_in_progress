import numpy as np
from scipy import linalg
from scipy.spatial import distance
from scipy.stats import pearsonr


import torch
import torch.nn as nn 

def make_init_mask(model, is_print=False):
    '''
    Makes the initial pruning mask for the given model. For example, for LeNet-5 architecture it return a list of
    5 arrays, each array is the same size of each layer's weights and with all 1 entries. We do not prune bias
    
    :param model: a pytorch model 
    :return mask: a list of pruning masks 
    '''
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            step = step + 1
    mask = [None]* step 

    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            if is_print:
                print(f'step {step}, shape: {mask[step].shape}')
            step = step + 1
            
    return mask

def fake_prune(percent, model, mask):
    '''
    This function derives the new pruning mask, it put 0 for the weights under the given percentile
    
    :param percent: pruning percent 
    :param model: a pytorch model 
    :param mask: the pruning mask 
    
    :return mask: updated pruning mask
    '''
    # Calculate percentile value
    step = 0
    for name, param in model.named_parameters():
        
        # We do not prune bias term
        if 'weight' in name:
            weight_dev = param.device
            
            tensor = param.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor * mask[step])] # flattened array of nonzero values
            percentile_value = np.percentile(abs(alive), percent)
            
            new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

            # Apply mask
            mask[step] = new_mask
            step += 1

    return mask

def real_prune(model, mask):
    '''
    This function applies the derived mask. It zeros the weights needed to be pruned based on the updated mask
    
    :param model: a pytorch model 
    :param mask: pruning mask 
    
    :return state_dict: updated (pruned) model state_dict
    '''
    # Calculate percentile value
    step = 0
    for name, param in model.named_parameters():

        # We do not prune bias term
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            weight_dev = param.device

            # Apply new weight and mask
            param.data = torch.from_numpy(tensor * mask[step]).to(weight_dev)
            step += 1
            
    return model.state_dict()

def print_pruning(model, is_print = False):
    '''
    This function prints the pruning percentage and status of a given model 
    
    :param model: a pytorch model 
    
    :return pruning percentage, number of remaining weights: 
    '''
    nonzero = 0
    total = 0
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
    if is_print: 
        print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total},'
              f'Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:3.2f}% pruned)')
    return 100 * (total-nonzero) / total, nonzero

def dist_masks(m1, m2): 
    '''
    Calculates hamming distance of two pruning masks. It averages the hamming distance of all layers and returns it
    
    :param m1: pruning mask 1 
    :param m2: pruning mask 2 
    
    :return average hamming distance of two pruning masks: 
    '''
    temp_dist = []
    for step in range(len(m1)): 
        #1 - float(m1[step].reshape([-1]) == m2[step].reshape([-1])) / len(m2[step].reshape([-1]))
        temp_dist.append(distance.hamming(m1[step].reshape([-1]), m2[step].reshape([-1])))
    dist = np.mean(temp_dist)
    return dist



def calculate_avg_10_percent_personalized_weights_each_layer(mask_list,n_conv_layer=0):
    n_client = len(mask_list)
    n_layer = len(mask_list[0])



    personalized_percentage_list = []

    for l in range(n_layer):
        personalized_percentage_list.append([])

    for c_idx in range(n_client):
        #print("sample mask",mask_list[c_idx][3][0])
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
                                if ref_c_idx != c_idx:
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
                                        if ref_c_idx != c_idx:
                                            if mask_list[ref_c_idx][l][i][j][k][m] == 1:
                                                overlap_num += 1
                                    if overlap_num <= 0.1 *(n_client):
                                        
                                        personalized_num += 1

            
            #print("total_unpruned_weights_num = ",total_unpruned_weights_num)
            personalized_percentage_list[l].append(personalized_num/total_unpruned_weights_num)
    
    personalized_parameters_rate_list = []
    for l in range(n_layer):
        print("layer idx =",l)
        print("average percentage of personalized weights = ", sum(personalized_percentage_list[l])/n_client)
        personalized_parameters_rate_list.append(sum(personalized_percentage_list[l])/n_client)

    return personalized_parameters_rate_list


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


def calculate_correlation_between_label_similarity_and_network_similarity(users_train_labels,mask_list,n_conv_layer=0):
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
                label_1 = np.unique(users_train_labels[c_idx])
                label_2 = np.unique(users_train_labels[ref_c_idx])

                label_similarity = len(set(label_1)&set(label_2))
                label_similarity_list.append(label_similarity)

                network_similarity_list.append(affinity_list[ref_c_idx])

    corr, _ = pearsonr(label_similarity_list, network_similarity_list)

    print("correlation between label similarity and network similarity = ",corr)
    return corr


