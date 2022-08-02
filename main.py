import numpy as np

import copy
import os 
import gc 
import pickle
import csv
from datetime import date

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.data import *
from src.models import *
from src.pruning import *
from src.sub_fedavg import * 
from src.client import * 
from src.utils.options_u import args_parser 
from src.ours.our_algorithm_utils import *
from src.fedspa.utils import *
from src.fedspa.rigil import *

today = date.today()



args = args_parser()




if args.algorithm == "fedspa":
    torch.set_default_tensor_type('torch.DoubleTensor')

args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

try:
    torch.cuda.set_device(args.gpu) ## Setting cuda on GPU 
except:
    print("no gpu")
    pass
## Data partitioning section 

if args.dataset == 'cifar10':
    data_dir = '../data/cifar10/'
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                   transform=apply_transform)

    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)
    
    nclass_cifar10 = args.nclass
    nsamples_cifar10 = args.nsample_pc
    
    if args.noniid: 
        if args.shard:
            if args.load_data:
                print(" load non IID dataset")
                print(f'--CIFAR-10 Non-IID-- {args.nclass} random Shards, Sample per shard {args.nsample_pc}')

                # Load dataset
                file_name_train = 'src/data/'  + str(args.dataset) + "/train.p"
                with open(file_name_train, 'rb') as fp:
                    user_groups_train = pickle.load(fp)
                
                file_name_test = 'src/data/'  + str(args.dataset) + "/test.p"
                with open(file_name_test, 'rb') as fp:
                    user_groups_test = pickle.load(fp)
                
                file_name_val = 'src/data/'  + str(args.dataset) + "/val.p"
                with open(file_name_val, 'rb') as fp:
                    user_groups_val = pickle.load(fp)

            else:
                print(f'--CIFAR-10 Non-IID-- {args.nclass} random Shards, Sample per shard {args.nsample_pc}')
                user_groups_train, user_groups_test, user_groups_val = noniid_shard(args.dataset, train_dataset, test_dataset, 
                                args.num_users, nclass_cifar10, nsamples_cifar10, args.split_test)
            
        elif args.label: 
            print(f'--CIFAR-10 Non-IID-- {args.nclass} random Label, Sample per label {args.nsample_pc}')
            user_groups_train, user_groups_test = \
            noniid_label(args.dataset, train_dataset, test_dataset, args.num_users, nclass_cifar10,
                                 nsamples_cifar10, args.split_test)
            
        else: 
            exit('Error: unrecognized partitioning type')
    else: 
        print(f'--CIFAR-10 IID-- Split Test {args.split_test}')
        user_groups_train, user_groups_test = \
        iid(args.dataset, train_dataset, test_dataset, args.num_users, args.split_test)
            
elif args.dataset == 'cifar100':
    data_dir = '../data/cifar100/'
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])])
    
    train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=apply_transform)
    
    test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=apply_transform)
    
    nclass_cifar100 = args.nclass
    nsamples_cifar100 = args.nsample_pc
    
    if args.noniid: 
        if args.shard:
            print(f'--CIFAR-100 Non-IID-- {args.nclass} random Shards, Sample per shard {args.nsample_pc}')
            user_groups_train, user_groups_test, user_groups_val = noniid_shard(args.dataset, train_dataset, test_dataset, 
                        args.num_users, nclass_cifar100, nsamples_cifar100, args.split_test)
            
        elif args.label: 
            print(f'--CIFAR-100 Non-IID-- {args.nclass} random Labels, Sample per label {args.nsample_pc}')
            user_groups_train, user_groups_test = \
            noniid_label(args.dataset, train_dataset, test_dataset, args.num_users, nclass_cifar100,
                                 nsamples_cifar100, args.split_test)
        else: 
            exit('Error: unrecognized partitioning type')
    else:
        print(f'--CIFAR-100 IID-- Split Test {args.split_test}')
        user_groups_train, user_groups_test = \
        iid(args.dataset, train_dataset, test_dataset, args.num_users, args.split_test)
            
elif args.dataset == 'mnist': 
    data_dir = '../data/mnist/'
    apply_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)
    
    nclass_mnist = args.nclass
    nsamples_mnist = args.nsample_pc
    
    if args.noniid: 
        if args.shard:
            print(f'--MNIST Non-IID-- {args.nclass} random Shards, Sample per shard {args.nsample_pc}')
            user_groups_train, user_groups_test, user_groups_val = noniid_shard(args.dataset, train_dataset, test_dataset, 
                            args.num_users, nclass_mnist, nsamples_mnist, args.split_test)
        elif args.label: 
            print(f'--MNIST Non-IID-- {args.nclass} random Labels, Sample per label {args.nsample_pc}')
            user_groups_train, user_groups_test = \
            noniid_label(args.dataset, train_dataset, test_dataset, args.num_users, nclass_mnist,
                                 nsamples_mnist, args.split_test)        
        else: 
            exit('Error: unrecognized partitioning type')
    else: 
        print(f'--MNIST IID-- Split Test {args.split_test}')
        user_groups_train, user_groups_test = \
        iid(args.dataset, train_dataset, test_dataset, args.num_users, args.split_test)
        
## 
## Checking the partitions (total sample and labels for each client)

users_train_labels = {i: [] for i in range(args.num_users)}
users_test_labels = {i: [] for i in range(args.num_users)}

train_targets = np.array(train_dataset.targets)
test_targets = np.array(test_dataset.targets)

for i in range(args.num_users):
    ## Train Data for Each Client 
    train_count_per_client = 0 
    label = train_targets[user_groups_train[i]]
    train_count_per_client += len(label)
    label = set(label)
    users_train_labels[i] = list(label)
    
    # Test Data for Each Client 
    test_count_per_client = 0 
    label = test_targets[user_groups_test[i]]
    test_count_per_client += len(label)
    label = set(label)
    users_test_labels[i] = list(label) 
    
    #print(f'Client: {i}, Train Labels: {users_train_labels[i]}, Test Labels: {users_test_labels[i]},'
          #f' Num Train: {train_count_per_client}, Num Test: {test_count_per_client}')
        
## 
# build model
print(f'MODEL: {args.model}, Dataset: {args.dataset}')

users_model = []
if args.model == 'lenet5' and args.dataset == 'cifar10':
    net_glob = LeNet5Cifar10().to(args.device)
    net_glob.apply(weight_init)
    users_model = [LeNet5Cifar10().to(args.device).apply(weight_init) for _ in range(args.num_users)]
elif args.model == 'lenet5' and args.dataset == 'cifar100':
    net_glob = LeNet5Cifar100().to(args.device)
    net_glob.apply(weight_init)
    users_model = [LeNet5Cifar100().to(args.device).apply(weight_init) for _ in range(args.num_users)]
elif args.model == 'lenet5' and args.dataset == 'mnist':
    net_glob = LeNet5Mnist().to(args.device)
    net_glob.apply(weight_init)
    users_model = [LeNet5Mnist().to(args.device).apply(weight_init) for _ in range(args.num_users)]
elif args.model == 'fl' and args.dataset == 'cifar10':
    net_glob = FLCifar10().to(args.device)
    net_glob.apply(weight_init)
    users_model = [FLCifar10().to(args.device).apply(weight_init) for _ in range(args.num_users)]


if args.load_initial:
    file_path = "src/data/weights/" + str(args.model)  + str("_seed_") + str(args.seed) + ".pt"
    initial_state_dict = torch.load(file_path)

    net_glob.load_state_dict(initial_state_dict)
else:
    # save initial weights
    file_path = "src/data/weights/" + str(args.model)  + str("_seed_") + str(args.seed) + ".pt"
    torch.save(net_glob.state_dict(),file_path)

initial_state_dict = copy.deepcopy(net_glob.state_dict())
server_state_dict = copy.deepcopy(net_glob.state_dict())



for i in range(args.num_users):
    users_model[i].load_state_dict(initial_state_dict)
    
## 
mask_init = make_init_mask(net_glob)

clients = []
    
for idx in range(args.num_users):
    clients.append(Client_Sub_Un(idx, copy.deepcopy(users_model[idx]), args.local_bs, args.local_ep, 
               args.lr, args.momentum, args.device, copy.deepcopy(mask_init), 
               args.pruning_target, train_dataset, user_groups_train[idx], 
               test_dataset, user_groups_test[idx],
               test_dataset, user_groups_val[idx])) 
    


if args.algorithm == "fedspa":
    sparsities_list = fedspa_mask_initialization(clients,args.pruning_target)
    S = []
    for item in sparsities_list[0].items():
        S.append(item[1])

    pruner_state_dict_list = []
    pruner_list = []
    for idx in range(args.num_users):

        W, _linear_layers_mask = get_W(clients[idx].get_net(), return_linear_layers_mask=True)
        N = [torch.numel(w) for w in W]
        obj = {
            'dense_allocation': 1-args.pruning_target/100,
            'S': S,
            'N': N,
            'hyperparams': {
                'delta_T': args.local_ep*2,
                'alpha': args.alpha,
                'T_end': args.rounds*args.local_ep*2,
                'ignore_linear_layers': False,
                'static_topo': False,
                'sparsity_distribution': "ERK",
                'grad_accumulation_n': 1,
            },
            'step': 0,
            'rigl_steps': 0,
            'backward_masks': clients[idx].get_mask(),
            '_linear_layers_mask': clients[idx].get_mask()[args.n_conv_layer:],
        }
        pruner_state_dict_list.append(obj)
        pruner_list.append(None)

## 
loss_train = []

init_tracc_pr = []  # initial train accuracy for each round 
final_tracc_pr = [] # final train accuracy for each round 

init_tacc_pr = []  # initial test accuarcy for each round 
final_tacc_pr = [] # final test accuracy for each round

init_tloss_pr = []  # initial test loss for each round 
final_tloss_pr = [] # final test loss for each round 

clients_best_acc = [0 for _ in range(args.num_users)]
w_locals, loss_locals = [], []
masks = []

init_local_tacc = []       # initial local test accuracy at each round 
final_local_tacc = []  # final local test accuracy at each round 

init_local_tloss = []      # initial local test loss at each round 
final_local_tloss = []     # final local test loss at each round 

ckp_avg_tacc = []
ckp_avg_pruning = []
ckp_avg_best_tacc_before = []
ckp_avg_best_tacc_after = []

# prepare columns to save results
csv_fields_each_round = ["round","num_users","frac","local_ep","local_bs","bs","lr","momentum","warmup_epoch","model","ks","in_ch","dataset","nclass","nsample_pc","noniid","pruning_percent","pruning_target","dist_thresh_fc","acc_thresh","weight-decay","seed","algorithm","avg_final_tacc","personalized_parameters_percentage","corr_label_network_similarity"]

# Create affinity matrix and select related clients
if args.algorithm == "ours":
    selected_idx_mat = create_selected_idx_mat(clients,args.n_conv_layer,args.parameter_to_multiply_avg) 
    print("selected_idx_mat = ",selected_idx_mat)

for iteration in range(args.rounds):

    print(f'###### ROUND {iteration+1} ######')
        
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    
    if args.is_print:
        #print(f'###### ROUND {iteration+1} ######')
        print(f'Clients {idxs_users}')

    if args.algorithm == "ours":
        updated_weights_list_with_pruning_status = []
    
    if args.algorithm == "fedspa":
        U_t_list = []
    
    for idx in idxs_users:
                    
        if (iteration+1 > 1) and (args.algorithm != "local_training"):
            dic = Sub_FedAvg_U_initial(copy.deepcopy(clients[idx].get_mask()), 
                                     copy.deepcopy(clients[idx].get_net()), server_state_dict)
            
            clients[idx].set_state_dict(dic) 
        
        loss, acc = clients[idx].eval_test()        
            
        init_local_tacc.append(acc)
        init_local_tloss.append(loss)
        
        if args.algorithm == "sub_fedavg":
            loss = clients[idx].train(args.pruning_percent, args.dist_thresh, args.acc_thresh, is_print=False)
        elif args.algorithm == "ours":
            mask_list = []
            for i in range(args.num_users):
                mask_list.append(copy.deepcopy(clients[i].get_mask()))
            loss,weights_list,prune = clients[idx].new_algorithm_client_update(iteration,args.delta_r,args.alpha,args.rounds,mask_list,selected_idx_mat[idx],args.n_conv_layer,args.acc_thresh,args.early_stop_sparse_tr,args.regrowth_param)
            updated_weights_list_with_pruning_status.append((weights_list,prune))
        elif args.algorithm == "fedspa":
            loss,U_t,pruner_state_dict,pruner = clients[idx].fedspa_client_update(pruner_state_dict_list[idx],args.pruning_target,args.rounds,args.alpha,pruner_list[idx])
            U_t_list.append(U_t)
            pruner_state_dict_list[idx] = copy.deepcopy(pruner_state_dict)
            pruner_list[idx] = pruner
        elif args.algorithm == "local_training" or args.algorithm == "fedavg":
            loss = clients[idx].local_train()
                    
        masks.append(copy.deepcopy(clients[idx].get_mask()))     
        w_locals.append(copy.deepcopy(clients[idx].get_state_dict()))
        loss_locals.append(copy.deepcopy(loss))
                        
        loss, acc = clients[idx].eval_test()
        
        if acc > clients_best_acc[idx]:
            clients_best_acc[idx] = acc
  
        final_local_tacc.append(acc)
        final_local_tloss.append(loss)

    if args.algorithm == "ours":
        mask_list = []
        pruned_rate_list = []
        for idx in idxs_users:
            mask_list.append(copy.deepcopy(clients[idx].get_mask()))
            pruned_rate_list.append(copy.deepcopy(clients[idx].get_pruning()))

        mask_list,pruned_rate_list = global_multi_criteria_pruning(updated_weights_list_with_pruning_status,mask_list,args.lambda_value,args.pruning_percent,args.n_conv_layer,pruned_rate_list,args.pruning_target)  
        counter = 0
        for idx in idxs_users:
            if updated_weights_list_with_pruning_status[counter][1] == True:
                clients[idx].set_mask(mask_list[counter])
                masks[counter] = copy.deepcopy(mask_list[counter])
                clients[idx].set_pruned(pruned_rate_list[counter])
                clients[idx].update_weights()
                w_locals[counter] = copy.deepcopy(clients[idx].get_state_dict())  

            counter += 1
                 
    if (args.algorithm != "fedspa") and (args.partial_global_update == False) and (args.algorithm != "local_training"):
        server_state_dict = Sub_FedAVG_U(server_state_dict, w_locals, masks)
    elif args.partial_global_update == True:
        local_averaged_server_state_dict = Sub_FedAVG_U(server_state_dict, w_locals, masks)
        server_state_dict = weighted_global_model_update(server_state_dict,local_averaged_server_state_dict,args.frac)
    elif args.algorithm == "local_training":
        pass
    else:
        server_state_dict = fedspa_global_model_update(server_state_dict,U_t_list)

    if args.algorithm == "ours":
        server_state_dict = fill_zero_weights(server_state_dict,args.n_conv_layer,layer_wise=args.layer_wise_fill_weights)
    
    # print loss
    loss_avg = sum(loss_locals) / len(loss_locals)
    avg_init_tloss = sum(init_local_tloss) / len(init_local_tloss)
    avg_init_tacc = sum(init_local_tacc) / len(init_local_tacc)
    avg_final_tloss = sum(final_local_tloss) / len(final_local_tloss)
    avg_final_tacc = sum(final_local_tacc) / len(final_local_tacc)
    
    if args.is_print:    
        print('## END OF ROUND ##')
        template = 'The Number Of Round  {:.3f} , Average Train loss {:.3f}'
        print(template.format(iteration+1, loss_avg))

        template = "AVG Init Test Loss: {:.3f}, AVG Init Test Acc: {:.3f}"
        print(template.format(avg_init_tloss, avg_init_tacc))

        template = "AVG Final Test Loss: {:.3f}, AVG Final Test Acc: {:.3f}"
        print(template.format(avg_final_tloss, avg_final_tacc))

        if iteration%args.print_freq == 0:
            print('--- PRINTING ALL CLIENTS STATUS ---')
            best_acc_before_pruning = []
            pruning_state = []
            current_acc = []
            for k in range(args.num_users):
                best_acc_before_pruning.append(clients[k].get_best_acc())
                pruning_state.append(clients[k].get_pruning())
                loss, acc = clients[k].eval_test() 
                current_acc.append(acc)
                
                template = ("Client {:3d}, labels {}, count {}, pruning_state {:3.3f}, "
                        "best_acc_befor_pruning {:3.3f}, after_pruning {:3.3f}, current_acc {:3.3f} \n")
                
                print(template.format(k, users_train_labels[k], clients[k].get_count(), pruning_state[-1], 
                                    best_acc_before_pruning[-1], clients_best_acc[k], current_acc[-1]))
                
            template = ("Round {:1d}, Avg Pruning {:3.3f}, Avg current_acc {:3.3f}, "
                        "Avg best_acc_before_pruning {:3.3f}, after_pruning {:3.3f}")
            
            print(template.format(iteration+1, np.mean(pruning_state), np.mean(current_acc), 
                                np.mean(best_acc_before_pruning), np.mean(clients_best_acc)))
            
            ckp_avg_tacc.append(np.mean(current_acc))
            ckp_avg_pruning.append(np.mean(pruning_state))
            ckp_avg_best_tacc_before.append(np.mean(best_acc_before_pruning))
            ckp_avg_best_tacc_after.append(np.mean(clients_best_acc))

            # Calculate personalized parameters ratio
            mask_list = []
            
            for k in range(args.num_users):
                mask_list.append(clients[k].get_mask())
                # print("sample mask",clients[k].get_mask()[0][0])
            personalized_parameters_ratio_list = calculate_avg_10_percent_personalized_weights_each_layer(mask_list,args.n_conv_layer)
            
            # Calculate correlation between label and network similarity
            corr_label_and_network_similarity = calculate_correlation_between_label_similarity_and_network_similarity(users_train_labels,mask_list,args.n_conv_layer)
            
            csv_fields_each_round = ["round","num_users","frac","local_ep","local_bs","bs","lr","momentum","warmup_epoch","model","ks","in_ch","dataset","nclass","nsample_pc","noniid","pruning_percent","pruning_target","dist_thresh_fc","acc_thresh","seed","algorithm","avg_final_tacc","personalized_parameters_percentage","corr_label_network_similarity","date","delta_r","alpha","regrowth_param","parameter_to_multiply_avg","lamda_value"]
            csv_rows_each_round = [[str(iteration),str(args.num_users),str(args.frac),str(args.local_ep),str(args.local_bs),str(args.bs),str(args.lr),str(args.momentum),str(args.warmup_epoch),str(args.model),str(args.ks),str(args.in_ch),str(args.dataset),str(args.nclass),str(args.nsample_pc),str(args.noniid),str(args.pruning_percent),str(args.pruning_target),str(args.dist_thresh),str(args.acc_thresh),str(args.seed),str(args.algorithm),np.mean(current_acc),str(personalized_parameters_ratio_list),str(corr_label_and_network_similarity),today,args.delta_r,args.alpha,args.regrowth_param,args.parameter_to_multiply_avg,args.lambda_value]]
            with open('src/data/log/training_log.csv', 'a') as f:
        
                # using csv.writer method from CSV package
                write = csv.writer(f)
                
                write.writerows(csv_rows_each_round)

    # update learning rate
    if args.lr_decay != 1:
        for k in range(args.num_users):
            clients[k].lr_decay(args.lr_decay)

        
    loss_train.append(loss_avg)
    
    init_tacc_pr.append(avg_init_tacc)
    init_tloss_pr.append(avg_init_tloss)
    
    final_tacc_pr.append(avg_final_tacc)
    final_tloss_pr.append(avg_final_tloss)
    
    ## clear the placeholders for the next round 
    masks.clear()
    w_locals.clear()
    loss_locals.clear()
    init_local_tacc.clear()
    init_local_tloss.clear()
    final_local_tacc.clear()
    final_local_tloss.clear()
    
    ## calling garbage collector 
    gc.collect()
    
## Printing Final Test and Train ACC / LOSS
test_loss = []
test_acc = []
train_loss = []
train_acc = []


for idx in range(args.num_users):  
    if args.algorithm == "fedavg":
        dic = Sub_FedAvg_U_initial(copy.deepcopy(clients[idx].get_mask()), 
                                     copy.deepcopy(clients[idx].get_net()), server_state_dict)
            
        clients[idx].set_state_dict(dic) 

    loss, acc = clients[idx].eval_test()
        
    test_loss.append(loss)
    test_acc.append(acc)
    
    loss, acc = clients[idx].eval_train()
    
    train_loss.append(loss)
    train_acc.append(acc)

test_loss = sum(test_loss) / len(test_loss)
test_acc = sum(test_acc) / len(test_acc)

train_loss = sum(train_loss) / len(train_loss)
train_acc = sum(train_acc) / len(train_acc)

print(f'Train Loss: {train_loss}, Test_loss: {test_loss}')
print(f'Train Acc: {train_acc}, Test Acc: {test_acc}')

# save final results

# Calculate personalized parameters ratio
mask_list = []
for k in range(args.num_users):
    mask_list.append(clients[k].get_mask())
personalized_parameters_ratio_list = calculate_avg_10_percent_personalized_weights_each_layer(mask_list,args.n_conv_layer)

# Calculate correlation between label and network similarity
corr_label_and_network_similarity = calculate_correlation_between_label_similarity_and_network_similarity(users_train_labels,mask_list,args.n_conv_layer)

csv_fields_each_round = ["round","num_users","frac","local_ep","local_bs","bs","lr","momentum","warmup_epoch","model","ks","in_ch","dataset","nclass","nsample_pc","noniid","pruning_percent","pruning_target","dist_thresh_fc","acc_thresh","seed","algorithm","avg_final_tacc","personalized_parameters_percentage","corr_label_network_similarity","date","delta_r","alpha","regrowth_param","parameter_to_multiply_avg","lambda_value"]
csv_rows_each_round = [[args.rounds,args.num_users,args.frac,args.local_ep,args.local_bs,args.bs,args.lr,args.momentum,args.warmup_epoch,args.model,args.ks,args.in_ch,args.dataset,args.nclass,args.nsample_pc,args.noniid,args.pruning_percent,args.pruning_target,args.dist_thresh,args.acc_thresh,args.seed,args.algorithm,test_acc,str(personalized_parameters_ratio_list),corr_label_and_network_similarity,today,args.delta_r,args.alpha,args.regrowth_param,args.parameter_to_multiply_avg,args.lambda_value]]
with open('src/data/log/final_results.csv', 'a') as f:

    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(csv_rows_each_round)

# save final masks and weights
for idx in range(args.num_users):
    final_mask = clients[idx].get_mask()
    final_weights = clients[idx].get_state_dict()

    file_name_mask = "src/data/masks/" + str(args.algorithm) +"_"+str(args.model) + "_" + str(args.num_users)  + "_" + str(args.dataset) + "_" + str(args.seed) + "_client_id_" + str(idx) + ".pickle"
    with open(file_name_mask, 'wb') as fp:
        pickle.dump(final_mask, fp)

    file_name_weights = "src/data/weights/" + str(args.algorithm) +"_"+str(args.model)+ "_" + str(args.num_users) + "_" + str(args.dataset) + "_" + str(args.seed) + "_client_id_" + str(idx) + ".pickle"
    with open(file_name_weights, 'wb') as fp:
        pickle.dump(final_weights, fp)

    file_name_weights = "src/data/labels/" + str(args.algorithm) +"_"+str(args.model)+ "_" + str(args.num_users) + "_" + str(args.dataset) + "_" + str(args.seed) + "_client_id_" + str(idx) + ".pickle"
    label = np.unique(users_train_labels[idx])
    with open(file_name_weights, 'wb') as fp:
        pickle.dump(label, fp)