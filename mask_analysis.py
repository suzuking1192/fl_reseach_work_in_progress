import argparse
import pickle
from src.pruning.unstructured import *
# Prepare parser

def args_parser_mask():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--algorithm', type=str, default='sub_fedavg', help='model name')
    parser.add_argument('--model', type=str, default='lenet5', help='model name')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        help="name of dataset: mnist, cifar10, cifar100")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--n_conv_layer', type=int, default=2, help='number of conv layers')
    parser.add_argument('--percentage', type=float, default=0.2, help='percentage of selected clients')
    parser.add_argument('--top', type=str, help='whether we select top overlapped networks or worst')

    args = parser.parse_args()
    return args

args = args_parser_mask()

# Import masks

mask_list = []
label_list = []
for idx in range(args.num_users):
    file_name_mask = "src/data/masks/" + str(args.algorithm) +"_"+str(args.model) + "_" + str(args.num_users)  + "_" + str(args.dataset) + "_" + str(args.seed) + "_client_id_" + str(idx) + ".pickle"
    # open a file, where you stored the pickled data
    file = open(file_name_mask, 'rb')

    # dump information to that file
    mask = pickle.load(file)

    mask_list.append(mask)


    # Import labels
    file_name_weights = "src/data/labels/" + str(args.algorithm) +"_"+str(args.model)+ "_" + str(args.num_users) + "_" + str(args.dataset) + "_" + str(args.seed) + "_client_id_" + str(idx) + ".pickle"
    
    file = open(file_name_weights, 'rb')
    label = pickle.load(file)

    label_list.append(label)



# Calculate mask difference 

n_client = len(mask_list)

top_label_similarity_list = []

select_users_num = int(n_client*args.percentage)

for c_idx in range(n_client):
    affinity_list = calculate_affinity_based_on_network(mask_list[c_idx],mask_list,args.n_conv_layer)
    #print("affinity_list = ",affinity_list)
    label_1 = label_list[c_idx]
    print("label_base = ",label_1)
    
    # Get top 6 indicies based on affinity measures
    

    if args.top == True:
        selected_idx = sorted(range(len(affinity_list)), key=lambda i: affinity_list[i])[-select_users_num:]
    else:

        selected_idx = sorted(range(len(affinity_list)), key=lambda i: affinity_list[i])[:select_users_num]

    for ref_c_idx in selected_idx:
        if c_idx == ref_c_idx:
            pass
        else:
            label_2 = label_list[ref_c_idx]

            print("selected label",label_2)

            label_similarity = len(set(label_1)&set(label_2))

            top_label_similarity_list.append(label_similarity)

    
print("average label overlap in top 5 overlapped network",sum(top_label_similarity_list)/len(top_label_similarity_list))
