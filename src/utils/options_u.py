import argparse
    
## CIFAR-10 has 50000 training images (5000 per class), 10 classes, 10000 test images (1000 per class)
## CIFAR-100 has 50000 training images (500 per class), 100 classes, 10000 test images (100 per class)
## MNIST has 60000 training images (min: 5421, max: 6742 per class), 10000 test images (min: 892, max: 1135
## per class) --> in the code we fixed 5000 training image per class, and 900 test image per class to be 
## consistent with CIFAR-10 

## CIFAR-10 Non-IID 250 samples per label for 2 class non-iid is the benchmark (500 samples for each client)

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--rounds', type=int, default=300, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--nclass', type=int, default=2, help="classes or shards per user")
    parser.add_argument('--nsample_pc', type=int, default=250, 
                        help="number of samples per class or shard for each client")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--warmup_epoch', type=int, default=0, help="the number of pretrain local ep")

    # model arguments
    parser.add_argument('--model', type=str, default='lenet5', help='model name')
    parser.add_argument('--ks', type=int, default=5, help='kernel size to use for convolutions')
    parser.add_argument('--in_ch', type=int, default=3, help='input channels of the first conv layer')
    parser.add_argument('--algorithm', type=str, default='sub_fedavg', help='model name')
    parser.add_argument('--n_conv_layer', type=int, default=2, help='number of conv layers')
    parser.add_argument('--n_layer', type=int, default=5, help='number of conv layers')
    parser.add_argument('--lr_decay', type=float, default=1, help="learning rate decay")
    parser.add_argument('--layer_wise_fill_weights', type=bool, default=False, help="whether filling zeros in global weights are handled layerwise or not")
    parser.add_argument('--early_stop_sparse_tr', type=int, default=1000, help="when to stop sparse training")

    # algorithm arguments
    parser.add_argument('--parameter_to_multiply_avg', type=float, default=0.8, help='parameter_to_multiply_avg')
    parser.add_argument('--delta_r', type=int, default=20, help='frequency of sparse training')
    parser.add_argument('--alpha', type=float, default=0.5, help='initial value of pruning')
    parser.add_argument('--lambda_value', type=float, default=1, help='weight of standard deviation in multi pruning')
    parser.add_argument('--partial_global_update', type=bool, default=False, help='whether a global model is updated partially in each round')
    parser.add_argument('--regrowth_param', type=float, default=0.5, help='percentage of regrowth from similar networks')
    parser.add_argument('--clustering', type=str, default="k_means", help='Clustering method')
    parser.add_argument('--n_cluster', type=int, default=3, help='number of clusters')
    parser.add_argument('--weight_regularization', type=float, default=0.001, help='weight_regularization')
    parser.add_argument('--transfer', type=str, default="positive", help='whether we observe positive transfer or negative trasnfer')
    parser.add_argument('--mask_sparse_initialization', type=bool, default=False, help='whether we start from sparse masks')


    # dataset partitioning arguments
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        help="name of dataset: mnist, cifar10, cifar100")
    parser.add_argument('--noniid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--load_data', action='store_true', help='whether load existing data or not')
    parser.add_argument('--shard', action='store_true', help='whether non-i.i.d based on shard or not')
    parser.add_argument('--label', action='store_true', help='whether non-i.i.d based on label or not')
    parser.add_argument('--split_test', action='store_true', 
                        help='whether split test set in partitioning or not')
    
    # pruning arguments 
    parser.add_argument('--pruning_percent', type=float, default=10, 
                        help="Pruning percent for layers (0-100)")
    parser.add_argument('--pruning_target', type=int, default=30, 
                        help="Total Pruning target percentage (0-100)")
    parser.add_argument('--dist_thresh', type=float, default=0.0001, 
                        help="threshold for fcs masks difference ")
    parser.add_argument('--acc_thresh', type=int, default=50, 
                        help="accuracy threshold to apply the derived pruning mask")
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
    # other arguments 
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--is_print', action='store_true', help='verbose print')
    parser.add_argument('--print_freq', type=int, default=100, help="printing frequency during training rounds")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--load_initial', action='store_true', help='define initial model path')
    #parser.add_argument('--results_save', type=str, default='/', help='define fed results save folder')
    #parser.add_argument('--start_saving', type=int, default=0, help='when to start saving models')

    args = parser.parse_args()
    return args
