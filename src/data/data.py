
from cmath import nan
import copy
import numpy as np 
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pickle

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def noniid_shard(dataset_name, train_dataset, test_dataset, num_users, n_class, nsamples_pc, split_test = False):
    """
    Partitioning cifar10 non-iid amongst clients based on number of shards. For example if n_class is 2,
    each partition will have 2 random shards --> if may end up with 2 random labels or the both shards have 
    the same label, then the client will have one label. 
    
    :param dataset_name: name of dataset
    :param train_dataset: pytorch train dataset 
    :param test_dataset: pytorch test dataset
    :param num_users: number of users to partition dataset 
    :param n_class: number of random labels to be assigned to each client 
    :param nsamples_pc: number of samples per class (label)
    :param split_test: splitting test data amongst clients --> if False, then clients will have the 
     full test data based of the labels they have! 
    
    :return: users_train_groups, users_test_groups
    """
    
    if dataset_name == 'cifar10':
        num_classes = 10
        img_train_pc = 5000 
        img_test_pc = 1000
    elif dataset_name == 'cifar100':
        num_classes = 100
        img_train_pc = 500
        img_test_pc = 100
    elif dataset_name == 'mnist':
        num_classes = 10
        img_train_pc = 5000 
        img_test_pc = 900
        
    num_shards_train, num_imgs_train_per_shard = int(len(train_dataset)/nsamples_pc), nsamples_pc
    num_imgs_test_per_client, num_imgs_test_total = img_test_pc, len(test_dataset)
    
    ## checking 
    assert(n_class * num_users <= num_shards_train)
    assert(n_class <= num_classes)
    
    idx_class = [i for i in range(num_classes)]
    idx_shard = [i for i in range(num_shards_train)]
    
    dict_users_train = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_test = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    num_samples_test_per_class = int(img_test_pc/num_users)
    num_shards_test_per_class = int(img_test_pc/num_samples_test_per_class)
    idx_shards_test_y = {j: [i for i in range(num_shards_test_per_class)] for j in range(num_classes)}
    idx_test_y = {i: [] for i in range(num_classes)}

    # sort train data based on labels 
    idxs_train = np.arange(num_shards_train*num_imgs_train_per_shard)
    labels_train = np.array(train_dataset.targets)
    idxs_labels_train = np.vstack((idxs_train, labels_train))
    idxs_labels_train = idxs_labels_train[:, idxs_labels_train[1, :].argsort()]
    idxs_train = idxs_labels_train[0, :]
    labels_train = idxs_labels_train[1, :]
    
    # sort test data based on labels 
    idxs_test = np.arange(num_imgs_test_total)
    labels_test = np.array(test_dataset.targets)
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]
    labels_test = idxs_labels_test[1, :]
    
    for i in range(num_classes):
        idx_test_y[i] = idxs_test[np.where(labels_test == i)[0]]
    
    # divide and assign
    for i in range(num_users):
        user_labels = np.array([])
        # avoid client having only one class
        while len(np.unique(user_labels)) < 2:

            user_labels = np.array([])
            dict_users_train[i] = np.array([], dtype='int64')
            rand_set = set(np.random.choice(idx_shard, n_class, replace=False))
          
            for rand in rand_set:
                dict_users_train[i] = np.concatenate((dict_users_train[i], \
                        idxs_train[rand*num_imgs_train_per_shard:(rand+1)*num_imgs_train_per_shard]), axis=0)
                
                user_labels = np.concatenate((user_labels, labels_train[rand*num_imgs_train_per_shard:(rand+1)*num_imgs_train_per_shard]), axis=0)
        idx_shard = list(set(idx_shard) - rand_set) 
        user_labels_set = set(user_labels)
        for label in user_labels_set:
            if split_test: 
                rand_shard = set(np.random.choice(idx_shards_test_y[label], 1, replace=False))
                idx_shards_test_y[label] = list(set(idx_shards_test_y[label]) - rand_shard)
                shard = list(rand_shard)[0]
                
                iidxx = idx_test_y[label][shard*num_samples_test_per_class:(shard+1)*num_samples_test_per_class]
            else: 
                iidxx = idx_test_y[label]
                
            dict_users_test[i] = np.concatenate((dict_users_test[i], iidxx), axis=0)
            
        #print(set(labels_test_raw[dict_users_test[i].astype(int)]))

        # divide test and validation data
        percentage_of_val = 0.2
        num_test_pc = len(dict_users_test[i])
        dict_users_val[i] = copy.deepcopy(dict_users_test[i][: int(num_test_pc*percentage_of_val)])
        dict_users_test[i] = dict_users_test[i][int(num_test_pc*percentage_of_val):]
    
    # Save data
    file_name_train = 'src/data/'  + str(dataset_name) + "/train.p"
    with open(file_name_train, 'wb') as fp:
        pickle.dump(dict_users_train, fp)
    
    file_name_test = 'src/data/'  + str(dataset_name) + "/test.p"
    with open(file_name_test, 'wb') as fp:
        pickle.dump(dict_users_test, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_name_val = 'src/data/'  + str(dataset_name) + "/val.p"
    with open(file_name_val, 'wb') as fp:
        pickle.dump(dict_users_val, fp, protocol=pickle.HIGHEST_PROTOCOL)

    return dict_users_train, dict_users_test,dict_users_val


def noniid_label(dataset_name, train_dataset, test_dataset, num_users, n_class, nsamples_pc, split_test = False):
    """
    Partitioning Dataset non-iid amongst clients based on number of labels. For example if n_class is 2,
    each partition will have exactly 2 random labels. 
    
    :param dataset_name: name of dataset
    :param train_dataset: pytorch train dataset 
    :param test_dataset: pytorch test dataset
    :param num_users: number of users to partition dataset 
    :param n_class: number of random labels to be assigned to each client 
    :param nsamples_pc: number of samples per class (label)
    
    :return: users_train_groups, users_test_groups
    """
    if dataset_name == 'cifar10':
        num_classes = 10
        img_train_pc = 5000 
        img_test_pc = 1000
    elif dataset_name == 'cifar100':
        num_classes = 100
        img_train_pc = 500
        img_test_pc = 100
    elif dataset_name == 'mnist':
        num_classes = 10
        img_train_pc = 5000 
        img_test_pc = 900
        
    num_samples_train_per_class = nsamples_pc
    num_shards_train_per_class = int(img_train_pc/num_samples_train_per_class)
    idx_shards_train_y = {j: [i for i in range(num_shards_train_per_class)] for j in range(num_classes)}

    num_samples_test_per_class = int(img_test_pc/num_users)
    num_shards_test_per_class = int(img_test_pc/num_samples_test_per_class)
    idx_shards_test_y = {j: [i for i in range(num_shards_test_per_class)] for j in range(num_classes)}

    users_train_idx = {i: np.array([], dtype='int64') for i in range(num_users)}
    users_train_y = {i: [] for i in range(num_users)}
    users_count_train = {i: {} for i in range(num_users)} # For check

    users_test_idx = {i: np.array([], dtype='int64') for i in range(num_users)}
    users_test_y = {i: [] for i in range(num_users)}
    users_count_test = {i: {} for i in range(num_users)}

    count_train_y = {i:0 for i in range(num_classes)} # For check 
    idx_train_y = {i: [] for i in range(num_classes)}

    count_test_y = {i:0 for i in range(num_classes)} # For check 
    idx_test_y = {i: [] for i in range(num_classes)}

    # sort train labels
    idxs_train = np.arange(len(train_dataset))
    labels_train = np.array(train_dataset.targets)
    idxs_labels = np.vstack((idxs_train, labels_train))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs_train_x = copy.deepcopy(idxs_labels[0,:])
    idxs_train_y = copy.deepcopy(idxs_labels[1,:])

    # sort test labels
    idxs_test = np.arange(len(test_dataset))
    labels_test = np.array(test_dataset.targets)
    idxs_labels = np.vstack((idxs_test, labels_test))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs_test_x = copy.deepcopy(idxs_labels[0,:])
    idxs_test_y = copy.deepcopy(idxs_labels[1,:])

    # checking 
    for i in range(num_classes):
        count_train_y[i] = len(np.where(idxs_train_y == i)[0])
        idx_train_y[i] = idxs_train_x[np.where(idxs_train_y == i)[0]]

        count_test_y[i] = len(np.where(idxs_test_y == i)[0])
        idx_test_y[i] = idxs_test_x[np.where(idxs_test_y == i)[0]]

    #print(f'Count_train_y: {count_train_y}, Count_test_y: {count_test_y}')
    
    # divide and assign
    total_labels = np.arange(num_classes)
    for i in range(num_users):
        if len(total_labels) == 1: 
            rand_labels = total_labels
        else:
            rand_labels = set(np.random.choice(total_labels, n_class, replace=False))
        
        #print(f'Total Label: {total_labels}, rand_labels: {rand_labels}')
        
        count_train_data = 0 
        count_test_data = 0 
        for label in rand_labels:
            ## Train Data 
            rand_shard = set(np.random.choice(idx_shards_train_y[label], 1, replace=False))
            idx_shards_train_y[label] = list(set(idx_shards_train_y[label]) - rand_shard)

            shard = list(rand_shard)[0]

            if len(idx_shards_train_y[label]) == 0: 
                total_labels = np.setdiff1d(total_labels, np.array(label))
                print(f'Label {label} is Done!!!')
            ## assigning train data 
            iidxx = idx_train_y[label][shard*num_samples_train_per_class:(shard+1)*num_samples_train_per_class]
            users_train_idx[i] = np.concatenate((users_train_idx[i], iidxx), axis=0)
            users_train_y[i] = np.concatenate((users_train_y[i], np.array(train_dataset.targets)[iidxx]), axis=0)
            users_count_train[i][label] = len(users_train_idx[i]) - count_train_data
            count_train_data = len(users_train_idx[i])

            ## assigning test data 
            rand_shard = set(np.random.choice(idx_shards_test_y[label], 1, replace=False))
            idx_shards_test_y[label] = list(set(idx_shards_test_y[label]) - rand_shard)
            shard = list(rand_shard)[0]
            
            if split_test: 
                ## uncomment the following line if you want each user has a sub-set of the test data 
                ## otherwise each user will have all the test data for the labels that it  has 
                iidxx = idx_test_y[label][shard*num_samples_test_per_class:(shard+1)*num_samples_test_per_class]
            else: 
                iidxx = idx_test_y[label]
            users_test_idx[i] = np.concatenate((users_test_idx[i], iidxx), axis=0)
            users_test_y[i] = np.concatenate((users_test_y[i], np.array(test_dataset.targets)[iidxx]), axis=0)
            users_count_test[i][label] = len(users_test_idx[i]) - count_test_data
            count_test_data = len(users_test_idx[i])

        user_labels_set = set(users_train_y[i])
        users_train_y[i] = user_labels_set

        user_labels_set = set(users_test_y[i])
        users_test_y[i] = user_labels_set

    #print(f'Users_count_train: {users_count_train}')
    #print(f'Users_count_test: {users_count_test}')

    return users_train_idx, users_test_idx

def iid(dataset_name, train_dataset, test_dataset, num_users, split_test = False):
    """
    Partitioning Dataset I.I.D. amongst clients. Each client will have all the labels  
    
    :param dataset_name: name of dataset
    :param train_dataset: pytorch train dataset 
    :param test_dataset: pytorch test dataset
    :param num_users: number of users to partition dataset 
    
    :return: users_train_groups, users_test_groups
    """
    num_items = int(len(train_dataset)/num_users)
    all_idxs = np.arange(len(train_dataset))
    idxs_test = np.arange(len(test_dataset))
    num_items_test = int(len(test_dataset)/num_users)
    
    users_train_idx = {i: np.array([], dtype='int64') for i in range(num_users)}
    users_test_idx = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    for i in range(num_users):
        ## assigning train data 
        selected = set(np.random.choice(all_idxs, num_items, replace=False))
        users_train_idx[i] = np.concatenate((users_train_idx[i], list(selected)), axis=0)
        #dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - selected)
        
        ## assigning test data 
        if args.split_test:
            test_selected = set(np.random.choice(idxs_test, num_items_test, replace=False))
            idxs_test = list(set(idxs_test) - test_selected)
            test_selected = list(test_selected)
        else: 
            test_selected = list(idxs_test)
        users_test_idx[i] = np.concatenate((users_test_idx[i], test_selected), axis=0)
        
    return users_train_idx, users_test_idx

def noniid_dirichlet(dataset_name, train_dataset, test_dataset, num_users, n_class, nsamples_pc, split_test = False):
    """
    Partitioning cifar10 non-iid amongst clients based on number of shards. For example if n_class is 2,
    each partition will have 2 random shards --> if may end up with 2 random labels or the both shards have 
    the same label, then the client will have one label. 
    
    :param dataset_name: name of dataset
    :param train_dataset: pytorch train dataset 
    :param test_dataset: pytorch test dataset
    :param num_users: number of users to partition dataset 
    :param n_class: number of random labels to be assigned to each client 
    :param nsamples_pc: number of samples per class (label)
    :param split_test: splitting test data amongst clients --> if False, then clients will have the 
     full test data based of the labels they have! 
    
    :return: users_train_groups, users_test_groups
    """
    
    if dataset_name == 'cifar10':
        num_classes = 10
        img_train_pc = 5000 
        img_test_pc = 1000
        alpha = 0.3
        
    elif dataset_name == 'cifar100':
        num_classes = 100
        img_train_pc = 500
        img_test_pc = 100
        alpha = 0.2
    elif dataset_name == 'mnist':
        num_classes = 10
        img_train_pc = 5000 
        img_test_pc = 900
        alpha = 0.3
    dir_test_per_user = int(img_test_pc*0.2*num_classes*0.8)
    dir_val_per_user = int(img_test_pc*0.2*num_classes*0.2)

        
    # num_shards_train, num_imgs_train_per_shard = int(len(train_dataset)/nsamples_pc), nsamples_pc
    num_imgs_test_per_client, num_imgs_test_total = img_test_pc, len(test_dataset)
    
    ## checking 
    # assert(n_class * num_users <= num_shards_train)
    assert(n_class <= num_classes)
    
    idx_class = [i for i in range(num_classes)]
    # idx_shard = [i for i in range(num_shards_train)]
    
    dict_users_train = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_test = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    # num_samples_test_per_class = int(img_test_pc/num_users)
    # num_shards_test_per_class = int(img_test_pc/num_samples_test_per_class)
    # idx_shards_test_y = {j: [i for i in range(num_shards_test_per_class)] for j in range(num_classes)}
    idx_train_y = {i: [] for i in range(num_classes)}
    idx_test_y = {i: [] for i in range(num_classes)}
    idx_val_y = {i: [] for i in range(num_classes)}
    n_sample_per_user = nsamples_pc * n_class
    # sort train data based on labels 
    idxs_train = np.arange(img_train_pc*num_classes)
    labels_train = np.array(train_dataset.targets[:img_train_pc*num_classes])
    idxs_labels_train = np.vstack((idxs_train, labels_train))
    idxs_labels_train = idxs_labels_train[:, idxs_labels_train[1, :].argsort()]
    idxs_train = idxs_labels_train[0, :]
    labels_train = idxs_labels_train[1, :]
    
    # sort test data based on labels 
    percentage_of_val = 0.2
    idxs_test = np.arange(num_imgs_test_total*(1-percentage_of_val))
    n_test = len(idxs_test)
    labels_test = np.array(test_dataset.targets[:n_test])
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]
    labels_test = idxs_labels_test[1, :]

    # sort val data based on labels 
    idxs_val = np.arange(num_imgs_test_total*percentage_of_val)
    labels_val = np.array(test_dataset.targets[n_test:])
    idxs_labels_val = np.vstack((idxs_val, labels_val))
    idxs_labels_val = idxs_labels_val[:, idxs_labels_val[1, :].argsort()]
    idxs_val = idxs_labels_val[0, :]
    labels_val = idxs_labels_val[1, :]
    
    for i in range(num_classes):
        idx_train_y[i] = idxs_train[np.where(labels_train == i)[0]]
        idx_test_y[i] = idxs_test[np.where(labels_test == i)[0]]
        idx_val_y[i] = idxs_val[np.where(labels_val == i)[0]]
    
    # assign
    cls_priors = np.random.dirichlet(alpha=[alpha] * num_classes, size=num_users)
    for i in range(num_users):
        
        
        # train dataset
        n_data_per_class_train = np.ceil(cls_priors[i]*n_sample_per_user).astype(int)
        if sum(n_data_per_class_train) > n_sample_per_user:
            diff = sum(n_data_per_class_train) - n_sample_per_user
            max_ele = max(n_data_per_class_train)
            index_max_value = np.where(n_data_per_class_train==max_ele)[0][0]
            reduce_class = index_max_value
            n_data_per_class_train[reduce_class] = n_data_per_class_train[reduce_class] -diff

        for cls in range(num_classes):
            
            assign_num_cls = n_data_per_class_train[cls]
            if assign_num_cls>0:
                

                train_idx_target_cls = np.random.choice(idx_train_y[cls],assign_num_cls)
                
                idx_train_y[cls]= np.setdiff1d(idx_train_y[cls], train_idx_target_cls)
                dict_users_train[i] = np.concatenate((dict_users_train[i], train_idx_target_cls.astype(int)), axis=0)

        # test dataset
        n_data_per_class_test = np.ceil(cls_priors[i]*dir_test_per_user).astype(int)
        if sum(n_data_per_class_test) > dir_test_per_user:
            diff = sum(n_data_per_class_test) - dir_test_per_user
            max_ele = max(n_data_per_class_test)
            index_max_value = np.where(n_data_per_class_test==max_ele)[0][0]
            reduce_class = index_max_value
            n_data_per_class_test[reduce_class] = n_data_per_class_test[reduce_class] -diff

        for cls in range(num_classes):
            
            assign_num_cls = n_data_per_class_test[cls]
            if assign_num_cls>0:
                test_idx_target_cls = np.random.choice(idx_test_y[cls],assign_num_cls)
                
                
                dict_users_test[i] = np.concatenate((dict_users_test[i], test_idx_target_cls.astype(int)), axis=0)
                    
        # val dataset
        n_data_per_class_val = np.ceil(cls_priors[i]*dir_val_per_user).astype(int)
        if sum(n_data_per_class_val) > dir_val_per_user:
            diff = sum(n_data_per_class_val) - dir_val_per_user
            max_ele = max(n_data_per_class_val)
            index_max_value = np.where(n_data_per_class_val==max_ele)[0][0]
            reduce_class = index_max_value
            n_data_per_class_val[reduce_class] = n_data_per_class_val[reduce_class] -diff

        for cls in range(num_classes):
            
            assign_num_cls = n_data_per_class_val[cls]
            if assign_num_cls>0:

                val_idx_target_cls = np.random.choice(idx_val_y[cls],assign_num_cls)
                
                
                dict_users_val[i] = np.concatenate((dict_users_val[i], val_idx_target_cls.astype(int)), axis=0)
                    
        
        



    
    # Save data
    file_name_train = 'src/data/'  + str(dataset_name) + "/num_user_"+str(num_users) + "dirichlet" + "train.p"
    with open(file_name_train, 'wb') as fp:
        pickle.dump(dict_users_train, fp)
    
    file_name_test = 'src/data/'  + str(dataset_name)+ "/num_user_"+str(num_users) + "dirichlet" + "test.p"
    with open(file_name_test, 'wb') as fp:
        pickle.dump(dict_users_test, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_name_val = 'src/data/'  + str(dataset_name)+ "/num_user_"+str(num_users) + "dirichlet" + "val.p"
    with open(file_name_val, 'wb') as fp:
        pickle.dump(dict_users_val, fp, protocol=pickle.HIGHEST_PROTOCOL)

    return dict_users_train, dict_users_test,dict_users_val
