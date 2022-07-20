import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split

def create_pathological_heterogeneous_datasets_specific(train_x,train_y,test_x,test_y,n_data,n_class_list):
    
    def select_sample(x,y,num_data):
        specifiec_classes_train_x = []
        specifiec_classes_train_y = []
        for i in range(len(x)):
            if y[i] in n_class_list:
                specifiec_classes_train_x.append(x[i])
                specifiec_classes_train_y.append(y[i])
        selected_id = random.sample(set(range(0,len(specifiec_classes_train_x))), num_data)
        selected_train_x = []
        selected_train_y = []
        for i in selected_id:
            selected_train_x.append(specifiec_classes_train_x[i])
            selected_train_y.append(specifiec_classes_train_y[i])
            
        return selected_train_x,selected_train_y
    
    selected_train_x,selected_train_y = select_sample(train_x,train_y,n_data)
    
    def select_all(x,y):
        specifiec_classes_train_x = []
        specifiec_classes_train_y = []
        for i in range(len(x)):
            if y[i] in n_class_list:
                specifiec_classes_train_x.append(x[i])
                specifiec_classes_train_y.append(y[i])
        return specifiec_classes_train_x,specifiec_classes_train_y

    selected_raw_test_x,selected_raw_test_y = select_all(test_x,test_y)

    selected_val_x, selected_test_x, selected_val_y, selected_test_y = train_test_split(selected_raw_test_x, selected_raw_test_y, test_size=0.8, random_state=1)
    
    # val_index = int(0.2*len(test_x))
    # print("val_index = ",val_index)
    # selected_val_x,selected_val_y = select_all(test_x[:val_index],test_y[:val_index])
    
    print("len(selected_val_x)=",len(selected_val_x))
    
    # selected_test_x,selected_test_y = select_all(test_x[val_index:],test_y[val_index:])
    print("len(selected_test_x)=",len(selected_test_x))
    #print(selected_train_x,selected_train_y)
    #print(selected_test_x,selected_test_y)

    return np.asarray(selected_train_x)/255.0,np.asarray(selected_train_y),np.asarray(selected_test_x)/255.0,np.asarray(selected_test_y),np.asarray(selected_val_x)/255.0,np.asarray(selected_val_y)
    
    
def create_non_iid_data(dataset_idx,dataset_id,labels,n_clients,n_data):
    # Initialization

    train = {}
    test = {}
    val = {}

    (cifar10_x_train, cifar10_y_train), (cifar10_x_test, cifar10_y_test) = tf.keras.datasets.cifar10.load_data()

    for i in range(n_clients):
        
        
        train[i] = {}
        test[i] = {}
        val[i] = {}
        train[i][0],train[i][1],test[i][0],test[i][1],val[i][0],val[i][1] =create_pathological_heterogeneous_datasets_specific(cifar10_x_train, cifar10_y_train,cifar10_x_test, cifar10_y_test,n_data,labels)
        
        
        
        filename_train_x = "data/cifar10_"+ str(dataset_idx) +   "/dataset_id=" + str(dataset_id) + "train_x" + "client_id=" + str(i)
        np.save(filename_train_x, 
            train[i][0])
        
        filename_train_y = "data/cifar10_"+ str(dataset_idx) +   "/dataset_id=" + str(dataset_id) + "train_y"+ "client_id=" + str(i)
        np.save(filename_train_y, 
            train[i][1])
        
        filename_test_x = "data/cifar10_"+ str(dataset_idx) +   "/dataset_id=" + str(dataset_id) + "test_x"+ "client_id=" + str(i)
        np.save(filename_test_x, 
            test[i][0])
        
        filename_test_y = "data/cifar10_"+ str(dataset_idx) +   "/dataset_id=" + str(dataset_id) + "test_y"+ "client_id=" + str(i)
        np.save(filename_test_y, 
            test[i][1])
        
        filename_val_x = "data/cifar10_"+ str(dataset_idx) +   "/dataset_id=" + str(dataset_id) + "val_x"+ "client_id=" + str(i)
        np.save(filename_val_x, 
            val[i][0])
        
        filename_val_y = "data/cifar10_"+ str(dataset_idx) +   "/dataset_id=" + str(dataset_id) + "val_y"+ "client_id=" + str(i)
        np.save(filename_val_y, 
            val[i][1])
        
        




# Create local client dataset
# CIFAR 10
# Create dataset

cifar10_1_labels = [0,1]
cifar10_2_labels = [0,2]
cifar10_3_labels = [2,3]
cifar10_4_labels = [0,1,2,3,4]
cifar10_5_labels = [3,4,5,6,7]
cifar10_6_labels = [0,1]
cifar10_7_labels = [0,2]
cifar10_8_lables = [4,5]
cifar10_9_labels = [0,1,2,3,4]
cifar10_10_labels = [1,2,3,4,5]
cifar10_11_labels = [2,3,4,5,6]
cifar10_12_labels = [3,4,5,6,7]
cifar10_13_labels = [4,5,6,7,8]
cifar10_14_labels = [5,6,7,8,9]
cifar10_15_labels = [0,1]
cifar10_16_labels = [1,2]
cifar10_17_labels = [3,4]
cifar10_18_labels = [4,5]



#create_non_iid_data(dataset_idx=18,dataset_id=8,labels=cifar10_18_labels,n_clients=5,n_data=40)


def create_multiple_type_non_iid_data(dataset_idx,dataset_id,n_class,n_sub_class,n_clients,n_data):
    # Initialization

    train = {}
    test = {}
    val = {}

    (cifar10_x_train, cifar10_y_train), (cifar10_x_test, cifar10_y_test) = tf.keras.datasets.cifar10.load_data()

    for i in range(n_clients):
        
        labels = random.sample(range(n_class),n_sub_class)
        
        train[i] = {}
        test[i] = {}
        val[i] = {}
        train[i][0],train[i][1],test[i][0],test[i][1],val[i][0],val[i][1] =create_pathological_heterogeneous_datasets_specific(cifar10_x_train, cifar10_y_train,cifar10_x_test, cifar10_y_test,n_data,labels)
        
        
        
        filename_train_x = "data/cifar10_"+ str(dataset_idx) +   "/dataset_id=" + str(dataset_id) + "train_x" + "client_id=" + str(i)
        np.save(filename_train_x, 
            train[i][0])
        
        filename_train_y = "data/cifar10_"+ str(dataset_idx) +   "/dataset_id=" + str(dataset_id) + "train_y"+ "client_id=" + str(i)
        np.save(filename_train_y, 
            train[i][1])
        
        filename_test_x = "data/cifar10_"+ str(dataset_idx) +   "/dataset_id=" + str(dataset_id) + "test_x"+ "client_id=" + str(i)
        np.save(filename_test_x, 
            test[i][0])
        
        filename_test_y = "data/cifar10_"+ str(dataset_idx) +   "/dataset_id=" + str(dataset_id) + "test_y"+ "client_id=" + str(i)
        np.save(filename_test_y, 
            test[i][1])
        
        filename_val_x = "data/cifar10_"+ str(dataset_idx) +   "/dataset_id=" + str(dataset_id) + "val_x"+ "client_id=" + str(i)
        np.save(filename_val_x, 
            val[i][0])
        
        filename_val_y = "data/cifar10_"+ str(dataset_idx) +   "/dataset_id=" + str(dataset_id) + "val_y"+ "client_id=" + str(i)
        np.save(filename_val_y, 
            val[i][1])
        
        

create_multiple_type_non_iid_data(dataset_idx=20,dataset_id=1,n_class=10,n_sub_class=2,n_clients=100,n_data=40)