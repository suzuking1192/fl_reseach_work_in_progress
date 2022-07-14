from keras.datasets import mnist
from matplotlib import pyplot
import tensorflow as tf
import random
from tensorflow.keras import datasets, layers, models
import numpy as np
from hungarian_algorithm import algorithm
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
from datetime import datetime
import csv
from scipy.stats import pearsonr
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dropout,Input
import numpy as np
import random
from scipy.optimize import linear_sum_assignment
import scipy
from tensorflow.keras import initializers
from keras.layers import BatchNormalization
from utils import lenet5, client_model_initialization_lenet5_no_enc
from lottery_fl_non_iid import client_model_initialization_multiple_fc_for_local_training

(cifar_x_train, cifar_y_train), (cifar_x_test, cifar_y_test) = tf.keras.datasets.cifar10.load_data()

# local training

# Create models

def cifar100_cnn_model(n_class):
    input_img = keras.Input(shape=(32,32,3))
    
    encoder_1 = models.Sequential()
    layer_input = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3),kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.005, seed=0))
    encoder_1.add(layer_input)
    encoder_1.add(layers.MaxPooling2D((2, 2)))
    
    encoder_2 = models.Sequential()
    layer_input = layers.Conv2D(64, (3, 3), activation='relu',kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.005, seed=0))
    encoder_2.add(layer_input)
    encoder_2.add(layers.MaxPooling2D((2, 2)))
    
    encoder_3 = models.Sequential()
    layer_input = layers.Conv2D(64, (3, 3), activation='relu',kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.005, seed=0))
    encoder_3.add(layer_input)
    encoder_3.add(layers.Flatten())
    
    encoder_4 = models.Sequential()
    encoder_4.add(layers.Dense(64, activation='relu',kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.005, seed=0)))
    
    classification_layer = models.Sequential()
    classification_layer.add(layers.Dense(n_class,activation="softmax",kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.005, seed=0)))
    
    output_1 = encoder_1(input_img)
    output_2 = encoder_2(output_1)
    output_3 = encoder_3(output_2)
    output_4 = encoder_4(output_3)
    output = classification_layer(output_4)
    
    
    final_model = keras.Model(input_img,output)
    
    return final_model,encoder_1,encoder_2,encoder_3,encoder_4,classification_layer
    

def cifar100_single_fl_model(n_class,n_neurons,initialize=False):
    input_img = keras.Input(shape=(32,32,3))
    encoder_1 = models.Sequential()
    encoder_1.add(layers.Flatten())
    if initialize == False:
        encoder_1.add(layers.Dense(n_neurons, activation='relu',kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.004, seed=0)))
    else:
        encoder_1.add(layers.Dense(n_neurons, activation='sigmoid',kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=0),bias_initializer=initializers.Zeros(),))
    #encoder_1.add(BatchNormalization())
    
    classification_layer = models.Sequential()
    classification_layer.add(layers.Dense(n_class,activation='softmax',kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.004, seed=0)))
    
    output_1 = encoder_1(input_img)
    output = classification_layer(output_1)
    
    final_model = keras.Model(input_img,output)
    
    return final_model,encoder_1,classification_layer

def client_model_initialization_single_fl_with_enc(n_client,n_class,n_neurons):
    client_models = {}
    
    for i in range(n_client):
        client_models[i] = cifar100_single_fl_model(n_class,n_neurons)
        
    return client_models

def client_model_initialization_single_fl(n_client,n_class,n_neurons):
    client_models = {}
    
    for i in range(n_client):
        client_models[i],_,_ = cifar100_single_fl_model(n_class,n_neurons)
        
    return client_models

def client_model_initialization(n_client,n_class,n_neurons):
    client_models = {}
    
    for i in range(n_client):
        client_models[i],_,_,_,_,_ = cifar100_cnn_model(n_class)
        
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


def local_training(dataset_name_list,n_client,dataset_id,epoch_per_round,n_round,n_class,client_model_initialization,model_architecture_name,n_neurons):
    #opt = keras.optimizers.SGD(learning_rate=0.01,momentum=0.5)
    #opt = keras.optimizers.SGD(learning_rate=0.001,momentum=0.5)

    decayrate = 0
    batch_size = 32
    
    for dataset_name in dataset_name_list:
        client_models = client_model_initialization(n_client,n_class,n_neurons)
        client_train,client_test = load_data_for_clients(dataset_name,n_client,dataset_id)
        
        accuracy_dict ={}
        
        for i_round in range(n_round):
            print("round: ", i_round)
            accuracy_dict[i_round] = {}
            
            for c_id in range(n_client):

                opt = keras.optimizers.SGD(learning_rate=0.01*(1/(1+decayrate*epoch_per_round))**i_round,momentum=0.5)

                # if i_round >= 50:
                #     print("step decay")
                #     opt = keras.optimizers.SGD(learning_rate=0.01/(4**(int(i_round/50))),momentum=0.5)


                client_models[c_id].compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
                client_models[c_id].fit(client_train[c_id][0],client_train[c_id][1],epochs=epoch_per_round,verbose=0,batch_size=batch_size)
                
                accuracy = calculate_accuracy(client_models[c_id],client_test[c_id][0],client_test[c_id][1])
                print("accuracy = ",accuracy)
                accuracy_dict[i_round][c_id] = accuracy
        
        res = []
        for i in range(n_round):
            avg = np.average(list(accuracy_dict[i].values()))
            res.append([str(dataset_name),dataset_id,n_class,n_client,i,epoch_per_round,'local_training',avg,model_architecture_name])
            print("average accuracy = ", avg)
        # res_df = pd.DataFrame(res,columns=['dataset','dataset_id','n_class','n_client','round','local_epoch','fl_algorithm','avg_test_accuracy','model_architecture'])
        # res_df['datetime'] = datetime.now()
        # current_res_df = pd.read_csv('result/neuron_level_fl_test.csv')
        # new_res_df = pd.concat([res_df,current_res_df],ignore_index=True)
        # new_res_df.to_csv('result/neuron_level_fl_test.csv')
        
        
dataset_name_list = ["cifar10_20"]
n_client = 10
dataset_id = 1
epoch_per_round = 10
n_round = 100
n_class = 10
n_neurons = 32

#local_training(dataset_name_list,n_client,dataset_id,epoch_per_round,n_round,n_class,client_model_initialization_lenet5_no_enc,"lenet5",n_neurons)

local_training(dataset_name_list,n_client,dataset_id,epoch_per_round,n_round,n_class,client_model_initialization_single_fl,"singlefl",n_neurons)

#local_training(dataset_name_list,n_client,dataset_id,epoch_per_round,n_round,n_class,client_model_initialization,"cnn",n_neurons)
