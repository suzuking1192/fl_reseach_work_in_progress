import pandas as pd
import numpy as np

X = pd.read_csv("l2_dist_list.csv",header=None)
y = pd.read_csv("positive_transfer_list.csv",header=None)
y = y.transpose()

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)
print("X=",X)
print("y=",y)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LinearRegression

# selected_columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
# selected_columns = [12,14,16]
selected_columns = [0,1,2,3,4,5,6,7,8,9]

reg = LinearRegression().fit(X_train[selected_columns], y_train)
# score = reg.score(X[selected_columns], y)
# print("score = ",score)

print("reg.coef_",reg.coef_)

print("reg.intercept_",reg.intercept_)

score = reg.score(X_train[selected_columns], y_train)
print("Linear regression train score = ",score)

score = reg.score(X_test[selected_columns], y_test)

print("Linear regression test score = ",score)

import scipy.stats as stats
r = stats.pearsonr(X[10], y)
print("pearson correlation= ",r)

import matplotlib.pyplot as plt
plt.scatter(X[10], y, alpha=0.5)
plt.show()



from sklearn.ensemble import RandomForestRegressor
  
 # create regressor object
regressor = RandomForestRegressor(n_estimators = 20,max_depth=30, random_state = 0)
  
# fit the regressor with x and y data
regressor.fit(X_train[selected_columns], y_train)  
score = regressor.score(X_train[selected_columns], y_train)
print("Random Forest train score = ",score)
print("feature importance",regressor.feature_importances_)

score = regressor.score(X_test[selected_columns], y_test)
print("Random Forest test score = ",score)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train[selected_columns], y_train)

score = regressor.score(X_train[selected_columns], y_train)
print("SVM train score = ",score)
score = regressor.score(X_test[selected_columns], y_test)
print("SVM test score = ",score)



# Learn positive transfer 

# num_user = 10

# local_weights = []

# for c_idx in range(num_user):
#     numpy_filename = "src/data/weights/local_weights_array_for_positive_transfer_training_" + str(c_idx) +".npy" 
#     weights = np.load(numpy_filename, allow_pickle=True)

#     len_weights = len(weights)
#     tmp = []
#     tmp = np.array(tmp)
#     for i in range(len_weights):
        
#         tmp = np.concatenate((tmp, weights[i].flatten()))

#     local_weights.append(tmp.flatten())


# X_weights_1 = []
# X_weights_2 = []

# for c_idx in range(num_user):
#     for ref_idx in range(num_user):
#         if c_idx != ref_idx:

            
#             X_weights_1.append(local_weights[c_idx])
#             X_weights_2.append(local_weights[ref_idx])

            

# X_weights_1 = np.array(X_weights_1)
# X_weights_2 = np.array(X_weights_1)

# len_weights = len(X_weights_1[0])

# from keras.layers import Input, Concatenate, Conv2D, Flatten, Dense
# from keras.models import Model

# # Define two input layers
# w1_input = Input((len_weights,))
# w2_input = Input((len_weights,))

# emb_1_1 = Dense(1024)(w1_input)
# emb_1_2 = Dense(512)(emb_1_1)
# emb_1_3 = Dense(256)(emb_1_2)
# emb_2_1 = Dense(1024)(w2_input)
# emb_2_2 = Dense(512)(emb_2_1)
# emb_2_3 = Dense(256)(emb_2_2)

# # Concatenate the convolutional features and the vector input
# concat_layer= Concatenate()([emb_1_3, emb_2_3])
# fl_1 = Dense(126)(concat_layer)
# fl_2 = Dense(64)(fl_1)
# fl_3 = Dense(16)(fl_2)
# output = Dense(1)(fl_3)

# # define a model with a list of two inputs
# model = Model(inputs=[w1_input, w2_input], outputs=output)

# from tensorflow.keras.optimizers import SGD

# optimizer = SGD(lr=0.01, momentum=0.9, clipvalue=1.0)
# model.compile(loss='mse', optimizer=optimizer)

# n_x = len(X_weights_1)

# history = model.fit((X_weights_1[:int(n_x*0.8)], X_weights_2[:int(n_x*0.8)]),y[:int(n_x*0.8)], 
#                                               batch_size=32, 
#                                               epochs=100,validation_data=((X_weights_1[int(n_x*0.8):], X_weights_2[int(n_x*0.8):]),y[int(n_x*0.8):]))



# import matplotlib.pyplot as plt
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('NN loss')
# plt.ylabel('mse loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()