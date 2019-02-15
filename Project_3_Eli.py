#name comment
from keras import models
from keras import layers
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

#General Variables
Verify_Step = 21

#Training Data
X = np.array([[1, 5],
              [2, 4],
              [7, 7],
              [4, 6],
              [6, 4],
              [6, 9],
              [4, 2],
              [8, 6],
              [5, 5],
              [3, 8]])

X_norm = np.divide((X-0),(10-0)) #works since range is 0 to 10

#[1, 0] = red 
#[0, 1] = blue
R = np.append(np.tile(np.array([1, 0]), [5, 1]), np.tile(np.array([0, 1]), [5, 1]), axis=0)

#Set Up Network
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(2,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.compile(optimizer='sgd', loss='mse')

#Train Network
model.fit(X_norm, R, epochs=200, batch_size=10)
model.save('color_model.h5')

#verify network
x_val = np.linspace(0, 10, Verify_Step)
y_val = np.linspace(0, 10, Verify_Step)
x_val, y_val = np.meshgrid(x_val, y_val)

x_y_val = []
for i in range(Verify_Step):
    x_val_individual = x_val[i][:, np.newaxis]
    y_val_individual = y_val[i][:, np.newaxis]
    x_y_val.append(np.concatenate((x_val_individual, y_val_individual), axis=1))

x_y_val = np.divide((np.concatenate(x_y_val, axis=0)-0),(10-0))

#test network
model = load_model('color_model.h5')
#z_pre = model.predict(x_y_val)

#plot result

for i in range(x_y_val.shape[0]):
    Location = x_y_val[i]
    Prediction = model.predict(np.array([Location]))
    if Prediction[0][0] > Prediction[0][1]:
        plt.plot(np.multiply(Location[0], 10), np.multiply(Location[1], 10), 'r.')
    else:
        plt.plot(np.multiply(Location[0], 10), np.multiply(Location[1], 10), 'b.')

#plot training data
for i in range(10):
    Location = X[i]
    Result = R[i]    
    if Result[0] > Result[1]:
        plt.plot(Location[0], Location[1], 'ro')
    else:
        plt.plot(Location[0], Location[1], 'bo')
    
plt.xticks(np.arange(0,11))
plt.xlim(0,10)
plt.yticks(np.arange(0,11))
plt.ylim(0,10)
plt.show()
    
    




