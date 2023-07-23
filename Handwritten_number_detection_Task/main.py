from os import access
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
data = tf.keras.datasets.mnist


(x_train , y_train), (x_test, y_test) = data.load_data()

x_train= tf.keras.utils.normalize(x_train, axis =1)
x_test= tf.keras.utils.normalize(x_test, axis =1)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (28,28)))
model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train, epochs=5)


loss,accuracy=model.evaluate(x_train,y_train)
print(accuracy)
print(loss)


for x in range(1,6):
    # now we are going to read images it with open cv

    img=cv.imread(f'{x}.png')[:,:,0]#all of it and 1st and last one
    img=np.invert(np.array([img]))#invert black to white in images so that model wont get confues
    prediction=model.predict(img)
    print("----------------")
    print("The predicted value is : ",np.argmax(prediction))
    print("----------------")
    plt.imshow(img[0],cmap=plt.cm.binary)#change the color in black and white
    plt.show()


