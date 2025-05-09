import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
import tensorflow  as tf
from sklearn.metrics import ConfusionMatrixDisplay
import string
from termcolor import colored
import matplotlib.ticker as ticker

##########################################
# Global variables

inputShape = (28,28)
inputShapeConv = (28,28,1)
nNeurons = np.full((16), 128)
lastClassNumber = 36
nOutputNeurons = 36
nEpochs = 50
kernel_regularization = tf.keras.regularizers.L2(l2 = 0.001)

##########################################
##########################################
# Function to load the EMNIST dataset

def getEMNIST(datasetName, datasetPath):

    fileName = datasetPath+"/"+datasetName+'-images-idx3-ubyte'
    features = idx2numpy.convert_from_file(fileName)

    fileName = datasetPath+"/"+datasetName+'-labels-idx1-ubyte'
    labels = idx2numpy.convert_from_file(fileName)
    return (features, labels)

###########################################
###########################################
# Function to preprocessing the EMNIST dataset

def preprocessData(x, y):
    mask = y < lastClassNumber
    x = x[mask]/255.0
    y = y[mask]
    return x, y

###########################################
###########################################
# Function to return model

def getModel(inputShape, nOutputNeurons, nNeurons, hiddenActivation='relu', outputActivation='softmax'):
    inputs = tf.keras.Input(shape=inputShape, name="features")
    x = tf.keras.layers.Flatten()(inputs)
    for iLayer, n in enumerate(nNeurons):
        x = tf.keras.layers.Dense(n, activation=hiddenActivation, 
                                  kernel_initializer='he_normal',
                                  bias_initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1),
                                  kernel_regularizer=kernel_regularization,
                                  name="layer_"+str(iLayer))(x)
        #x = tf.keras.layers.Dropout(name="dropout_layer_"+str(iLayer), rate = 0.2)(x)
            
    outputs = tf.keras.layers.Dense(nOutputNeurons, activation=outputActivation, name = "output")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="DNN")
    return model
############################################
############################################
# Function to return conv model

def getConvModel(inputShape, nOutpuNeurons):
    model = tf.keras.Sequential(layers = [tf.keras.Input(shape=inputShape),
                                      tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation="relu", padding="same"),
                                      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                                      tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation="relu", padding="same"),
                                      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                                      tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation="relu", padding="same"),
                                      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                                      tf.keras.layers.Flatten(),
                                      tf.keras.layers.Dense(128, activation="relu"),
                                      tf.keras.layers.Dropout(0.5),
                                      tf.keras.layers.Dense(64, activation="relu"),
                                      tf.keras.layers.Dropout(0.5),
                                      tf.keras.layers.Dense(nOutpuNeurons, name = "output_layer", activation="softmax")],
                             name="conv")
    return model


############################################
############################################
# Function to compile and train model

def trainModel(model, x, y, nEpochs):
    batchSize = 2048
    initial_learning_rate = 0.001
    nStepsPerEpoch = len(x)/batchSize
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                        decay_steps=10*nStepsPerEpoch,
                        decay_rate=0.95,
                        staircase=True)
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, min_delta=1E-3)
    callbacks = [early_stop_callback]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(x, y, batch_size=batchSize, epochs=nEpochs, validation_split=0.1, verbose=1, callbacks=callbacks)
    plotTrainHistory(history)
    return model

############################################
############################################
# Function to plot loss function

def plotTrainHistory(history):
    if history.history['val_loss'] is None:
        fig, axis = plt.subplots(1, 2, figsize=(20, 5))
        axis[0].plot(history.history['loss'], label='loss')
        axis[0].set_xlabel('Epoch')
        axis[0].set_ylabel('Error')
        axis[0].legend()
        axis[1].plot(history.history['loss'], label='loss')
        axis[1].set_yscale('log')
        axis[1].set_xlabel('Epoch')
        axis[1].set_ylabel('Error')
        axis[1].legend()
    else:
        fig, axis = plt.subplots(1, 2, figsize=(20, 5))
        axis[0].plot(history.history['loss'], label='loss')
        axis[0].plot(history.history['val_loss'], label='val_loss')
        axis[0].set_xlabel('Epoch')
        axis[0].set_ylabel('Error')
        axis[0].legend()
        axis[1].plot(history.history['loss'], label='loss')
        axis[1].plot(history.history['val_loss'], label='val_loss')
        axis[1].set_yscale('log')
        axis[1].set_xlabel('Epoch')
        axis[1].set_ylabel('Error')
        axis[1].legend()
    pass

############################################
############################################
# Function to plot Confusion Matrix

def plotMNIST_CM(y, y_pred, label):
    fig, axis = plt.subplots(1, 1, figsize=(12,12))
    
    numbers = [str(i) for i in range(0,10)]
    big_letters = list(string.ascii_uppercase)
    labels = numbers + big_letters

    ConfusionMatrixDisplay.from_predictions(y, y_pred, normalize="true", include_values = False, ax=axis, display_labels=labels)
    
    axis.set_title(label + " dataset")
    
#############################################
#############################################
# Function to plot MNIST digits
def plotMNIST1(x, y,y_pred):
    
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    ax = ax.flatten()  
    random_indices = np.random.choice(len(x), 4, replace=False)  
    for i, idx in enumerate(random_indices):
        ax[i].imshow(x[idx], cmap='Greens')
        ax[i].set_title(f"True: {y[idx]}\nPred: {y_pred[idx]}")
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()



def plotMNIST(x, y, y_pred):
    indices = np.random.default_rng().integers(0, len(x), (4))
    fig, axes = plt.subplots(2, 2, figsize=(4.5,4.5))
    numbers = [str(i) for i in range(10)]
    big_letters = list(string.ascii_uppercase)
    labels = numbers + big_letters
    y = [labels[i] for i in y]
    y_pred = [labels[i] for i in y_pred]

    for index, axis in zip(indices, axes.flatten()):
        title = "T: {}/P: {}".format(y[index],y_pred[index])
        axis.imshow(x[index].T, cmap=plt.get_cmap('CMRmap'), label="A")
        axis.set_title(title)

    #axes[0,1].legend(bbox_to_anchor=(1.5,1), loc='upper left', title="Label: True/Predicted")
    plt.subplots_adjust(bottom=0.15, left=0.05, right=0.95, wspace=0.4, hspace=0.5)
    
#############################################
#############################################
# Function to decoding message

def decodeMessage(model, encoded):
    
    mask_space = np.sum(encoded, axis=(1,2))>0
    y_pred = model.predict(encoded)
    decoded = tf.argmax(y_pred, axis=1).numpy()
    decoded = decoded.astype(int)
    
    numbers = [str(i) for i in range(0,10)]
    big_letters = list(string.ascii_uppercase)
    labels = numbers + big_letters
    
    decoded = [labels[i] for i in decoded]
    decoded = np.where(mask_space, decoded, " ")
    decoded = ''.join(decoded)
    print(colored("Decoded message is:","blue"),decoded)
    
    fig, axis = plt.subplots(1, 17, figsize=(16,6))
    for i in range(len(y_pred)):
        axis[i].imshow(encoded[i].T, cmap=plt.get_cmap('CMRmap'), label="A")
        axis[i].set_title("P: {}".format(decoded[i]))
        axis[i].xaxis.set_major_locator(ticker.NullLocator())
        axis[i].yaxis.set_major_locator(ticker.NullLocator())
    
    return y_pred