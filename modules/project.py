#Plik z funkcjami do projektu

#biblioteki
import tensorflow as tf
from termcolor import colored
import matplotlib.pyplot as plt
import numpy as np
import importlib
import sys
sys.path.append("/Users/jakubkusmierski/Desktop/Uczenie_Maszynowe_2/modules")

#Private functions
import plotting_functions as plf
importlib.reload(plf);

##########################################
batchSize = 32
nStepsPerEpoch = int(35586/batchSize) #35586 - liczba próbek w zbiorze treningowym
buffor_size = 2048

##########################################
##########################################
# Function to smeard MET
def add_smeared_MET(features, label, minval, maxval, mean, stddev, rho_min, rho_max):
    trueMETx = features[:, 10]
    trueMETy = features[:, 11]

    sigma_x = tf.random.uniform(shape=tf.shape(trueMETx), minval=minval, maxval=maxval)
    sigma_y = tf.random.uniform(shape=tf.shape(trueMETy), minval=minval, maxval=maxval)
    rho = tf.random.uniform(shape=tf.shape(trueMETx), minval=rho_min, maxval=rho_max)

    noise_x = tf.random.normal(shape=tf.shape(trueMETx), mean=mean, stddev=stddev)
    noise_y = tf.random.normal(shape=tf.shape(trueMETy), mean=mean, stddev=stddev)

    smeared_METx = trueMETx + sigma_x * noise_x
    smeared_METy = trueMETy + sigma_y * (rho * noise_x + tf.sqrt(1 - tf.square(rho)) * noise_y)

    var_xx = tf.square(sigma_x)
    var_yy = tf.square(sigma_y)
    cov_xy = rho * sigma_x * sigma_y

    # Rozmyte dane jako nowa paczka kolumn
    new_MET_info = tf.stack([smeared_METx, smeared_METy, var_xx, var_yy, cov_xy], axis=1)

    # Doklejamy nowe cechy na koniec, ale nic nie usuwamy
    features_augmented = tf.concat([features, new_MET_info], axis=1)

    return features_augmented, label
##########################################
##########################################
#Function to remove trueMET features
def remove_true_MET(features, label):
    features_cleaned = tf.concat([features[:, :10], features[:, 12:]], axis=1)
    return features_cleaned, label
##########################################
##########################################
#Loss function
def loss_test(y_true, y_pred):
    #loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)*10
    #spoko wyniki huber 20 plo mape i batchsize 32
    loss = tf.keras.losses.Huber(delta=4)(y_true, y_pred)
    #loss += tf.keras.losses.MeanAbsolutePercentageError()(y_true, y_pred)
    #loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    return loss
###########################################
##########################################
# Function to compile and train model
def trainModel(model, nEpochs, train_data, val_data, nStepsPerEpoch=nStepsPerEpoch):
   
    initial_learning_rate = 1E-3
    loss_fn = loss_test
    #loss_fn = tf.keras.losses.Huber(delta=20)
    
      
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                    decay_steps=nStepsPerEpoch*10,
                    decay_rate=0.80,
                    staircase=False)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), 
                loss=loss_fn, metrics=['accuracy'])
    
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', 
                                                           patience=30, verbose=1)
    callbacks = [early_stop_callback]
    #callbacks = []
    
    history = model.fit(train_data,
                        validation_data=val_data,
                        epochs=nEpochs, 
                        callbacks=callbacks,
                        verbose=1)
    
    plf.plotTrainHistory(history)
    print(colored("Evaluation on training dataset:","blue"))
    model.evaluate(val_data, verbose=1)
    
    return model
    
    
##########################################
##########################################
#Function to split inputs 1
def split_inputs(features, labels):
    return {
        'particles_input': features[:10],
        'met_input': features[10:]
    }, labels
    
##########################################
##########################################
#Function to split inputs 2
def split_inputs2(features, labels):
    return {
        'particles_input1': features[:5],
        'particles_input2': features[5:10],
        'met_input': features[10:]
    }, labels
##########################################
##########################################
# Function to plotting H mass
def plot_H_mass(model, val_data):
    
    true_H_m = []
    pred_H_m = []

    for x_batch, y_batch in val_data:
        predictions = model(x_batch, training=False)
        true_H_m.extend(y_batch[:, 0].numpy())        
        pred_H_m.extend(predictions[:, 0].numpy())



    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Wykres 1: True H.m
    axs[0].hist(true_H_m, bins=60, alpha=0.7, color='blue')
    axs[0].set_title("True H.m")
    axs[0].set_xlabel("H.m [GeV]")
    axs[0].set_ylabel("Density")
    axs[0].grid(True)
    axs[0].set_ylim(0, 600)

    # Wykres 2: Predicted H.m
    axs[1].hist(pred_H_m, bins=60, alpha=0.7, color='orange')
    axs[1].set_title("Predicted H.m")
    axs[1].set_xlabel("H.m [GeV]")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
    
##########################################
##########################################
#Function to plot predict vs true
def true_vs_predict_H_mass(model, val_data):
    true_H_m = []
    pred_H_m = []
    true_H_pt = []
    pred_H_pt = []

    for x_batch, y_batch in val_data:
        predictions = model(x_batch, training=False)
        true_H_m.extend(y_batch[:, 0].numpy())        
        pred_H_m.extend(predictions[:, 0].numpy())
        true_H_pt.extend(y_batch[:, 1].numpy())
        pred_H_pt.extend(predictions[:, 1].numpy())
        
    fig, axis = plt.subplots(1, 2, figsize=(14, 5))
    axis[0].scatter(true_H_m, pred_H_m, alpha=0.3)
    axis[0].plot([0, 1000], [0, 1000], color='red', linestyle='--', label='Ideal')
    axis[0].set_xlabel("True H.m")
    axis[0].set_ylabel("Predicted H.m")
    axis[0].set_title("True vs Predicted H.m")
    axis[0].legend()
    axis[0].grid(True)
    
    axis[1].scatter(true_H_pt, pred_H_pt, alpha=0.3)
    axis[1].plot([0, 1000], [0, 1000], color='red', linestyle='--', label='Ideal')
    axis[1].set_xlabel("True H.pt")
    axis[1].set_ylabel("Predicted H.pt")
    axis[1].set_title("True vs Predicted H.pt")
    axis[1].legend()
    axis[1].grid(True)
    
    plt.show()
    
##########################################
##########################################
#Function to plot all labels
def plot_all_labels(model, val_data, output_names):
    true_vals = [[] for _ in range(len(output_names))]
    pred_vals = [[] for _ in range(len(output_names))]

    # Zbieranie danych
    for x_batch, y_batch in val_data:
        predictions = model(x_batch, training=False)
        for i in range(len(output_names)):
            true_vals[i].extend(y_batch[:, i].numpy())
            pred_vals[i].extend(predictions[:, i].numpy())
            
        
    print(colored("True Higgs mass: ", 'green'), colored(true_vals[0][1], 'cyan'),colored(' and predicted:', 'green'), colored(pred_vals[0][1], 'red'))

    # Tworzenie wykresów
    n_outputs = len(output_names)
    n_cols = 3
    n_rows = (n_outputs + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = axs.flatten()

    for i in range(n_outputs):
        axs[i].hist(true_vals[i], bins=60, alpha=0.6, label='True', color='blue')
        axs[i].hist(pred_vals[i], bins=60, alpha=0.6, label='Predicted', color='orange')
        axs[i].set_title(output_names[i])
        axs[i].set_xlabel('[GeV]' if 'm' in output_names[i] or 'pt' in output_names[i] else '')
        axs[i].grid(True)
        axs[i].legend()

    # Ukryj puste subploty jeśli są
    for i in range(n_outputs, len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()
    
##########################################
##########################################
#Function to plot MET
def plot_MET(dataset):
    true_metx_vals = []
    smeared_metx_vals = []
    true_mety_vals = []
    smeared_mety_vals = []

    for features, _ in dataset.take(50): 
        true_metx_vals.extend(features[:, 10].numpy())
        smeared_metx_vals.extend(features[:, 12].numpy())
        true_mety_vals.extend(features[:, 11].numpy())
        smeared_mety_vals.extend(features[:, 13].numpy())
        
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # METx - skala liniowa
    axes[0, 0].hist(true_metx_vals, bins=100, alpha=0.6, label='trueMETx')
    axes[0, 0].hist(smeared_metx_vals, bins=100, alpha=0.6, label='smearedMETx')
    axes[0, 0].set_title('METx – skala liniowa')
    axes[0, 0].set_xlabel('MET x [GeV]')
    axes[0, 0].set_ylabel('Liczba zdarzeń')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # METy - skala liniowa
    axes[0, 1].hist(true_mety_vals, bins=100, alpha=0.6, label='trueMETy')
    axes[0, 1].hist(smeared_mety_vals, bins=100, alpha=0.6, label='smearedMETy')
    axes[0, 1].set_title('METy – skala liniowa')
    axes[0, 1].set_xlabel('MET y [GeV]')
    axes[0, 1].set_ylabel('Liczba zdarzeń')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # METx - skala logarytmiczna
    axes[1, 0].hist(true_metx_vals, bins=100, alpha=0.6, label='trueMETx')
    axes[1, 0].hist(smeared_metx_vals, bins=100, alpha=0.6, label='smearedMETx')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('METx – skala logarytmiczna')
    axes[1, 0].set_xlabel('MET x [GeV]')
    axes[1, 0].set_ylabel('log(Liczba zdarzeń)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, which='both')

    # METy - skala logarytmiczna
    axes[1, 1].hist(true_mety_vals, bins=100, alpha=0.6, label='trueMETy')
    axes[1, 1].hist(smeared_mety_vals, bins=100, alpha=0.6, label='smearedMETy')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('METy – skala logarytmiczna')
    axes[1, 1].set_xlabel('MET y [GeV]')
    axes[1, 1].set_ylabel('log(Liczba zdarzeń)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, which='both')

    plt.tight_layout()
    plt.show()
    
##########################################
##########################################
#Pull ploting
def plot_all_pulls(model, val_data, output_names):
    true_vals = [[] for _ in range(len(output_names))]
    pred_vals = [[] for _ in range(len(output_names))]

    # Zbieranie danych
    for x_batch, y_batch in val_data:
        predictions = model(x_batch, training=False)
        for i in range(len(output_names)):
            true_vals[i].extend(y_batch[:, i].numpy())
            pred_vals[i].extend(predictions[:, i].numpy())

    # Obliczanie pullów
    pulls = []
    for i in range(len(output_names)):
        true = np.array(true_vals[i])
        pred = np.array(pred_vals[i])
        sigma = np.std(pred - true)
        if sigma == 0:
            sigma = 1e-6  # zapobiega dzieleniu przez 0
        pull = (pred - true) / sigma
        pulls.append(pull)

        print(colored(f"[{output_names[i]}] sigma = {sigma:.4f}", 'cyan'))

    # Rysowanie
    n_outputs = len(output_names)
    n_cols = 3
    n_rows = (n_outputs + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = axs.flatten()

    for i in range(n_outputs):
        axs[i].hist(pulls[i], bins=60, range=(-5, 5), color='purple', alpha=0.7, histtype='stepfilled', density=True)
        axs[i].set_title(f"Pull: {output_names[i]}")
        axs[i].set_xlabel("Pull value")
        axs[i].set_ylabel("Normalized count")
        axs[i].grid(True)
        axs[i].axvline(0, color='black', linestyle='--')

    # Ukryj puste subploty jeśli są
    for i in range(n_outputs, len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()