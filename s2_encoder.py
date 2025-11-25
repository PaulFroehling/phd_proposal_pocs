'''
Training script for S²-->R³ regression model, regularzied by pullback metric.
'''

import os
import mlflow
import tensorflow as tf
import numpy as np
from geodesics.s2 import s2_geodesic
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

print("Available Devices:", tf.config.list_physical_devices())
print("GPU is used:", tf.config.list_physical_devices('GPU'))

np.random.seed(42)
tf.random.set_seed(42)

geo = s2_geodesic()

R = 1
N_EPOCHS =1000
N_TRAINING_SAMPLES = 20
N_VALIDATION_SAMPLES = 400
VARIATION_DELTA = tf.constant(1e-3)
REG_LAMBDA = 2.0
LR = 1e-3

TRAINING_AND_VALIDATION_DATA_PATH = "data/training_and_validation/s2_encoder/training_validation_seed_42.npz"

MLFLOW_HOST = "127.0.0.1"
MLFLOW_PORT = "8080"
MLFLOW_EXPERIMENT_NAME = "Neural S2 Encoder"


mlflow.set_tracking_uri(uri=f"http://{MLFLOW_HOST}:{MLFLOW_PORT}")
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def define_model(LR:float) -> tuple[tf.keras.Model, tf.keras.optimizers.Adam]:
    '''Defines the regression model from spherical polar coordinates to carthesian R³. 

    Args:
        - LR (float): Learning rate
    Returns:
        -model (tf.keras.Model): Model definition
        -opt   (tf.keras.Optimizer): Optimizer to train with
    '''
    
    activation = "tanh"
    inputs = tf.keras.Input(shape=(2,))
    x = tf.keras.layers.Dense(64, activation=activation)(inputs)
    x = tf.keras.layers.Dense(32, activation=activation)(x)
    x = tf.keras.layers.Dense(16, activation=activation)(x)
    x = tf.keras.layers.Dense(16, activation=activation)(x)
    outputs = tf.keras.layers.Dense(3)(x)
    model = tf.keras.Model(inputs, outputs)
    model.summary()

    opt = tf.keras.optimizers.AdamW(
        learning_rate=LR,
    )
    
    return model, opt


def create_training_and_validation_data() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    ''' Randomly samples points on the 2-sphere and creates labels in R³. 
    Also, points for regularization are sampled.

    Args:
        -
    Returns:
        training_cart_points (np.ndarray): Carth. coordinates for training
        training_sphere_points (np.ndarray):  S² coordinates for training
        validation_cart_points (np.ndarray): Carth. coordinates for validation
        validation_sphere_points (np.ndarray): S² coordinates for validation
        reg_sphere_points (np.ndarray): Points for regularization via pullback metric
    '''
    training_cart_points = []
    training_sphere_points = []

    validation_cart_points = []
    validation_sphere_points = []

    reg_sphere_points = []

    for _ in range(N_TRAINING_SAMPLES):
        theta   = np.random.uniform(0, np.pi)
        phi = np.random.uniform(-np.pi, np.pi)
        training_cart_points.append([*geo.parameterform_s2(theta,phi, R)])
        training_sphere_points.append([theta, phi])

    for _ in range(N_VALIDATION_SAMPLES):
        theta   = np.random.uniform(0, np.pi)
        phi = np.random.uniform(-np.pi, np.pi)  
        validation_cart_points.append(geo.parameterform_s2(theta,phi, R))
        validation_sphere_points.append([theta, phi])

    for _ in range(300):
        theta   = np.random.uniform(0, np.pi)
        phi = np.random.uniform(-np.pi, np.pi)  
        reg_sphere_points.append([theta, phi])

    training_cart_points = tf.constant(training_cart_points, dtype=tf.float32)
    training_sphere_points = tf.constant(training_sphere_points, dtype=tf.float32)
    validation_cart_points = tf.constant(validation_cart_points, dtype=tf.float32)
    validation_sphere_points = tf.constant(validation_sphere_points, dtype=tf.float32)
    reg_sphere_points = tf.constant(reg_sphere_points, dtype=tf.float32)

    return training_cart_points, training_sphere_points, validation_cart_points, validation_sphere_points, reg_sphere_points
        

def parameterform_s2_tf(theta_phis:tf.Tensor, R:float) -> tf.Tensor:
    ''' Parameterform for the 2-sphere, transforming spherical polar coordinates to R³

    Args:
        - theta_phis (np.ndarray): List of theta and phi values
        - R (float): Radius of the sphere

    Returns:
        -carth_coords (tf.Tensor): Carthesian coords for spherical coords. 
    '''
    theta = theta_phis[:, 0]
    phi   = theta_phis[:, 1]
    x = R * tf.sin(theta) * tf.cos(phi) 
    y = R * tf.sin(theta) * tf.sin(phi)
    z = R * tf.cos(theta)
    carth_coords = tf.stack([x, y, z], axis=1)
    return carth_coords


def geometric_regularization(model:tf.keras.Model, reg_sphere_points: tf.Tensor) -> tf.Tensor:
    ''' Calculates the regularzation via the pullback metric.
    The latter can be achieved by calculating the discrete partial derivative for θ and Φ.

    Args:
        - model (tf.keras.Model): Model to be regularized
        - reg_sphere_points (tf.Tensor): Points used for regularization

    Returns:
        - diff_pb_s2_metric (tf.Tensor): Mean of the distances between pullback metric and S² metric tensor. 
    '''
    
    #Variation in θ direction
    training_sphere_varied_theta = tf.stack([
        reg_sphere_points[:, 0] + VARIATION_DELTA,
        reg_sphere_points[:, 1]
    ], axis=1)

    #Variation in Φ direction.
    training_sphere_varied_phi = tf.stack([
        reg_sphere_points[:, 0],
        reg_sphere_points[:, 1] + VARIATION_DELTA
    ], axis=1)

    ouput_unvaried = model(reg_sphere_points)
    output_varied_theta = model(training_sphere_varied_theta)
    output_varied_phi   = model(training_sphere_varied_phi)

    dist_theta = output_varied_theta - ouput_unvaried
    dist_phi   = output_varied_phi   - ouput_unvaried 

    g_tt = tf.reduce_sum((dist_theta * dist_theta)/VARIATION_DELTA**2, axis=1)
    g_pp = tf.reduce_sum((dist_phi * dist_phi)/VARIATION_DELTA**2, axis=1)
    g_tp = tf.reduce_sum((dist_theta * dist_phi)/VARIATION_DELTA**2, axis=1)

    pullback_metrics = tf.stack([tf.stack([g_tt, g_tp], axis=1),tf.stack([g_tp, g_pp], axis=1)], axis=1)
    thetas = reg_sphere_points[:, 0]

    g_pp_target = R**2 * tf.sin(thetas)**2

    metric_tensors = tf.stack([
        tf.stack([R**2*tf.ones_like(thetas), tf.zeros_like(thetas)], axis=1),
        tf.stack([tf.zeros_like(thetas), g_pp_target], axis=1)
    ], axis=1)

    diff_pb_s2_metric = tf.reduce_mean(tf.square(pullback_metrics-metric_tensors))

    return diff_pb_s2_metric


def log_state(ep:int, LR:float, loss:tf.Tensor, val_loss:tf.Tensor, geo_reg:tf.Tensor) -> None:
    '''Logs parameters of the current training iteration

    Args:
        - ep (int): Current epoche
        - LR (float): Learning rate
        - loss (tf.Tensor): L2 training loss (+ regularization)
        - val_loss (tf.Tensor): Validation loss
        - geo_reg (tf.Tensor): Regularization value/loss

    Returns:
        -None
    '''

    print(f"Ep:{ep} | Loss:{loss} | LR: {LR} | Geo_Reg:{geo_reg}| Val Loss: {val_loss}")
    mlflow.log_metric("Training Loss", loss, step=ep)
    mlflow.log_metric("Validation Loss", val_loss, step=ep)
    mlflow.log_metric("Regularization Value", geo_reg, step=ep)


def log_params() -> None:
    '''Logs fixed parameter/constants to MLFlow

    Args:
        -
    Returns:
        -
    '''
    mlflow.log_param("Variation Delta", VARIATION_DELTA)
    mlflow.log_param("Regularization Lambda", REG_LAMBDA)


#PREP
training_cart_points, training_sphere_points, validation_cart_points, validation_sphere_points, reg_sphere_points= create_training_and_validation_data()

print(f"Training cart points shape:{training_cart_points.shape}")
print(f"Training spheric points shape:{training_sphere_points.shape}")

model, optimizer = define_model(LR)


#TRAINING LOOP

with mlflow.start_run() as run:
    log_params()
    for ep in range(N_EPOCHS):
        if ep > 400:
            LR *= 0.9999
            optimizer.learning_rate.assign(LR)
        with tf.GradientTape() as tape:
            if ep >= 0:
                geo_reg = geometric_regularization(model, reg_sphere_points)
            else:
                geo_reg = 0

            predicted_points = model(training_sphere_points)
        
            loss = tf.reduce_mean(tf.reduce_sum((predicted_points - training_cart_points)**2, axis=1))
            loss += REG_LAMBDA * geo_reg

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        validation_out = model(validation_sphere_points)
        val_loss = tf.reduce_mean(tf.reduce_sum((validation_out - validation_cart_points)**2, axis=1))
        log_state(ep, LR, loss, val_loss, geo_reg)
        

    mlflow.log_artifact("./s2_encoder.py")
    mlflow.tensorflow.log_model(model, name="s2_encoder")


#PLOT
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = 'blue'
ax.scatter(validation_out[:, 0], validation_out[:, 1], validation_out[:, 2], c=colors, marker='o', alpha =0.2, s=20.)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_box_aspect([1, 1, 1])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = 'blue'
ax.scatter(validation_cart_points[:, 0], validation_cart_points[:, 1], validation_cart_points[:, 2], c=colors, marker='o', alpha =0.2, s=20)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_box_aspect([1, 1, 1])
plt.show()