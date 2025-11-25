'''
Training script for a regression model, learning geodesics on S²
'''

import os
import mlflow
import tensorflow as tf
import numpy as np
from geodesics.s2 import s2_geodesic
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
print("Available Devices:", tf.config.list_physical_devices())
print("GPU is used:", tf.config.list_physical_devices('GPU'))

np.random.seed(42)
tf.random.set_seed(42)

geo = s2_geodesic()

R = 1
N_EPOCHS =30
N_SAMPLES = 1000
N_GEODESIC_POINTS = 400+2
FRAC_TRAIN_SAMPLES = 0.8
GEODESICS_BATCH_SIZE = 100
POINTS_BATCH_SIZE = 10
REGULARIZATION_LAMBDA = 0.5
LR = 1e-2
LOSS_FUNCTION = "spheric"
TRAINING_AND_VALIDATION_DATA_PATH = "data/training_and_validation/training_validation_seed_42.npz"

MLFLOW_HOST = "127.0.0.1"
MLFLOW_PORT = "8080"
mlflow.set_tracking_uri(uri=f"http://{MLFLOW_HOST}:{MLFLOW_PORT}")
mlflow.set_experiment("Neural S2 Geodesic")

class L2Normalize(tf.keras.layers.Layer):
    '''Used to retract points to the sphere of radius 1'''
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

def define_model(LR:int) -> tuple[tf.keras.Model, tf.keras.optimizers.Adam]:
    '''Defines the neural network for predicting geodesics. 

    Args:
        - LR (float): Learning rate
    Returns:
        -model (tf.keras.Model): Model definition
        -opt   (tf.keras.Optimizer): Optimizer to train with
    '''
    
    activation = "tanh"
    inputs = tf.keras.Input(shape=(7,))
    x1 = tf.keras.layers.Dense(64, activation=activation)(inputs)
    x = tf.keras.layers.Dense(64, activation=activation)(x1)
    x = tf.keras.layers.Dense(64, activation=activation)(x)
    x = tf.keras.layers.Dense(64, activation=activation)(x)
    x = tf.keras.layers.Dense(64, activation=activation)(x)
    x = tf.keras.layers.Dense(3)(x)
    outputs = L2Normalize()(x)
    model = tf.keras.Model(inputs, outputs)
    model.summary()

    opt = tf.keras.optimizers.AdamW(
        learning_rate=LR,
    )
    
    return model, opt

#TODO: Sample so that start and target is always part of the training data
def create_training_data(n_samples:int, n_geodesic_points:int) -> None:
    ''' Creates random start and target points for geodesics associated with a parameter value t, that is used for prediction.
    (Eventhough the sampling is quite simple like this, points are not equally distributed over the sphere. But ok for testing purpose)
    Results are saved to disk

    Args:
        - n_samples (int): Number of samples to create
        - n_geodesic_points (int): Number of points created between start and target. 
    Returns:
        -
    '''
    start_points  = []
    target_points = []
    train_indices = []
    train_points  = []
    val_indicies  = []
    val_points    = []

    for _ in range(n_samples):
        theta_start  = np.random.random() * np.pi
        theta_target = theta_start + np.random.random() * (np.pi - theta_start)

        phi_start  = np.random.random() * 2 * np.pi - np.pi
        phi_target = phi_start + np.random.random() * np.pi

        geodesic = geo.calculate_geodesic(phi_start, phi_target, theta_start, theta_target, R, n_points=n_geodesic_points)

        geodesics.append(geodesic)
        all_indices = np.array(range(n_geodesic_points))
        prediction_indices = all_indices[5:-5]
        #prediction_indices = list(range(n_geodesic_points))
        np.random.shuffle(prediction_indices)
        n_training_samples = int(n_geodesic_points * FRAC_TRAIN_SAMPLES)
        batch_train_indices = np.concat([prediction_indices[:n_training_samples], np.setdiff1d(all_indices, prediction_indices)])
        train_indices.append(batch_train_indices)
        val_indicies.append(prediction_indices[n_training_samples:])
        train_points.append(geodesic[batch_train_indices])
        val_points.append(geodesic[prediction_indices[n_training_samples:]])

        start_points.append([*geo.parameterform_s2(theta_start, phi_start, R)])
        target_points.append([*geo.parameterform_s2(theta_target, phi_target, R)])

    train_indices = np.array(train_indices)/(N_GEODESIC_POINTS-1)
    val_indicies = np.array(val_indicies)/(N_GEODESIC_POINTS-1)

    save_training_data_to_disk(geodesics, np.array(start_points), np.array(target_points), np.array(train_indices), np.array(val_indicies), np.array(train_points), np.array(val_points))


def save_training_data_to_disk(geodesics, start_points, target_points, train_indices, val_indices, train_points, val_points) -> None:
    ''' Creates random start and target points for geodesics associated with a parameter value t, that is used for prediction.
    (Eventhough the sampling is quite simple like this, points are not equally distributed over the sphere. But ok for testing purpose)
    Results are saved to disk

    Args:
        - geodesics (np.ndarray): Geodesics - i.e. list of n_geodesic_points between start and target
        - start_points (np.ndarray): Array of start points
        - target_points (np.ndarray): Array of target points
        - train_indices (np.ndarray): Array of training indices, used for prediction
        - val_indices (np.ndarray): Array of validation indices used for prediction
        - train_points (np.ndarray): Array of training points - i.e. points at index position in train_indices
        - val_points (np.ndarray): Array of validation points - i.e. points at inidex position in validation_indices

    Returns:
        -
    '''
    np.savez("data/training_and_validation/training_validation_seed_42.npz",
         geodesics=geodesics,
         start_points=start_points,
         target_points=target_points,
         train_indices=train_indices,
         val_indices=val_indices,
         train_points=train_points,
         val_points=val_points)
    
def load_training_and_validation_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ''' Restores training and validation data from disk, if exists. Otherwise it creates training data and loads if afterwards.

    Args:
        -
    Returns: 
        - geodesics (np.ndarray): Geodesics - i.e. list of n_geodesic_points between start and target
        - start_points (np.ndarray): Array of start points
        - target_points (np.ndarray): Array of target points
        - train_indices (np.ndarray): Array of training indices, used for prediction
        - val_indices (np.ndarray): Array of validation indices used for prediction
        - train_points (np.ndarray): Array of training points - i.e. points at index position in train_indices
        - val_points (np.ndarray): Array of validation points - i.e. points at inidex position in validation_indices

    '''
    if os.path.exists(TRAINING_AND_VALIDATION_DATA_PATH) == False:
        print("Creating new data...!")
        create_training_data(n_samples=N_SAMPLES, n_geodesic_points=N_GEODESIC_POINTS)
    else:
        print("Loading existing data...!")

    data = np.load(TRAINING_AND_VALIDATION_DATA_PATH)
    geodesics     = data['geodesics']
    start_points  = data['start_points']
    target_points = data['target_points']
    train_indices = data['train_indices']
    val_indices   = data['val_indices']
    train_points  = data['train_points']
    val_points    = data['val_points']

    return geodesics, start_points, target_points, train_indices, val_indices, train_points, val_points


def parameterform_s2_tf(theta_phis:tf.Tensor, R:int) -> tf.Tensor:
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
    return tf.stack([x, y, z], axis=1)


def eucl_distance(predictions:tf.Tensor, groundtruth:tf.Tensor) -> tf.Tensor:
    '''Calculates the euclidean distance between a prediction vector and groundtruth vector

    Args:
        - predictions (tf.Tensor): Prediction from model 
        - groundtruth (tf.Tensor): Groundtruth from training/validation/test data
    Returns:
        - result (tf.Tensor): Euclidean distance between inputs
    '''

    diff = tf.norm(predictions - groundtruth, axis=1)
    return tf.reduce_mean(diff)


def spheric_distance(predictions:tf.Tensor, groundtruth:tf.Tensor, R:float=1.0) -> tf.Tensor:
    '''Calculates the spherical distance between a prediction vector and groundtruth vector

    Args:
        - predictions (tf.Tensor): Prediction from model 
        - groundtruth (tf.Tensor): Groundtruth from training/validation/test data
        - R (float): Radius 
    Returns:
        - result (tf.Tensor): Spherical distance between inputs
    '''
    dot_product = tf.reduce_sum(predictions * groundtruth, axis=1)
    clipped_dot_product = tf.clip_by_value(dot_product/R**2,-0.999999,0.999999)
    distance = R*tf.reduce_mean(tf.acos(clipped_dot_product))
    return distance


def compute_loss(predictions:tf.Tensor, groundtruth:tf.Tensor, type:str='euclidean'):
    ''' Computes the loss depending on the input value for type. Can be euclidean or spheric. 

    Args:
        - predictions (tf.Tensor): Prediction from model 
        - groundtruth (tf.Tensor): Groundtruth from training/validation/test data
        - type        (str)      : eulicdean or spheric
        
    Returns:
        -result (tf.Tensor): Euclidean or spherical distance between inputs
    '''
    loss_functions = {'euclidean': eucl_distance, 'spheric': spheric_distance}
    if type not in loss_functions:
        raise ValueError("Unknown Loss Type")
    return loss_functions[type](predictions, groundtruth)
    

def create_batch(start_target_points:np.ndarray, indices:np.ndarray) -> tf.Tensor:
    ''' Creates a batch for the training loop. Since there are multiple indices for one start-target combination, each combination is assigned one of the indices.

    Args:
        - start_target_points (np.ndarray): List of start and target points
        - indices (np.ndarray): List of indices between start and target
    Returns:
        - result (tf.Tensor): Stacked version of start-target combinations and single indices
    '''
    batch = []
    for i in range(start_target_points.shape[0]):
        for j in range(indices.shape[1]):
            batch.append(np.hstack([start_target_points[i],indices[i,j]]))

    return tf.constant(batch, dtype=tf.float32)

def perform_validation(model:tf.keras.Model, p_start:np.ndarray, p_target:np.ndarray, validation_indices:np.ndarray, validation_points:np.ndarray) -> tf.Tensor:
    ''' Computes validation loss from validation samples

    Args:
        - model (tf.keras.Model): Current state of the model
        - p_start (np.ndarray): Start points
        - p_target (np.ndarray): Target points
        - validation_indices (np.ndarray): Validation indices
        - validation_points (np.ndarray): Corresponding validation points for validation indices
    Returns:
        - validation_loss (tf.Tensor): Resulting validation loss for given inputs
    '''
    stacked_start_target = np.hstack([p_start, p_target])
    validation_input = create_batch(stacked_start_target, validation_indices)
    predictions = model(validation_input)
    validation_points = validation_points.reshape(-1,3)
    validation_loss = compute_loss(predictions, validation_points, LOSS_FUNCTION)
    
    return validation_loss

def geometric_regularization(model: tf.keras.Model, stacked_start_target:np.array) -> tf.Tensor:
    '''Regularization function that penalizes varying speeds for predicted geodesics.

    Args:
        - model (tf.keras.Model): Current state of the model
        - stacked_start_target (np.ndarray): Combination of start-target points and indices on the geodesic curve
    
    Returns:
        - reg (tf.Tensor): Regularization value/loss
    '''
    n_t_steps = 50
    n_batches = stacked_start_target.shape[0]

    t_vals = tf.cast(tf.linspace(0.0, 1.0, n_t_steps), dtype=tf.float32)
    t_vals = tf.reshape(t_vals, [1, n_t_steps, 1]) 
    t_vals = tf.tile(t_vals, [n_batches, 1, 1])

    dt = tf.constant(1/(n_t_steps-1))
    stacked_start_target = tf.constant(stacked_start_target, dtype=tf.float32)
    repeated_stack = tf.reshape(stacked_start_target, shape=[n_batches, 1, 6])
    repeated_stack = tf.tile(repeated_stack, [1, n_t_steps, 1])

    model_input = tf.concat([repeated_stack, t_vals], axis=2)
    model_input = tf.reshape(model_input, [n_batches*n_t_steps, 7])
    predicted_points = tf.reshape(model(model_input), shape=[n_batches, n_t_steps, 3])
    diffs = predicted_points[:,1:,:] - predicted_points[:,:-1,:]
    speeds = tf.norm(diffs, axis=2)/dt
    avg_speed = tf.reduce_mean(speeds, axis=1, keepdims=True)
    reg = tf.reduce_mean(tf.square(speeds-avg_speed))

    return reg


def log_params():
    '''Logs fixed paramters/constants to MLFlow
    Args:
        - 
    Returns:
        -
    '''
    mlflow.log_param("Loss function", LOSS_FUNCTION)
    mlflow.log_param("Regularisation λ", REGULARIZATION_LAMBDA)
    mlflow.log_param("Start points shape", start_points.shape)
    mlflow.log_param("Target points shape", target_points.shape)
    mlflow.log_param("Train indices shape", train_indices.shape)
    mlflow.log_param("Train points shape", train_points.shape)
    mlflow.log_param("Val indices shape", val_indicies.shape)
    mlflow.log_param("Val points shape", val_points.shape)


def plot_sample_geodesic_for_trained_model(model: tf.keras.Model):
    ''' Plots a sample geodesic after training.
    Args:
        - model (tf.keras.Model): Trained model
    
    Returns:
        -
    '''
    phi_1, theta_1 = 0.8 * np.pi, 0.4 * np.pi 
    phi_2, theta_2 = 2 * np.pi - 1.2 * np.pi, 0.8 * np.pi

    eucl_p1 = geo.parameterform_s2(theta_1, phi_1, 1)
    eucl_p2 = geo.parameterform_s2(theta_2, phi_2, 1)
    neural_geodesic = []

    eucl_s2, _ = geo.generate_sphere_data(100, 1)

    for i in range(N_GEODESIC_POINTS):
        model_input = tf.constant(np.hstack([eucl_p1, eucl_p2, i/(N_GEODESIC_POINTS-1)]).reshape(1, -1))
        model_out = model(model_input)
        neural_geodesic.append(np.squeeze(model_out.numpy()))

    neural_geodesic = np.array(neural_geodesic)
    geo.create_3d_scatter_plot(eucl_s2, [eucl_p1], [eucl_p2], [neural_geodesic])


def log_state(ep:int, LR:float, g:int, mean_loss:tf.Tensor, val_loss:tf.Tensor, mean_geo_reg:tf.Tensor) -> None:
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

    mlflow.log_metric("training_loss_l2", mean_loss, step=ep*n_geodesic_batches + g)
    mlflow.log_metric("validation_loss_l2", val_loss, step=ep*n_geodesic_batches + g)
    mlflow.log_metric("regularization_loss", geo_reg, step=ep*n_geodesic_batches + g)
    print(f"Ep:{ep} | Geodesic_Batch:{g} | Loss:{mean_loss} | LR: {LR} | Validation Loss: {val_loss:.3f} | Regularization_Term: {mean_geo_reg} | Lambda_g: {1}")

#PREP

geodesics, start_points, target_points, train_indices, val_indicies, train_points, val_points = load_training_and_validation_data()
model, optimizer = define_model(LR)

#TRAINING LOOP

with mlflow.start_run() as run:
    log_params()
    for ep in range(N_EPOCHS):
        n_geodesic_batches = int(N_SAMPLES/GEODESICS_BATCH_SIZE - 1)
        for g in range(n_geodesic_batches):
            if ep>=4:
                LR*=0.99
                optimizer.learning_rate.assign(LR)
            g_batch_start_points = start_points[g*GEODESICS_BATCH_SIZE : (g + 1) * GEODESICS_BATCH_SIZE]
            g_batch_target_points = target_points[g*GEODESICS_BATCH_SIZE : (g + 1) * GEODESICS_BATCH_SIZE]
            g_batch_train_indices = train_indices[g*GEODESICS_BATCH_SIZE : (g + 1) * GEODESICS_BATCH_SIZE]
            g_batch_train_points = train_points[g*GEODESICS_BATCH_SIZE : (g + 1) * GEODESICS_BATCH_SIZE]
            n_train_cols = g_batch_train_points.shape[1]
            n_point_batches = n_train_cols // POINTS_BATCH_SIZE
            losses = []
            geo_regs = []
            for p in range(n_point_batches):
                p_batch_train_indices = g_batch_train_indices[:, p*POINTS_BATCH_SIZE : (p + 1) * POINTS_BATCH_SIZE]
                p_batch_train_points = tf.constant(g_batch_train_points[:, p*POINTS_BATCH_SIZE : (p + 1) * POINTS_BATCH_SIZE].reshape(-1,3), dtype=tf.float32)

                stacked_start_target = np.hstack([g_batch_start_points, g_batch_target_points])
                input_batch = create_batch(stacked_start_target, p_batch_train_indices)
                with tf.GradientTape() as tape:
                    if ep >= 0:
                        geo_reg = geometric_regularization(model, stacked_start_target)

                    else:
                        geo_reg = 0
                    
                    geo_regs.append(geo_reg)

                    predicted_points = model(input_batch)
                
                    loss = compute_loss(predicted_points, p_batch_train_points, type=LOSS_FUNCTION)
                    loss += REGULARIZATION_LAMBDA*geo_reg
                    losses.append(loss)

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            val_loss = perform_validation(model, start_points, target_points, val_indicies, val_points)    
            mean_loss = np.mean(losses)
            mean_geo_reg = np.mean(geo_regs)
            log_state(ep, LR, g, mean_loss, val_loss, mean_geo_reg)


    mlflow.log_artifact("./geodesic_sphere_nn.py")
    mlflow.tensorflow.log_model(model, name="s2_geodesic_model")

    plot_sample_geodesic_for_trained_model(model)