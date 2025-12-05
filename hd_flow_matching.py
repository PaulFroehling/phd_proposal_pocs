'''The following experiment tries to replicates the approach of the Paper
"Generative Modeling of Discrete Joint Distributions by E-Geodesic Flow Matching on Assignment Manifolds"
by Bastian Boll1 Daniel Gonzalez-Alvarado1 Christoph Schnörr (https://arxiv.org/pdf/2402.07846)
'''

import os
import mlflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
print("Available Devices:", tf.config.list_physical_devices())
print("GPU is used:", tf.config.list_physical_devices('GPU'))

MLFLOW_HOST = "127.0.0.1"
MLFLOW_PORT = "8080"
MLFLOW_EXPERIMENT_NAME = "Flow Matching Reg"
N_EPOCHS = 1000
LR = 1e-3

N_GEODESICS = 1000
N_GEODESIC_POINTS = 100
N_DIMS_EMBEDDING = 4

FRAC_TRAIN = 0.7

TRAINING_AND_VALIDATION_DATA_PATH = "data/training_and_validation/flow_map_toy/training_validation.npz"


def generate_factorizing_distributions(n_samples:int) -> np.ndarray:
    """_Utilty function for generating random segre embedded factorizing distribution. 
    Mainly used for plotting the assignment manifold. 

    Args:
        n_samples (int): Number of samples to be generated

    Returns:
        np.ndarray: Array of generated factorizing distributions used for plotting
    """    

    dists = []
    
    for _ in range(n_samples):
        v = np.random.rand()
        w = np.random.rand()
        fact_dist = np.array([v*w, v*(1-w), (1-v)*w, (1-v)*(1-w)])
        dists.append(fact_dist)

    return np.array(dists)


def define_model(LR:int) -> tuple[tf.keras.Model, tf.keras.optimizers.Adam]:
    """Defines the neural network for Flow Matching

    Args:
        - LR (float): Learning rate
    Returns:
        -model (tf.keras.Model): Model definition
        -opt   (tf.keras.Optimizer): Optimizer to train with
    """
    
    activation = "relu6"
    inputs = tf.keras.Input(shape=(4,))
    x = tf.keras.layers.Dense(64, activation=activation)(inputs)
    x = tf.keras.layers.Dense(64, activation=activation)(x)
    x = tf.keras.layers.Dense(64, activation=activation)(x)
    x = tf.keras.layers.Dense(64, activation=activation)(x)
    outputs = tf.keras.layers.Dense(4)(x)
    model = tf.keras.Model(inputs, outputs)
    model.summary()

    opt = tf.keras.optimizers.AdamW(
        learning_rate=LR,
    )
    
    return model, opt


def load_training_and_validation_data(force_overwrite:bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Restores training and validation data from disk, if exists. Otherwise it creates training data and loads if afterwards.

    Args:
        -
    Returns: 
        - geodesics (np.ndarray): Geodesics - i.e. list of n_geodesic_points between start and target
        - start_points (np.ndarray): Array of start points
        - target_points (np.ndarray): Array of target points
        - train_indices (np.ndarray): Array of training indices, used for prediction
    """
    if os.path.exists(TRAINING_AND_VALIDATION_DATA_PATH) == False:
        print("Creating new data...!")
        generate_training_data(N_GEODESICS, N_GEODESIC_POINTS)
    else:
        print("Loading existing data...!")

    data = np.load(TRAINING_AND_VALIDATION_DATA_PATH)
    x_train = data['x_train']
    vts_train = data['vts_train']
    x_val = data['x_val']
    vts_val = data['vts_val']

    return x_train, vts_train, x_val, vts_val


def save_training_data_to_disk(x_train, vts_train, x_val, vts_val) -> None:
    """ Creates random start and target points for geodesics associated with a parameter value t, that is used for prediction.
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
    """
    np.savez(TRAINING_AND_VALIDATION_DATA_PATH,
         x_train=x_train,
         vts_train=vts_train,
         x_val=x_val,
         vts_val=vts_val)


def plot_flow(flows:np.ndarray, fact_dists:np.ndarray) -> None:
    """Plotting flows, assignment manifold (by points) and simplex

    Args:
        flows (np.ndarray): List of geodesic flows
        fact_dists (np.ndarray): Some factorizing dists embedded and ready to plot
    """     

    simplex_dists = []
    x = np.linspace(0,1,100)
    n=10

    for x_val in x:
        y = np.linspace(0, 1-x_val, n)
        for y_val in y:
            z = np.linspace(0, 1-(y_val+x_val), n)
            for z_val in z:
                if x_val+y_val+z_val <=1:
                    simplex_dists.append([x_val, y_val, z_val])        

    simplex_dists = np.array(simplex_dists)
    fact_dists = np.array(fact_dists)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(simplex_dists[:,0], simplex_dists[:,1], simplex_dists[:,2], alpha=0.4)
    #x.scatter(fact_dists[:,0], fact_dists[:,1], fact_dists[:,2], c="red")
    for flow in flows:
        ax.scatter(flow[:,0], flow[:,1], flow[:,2], c="black", s=5)
    plt.show()               


def calc_segre_embedding(w:np.ndarray) -> np.ndarray:
    """Calculates the segre embedding of c=2, n=2 distributions, defining joint distributions of two marginals

    Args:
        w (np.ndarray): 2x2 matrix of marginal distributions

    Returns:
        np.ndarray: Joint distr.
    """    
    w1, w2 = w.reshape(2,2)[:, 0]
    return np.array([w1 * w2, w1 * (1-w2), (1-w1) * w2, (1-w1) * (1-w2)])


def sample_c2_n2_data(n_samples:int) -> np.ndarray:
    samples = []
    for _ in range(n_samples):
        samples.append([np.random.randint(2), np.random.randint(2)])

    return samples


def sample_tangent_vector() -> np.ndarray:
    """Randomly samples a tangent vector for c=2, n=2 from std normal distribution
    Returns:
        np.ndarray: Resulting tangent vector
    """    
    rnd1 = np.random.randn()
    rnd2 = np.random.randn()
    return np.array([[rnd1, -rnd1], [rnd2, -rnd2]])


def exp_map(w:np.ndarray, v:np.ndarray) -> np.ndarray:
    """Calculates the exponent map for a point on the manifold w and a direction v

    Args:
        w (np.ndarray): Point on manifold
        v (np.ndarray): Direction (tangent vector)

    Returns:
        np.ndarray: Resulting point on the manifold
    """    
    exp_v1 = np.exp(v[0])
    exp_v2 = np.exp(v[1])
    map_1 = (exp_v1 * w[0]) / np.dot(exp_v1,w[0])
    map_2 = (exp_v2 * w[1]) / np.dot(exp_v2,w[1])
    total_map = np.array([map_1, map_2])
    return total_map


def generate_start_dists(n_samples:int, barycenter:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generates start distribution by randomly sampling tangent vectors at the barycenter of the assignment manifold.

    Args:
        n_samples (int): Number of start distributions
        barycenter (np.ndarray): Barycenter of the manifold

    Returns:
        tuple[np.ndarray, np.ndarray]: Samples at the manfold and tangent vectors in the tangent space of the barycenter
    """    
    samples = []
    tang_samples = []
    for _ in range(n_samples):
        v_tang = sample_tangent_vector()
        v_tang_exp = exp_map(barycenter, v_tang)
        tang_samples.append(v_tang)
        samples.append(v_tang_exp)
    
    return np.array(samples), np.array(tang_samples)


def apply_riemann_map(W:np.ndarray, v:np.ndarray) -> np.ndarray:
    """Draws velocity prediction from the neural network to the geometry of the manifold. 
    Formular 2b) of the paper

    Args:
        W (np.ndarray): Point on the manifold (format (2x2 matrix)
        v (np.ndarray): Tangent vector from the model (2x2 matrix)

    Returns:
        np.ndarray: Corrected velocity prediction
    """    
    R = np.zeros_like(W)
    for i in range(2):
        Wi = W[i]     
        vi = v[i]      
        inner = np.dot(Wi, vi)
        R[i] = Wi * vi - Wi * inner
    return R


def log_map(b:np.ndarray, x:np.ndarray) -> np.ndarray:
    """Logarithmic map, pushing a point x from the assignment manifold to the tangent space of b

    Args:
        b (np.ndarray): Reference point of the tangent space
        x (np.ndarray): Point to be pushed to the tangent space

    Returns:
        np.ndarray: Point in tangent space of b
    """    
    v = np.zeros_like(x)
    for i in range(2):
        u = np.log(x[i]) - np.log(b[i]) 
        v[i] = u - u.mean()              
    return v


def calculate_q_beta(beta:np.ndarray, eta:float = 1e-2) -> np.ndarray:
    """Drawing data samples (diracs) to the manifold, while encoding direction from bary center

    Args:
        beta (np.ndarray): Data sample
        eta (float, optional): Small value of divergence. Defaults to 1e-2.

    Returns:
        np.ndarray: Target point for trainign
    """    
    q_beta = eta*np.array([[0.5,0.5],[0.5,0.5]]) + (1-eta)*beta

    return q_beta


def generate_target_dists(n_samples:int, barycenter:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generates an array of target samples for geodesics

    Args:
        n_samples (int): Number of samples
        barycenter (np.ndarray): Barycenter of the manifold as reference

    Returns:
        tuple[np.ndarray, np.ndarray]: Two arrays for target distributions and the corresponding logarithmic map
    """   
    samples = sample_c2_n2_data(n_samples)
    target_dists = []
    log_target_dists = []
    for s in samples:
        marg_1 = [1,0] if s[0] == 0 else [0, 1]
        marg_2 = [1,0] if s[1] == 0 else [0, 1]

        distr = np.array([marg_1, marg_2])
        q_beta = calculate_q_beta(distr)
        q_beta_log = log_map(barycenter, q_beta)
        target_dists.append(q_beta)
        log_target_dists.append(q_beta_log)
    
    return np.array(target_dists), np.array(log_target_dists)


def compute_tangent_interpolation(v1:np.ndarray, v2:np.ndarray, t:int) -> np.ndarray:
    """Computes linear interpolation in tangent space between v1 and v2

    Args:
        v1 (np.ndarray): Tagent vector 1
        v2 (np.ndarray): Tangent vector 2
        t (int): "Timestept" t

    Returns:
        np.ndarray: Resulting tangent vector for t
    """    
    v1 = v1.flatten()
    v2 = v2.flatten()
    interp = v1 + t*(v2-v1)
    return interp.reshape(2,2)


def generate_training_data(n_geodesics:int, n_geodesic_points:int) -> None:
    """Generates a new set of training and validation data

    Args:
        n_geodesics (int): Number of geodesics
        n_geodesic_points (int): Number of points for each geodesic
    """    
    x_train = []
    vts_train = []
    x_val = []
    vts_val = []
    barycenter = np.array([[0.5, 0.5], [0.5, 0.5]])
    _, target_dists_log = generate_target_dists(n_geodesics, barycenter)
    _, start_dists_log = generate_start_dists(n_geodesics, barycenter)

    for i in range(n_geodesics):
        ts = np.clip(np.random.random(n_geodesic_points),0,1)
        for t_idx, t in enumerate(ts):
            v_t = compute_tangent_interpolation(start_dists_log[i], target_dists_log[i], t) 
            w_t = exp_map(barycenter, v_t)
            model_target = (target_dists_log[i] - start_dists_log[i]).flatten()
            if t_idx < int(n_geodesic_points * FRAC_TRAIN):
                x_train.append(w_t.flatten())
                vts_train.append(model_target)
            else:
                x_val.append(w_t.flatten())
                vts_val.append(model_target)    
            
    x_train = tf.constant(x_train, dtype=tf.float32)
    vts_train = tf.constant(vts_train, dtype=tf.float32)
    x_val = tf.constant(x_val, dtype=tf.float32)
    vts_val = tf.constant(vts_val, dtype=tf.float32)

    save_training_data_to_disk(x_train, vts_train, x_val, vts_val)


def log_state(ep:int, LR:float, mean_loss:tf.Tensor, val_loss:tf.Tensor, reg:tf.Tensor) -> None:
    """Utility function for logging during training

    Args:
        ep (int): Epoche
        LR (float): Learning Rate
        mean_loss (tf.Tensor): Training Loss
        val_loss (tf.Tensor): Validation Loss
        reg (tf.Tensor): Regulariazion term
    """    
    print(f"Ep: {ep} | Loss: {mean_loss:.5} | LR: {LR:.4} | Val_Loss: {val_loss:.4}| Reg: {reg:.4}")


def integrate_flow(model:tf.keras.Model, n_geodesics:int, n_geodesic_points:int = 100):
    """Generates an array of flows, by integrates the result from the model using a Euler integration

    Args:
        model (tf.keras.Model): Trained model
        n_geodesics (int): Number of geodesics
        n_geodesic_points (int, optional): Number of points per geodesic. Defaults to 100.

    Returns:
        _type_: _description_
    """    
    dt = 1 / n_geodesic_points
    flows=[]
    barycenter = np.array([[0.5, 0.5], [0.5, 0.5]])
    p_start, p_start_t = generate_start_dists(n_geodesics, barycenter)
    for i in range(n_geodesics):
        p_ts = []
        p_t = p_start[i].flatten().copy()
        for t in range(n_geodesic_points):
            v_t = model(tf.constant([p_t], dtype=tf.float32)).numpy()
            v_t = apply_riemann_map(p_t.reshape(2,2), v_t.reshape(2,2)).reshape(4)
            p_t += v_t*dt
            p_t_embed = calc_segre_embedding(p_t)
            p_ts.append(p_t_embed.copy())
        
        flows.append(p_ts)
    for i in range(n_geodesic_points):
        np.savetxt(f"animation_data/points_{i}.csv", np.array(flows)[:,i], delimiter=",")
    return np.array(flows)


def reg(predicted_velocity_train:tf.Tensor) -> tf.Tensor:
    """Regularization approach forcing the model to learn vectors of sum 0

    Args:
        predicted_velocity_train (tf.Tensor): Model output

    Returns:
        tf.Tensor: Sum over components
    """    

    sum1 = tf.reduce_mean(tf.nn.sigmoid(tf.reduce_sum(predicted_velocity_train[:,:2], axis=1)))
    sum2 = tf.reduce_mean(tf.nn.sigmoid(tf.reduce_sum(predicted_velocity_train[:,2:], axis=1)))
    return sum1 + sum2


#TRAINING LOOP
x_train, vts_train, x_val, vts_val = load_training_and_validation_data()
model, optimizer = define_model(LR)

for ep in range(N_EPOCHS):
    if ep >= 350 and ep % 10 == 0:
        LR *= 0.99
        optimizer.learning_rate.assign(LR)

    with tf.GradientTape() as tape:
        
        predicted_velocity_train = model(x_train)
        geo_reg = reg(predicted_velocity_train)
        loss = tf.reduce_mean((predicted_velocity_train - vts_train)**2)

        if ep < 1600:
            reg_lambda = 0
        else:
            reg_lambda = 0.2
        loss += reg_lambda * geo_reg

    predicted_velocity_val =  model(x_val)
    val_loss = tf.reduce_mean((predicted_velocity_val - vts_val)**2)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    log_state(ep, LR, loss, val_loss, geo_reg)        


fact_dists = generate_factorizing_distributions(600)
flows = integrate_flow(model, 600, 100)

plot_flow(flows, fact_dists)