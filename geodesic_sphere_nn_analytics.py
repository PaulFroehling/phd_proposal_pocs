import os
import numpy as np
import tensorflow as tf
from geodesics.s2 import s2_geodesic
import matplotlib.pyplot as plt

np.random.seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

R = 1
N_GEODESIC_POINTS = 400+2
N_T_STEPS =10
DT = 1/(N_T_STEPS-1)
N_SAMPLES = 200

geo = s2_geodesic()


class L2Normalize(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)


def calc_average_stability(model_outputs:np.ndarray) -> np.ndarray:
    ''' Calculates to which degree points on a geodesic have constant "speed"
    Args:
        - model_outputs (np.ndarray): Model output from geodesic model
    Returns:
        - speed_stability (np.ndarray): Stability/consistency of speed. Difference from average should be 0 in the optimal case. 

    '''
    speeds = model_outputs[1:] - model_outputs[:-1]
    speeds = np.linalg.norm(speeds, axis=1)/DT
    avg_speeds = np.mean(speeds)
    speed_stability = (speeds - avg_speeds)**2

    return np.mean(speed_stability)

def create_boxplot_visualization(model_reg:tf.keras.Model, model_non_reg:tf.keras.Model) -> None:
    ''' Visualizes the differences in the "speed" of a geodesic for a regularzied and non-regularized model.

    Args:
        - model_reg (tf.keras.Model): Regularized model
        - model_non_reg (tf.keras.Model): Model without regularization

    Returns:
        - 
    '''
    speed_diffs_reg = []
    speed_diffs_non_reg = []

    t_vals = np.linspace(0.0, 1.0, N_T_STEPS)

    for _ in range(N_SAMPLES):
            theta_1  = np.random.random() * np.pi
            theta_2 = theta_1 + np.random.random() * (np.pi - theta_1)
            phi_1  = np.random.random() * 2 * np.pi - np.pi
            phi_2 = phi_1 + np.random.random() * np.pi
            
            eucl_1 = geo.parameterform_s2(theta_1, phi_1, R)
            eucl_2 = geo.parameterform_s2(theta_2, phi_2, R)
            outputs_reg = []
            outputs_non_reg = []
            for t in t_vals:
                model_input = tf.constant(np.hstack([eucl_1, eucl_2, t]).reshape(1, -1))
                outputs_reg.append(np.squeeze(model_reg(model_input)))
                outputs_non_reg.append(np.squeeze(model_non_reg(model_input)))

            outputs_reg = np.array(outputs_reg)
            outputs_non_reg = np.array(outputs_non_reg)

            speed_diffs_reg.append(calc_average_stability(outputs_reg))
            speed_diffs_non_reg.append(calc_average_stability(outputs_non_reg))


    plt.boxplot([speed_diffs_reg,speed_diffs_non_reg], labels=["Speed Differences - Regularized", "Speed Differences - Not Regularized"], showfliers=False)
    plt.show()


#Load models (regularized & non-regularized)
model_reg = tf.keras.models.load_model(
   "models/geodesics/regularized.keras",
    custom_objects={"L2Normalize": L2Normalize},
    compile=False
)

model_non_reg = tf.keras.models.load_model(
   "models/geodesics/no_regularization.keras",
    custom_objects={"L2Normalize": L2Normalize},
    compile=False
)


create_boxplot_visualization(model_reg, model_non_reg)


