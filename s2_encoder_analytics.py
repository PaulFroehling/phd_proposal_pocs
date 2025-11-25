"""
Evaluation script for the S² → R³ regression model.

This script computes the pullback metric of the trained encoder 
and compares it to the analytical S² metric tensor. It evaluates:

• pointwise metric tensor approximation via finite differences
• distribution of metric tensor components
• heatmap of Frobenius norm errors

This script is part of the geometric regularization proof-of-concept.
"""

import os
import math
import mlflow
import numpy as np
from numpy import pi
import tensorflow as tf
import mlflow.tensorflow
from geodesics.s2 import s2_geodesic
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

R = 1

geo = s2_geodesic()
N_GEODESIC_POINTS = 400+2

model_non_reg = tf.keras.models.load_model(
   "models/regression/no_regularization.keras"
)

model_reg = tf.keras.models.load_model(
   "models/regression/regularized.keras"
)

model = model_reg #Model used for analysis

delta = .001
n_samples_per_axis = 50
thetas = np.linspace(0,np.pi, n_samples_per_axis)
phis = np.linspace(-np.pi,np.pi, n_samples_per_axis)
g_est_tensors = []
g_est_coordinates = []
g_est_diffs = []
dists = []

#Calculate differences between pullback and S² metric:
for theta_1 in thetas:
    g_est_matrix_row= []
    for phi_1 in phis:
        phi_2, theta_2 = phi_1 + delta, theta_1 + delta

        model_out_1 = np.squeeze(model(tf.constant([[theta_1, phi_1]])))
        model_out_phi_var = np.squeeze(model(tf.constant([[theta_1, phi_2]])))
        model_out_theta_var = np.squeeze(model(tf.constant([[theta_2, phi_1]])))

        dist_phi_change = (model_out_phi_var - model_out_1)
        dist_theta_change = (model_out_theta_var - model_out_1)

        g_tt = np.dot(dist_theta_change, dist_theta_change)/delta**2 
        g_pp = np.dot(dist_phi_change, dist_phi_change)/delta**2
        g_tp = np.dot(dist_theta_change, dist_phi_change)/delta**2
        estimated_metric_tensor = np.array([[g_tt, g_tp], [g_tp, g_pp]])
        correct_metric_tensor = np.array([[R**2, 0],[0, R**2 * np.sin(theta_1)**2]])
        g_est_tensors.append([estimated_metric_tensor, correct_metric_tensor])
        g_est_coordinates.append([theta_1, phi_1])
        g_est_diffs.append(np.linalg.norm(estimated_metric_tensor-correct_metric_tensor, ord='fro'))


g_est_diffs = np.array(g_est_diffs)
g_est_tensors = np.array(g_est_tensors)
min_dist_idx = np.argmin(g_est_diffs)
max_dist_idx = np.argmax(g_est_diffs)


#Create Boxplots:
boxplot_vals_1 = [g_est_tensors[:,0,0,0], g_est_tensors[:,0,0,1], g_est_tensors[:,0,1,0], g_est_tensors[:,0,1,1]]
boxplot_vals_2 = [g_est_tensors[:,1,0,0], g_est_tensors[:,1,0,1], g_est_tensors[:,1,1,0], g_est_tensors[:,1,1,1]]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
labels =["θθ", "θΦ", "Φθ", "ΦΦ"]
axes[0].set_title("Distribution of Metric Tensor Values - Estimated", fontsize=20)
axes[0].boxplot(boxplot_vals_1, labels=labels)
axes[0].set_xlabel("Metric Tensor Position", fontsize=14)
axes[0].set_ylabel("Metric Tensor Values", fontsize=14)
axes[0].tick_params(axis='both', which='major', labelsize=25)
axes[1].set_title("Distribution of Metric Tensor Values - Correct Metric Tensor", fontsize=20)
axes[1].boxplot(boxplot_vals_2, labels=labels)
axes[1].set_xlabel("Metric Tensor Position", fontsize=14)
axes[1].set_ylabel("Metric Tensor Values", fontsize=14)
axes[1].tick_params(axis='both', which='major', labelsize=25)

fig.tight_layout()
plt.show()


#Create Heatmap:
print(np.min(g_est_diffs))
print(np.max(g_est_diffs))
g_est_diffs = g_est_diffs.reshape(n_samples_per_axis, n_samples_per_axis)
plt.imshow(g_est_diffs, cmap='hot', interpolation='nearest', vmin=0, vmax=1.1649207131875883)
plt.show()
