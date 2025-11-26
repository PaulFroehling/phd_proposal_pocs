import math
from scipy.integrate import cumulative_trapezoid
import scipy.special as special
import numpy as np
from numpy import sin, cos, tan, arctan, pi
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

class s2_geodesic():
    def __init__(self):
        pass

    def calculate_geodesic(self, phi_1:float, phi_2:float, theta_1:float, theta_2:float, R:int, n_points:int=100) -> np.ndarray:
        '''Calculate the geodesic between two points based on the solution from Euler-Lagrange $x_1$ '''
        
        phi_vals = np.linspace(phi_1, phi_2, n_points) #Linearly interpolate between phi_1 and phi_2
        alpha, beta = self.calc_alpha(phi_1, phi_2, theta_1, theta_2), self.calc_beta(phi_1, phi_2, theta_1, theta_2)
        if abs(self.calc_theta(alpha, beta, phi_1) - theta_1) > abs(self.calc_theta(-alpha, -beta, phi_1) - theta_1):
            alpha, beta = -alpha, -beta
        alpha_beta_term = alpha * cos(phi_vals) - beta * sin(phi_vals)

        sin_teta = 1/np.sqrt((alpha * sin(phi_vals) + beta * cos(phi_vals))**2 + 1)
        s = R * np.sqrt(alpha_beta_term**2 * sin_teta**4 + sin_teta**2)
            
        s = cumulative_trapezoid(s, phi_vals, initial=0)
        unif_s = np.linspace(0, s[-1], n_points)
        unif_phi = np.interp(unif_s, s, phi_vals)
        unif_theta = self.calc_theta(alpha, beta, unif_phi)
        geodesic = [[*self.parameterform_s2(theta, phi, R)] for theta, phi in zip(unif_theta, unif_phi)]
        geodesic.insert(0, self.parameterform_s2(theta_1, phi_1, R))
        geodesic.append(self.parameterform_s2(theta_2, phi_2, R))

        return np.array(geodesic)

    def cot(self, x):
        return 1/tan(x)

    def arccot(self, x):
        return pi/2 - arctan(x)


    def calc_alpha(self, phi1,phi2, theta1, theta2):
        return (self.cot(theta1) * cos(phi2) - self.cot(theta2) * cos(phi1)) / sin(phi1-phi2)

    def calc_beta(self, phi1,phi2, theta1, theta2):
        return (self.cot(theta2) * sin(phi1) - self.cot(theta1) * sin(phi2))/sin(phi1-phi2)

    def calc_theta(self, alpha, beta, phi):
        return self.arccot(alpha * sin(phi) + beta * cos(phi))

    def parameterform_s2(self, zenit, azimut, R):
        x = R * sin(zenit) * cos(azimut)
        y = R * sin(zenit) * sin(azimut)
        z = R * cos(zenit)
        return x,y,z
    
    def generate_sphere_data(self, n, r):
        zenit = np.linspace(0, pi, n)
        azimut = np.linspace(0, 2 * pi, n)
        zenit_m, azimut_m = np.meshgrid(zenit, azimut)
        zenit_m, azimut_m = zenit_m.flatten(), azimut_m.flatten()
        eucl_coords = []
        sphere_coords = []
        for i,j in zip(zenit_m, azimut_m):
            x,y,z = self.parameterform_s2(i,j,r)
            eucl_coords.append([x, y, z])
            sphere_coords.append([r, i, j])

        return np.array(eucl_coords), np.array(sphere_coords)

    def create_3d_scatter_plot(self, data:np.ndarray, start_points:np.ndarray, end_points:np.ndarray, geodesics:np.ndarray) -> None:
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
        colors = 'blue'
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, marker='o', alpha =0.2, s=0.8)
        
        for g in geodesics:
            ax.scatter(g[:, 0], g[:, 1], g[:, 2], c="purple", marker='o', alpha =0.2, s=40)
        for sp in start_points:
            ax.scatter(sp[0], sp[1], sp[2], c="yellow", marker='o', alpha =1.0, s=150)
        for ep in end_points: 
            ax.scatter(ep[0], ep[1], ep[2], c="blue", marker='o', alpha =1.0, s=150)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_box_aspect([1, 1, 1])
        plt.show()

    
    def example_function(self):
        phi_1, theta_1 = 0.7 * pi, 0.4 * pi 
        phi_2, theta_2 = 2 * pi - 1.2 * pi, 0.8 * pi

        phi_3, theta_3 = 0.2 * pi, 0.8 * pi 
        phi_4, theta_4 = 0.4 * pi, 0.9 * pi

        R = 1

        eucl_s2, _ = self.generate_sphere_data(100, R)
        p1_eucl = [*self.parameterform_s2(theta_1, phi_1, R)]
        p2_eucl = [*self.parameterform_s2(theta_2, phi_2, R)]

        p3_eucl = [*self.parameterform_s2(theta_3, phi_3, R)]
        p4_eucl = [*self.parameterform_s2(theta_4, phi_4, R)]

        start_points = [p1_eucl, p3_eucl]
        end_points = [p2_eucl, p4_eucl]

        geodesic_1 = self.calculate_geodesic(phi_1, phi_2, theta_1, theta_2, 1)
        geodesic_2 = self.calculate_geodesic(phi_3, phi_4, theta_3, theta_4, 1)

        self.create_3d_scatter_plot(eucl_s2, start_points, end_points, [geodesic_1, geodesic_2])


if __name__ == '__main__':
    geo = s2_geodesic()
    geo.example_function()

