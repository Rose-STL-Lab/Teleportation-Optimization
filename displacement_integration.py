# Script for computing the expected distance between w0 and a curve as a function of the curvature at w0.

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geom
from scipy.integrate import quad, nquad

# define integrand = dist((r\cos{\theta}, r\sin{\theta}), \gamma) r
def integrand(t, r, gamma):
    point = geom.Point(r * np.cos(t), r * np.sin(t))
    dist = point.distance(gamma) * r
    return dist

# set up color maps
N = 5
base_cmaps = ['Blues','Oranges']
n_base = len(base_cmaps)
colors = np.concatenate([plt.get_cmap(name)(np.linspace(0.2,0.8,N)) for name in base_cmaps])


# gamma: (x, kx^2)
plt.figure()
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors)
for r in [2e0, 1e0, 5e-1, 2e-1, 1e-1]:
    k = np.arange(0, 40, 0.3)
    int_dist = np.zeros_like(k)
    for i in range(len(int_dist)):
        # construct gamma, which is a parabola here
        x_arr = np.arange(-10, 10, 0.1)
        y_arr = k[i] * np.abs(x_arr**2)

        point_arr = []
        for p_idx in range(len(x_arr)):
            point_arr.append(geom.Point(x_arr[p_idx], y_arr[p_idx]))
        parabola = geom.LineString(point_arr)

        # compute \int_0^{2\pi} dist((r\cos{\theta}, r\sin{\theta}), \gamma) r d\theta
        int_dist[i] = quad(integrand, 0, 2*np.pi, args=(r, parabola))[0]

    # plot int_dist / (2 \pi r) / r. The additional r aligns all curves in one plot.
    plt.plot(2*k, int_dist / (2 * np.pi * r * r), linewidth=3, label='r={}'.format(r))

plt.xlabel(chr(954)+r'$ = 2k_1$', fontsize=26)
plt.ylabel(r'$\frac{1}{r}\mathbb{E}_{S_r} dist(w, \gamma_1)$', fontsize=26)
plt.xticks([0, 20, 40, 60, 80], fontsize=20)
plt.yticks([0.625, 0.675, 0.725, 0.775], fontsize=20)
plt.legend(fontsize=20)
plt.savefig('figures/curvature_displacement_integral_kx_sqr.pdf', dpi=400, bbox_inches='tight')


# gamma: x^2 + (y-k)^2 = k^2
plt.figure()
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors)
for r in [1e-1, 5e-2, 2e-2, 1e-2, 5e-3]:
    k = np.concatenate((np.arange(0.1, 1.0, 0.01), np.arange(1.0, 10.0, 0.1)))
    int_dist = np.zeros_like(k)
    for i in range(len(int_dist)):
        # construct gamma
        theta_arr = np.arange(0, 2*np.pi, 1e-3)
        x_arr = k[i] * np.cos(theta_arr)
        y_arr = k[i] * np.sin(theta_arr) + k[i]

        point_arr = []
        for p_idx in range(len(x_arr)):
            point_arr.append(geom.Point(x_arr[p_idx], y_arr[p_idx]))
        gamma = geom.LineString(point_arr)

        # compute \int_0^{2\pi} dist((r\cos{\theta}, r\sin{\theta}), \gamma) r d\theta
        int_dist[i] = quad(integrand, 0, 2*np.pi, args=(r, gamma), limit=500)[0]
        
    # plot int_dist / (2 \pi r) / r. The additional r aligns all curves in one plot.
    plt.plot(1 / k, int_dist / (2 * np.pi * r * r), linewidth=3, label='r={}'.format(r))

plt.xlabel(r'$\kappa = \frac{1}{k_2}$', fontsize=26)
plt.ylabel(r'$\frac{1}{r}\mathbb{E}_{S_r} dist(w, \gamma_2)$', fontsize=26)
plt.xticks(fontsize= 20)
plt.yticks([0.6, 0.61, 0.62, 0.63], fontsize=20)
plt.legend(fontsize=20)
plt.savefig('figures/curvature_displacement_integral_circle.pdf', dpi=400, bbox_inches='tight')
