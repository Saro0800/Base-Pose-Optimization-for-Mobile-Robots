from scipy.spatial.transform import Rotation as R
import numpy as np
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker

from reach_space_modeling.generate_pointcloud.gen_cloud_GUI import GenereatePointCloud
from reach_space_modeling.opt_problem.eqn_solv_opt import solve_eqn_prob, create_marker_msg
from base_optimization.problem_formulation_align import BasePoseOptProblem
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.termination.robust import RobustTermination
from pymoo.termination.ftol import SingleObjectiveSpaceTermination

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_point(ax, point, color='red', label='Point'):
    ax.scatter(point[0], point[1], point[2], color=color, label=label)


def plot_ellipsoid(ax, center, axes_lengths, color='blue', alpha=0.5):
    # Create a grid for the ellipsoid
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Parametric equations for the ellipsoid
    x = axes_lengths[0] * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = axes_lengths[1] * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = axes_lengths[2] * np.outer(np.ones_like(u), np.cos(v)) + center[2]

    # Plot the surface
    ax.plot_surface(x, y, z, color=color, alpha=alpha)


def plot_axes(ax, center, axes_lengths, euler_angles=None, colors=['red', 'green', 'blue']):
    # Compute rotation matrix from Euler angles
    if euler_angles is not None:
        rotation = R.from_euler('xyz', euler_angles).as_matrix()
    else:
        rotation = np.eye(3)

    # Plot the axes of the ellipsoid at the center
    origin = np.array(center)
    directions = rotation @ np.diag(axes_lengths)

    for i, color in enumerate(colors):
        start = origin
        end = origin + directions[:, i]
        ax.plot([start[0], end[0]], [start[1], end[1]], [
                start[2], end[2]], color=color, linewidth=2)


# create the GUI to generate the pointcloud
gen_cloud_GUI = GenereatePointCloud()
gen_cloud_GUI.create_GUI()
# gen_cloud_GUI.create_ros_node()
point_cloud = gen_cloud_GUI.points

# solve the optimization problem to get the parameters of the ellipsoid
opt_ell_param = solve_eqn_prob(
    point_cloud, "PSO", gen_cloud_GUI.arm_frt_j_name).X
a = opt_ell_param[0]
b = opt_ell_param[1]
c = opt_ell_param[2]
xC = opt_ell_param[3]
yC = opt_ell_param[4]
zC = opt_ell_param[5]

ell_center = np.array([xC, yC, zC])
ell_axis = np.array([a, b, c])

# define the desired pose of the end-effector with respect to
# the reference frame of the last joint of the arm
des_pos = np.array([1, -2, 0.1, np.deg2rad(0),
                   np.deg2rad(0), np.deg2rad(30)])

# define the optimization problem to find the optimal base pose
problem = BasePoseOptProblem(ell_center, ell_axis, des_pos, point_cloud, False)

# solve the optimization problem using the PSO algorithm
algorithm = PSO()
termination = RobustTermination(
    SingleObjectiveSpaceTermination(tol=pow(10, -6))
)

# solve the optimization problem
res = minimize(problem=problem,
               algorithm=algorithm,
               termination=termination,
               verbose=False,
               seed=1)

print("Optimal base pose: ", res.X)
res_center = [res.X[0], res.X[1], ell_center[2]]

# add a 3d visualization of the old and new ellipsoid position, and the desired pos
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the ellipsoids
plot_ellipsoid(ax, ell_center, ell_axis, color='blue', alpha=0.3)
plot_ellipsoid(ax, res_center, ell_axis, color='green', alpha=0.3)

# Plot the point
plot_point(ax, des_pos, color='red', label='Point')
plot_point(ax, ell_center, color='black', label='center_init')
plot_point(ax, res_center, color='black', label='center_final')

# Plot the axes of the ellipsoids
plot_axes(ax, center=ell_center, axes_lengths=np.transpose(
    [0.2, 0.2, 0.2]))
plot_axes(ax, center=res_center, axes_lengths=np.transpose(
    [0.2, 0.2, 0.2]), euler_angles=[0, 0, res.X[2]])
plot_axes(ax, center=des_pos[:3], axes_lengths=np.transpose(
    [0.2, 0.2, 0.2]), euler_angles=des_pos[3:])

# Set plot limits
# ax.set_xlim(-2, 2)
# ax.set_ylim(-2, 2)
# ax.set_zlim(-2, 2)

# Set labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_aspect('equal', adjustable='box')
ax.legend()

# Show the plot
plt.show()
