#!/usr/bin/env python3
import numpy as np
import math
from pymoo.core.problem import ElementwiseProblem
from scipy.spatial.transform import Rotation


class BasePoseOptProblem(ElementwiseProblem):
    def __init__(self, *args):
        # retrieve the passed arguments
        self.ell_center = args[0]
        self.ell_axis = args[1]
        self.des_pos = args[2]
        self.point_cloud = args[3]
        self.viz_res = args[4]

        # let us define:
        #   - R0 the fixed reference frame, initially coincident with the ref. frame
        #     of the last joint of the arm
        #   - R_ell the reference frame defined with respect to the center of the ellipsoid
        #   - R_des the reference frame defined with respect to the desired EE pose
        
        # Compute the homogeneous transformation matrix from R0 to R_des (constant, since the
        # desired pose is fixed with respect to R0)
        mat_R0_Rdes = np.zeros((4, 4))
        mat_R0_Rdes[:3, :3] = Rotation.from_euler('xyz', self.des_pos[3:]).as_matrix()
        mat_R0_Rdes[:3, 3] = self.des_pos[:3]
        mat_R0_Rdes[3, 3] = 1
        
        # compute the coordinate of the unitary vector of the x-axis of R_des with respect to R0
        self.x_versor_R0 = np.dot(mat_R0_Rdes, np.array([1, 0, 0, 1]))            
        
        # define the parameters of the optimization problem
        super().__init__(n_var=3,
                         n_obj=1,
                         n_ieq_constr=2,
                         xl=np.array([-1000, -1000, 0]),
                         xu=np.array([1000, 1000, np.pi*2]))

    def _evaluate(self, x, out, *args, **kwargs):
        # retrieve the ellipsoid equation equation paramters
        a = self.ell_axis[0]
        b = self.ell_axis[1]
        c = self.ell_axis[2]
        xc = self.ell_center[0]
        yc = self.ell_center[1]
        zc = self.ell_center[2]

        # retrieve the desired position of the end-effector
        xp = self.des_pos[0]
        yp = self.des_pos[1]
        zp = self.des_pos[2]
        x_ang = self.des_pos[3]
        y_ang = self.des_pos[4]
        z_ang = self.des_pos[5]
        
        # compute the homogeneous transfomration matrix from R_ell to R0
        mat_Rell_R0 = np.zeros((4, 4))
        mat_Rell_R0[:3, :3] = np.transpose(Rotation.from_euler('xyz', [0, 0, x[2]]).as_matrix())
        mat_Rell_R0[:3, 3] = -np.dot(np.transpose(Rotation.from_euler('xyz', [0, 0, x[2]]).as_matrix()), np.array([x[0], x[1], zc]))
        mat_Rell_R0[3, 3] = 1
        
        # compute the coordinate of the origin of R_des with respect to R_ell
        des_pos_Rell = np.dot(mat_Rell_R0, np.append(self.des_pos[:3], 1))
        # compute the cosine of the angle between the x-axis of R_ell and the projection on
        # the xy plane of R_ell of the vector with the position of the origin of R_des 
        # with respect to R_ell.
        cos_theta1 = np.dot([1,0,0], des_pos_Rell[:3])/(np.linalg.norm([1, 0, 0], 2) * np.linalg.norm(des_pos_Rell[:3], 2))
        
        # compute the coordinate of the versor of the x-axis of R_des with respect to R_ell
        x_versor_Rell = np.dot(mat_Rell_R0, self.x_versor_R0)
        # compute the cosine of the angle between the x-axis of R_ell and the projection on
        # the xy plane of R_ell of the vector with the position of the versor of the x-axis
        # of R_des with respect to R_ell.
        cos_theta2 = np.dot([1,0,0], x_versor_Rell[:3])/(np.linalg.norm([1, 0, 0], 2) * np.linalg.norm(x_versor_Rell[:3], 2))

        # compute the distance between the origin of R_des and R_ell
        dist = np.linalg.norm(des_pos_Rell[:3], 2)

        # constraints definition
        constrs = [((x[0]-xp)/a)**2 + ((x[1]-yp)/b)**2 + ((zc-zp)/c)**2 - 1, dist - 0.2]
        out["G"] = np.row_stack(constrs)

        # define the objective function
        out["F"] = (1 - cos_theta1) + (1 - cos_theta2)
