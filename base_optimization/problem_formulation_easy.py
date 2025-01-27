#!/usr/bin/env python3
import numpy as np
import math
from pymoo.core.problem import ElementwiseProblem

class BasePoseOptProblem(ElementwiseProblem):
    def __init__(self, *args):
        # retrieve the passed arguments
        self.ell_center = args[0]
        self.ell_axis = args[1]
        self.des_pos = args[2]
        self.point_cloud = args[3]
        self.viz_res = args[4]

        # define the parameters of the optimization problem
        super().__init__(n_var = 2,
                         n_obj = 1,
                         n_ieq_constr = 1,
                         xl = np.array([-1000, -1000]),
                         xu = np.array([1000, 1000]))

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

        # constraints definition
        constrs = [((x[0]-xp)/a)**2 + ((x[1]-yp)/b)**2 + ((xc-zp)/c)**2 - 1]
        out["G"] = np.row_stack(constrs)
    
        # copmute the distance between the center and the desired position
        out["F"] = ((x[0]-xp)**2 + (x[1]-yp)**2 + (zc-zp)**2)**2
        # out["F"] = 1