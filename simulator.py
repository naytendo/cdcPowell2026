import numpy as np
from trajectory import Trajectory
from core.signal import Signal

class Simulator:
    def __init__(self, model):
        self.model = model

    def run(self, x0, *, controller=None, ref=None, N=None):
        
        
        U = []

        x = x0.copy().reshape(-1)
        X = [x.copy()]

        for k in range(N):
            if controller is None:
                # open-loop rollout
                u = np.zeros(self.model.nu)
            else:
                # controller may or may not use ref
                u = controller.control(x, k)

            x = self.model.step(x, u, k).reshape(-1)

            U.append(u)
            X.append(x.copy())

        # wrap in Signals + Trajectory
        t = self.model.t
        X_sig = Signal(t=t, Y=np.array(X))
        U_sig = Signal(t=t[:-1], Y=np.array(U))

        return Trajectory(X=X_sig, U=U_sig)
    
    def rollout(self, x0, U):
        """
        Open-loop rollout with a fixed control sequence U (array or Signal).
        """
        if isinstance(U, Signal):
            U_arr = U.Y
        else:
            U_arr = U

        N = U_arr.shape[0]

        X = [x0.copy()]
        x = x0.copy()

        for k in range(N):
            u = U_arr[k]
            x = self.model.step(x, u)
            X.append(x.copy())

        t = self.model.t
        X_sig = Signal(t=t, Y=np.array(X))
        U_sig = Signal(t=t[:-1], Y=U_arr)

        return Trajectory(X=X_sig, U=U_sig)
    
    def simulate_tvlqr(self, x0, controller):
        return self.run(x0, controller=controller, N=controller.N)


