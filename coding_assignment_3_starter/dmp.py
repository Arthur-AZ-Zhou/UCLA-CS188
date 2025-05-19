import numpy as np
import scipy.interpolate

class CanonicalSystem:
    """
    Skeleton of the discrete canonical dynamical system.
    """
    def __init__(self, dt: float, ax: float = 1.0):
        """
        Args:
            dt (float): Timestep duration.
            ax (float): Gain on the canonical decay.
        """
        # Initialize time parameters
        self.dt: float = dt
        self.ax: float = ax
        self.run_time: float = 1.0 
        self.reset()

        self.timesteps: int = int(self.run_time / dt)  
        self.x: float = 1.0

    def reset(self) -> None:
        """
        Reset the phase variable to its initial value.
        """
        self.x = 1.0

    def __step_once(self, x, tau, ec):
        return x + (-self.ax * x * ec) * tau * self.dt

    def step(self, tau: float = 1.0, error_coupling: float = 1.0) -> float:
        """
        Advance the phase by one timestep.

        Returns:
            float: Updated phase value.
        """
        self.x = self.__step_once(self.x, tau, error_coupling)
        return self.x

    def rollout(self, tau: float = 1.0, ec: float = 1.0) -> np.ndarray:
        """
        Generate the entire phase sequence.

        Returns:
            np.ndarray: Array of phase values over time.
        """
        xs = np.zeros(self.timesteps)
        self.reset()
        for i in range(self.timesteps):
          xs[i] = self.step(tau, ec)

        return xs

class DMP:
    """
    Skeleton of the discrete Dynamic Motor Primitive.
    """
    def __init__(
        self,
        n_dmps: int,
        n_bfs: int,
        dt: float = 0.01,
        y0: float = 0.0,
        goal: float = 1.0,
        ay: float = 25.0,
        by: float = None
    ):
        """
        Args:
            n_dmps (int): Number of dimensions.
            n_bfs (int): Number of basis functions per dimension.
            dt (float): Timestep duration.
            y0 (float|array): Initial state.
            goal (float|array): Goal state.
            ay (float|array): Attractor gain.
            by (float|array): Damping gain.
        """
        # TODO: initialize parameters
        self.n_dmps: int = n_dmps
        self.n_bfs: int = n_bfs
        self.dt: float = dt
        
        if (np.isscalar(y0)):
            self.y0 = np.ones(n_dmps) * y0
        else:
            self.y0 = np.array(y0)

        if (np.isscalar(goal)):
            self.goal = np.ones(n_dmps) * goal
        else:
            self.goal = np.array(goal)
        
        if (isinstance(ay, int) or isinstance(ay, float)):
            self.ay = np.ones(n_dmps) * ay
        else:
            self.ay = np.array(ay)

        if by is None:
            self.by = np.ones(n_dmps) * (ay / 4.0)
        else:
            self.by = np.ones(n_dmps) * by

        self.w = np.zeros((n_dmps, n_bfs))  # weights
        self.cs = CanonicalSystem(dt)
        self.reset_state()

    def reset_state(self) -> None:
        """
        Reset trajectories and canonical system state.
        """
        self.y = self.y0.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
            
        self.cs.reset()

    def imitate(self, y_des: np.ndarray) -> np.ndarray:
        """
        Learn DMP weights from a demonstration.

        Args:
            y_des (np.ndarray): Desired trajectory, shape (D, T).

        Returns:
            np.ndarray: Interpolated demonstration (D x T').
        """
        # Handle 1D input
        if y_des.ndim == 1:
            y_des = y_des[np.newaxis, :]  # shape (1, T)

        T = y_des.shape[1]
        
        # Update canonical system timing
        self.cs.run_time = T * self.dt
        self.cs.timesteps = T
        self.timesteps = T
        self._generate_centers_widths()

        # Generate canonical phase trajectory
        x_track = self.cs.rollout()
        self.reset_state()

        # Compute derivatives
        dy_des = np.gradient(y_des, axis=1) / self.dt
        ddy_des = np.gradient(dy_des, axis=1) / self.dt

        # Compute basis functions
        psi = self._basis_function(x_track)  # shape (T, n_bfs)

        # Solve for weights
        for d in range(self.n_dmps):
            f_target = (
                ddy_des[d]
                - self.ay[d] * (self.by[d] * (self.goal[d] - y_des[d]) - dy_des[d])
            )
            for b in range(self.n_bfs):
                numer = np.sum(psi[:, b] * x_track * f_target)
                denom = np.sum(psi[:, b] * x_track**2)
                self.w[d, b] = numer / (denom + 1e-6)

        return y_des

    def rollout(
        self,
        tau: float = 1.0,
        error: float = 0.0,
        new_goal: np.ndarray = None
    ) -> np.ndarray:
        """
        Generate a new trajectory from the DMP.

        Args:
            tau (float): Temporal scaling.
            error (float): Feedback coupling.
            new_goal (np.ndarray, optional): Override goal.

        Returns:
            np.ndarray: Generated trajectory (T x D).
        """
        # TODO: implement dynamical update loop
        if new_goal is not None:
            self.goal = np.array(new_goal)

        self.reset_state()
        x_track = self.cs.rollout(tau, error)  # canonical system phase over time
        traj = np.zeros((self.timesteps, self.n_dmps))  # to store generated trajectory

        for t in range(self.timesteps):
            x = x_track[t]
            psi = self._basis_function(np.array([x]))[0]  # shape (n_bfs,)

            for d in range(self.n_dmps):
                # Forcing function
                f = np.dot(psi, self.w[d]) * x / (np.sum(psi) + 1e-6)

                #DMP acceleration
                self.ddy[d] = self.ay[d] * (self.by[d] * (self.goal[d] - self.y[d]) - self.dy[d]) + f
                self.dy[d] += self.ddy[d] * self.dt
                self.y[d] += self.dy[d] * self.dt

            traj[t] = self.y.copy()

        return traj

# ==============================
# DMP Unit test
# ==============================
if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Test canonical system
    cs = CanonicalSystem(dt=0.05)
    x_track = cs.rollout()
    plt.figure()
    plt.plot(x_track, label='Canonical x')
    plt.title('Canonical System Rollout')
    plt.xlabel('Timestep')
    plt.ylabel('x')
    plt.legend()

    # Test DMP behavior with a sine-wave trajectory
    dt = 0.01
    T = 1.0
    t = np.arange(0, T, dt)
    y_des = np.sin(2 * np.pi * 2 * t)

    dmp = DMP(n_dmps=1, n_bfs=50, dt=dt)
    y_interp = dmp.imitate(y_des)
    y_run = dmp.rollout()

    plt.figure()
    plt.plot(t, y_des, 'k--', label='Original')
    plt.plot(np.linspace(0, T, y_interp.shape[1]), y_interp.flatten(), 'b-.', label='Interpolated')
    plt.plot(np.linspace(0, T, y_run.shape[0]), y_run.flatten(), 'r-', label='DMP Rollout')
    plt.title('DMP Imitation and Rollout')
    plt.xlabel('Time (s)')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.show()