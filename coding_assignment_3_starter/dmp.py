import numpy as np
import scipy.interpolate

class CanonicalSystem:
    def __init__(self, dt: float, ax: float = 1.0):
        self.dt: float = dt
        self.ax: float = ax
        self.run_time: float = 1.0 
        self.reset()

        self.timesteps: int = int(self.run_time / dt)  
        self.x: float = 1.0

    def reset(self) -> None:
        self.x = 1.0

    def __step_once(self, x, tau, ec):
        return (x + (-self.ax * x * ec) * tau * self.dt)

    def step(self, tau: float = 1.0, error_coupling: float = 1.0) -> float:
        self.x = self.__step_once(self.x, tau, error_coupling)
        return self.x

    def rollout(self, tau: float = 1.0, ec: float = 1.0) -> np.ndarray:
        xs = np.zeros(self.timesteps)
        self.reset()
        for i in range(self.timesteps):
          xs[i] = self.step(tau, ec)

        return xs

class DMP:
    def __init__(self, n_dmps: int, n_bfs: int, dt: float = 0.01, y0: float = 0.0, goal: float = 1.0, ay: float = 25.0, by: float = None):
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
        self.y = self.y0.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)

        self.cs.reset()

    def _initialize_timing_and_phase(self, T):
        self.cs.run_time = T * self.dt
        self.cs.timesteps = T
        self.timesteps = T
        self.x_track = self.cs.rollout()

    def _compute_basis_functions(self):
        self.centers = np.exp(-self.cs.ax * np.linspace(0, 1, self.n_bfs))
        self.widths = 1.0 / (np.gradient(self.centers) ** 2)
        psi_track = np.zeros((self.timesteps, self.n_bfs))

        for t in range(self.timesteps):
            for b in range(self.n_bfs):
                psi_track[t, b] = np.exp(-self.widths[b] * (self.x_track[t] - self.centers[b])**2)
                
        self.psi_track = psi_track

    def _interpolate_trajectory(self, y_des):
        path = np.zeros((self.n_dmps, self.timesteps))
        x = np.linspace(0, self.cs.run_time, y_des.shape[1])

        for d in range(self.n_dmps):
            interp_fn = scipy.interpolate.interp1d(x, y_des[d])
            for t in range(self.timesteps):
                path[d, t] = interp_fn(t * self.dt)
                
        return path

    def _compute_derivatives(self, y_des_interp):
        dy_des = np.gradient(y_des_interp, axis=1) / self.dt
        ddy_des = np.gradient(dy_des, axis=1) / self.dt
        return dy_des, ddy_des

    def _compute_forcing_targets(self, y_des, dy_des, ddy_des):
        f_target = np.zeros((self.timesteps, self.n_dmps))

        for d in range(self.n_dmps):
            f_target[:, d] = ddy_des[d] - self.ay[d] * (self.by[d] * (self.goal[d] - y_des[d]) - dy_des[d])

        return f_target

    def _fit_weights(self, f_target):
        for d in range(self.n_dmps):
            k = self.goal[d] - self.y0[d]

            for b in range(self.n_bfs):
                numerator = np.sum(self.x_track * self.psi_track[:, b] * f_target[:, d])
                denominator = np.sum((self.x_track**2) * self.psi_track[:, b])
                self.w[d, b] = numerator / (denominator + 1e-6)

                if (1e-5 < abs(k)):
                    self.w[d, b] /= k

    def _compute_forcing_term(self, psi, x, goal):
        psi_sum = np.sum(psi) + 1e-10
        f = np.zeros(self.n_dmps)
        
        for d in range(self.n_dmps):
            f[d] = (np.dot(psi, self.w[d]) / psi_sum) * x * (goal[d] - self.y0[d])

        return f

    def _update_dynamics(self, f, goal, tau):
        self.ddy = (self.ay * (self.by * (goal - self.y) - self.dy * tau) + f * tau) * tau
        self.dy += self.ddy * self.dt / tau
        self.y += self.dy * self.dt / tau

    def imitate(self, y_des):
        if y_des.ndim == 1:
            y_des = y_des[np.newaxis, :]
        self.y_des = y_des
        self.y0 = y_des[:, 0].copy()
        self.goal = y_des[:, -1].copy()

        self._initialize_timing_and_phase(y_des.shape[1])
        self._compute_basis_functions()
        y_des_interp = self._interpolate_trajectory(y_des)
        dy_des, ddy_des = self._compute_derivatives(y_des_interp)
        f_target = self._compute_forcing_targets(y_des_interp, dy_des, ddy_des)
        self._fit_weights(f_target)

        return y_des_interp

    def rollout(self, tau: float = 1.0, error: float = 0.0, new_goal: np.ndarray = None) -> np.ndarray:
        if (new_goal is not None):
            self.goal = np.array(new_goal)

        self.reset_state()
        goal = self.goal

        timesteps = self.y_des.shape[1]
        x_track = self.x_track
        psi_track = self.psi_track

        y_track = np.zeros((timesteps, self.n_dmps))

        for t in range(timesteps):
            f = self._compute_forcing_term(psi_track[t], x_track[t], goal)
            self._update_dynamics(f, goal, tau)
            y_track[t] = self.y.copy()

        return y_track

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