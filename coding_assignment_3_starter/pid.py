'''

DO NOT MODIFY THIS FILE

'''
import numpy as np

class PID:
    def __init__(self, kp, ki, kd, target):
        """
        Initialize a variable-dimension PID controller.

        Args:
            kp (float or list): Proportional gain(s) per axis (or scalar).
            ki (float or list): Integral gain(s) per axis (or scalar).
            kd (float or list): Derivative gain(s) per axis (or scalar).
            target (tuple or array): Target position of any dimension.
        """
        self.kp = np.array(kp, dtype=float)
        self.ki = np.array(ki, dtype=float)
        self.kd = np.array(kd, dtype=float)
        self.target = np.array(target, dtype=float)

        self.integral = np.zeros_like(self.target)
        self.prev_error = np.zeros_like(self.target)
        self.last_error_magnitude = 0.0

    def reset(self, target=None):
        """
        Reset the internal state of the PID controller.

        Args:
            target (optional): New target to reset to.
        """
        self.integral = np.zeros_like(self.target)
        self.prev_error = np.zeros_like(self.target)
        self.last_error_magnitude = 0.0
        if target is not None:
            self.target = np.array(target, dtype=float)

    def get_error(self):
        """
        Returns:
            float: Magnitude of the last error vector.
        """
        return self.last_error_magnitude

    def update(self, current_pos, dt=0.02):
        """
        Compute the PID control signal.

        Args:
            current_pos (array-like): Current position of the system.
            dt (float): Time step since last update.

        Returns:
            np.ndarray: PID control signal.
        """
        current_pos = np.array(current_pos, dtype=float)
        error = self.target - current_pos
        self.last_error_magnitude = np.linalg.norm(error)

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else np.zeros_like(error)
        self.prev_error = error

        p_term = self.kp * error
        i_term = self.ki * self.integral
        d_term = self.kd * derivative

        return p_term + i_term + d_term
