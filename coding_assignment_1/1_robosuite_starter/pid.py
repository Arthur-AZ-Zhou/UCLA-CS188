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
        # self.kp = np.array(kp) if (isinstance(kp, (list, np.ndarray))) else (np.ones(len(target)) * kp)
        # self.ki = np.array(ki) if (isinstance(ki, (list, np.ndarray))) else (np.ones(len(target)) * ki)
        # self.kd = np.array(kd) if (isinstance(kd, (list, np.ndarray))) else (np.ones(len(target)) * kd)
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.target = np.array(target)
        self.prev_error = np.zeros_like(self.target)
        self.error_integral = np.zeros_like(self.target)
        self.magnitude_previous_error = 0
        
    def reset(self, target=None):
        """
        Reset the internal state of the PID controller.

        Args:
            target (optional): New target to reset to.
        """
        self.target = np.array(target)
        self.prev_error = np.zeros_like(self.target)
        self.error_integral = np.zeros_like(self.target)
        self.magnitude_previous_error = 0
        
    def get_error(self):
        """
        Returns:
            float: Magnitude of the last error vector.
        """
        return self.magnitude_previous_error

    def update(self, current_pos, dt):
        """
        Compute the PID control signal.

        Args:
            current_pos (array-like): Current position (any dimension).
            dt (float): Time delta in seconds.

        Returns:
            np.ndarray: Control output vector.
        """
        current_pos = np.array(current_pos)
        error = self.target - current_pos
        self.magnitude_previous_error = np.linalg.norm(error)
        
        # Update integral term
        self.error_integral += error * dt
        
        # Calculate derivative term
        derivative = (error - self.prev_error) / dt if dt > 0 else np.zeros_like(error)
        self.prev_error = error
        
        # Compute PID output
        output = self.kp * error + self.ki * self.error_integral + self.kd * derivative
        
        return output