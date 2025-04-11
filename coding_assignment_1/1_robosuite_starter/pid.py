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
        pass
        
        
    def reset(self, target=None):
        """
        Reset the internal state of the PID controller.

        Args:
            target (optional): New target to reset to.
        """
        pass
        
    def get_error(self):
        """
        Returns:
            float: Magnitude of the last error vector.
        """
        
        # return error
        pass

    def update(self, current_pos, dt):
        """
        Compute the PID control signal.

        Args:
            current_pos (array-like): Current position (any dimension).
            dt (float): Time delta in seconds.

        Returns:
            np.ndarray: Control output vector.
        """
        # return control
        pass
