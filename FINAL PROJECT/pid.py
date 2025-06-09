import numpy as np

class PID:
    def __init__(self, kp, ki, kd, target):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.prev_error = np.zeros_like(self.target)
        self.error_integral = np.zeros_like(self.target)
        self.magnitude_previous_error = 0
        
    def reset(self, target = None):
        self.target = target
        self.prev_error = np.zeros_like(self.target)
        self.error_integral = np.zeros_like(self.target)
        self.magnitude_previous_error = 0
        
    def get_error(self):
        return self.magnitude_previous_error

    def update(self, current_pos, dt):
        current_pos = np.array(current_pos)
        target = np.array(self.target)
        error = target - current_pos

        self.prev_error = error
        self.error_integral += error * dt
        self.magnitude_previous_error = np.linalg.norm(error) #sqrt(element1^2 + element2^2 + element3^3 + ...)
        
        return self.kp * error + self.ki * self.error_integral + self.kd * ((error - self.prev_error) / dt)