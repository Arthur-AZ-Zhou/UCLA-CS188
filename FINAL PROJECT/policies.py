import numpy as np
import robosuite as suite
import time
from pid import PID

class LiftPolicyRed(object):
    def __init__(self, obs):
        self.kp_precise = [2.0, 2.0, 2.0] #K_ for axes: (x, y, z)
        self.ki_precise = [0.05, 0.05, 0.05]
        self.kd_precise = [0.2, 0.2, 0.2]
        self.kp_lift = [1.0, 1.0, 0.5] #low = slower and smoother lifts, (x, y, z) again
        self.ki_lift = [0.01, 0.01, 0.01]
        self.kd_lift = [0.1, 0.1, 0.1]
        
        cube_pos = obs['cubeA_pos']
        
        self.target_height = cube_pos.copy() #grabs cube 10cm above ground
        self.target_height[2] += 0.1
        self.target_grasp = cube_pos.copy() #grab cube at exact coords sitting down
        self.target_lift = cube_pos.copy() #how high we lift the cube
        self.target_lift[2] += 0.5  
        
        self.pid_controller = PID(self.kp_precise, self.ki_precise, self.kd_precise, self.target_height) #init new PID controller

        self.phase = 0  #0: arm moves above, 1: lower arm to grab, 2: grab, 3: lift
        self.gripper_open = True
        self.grasp_start_time = 0
        self.grasp_duration = 0.5  # 0.5 seconds to ensure grasp
        
    def get_action(self, obs):
        eef_pos = obs['robot0_eef_pos'] #end-effector position
        current_cube_pos = obs['cubeA_pos']
        error = self.pid_controller.get_error() #can think of this as distance to target
        
        if (self.phase == 0 and error < 0.02):  #reached above
            self.phase = 1
            self.pid_controller.reset(self.target_grasp)

        elif (self.phase == 1 and error < 0.01):  #reached grab
            self.phase = 2
            self.gripper_open = False
            self.grasp_start_time = time.time()  

        elif (self.phase == 2 and self.grasp_duration < (time.time() - self.grasp_start_time)):
            self.phase = 3
            self.pid_controller = PID(self.kp_lift, self.ki_lift, self.kd_lift, self.target_lift)
        
        control = self.pid_controller.update(eef_pos, 0.01)
        
        action = np.zeros(7)
        action[:3] = control[:3]
        
        if (2 <= self.phase):
            action[-1] = 1 #closed gripper
        else:
            action[-1] = -1 #opened gripper
        
        return action
    
class LiftPolicyGreen(object):
    def __init__(self, obs):
        self.kp_precise = [2.0, 2.0, 2.0] #K_ for axes: (x, y, z)
        self.ki_precise = [0.05, 0.05, 0.05]
        self.kd_precise = [0.2, 0.2, 0.2]
        self.kp_lift = [1.0, 1.0, 0.5] #low = slower and smoother lifts, (x, y, z) again
        self.ki_lift = [0.01, 0.01, 0.01]
        self.kd_lift = [0.1, 0.1, 0.1]
        
        cube_pos = obs['cubeB_pos']
        
        self.target_height = cube_pos.copy() #grabs cube 10cm above ground
        self.target_height[2] += 0.1
        self.target_grasp = cube_pos.copy() #grab cube at exact coords sitting down
        self.target_lift = cube_pos.copy() #how high we lift the cube
        self.target_lift[2] += 0.5  
        
        self.pid_controller = PID(self.kp_precise, self.ki_precise, self.kd_precise, self.target_height) #init new PID controller

        self.phase = 0  #0: arm moves above, 1: lower arm to grab, 2: grab, 3: lift
        self.gripper_open = True
        self.grasp_start_time = 0
        self.grasp_duration = 0.5  # 0.5 seconds to ensure grasp
        
    def get_action(self, obs):
        eef_pos = obs['robot0_eef_pos'] #end-effector position
        current_cube_pos = obs['cubeB_pos']
        error = self.pid_controller.get_error() #can think of this as distance to target
        
        if (self.phase == 0 and error < 0.02):  #reached above
            self.phase = 1
            self.pid_controller.reset(self.target_grasp)

        elif (self.phase == 1 and error < 0.01):  #reached grab
            self.phase = 2
            self.gripper_open = False
            self.grasp_start_time = time.time()  

        elif (self.phase == 2 and self.grasp_duration < (time.time() - self.grasp_start_time)):
            self.phase = 3
            self.pid_controller = PID(self.kp_lift, self.ki_lift, self.kd_lift, self.target_lift)
        
        control = self.pid_controller.update(eef_pos, 0.01)
        
        action = np.zeros(7)
        action[:3] = control[:3]
        
        if (2 <= self.phase):
            action[-1] = 1 #closed gripper
        else:
            action[-1] = -1 #opened gripper
        
        return action
    
class StackPolicy(object):
    def __init__(self, obs):
        self.kp_precise = [1.2, 1.2, 1.5]
        self.ki_precise = [0.01, 0.01, 0.02]
        self.kd_precise = [0.05, 0.05, 0.08]
        self.kp_lift = [1.5, 1.5, 1.2]
        self.ki_lift = [0.02, 0.02, 0.01]
        self.kd_lift = [0.05, 0.05, 0.05]
        
        self.cube_pos_1 = obs['cubeA_pos'] #red
        self.cube_pos_2 = obs['cubeB_pos'] #green
        
        self.target_height_1 = self.cube_pos_1.copy()
        self.target_height_1[2] += 0.1  #grabs red cube 10cm off ground
        self.target_grasp_1 = self.cube_pos_1.copy()
        self.target_lift = self.cube_pos_1.copy()
        self.target_lift[2] += 0.1  #how high we lift
        self.target_height_2 = self.cube_pos_2.copy()
        self.target_height_2[2] += 0.1  #height we stack at
        
        self.pid_controller = PID(self.kp_precise, self.ki_precise, self.kd_precise, self.target_height_1)
        self.phase = 0  #0: move to red, 1: grab red, 2: lift red, 3: approach green, 4: release
        self.gripper_open = True
        self.grasp_start_time = 0
        self.grasp_duration = 0.5
        self.release_start_time = 0
        self.release_duration = 1
        
    def get_action(self, obs):
        eef_pos = obs['robot0_eef_pos']
        self.cube_pos_1 = obs['cubeA_pos']
        self.cube_pos_2 = obs['cubeB_pos']
        error = self.pid_controller.get_error()
        
        if (self.phase == 0 and error < 0.02):
            self.phase = 1
            self.pid_controller.reset(self.target_grasp_1)

        elif (self.phase == 1 and error < 0.01):
            self.phase = 2
            self.gripper_open = False
            self.grasp_start_time = time.time()

        elif (self.phase == 2 and self.grasp_duration < (time.time() - self.grasp_start_time)):
            self.phase = 3
            self.pid_controller = PID(self.kp_lift, self.ki_lift, self.kd_lift, self.target_lift)

        elif (self.phase == 3 and error < 0.03):
            self.phase = 4
            self.pid_controller = PID(self.kp_precise, self.ki_precise, self.kd_precise, self.target_height_2)

        elif (self.phase == 4 and error < 0.02):
            self.phase = 5
            self.pid_controller.reset(self.cube_pos_2 + np.array([0, 0, 0.075])) #set it down more gently
            self.release_start_time = time.time()

        elif (self.phase == 5):
            if (time.time() - self.release_start_time) > self.release_duration:
                self.phase = 6 #set to 6 to release gripper
                self.gripper_open = True
                self.settle_start_time = time.time()
        
        control = self.pid_controller.update(eef_pos, 0.01)
        
        action = np.zeros(7)
        action[:3] = control[:3]
        
        if (2 <= self.phase < 5): #keep gripper closed while moving
            action[-1] = 1
        else:
            action[-1] = -1 if self.gripper_open else 1
        
        return action