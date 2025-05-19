import numpy as np
from collections import defaultdict
from dmp import DMP
from pid import PID

class DMPPolicyWithPID:
    def __init__(self, square_pos, demo_path='demonstration_data.npz', dt=0.01, n_bfs=20):
        self.dt = dt

        raw = np.load(demo_path)
        demos = defaultdict(dict)
        for key in raw.files:
            prefix, trial, field = key.split('_', 2)
            demos[f"{prefix}_{trial}"][field] = raw[key]
        demo = demos['demo_98']

        self.ee_pos = demo['obs_robot0_eef_pos']
        self.ee_grasp = demo['actions'][:, -1:].astype(int)
        self.segments = self.detect_grasp_segments(self.ee_grasp)

        self.demo_obj_pos = demo['obs_object'][0, :3]
        self.new_obj_pos = square_pos
        self.pos_offset = self.new_obj_pos - self.demo_obj_pos
        

        self.setup_dmps(n_bfs)
        self.current_segment = 0
        self.current_step = 0
        self.pid = PID(kp=10.0, ki=0.5, kd=1.0, target=self.get_target_position(0, 0))

    def detect_grasp_segments(self, grasp_flags):
        segments = []
        start_idx = 0
        prev_grasp = grasp_flags[0, 0]
        
        for i in range(1, len(grasp_flags)):
            curr_grasp = grasp_flags[i, 0]
            
            if (curr_grasp != prev_grasp):
                segments.append((start_idx, i))
                start_idx = i
                prev_grasp = curr_grasp
        
        if (start_idx < len(grasp_flags)):
            segments.append((start_idx, len(grasp_flags)))
            
        return segments
        
    def setup_dmps(self, n_bfs):
        self.dmps = []
        self.segment_trajectories = []
        
        for i, (start, end) in enumerate(self.segments):
            segment_pos = self.ee_pos[start:end]
            
            dmp = DMP(n_dmps=3, n_bfs=n_bfs, dt=self.dt)
            dmp.imitate(segment_pos.T)
            
            self.dmps.append(dmp)
                        
    def get_target_position(self, segment_idx, step_idx):
        start, end = self.segments[segment_idx]
        
        if segment_idx == 0:
            if (step_idx + start < end):
                return self.ee_pos[start + step_idx] + self.pos_offset
            else:
                return self.ee_pos[end - 1] + self.pos_offset
        else:
            if (start + step_idx < end):
                return self.ee_pos[start + step_idx]
            else:
                return self.ee_pos[end - 1]

    def get_grasp_state(self, segment_idx):
        _, end = self.segments[segment_idx]

        if (end < len(self.ee_grasp)):
            return self.ee_grasp[end - 1, 0]
        else:
            return 0
            
    def get_action(self, robot_eef_pos):
        start, end = self.segments[self.current_segment]
        segment_length = end - start
        
        if (self.current_step >= segment_length - 1):
            if (self.current_segment < len(self.segments) - 1):
                self.current_segment += 1
                self.current_step = 0
            else:
                self.current_step = segment_length - 1
        
        target_pos = self.get_target_position(self.current_segment, self.current_step)
        self.pid.reset(target=target_pos)
        delta_pos = self.pid.update(robot_eef_pos, dt=self.dt)
        grasp = self.get_grasp_state(self.current_segment)
        self.current_step += 1
        
        action = np.zeros(7)
        action[:3] = delta_pos 
        action[6] = grasp  
        return action