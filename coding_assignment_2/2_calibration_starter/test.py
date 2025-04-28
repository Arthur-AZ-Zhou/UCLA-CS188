import numpy as np
import robosuite as suite
from policies import *          # Custom robot control policies
from camera_utils import *      # Utility functions: detect_red_blob, pixel_to_camera3d, etc.
import cv2

"""
DO NOT MODIFY THIS FILE
"""

# --------------------------------------------------
# Initialize the simulation environment
# --------------------------------------------------

# Create a Robosuite environment with a Panda robot performing the "Lift" task
env = suite.make(
    env_name="Lift",             # Test task
    robots="Panda",              # Use Panda robot
    has_renderer=False,          # No on-screen rendering
    has_offscreen_renderer=True, # Enable offscreen camera capture
    use_camera_obs=True,         # Use camera observations
    camera_names="agentview",    # Select the agentview camera
    camera_depths=True           # Enable depth images
)

# --------------------------------------------------
# Camera Intrinsics Calculation
# --------------------------------------------------

# Get field of view and image size
fovy = env.sim.model.cam_fovy[env.sim.model.camera_name2id("agentview")]
img_height = 256
img_width = 256

# Estimate intrinsics using pinhole camera model
fy = 0.5 * img_height / np.tan(0.5 * fovy * np.pi / 180)
fx = fy  # Assume square pixels
cx = img_width / 2
cy = img_width / 2

# Store camera intrinsics in a dictionary
cam_intrinsics = {
    "fx": fx, 
    "fy": fy,
    "cx": cx,
    "cy": cy
}

print("cam_intrinsics: ", cam_intrinsics)  # Display calculated intrinsics

# --------------------------------------------------
# Load the precomputed camera-to-robot transformation
# --------------------------------------------------


T_CR = run_hand_eye_calibration() # Run calibration
# T_CR = np.load("T_CR.npy") # Load 3x4 transformation matrix
T_CR_homogeneous = np.vstack([T_CR, [0, 0, 0, 1]]) # Convert to 4x4 homogeneous format
offset = get_calibration_offset()  # Offset of calibration

# --------------------------------------------------
# Main Loop: Attempt task multiple times and evaluate
# --------------------------------------------------

num_trials = 10
success_rate = 0

for _ in range(num_trials):
    obs = env.reset()  # Reset the environment
    print(obs.keys())  # Show available observation keys

    # Convert and flip RGB image for display
    obs_rgb = cv2.cvtColor(obs['agentview_image'], cv2.COLOR_BGR2RGB)
    obs_rgb = cv2.flip(obs_rgb, 0)
    obs_rgb = cv2.flip(obs_rgb, 1)

    # Convert and flip depth image
    obs_depth = obs['agentview_depth']
    obs_depth = cv2.flip(obs_depth, 0)
    obs_depth = cv2.flip(obs_depth, 1)

    # Detect the red object in the image
    red_target = detect_red_blob(obs_rgb)
    print(red_target)

    # Visualize detected red blob
    if red_target:
        cv2.circle(obs_rgb, red_target, 5, (0, 255, 0), -1)

    cv2.imshow('obs', obs_rgb)
    cv2.waitKey(1)

    # Skip this iteration if no red blob was found
    if red_target is None:
        print('ERROR, no target detected!')
        continue

    # --------------------------------------------------
    # Convert detected red pixel to robot frame coordinate
    # --------------------------------------------------

    cx, cy = red_target
    target_3d = pixel_to_camera3d(cx, cy, obs_depth, cam_intrinsics)  # In camera frame
    target_3d = list(target_3d) + [1.0]                            # Homogeneous coords
    target_3d = np.array(target_3d)
    target_in_base_homogeneous = T_CR_homogeneous @ target_3d     # Transform to robot base frame

    # --------------------------------------------------
    # Create a policy to move toward the detected target
    # --------------------------------------------------

    goal_position = target_in_base_homogeneous[:3] + offset
    policy = LiftPolicy(goal_position)

    # --------------------------------------------------
    # Execute the policy
    # --------------------------------------------------

    while True:
        action = policy.get_action_proprio(obs['robot0_eef_pos'])
        obs, reward, done, info = env.step(action)

        # Update and display visuals
        obs_rgb = cv2.cvtColor(obs['agentview_image'], cv2.COLOR_BGR2RGB)
        obs_rgb = cv2.flip(obs_rgb, 0)
        obs_rgb = cv2.flip(obs_rgb, 1)
        obs_depth = obs['agentview_depth']
        obs_depth = cv2.flip(obs_depth, 0)
        obs_depth = cv2.flip(obs_depth, 1)

        cv2.imshow('obs', obs_rgb)
        cv2.waitKey(1)

        # Check for task success
        if reward == 1.0:
            success_rate += 1
            print("SUCCESS TRIGGER================================================================================")
            break
        if done:
            break

# --------------------------------------------------
# Final evaluation
# --------------------------------------------------

success_rate /= num_trials  # Normalize over 10 total trials
print('success rate:', success_rate)
