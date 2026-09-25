import time

import mujoco
import mujoco.viewer
import numpy as np


# load model and data
m = mujoco.MjModel.from_xml_path('./Models/fish.xml')
d = mujoco.MjData(m)

# buoyancy stuff
body_id = m.body("fish").id
rho, g = 1000.0, 9.81
volume = m.body_subtreemass[body_id] / rho
cob_offset = np.array([0.2, 0, 0.03])  # center of buoyancy, 5 cm above the CoM, in body frame


def apply_buoyancy():
    F = np.array([0, 0, rho * g * volume])
    R = d.xmat[body_id].reshape(3, 3)
    r = R @ cob_offset                    # offset in world frame
    torque = np.cross(r, F)
    d.xfrc_applied[body_id, :3] = F
    d.xfrc_applied[body_id, 3:] = torque

# gait
freq = 1.5        # tail beats per second
amp = 0.4         # radians, bigger toward the tail
phase_lag = 0.8   # radians between neighboring joints

def gait(t, n_joints):
    for i in range(n_joints):
        a = amp * (i + 1) / n_joints          # amplitude grows toward the tail
        d.ctrl[i] = a * np.sin(2 * np.pi * freq * t - i * phase_lag)

# simulation
with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 30:
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    apply_buoyancy()
    gait(step_start - start, 5)
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
