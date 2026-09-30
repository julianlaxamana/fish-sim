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

mujoco.mj_forward(m, d)
R = d.xmat[body_id].reshape(3, 3)
com_body = R.T @ (d.subtree_com[body_id] - d.xipos[body_id])   # whole-fish CoM, in fish body frame
cob_offset = com_body + np.array([0, 0, 0.015])                  # CoB 3 cm above it


def apply_buoyancy():
    F = np.array([0, 0, rho * g * volume])
    R = d.xmat[body_id].reshape(3, 3)
    r = R @ cob_offset                    # offset in world frame
    torque = np.cross(r, F)
    d.xfrc_applied[body_id, :3] = F
    d.xfrc_applied[body_id, 3:] = torque


# actuator ids, looked up by name so their order in the XML doesn't matter
tail_ids = [m.actuator(n).id for n in ("tail_servo_1", "tail_servo_3", "tail_servo_5")]
fin_ids = [m.actuator(n).id for n in ("fin_servo_1", "fin_servo_2")]

# gait
freq = 2.5        # tail beats per second
amp = 0.4         # radians, bigger toward the tail
phase_lag = 0.8   # radians between neighboring joints


def gait(t):
    n = len(tail_ids)
    for i, aid in enumerate(tail_ids):
        a = amp * (i + 1) / n             # amplitude grows toward the tail
        d.ctrl[aid] = a * np.sin(2 * np.pi * freq * t - i * phase_lag)


# fin pitch control (W / S)
FIN_STEP = 0.05                           # radians per key press
fin_lo, fin_hi = m.actuator_ctrlrange[fin_ids[0]]
fin_angle = 0.0


def key_callback(keycode):
    global fin_angle
    print(keycode)
    if keycode == 265:
        fin_angle += FIN_STEP
    elif keycode == 264:
        fin_angle -= FIN_STEP
    else:
        return
    fin_angle = float(np.clip(fin_angle, fin_lo, fin_hi))
    print(f"fin angle: {fin_angle:+.2f} rad")


def apply_fins():
    for aid in fin_ids:                   # same command on both fins = pitch
        d.ctrl[aid] = fin_angle


# simulation
with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
    while viewer.is_running():
        step_start = time.time()

        apply_buoyancy()
        gait(d.time)
        apply_fins()
        mujoco.mj_step(m, d)

        viewer.sync()

        # rudimentary time keeping, will drift relative to wall clock
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
