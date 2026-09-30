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
cob_offset = com_body + np.array([0, 0, 0.015])                  # CoB 1.5 cm above it


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
freq = 3.5        # tail beats per second
amp = 0.7         # radians, bigger toward the tail
phase_lag = 1.2   # radians between neighboring joints


def gait(t):
    n = len(tail_ids)
    for i, aid in enumerate(tail_ids):
        a = amp * (i + 1) / n             # amplitude grows toward the tail
        d.ctrl[aid] = a * np.sin(2 * np.pi * freq * t - i * phase_lag)


# keys
KEY_UP, KEY_DOWN = 265, 264               # target pitch
KEY_P, KEY_O = 80, 79                                # more thrust

# pitch hold: fins steer the nose toward target_pitch
PITCH_STEP = 0.1                          # radians per key press
PITCH_MAX = 0.8
KP_PITCH, KD_PITCH = 3.0, 1.5
fin_lo, fin_hi = m.actuator_ctrlrange[fin_ids[0]]
target_pitch = 0.0
pitch_rate_dof = m.jnt_dofadr[m.body_jntadr[body_id]] + 4    # free joint angular vel about body y

# forward thrust control
FREQ_STEP = 0.1                         # newtons per key press
FREQ_MAX = 5.0                         # newtons per key press


def key_callback(keycode):
    global target_pitch, freq
    if keycode == KEY_UP:
        target_pitch = min(target_pitch + PITCH_STEP, PITCH_MAX)
        print(f"target pitch: {target_pitch:+.2f} rad")
    elif keycode == KEY_DOWN:
        target_pitch = max(target_pitch - PITCH_STEP, -PITCH_MAX)
        print(f"target pitch: {target_pitch:+.2f} rad")
    elif keycode == KEY_P:
        freq = min(freq + FREQ_STEP, FREQ_MAX)
        print(f"thrust: {freq:.1f} ")
    elif keycode == KEY_O:
        freq = max(freq - FREQ_STEP, 0)
        print(f"thrust: {freq:.1f} ")


def nose_pitch():
    R = d.xmat[body_id].reshape(3, 3)
    return np.arcsin(np.clip(-R[2, 0], -1, 1))   # head points along -x; positive = nose up


def apply_fins():
    err = target_pitch - nose_pitch()
    fin = KP_PITCH * err - KD_PITCH * d.qvel[pitch_rate_dof]
    fin = float(np.clip(fin, fin_lo, fin_hi))
    for aid in fin_ids:                   # same command on both fins = pitch
        d.ctrl[aid] = fin



# simulation
with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
    while viewer.is_running():
        step_start = time.time()

        apply_buoyancy()                  # must come first: it overwrites xfrc_applied
        gait(d.time)
        apply_fins()
        mujoco.mj_step(m, d)

        viewer.sync()

        # rudimentary time keeping, will drift relative to wall clock
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
