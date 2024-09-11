import time
import mujoco
import mujoco.viewer
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from tqdm import tqdm

model = mujoco.MjModel.from_xml_path('mjmodel.xml')
data = mujoco.MjData(model)

def generate_joint_angle_ranges(model, num_samples=6):
    joint_ranges = []
    for i in range(model.njnt):
        if model.jnt_limited[i]:
            joint_ranges.append(np.linspace(model.jnt_range[i][0], model.jnt_range[i][1], num_samples))
        else:
            joint_ranges.append(np.linspace(-np.pi, np.pi, num_samples))
    return joint_ranges

joint_angle_ranges = generate_joint_angle_ranges(model)

max_torques = np.full(model.nu, -np.inf)
max_torque_configs = [None] * model.nu
torque_data = []

total_combinations = np.prod([len(r) for r in joint_angle_ranges])

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()

    for angles in tqdm(product(*joint_angle_ranges), total=total_combinations, desc="Calculating torques"):
        data.qpos[:] = angles
        mujoco.mj_inverse(model, data)

        for i in range(model.nu):
            torque_data.append([i, angles[i], data.qfrc_inverse[i]])
            if data.qfrc_inverse[i] > max_torques[i]:
                max_torques[i] = data.qfrc_inverse[i]
                max_torque_configs[i] = np.copy(angles)

        viewer.sync()


with open('torques.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Joint', 'Angle', 'Torque'])
    writer.writerows(torque_data)

data_torques = pd.read_csv('torques.csv')

plt.figure(figsize=(10, 6))
sns.violinplot(x='Joint', y='Torque', data=data_torques)
plt.title('Torque distribution across joints')
plt.xlabel('Joint name')
plt.ylabel('Torque')
plt.savefig('torque_distribution.png')


print("Visualizing critical points...")
time.sleep(1)
data_visualize = mujoco.MjData(model)
with mujoco.viewer.launch_passive(model, data_visualize) as viewer:
    for config in max_torque_configs:
        data_visualize.qpos[:] = config
        mujoco.mj_inverse(model, data_visualize)
        viewer.sync()
        time.sleep(1)
