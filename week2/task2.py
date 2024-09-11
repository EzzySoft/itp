import mujoco
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import mujoco.viewer

def generate_all_configurations(nq, num_steps):
    configurations = []
    for step in range(num_steps ** nq):
        indices = np.unravel_index(step, (num_steps,) * nq)
        configuration = [np.linspace(model.jnt_range[j][0], model.jnt_range[j][1], num_steps)[indices[j]] for j in range(nq)]
        configurations.append(configuration)
    return configurations

model = mujoco.MjModel.from_xml_path('task2.xml')
data = mujoco.MjData(model)

num_steps = 15

configurations = generate_all_configurations(model.nq, num_steps)

angles = []
torques = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    for config in configurations:
        data.qpos[:] = config
        
        mujoco.mj_inverse(model, data)
        
        angles.append(data.qpos.copy())
        torques.append(data.qfrc_inverse.copy())
        viewer.sync()
        
        time.sleep(.001)

angles_df = pd.DataFrame(angles, columns=[f'Joint_{i+1}' for i in range(model.nq)])
torques_df = pd.DataFrame(torques, columns=[f'Joint_{i+1}_Torque' for i in range(model.nq)])

df = pd.concat([angles_df, torques_df], axis=1)
df.to_csv('robot_dynamics.csv', index=False)

print('Results have been saved to robot_dynamics.csv')

plt.figure(figsize=(12, 6))

angles_df = angles_df.map(lambda x: np.arctan2(np.sin(x), np.cos(x)))
torques_df = torques_df.map(lambda x: np.log(np.abs(x) + 1e-6))

torque_data = pd.melt(torques_df.reset_index(), id_vars='index', var_name='Joint', value_name='Torque')
sns.violinplot(x='Joint', y='Torque', data=torque_data)
plt.xticks(rotation=45)
plt.xlabel('Joint Name')
plt.ylabel('Torque')
plt.title('Distribution of Torques in Joints')

plt.savefig('torque_distribution.png')
plt.close()
