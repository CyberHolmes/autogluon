import os
import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame(columns=['time', 'util', 'scheduler', 'task', 'machine'])

for task in os.listdir('Resource_utilization'):
    task_dir = os.path.join('Resource_utilization', task)
    for scheduler in os.listdir(task_dir):
        scheduler_dir = os.path.join(task_dir, scheduler)
        for machine in os.listdir(scheduler_dir):
            machine_dir = os.path.join(scheduler_dir, machine)
            machine_resources = pd.read_csv(machine_dir, names=['time', 'util']) \
                .sort_values(by=['time'], axis=0) \
                .reset_index(drop=True)
            machine_resources['machine'] = machine.split('.')[0]
            machine_resources['scheduler'] = scheduler
            machine_resources['task'] = task
            data = pd.concat([data, machine_resources])

nrows = len(data['task'].unique())
ncols = len(data['machine'].unique())
count = 1

fig = plt.figure(figsize=(19, 6))

for row, task in enumerate(data['task'].unique()):
    for col, machine in enumerate(data['machine'].unique()):
        df = data[data['task'] == task]
        df = df[df['machine'] == machine]
        df = df.drop(columns=['machine', 'task'])
        axes = fig.add_subplot(nrows, ncols, count)
        for scheduler in data['scheduler'].unique():
            plt_data = df[df['scheduler'] == scheduler]
            axes.plot(plt_data['util'], label=scheduler)
            axes.legend(loc="upper right")
            axes.set_xlabel("Time")
            axes.set_ylabel("CPU Utilization")
            axes.grid(True)
        count += 1

plt.show()