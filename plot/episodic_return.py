import pandas as pd
import matplotlib.pyplot as plt

csv_files = [
    'pp_1s_1m.csv',
    'pp_1s_2m.csv',
    'mpc_1s_1m.csv',
    'mpc_1s_2m.csv'
]

labels = [
    'Pure Pursuit - 1 million',
    'Pure Pursuit - 2 million',
    'MPC - 1 million',
    'MPC - 2 million'
]

plt.figure(figsize=(10, 6))

for file, label in zip(csv_files, labels):
    df = pd.read_csv(file)
    plt.plot(df['Step'], df['Value'], label=label)

plt.xlabel('Timesteps')
plt.ylabel('Episodic Returns')
plt.title('Episodic Returns of BC Using Different Experts with Different Timesteps')
plt.legend()
plt.grid(True)
plt.savefig('combined_plot.png')
plt.show()
