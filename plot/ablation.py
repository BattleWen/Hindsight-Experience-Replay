import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data7 = pd.read_csv('35her_final.csv')
data8 = pd.read_csv('35her_future.csv')
data9 = pd.read_csv('35her_episode.csv')
data10 = pd.read_csv('40her_final.csv')
data11 = pd.read_csv('40her_future.csv')
data12 = pd.read_csv('40her_episode.csv')

# Extract data for plotting
steps7 = data7['Step']*50
mean_values7 = data7['n_bits: 35 - eval/success_count']
min_values7 = data7['n_bits: 35 - eval/success_count__MIN']
max_values7 = data7['n_bits: 35 - eval/success_count__MAX']

steps8 = data8['Step']*50
mean_values8 = data8['n_bits: 35 - eval/success_count']
min_values8 = data8['n_bits: 35 - eval/success_count__MIN']
max_values8 = data8['n_bits: 35 - eval/success_count__MAX']

steps9 = data9['Step']*50
mean_values9 = data9['n_bits: 35 - eval/success_count']
min_values9 = data9['n_bits: 35 - eval/success_count__MIN']
max_values9 = data9['n_bits: 35 - eval/success_count__MAX']

steps10 = data10['Step']*50
mean_values10 = data10['n_bits: 40 - eval/success_count']
min_values10 = data10['n_bits: 40 - eval/success_count__MIN']
max_values10 = data10['n_bits: 40 - eval/success_count__MAX']

steps11 = data10['Step']*50
mean_values11 = data11['n_bits: 40 - eval/success_count']
min_values11 = data11['n_bits: 40 - eval/success_count__MIN']
max_values11 = data11['n_bits: 40 - eval/success_count__MAX']

steps12 = data10['Step']*50
mean_values12 = data12['n_bits: 40 - eval/success_count']
min_values12 = data12['n_bits: 40 - eval/success_count__MIN']
max_values12 = data12['n_bits: 40 - eval/success_count__MAX']

print(len(steps7), len(mean_values7))
print(len(steps8), len(mean_values8))
print(len(steps9), len(mean_values9))
print(len(steps10), len(mean_values10))
print(len(steps11), len(mean_values11))
print(len(steps12), len(mean_values12))

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(8, 12))

# Plot for n=35
axs[0].plot(steps7, mean_values7, label='final', color='#e58b7b')
axs[0].fill_between(steps7, min_values7, max_values7, color='#e58b7b', alpha=0.2)
axs[0].plot(steps8, mean_values8, label='future', color='#f6e093')
axs[0].fill_between(steps8, min_values8, max_values8, color='#f6e093', alpha=0.2)
axs[0].plot(steps9, mean_values9, label='episode', color='#97b319')
axs[0].fill_between(steps9, min_values9, max_values9, color='#97b319', alpha=0.2)
axs[0].set_title('n=35')
axs[0].set_xlabel('Steps')
axs[0].set_ylabel('Success Ratio')
axs[0].legend()
axs[0].grid(True)

# Plot for n=40
axs[1].plot(steps10, mean_values10, label='final', color='#e58b7b')
axs[1].fill_between(steps10, min_values10, max_values10, color='#e58b7b', alpha=0.2)
axs[1].plot(steps11, mean_values11, label='future', color='#f6e093')
axs[1].fill_between(steps11, min_values11, max_values11, color='#f6e093', alpha=0.2)
axs[1].plot(steps12, mean_values12, label='episode', color='#97b319')
axs[1].fill_between(steps12, min_values12, max_values12, color='#97b319', alpha=0.2)
axs[1].set_title('n=40')
axs[1].set_xlabel('Steps')
axs[1].set_ylabel('Success Ratio')
axs[1].legend()
axs[1].grid(True)

plt.suptitle('Ablation Study on Different HER Strategies')
plt.tight_layout()
plt.show()
plt.savefig('ablation.pdf', format='pdf')
