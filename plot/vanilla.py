import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data7 = pd.read_csv('7.csv')
data8 = pd.read_csv('8.csv')
data9 = pd.read_csv('9.csv')
data10 = pd.read_csv('10.csv')
data11 = pd.read_csv('11.csv')

# Extract data for plotting
steps7 = data7['Step']*50
mean_values7 = data7['n_bits: 7 - eval/success_count']
min_values7 = data7['n_bits: 7 - eval/success_count__MIN']
max_values7 = data7['n_bits: 7 - eval/success_count__MAX']

steps8 = data8['Step']*50
mean_values8 = data8['n_bits: 8 - eval/success_count']
min_values8 = data8['n_bits: 8 - eval/success_count__MIN']
max_values8 = data8['n_bits: 8 - eval/success_count__MAX']

steps9 = data9['Step']*50
mean_values9 = data9['n_bits: 9 - eval/success_count']
min_values9 = data9['n_bits: 9 - eval/success_count__MIN']
max_values9 = data9['n_bits: 9 - eval/success_count__MAX']

steps10 = data10['Step']*50
mean_values10 = data10['n_bits: 10 - eval/success_count']
min_values10 = data10['n_bits: 10 - eval/success_count__MIN']
max_values10 = data10['n_bits: 10 - eval/success_count__MAX']

steps11 = data10['Step']*50
mean_values11 = data11['n_bits: 11 - eval/success_count']
min_values11 = data11['n_bits: 11 - eval/success_count__MIN']
max_values11 = data11['n_bits: 11 - eval/success_count__MAX']

# Plot the mean success count
plt.plot(steps7, mean_values7, label='n_bits = 7', color='#46788e')
plt.plot(steps8, mean_values8, label='n_bits = 8', color='#78b7c9')
plt.plot(steps9, mean_values9, label='n_bits = 9', color='#f6e093')
plt.plot(steps10, mean_values10, label='n_bits = 10', color='#e58b7b')
plt.plot(steps11, mean_values11, label='n_bits = 11', color='#97b319')

# Plot the min and max as a filled area
plt.fill_between(steps7, min_values7, max_values7, color='#46788e', alpha=0.3)
plt.fill_between(steps8, min_values8, max_values8, color='#78b7c9', alpha=0.3)
plt.fill_between(steps9, min_values9, max_values9, color='#f6e093', alpha=0.3)
plt.fill_between(steps10, min_values10, max_values10, color='#e58b7b', alpha=0.3)
plt.fill_between(steps11, min_values11, max_values11, color='#97b319', alpha=0.3)

# Add labels and title
plt.xlabel('Steps')
plt.ylabel('Success Ratio')
plt.title('Vanilla DQN performance in varying n')
# plt.legend(loc='lower right')
plt.legend()
plt.tight_layout()

plt.savefig('vanilla.pdf', format='pdf')
# Show the plot
plt.show()
