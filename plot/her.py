import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data7 = pd.read_csv('30her_future.csv')
data8 = pd.read_csv('35her_future.csv')
data9 = pd.read_csv('40her_future.csv')
data10 = pd.read_csv('45her_future.csv')

# Extract data for plotting
steps7 = data7['Step']*50
mean_values7 = data7['n_bits: 30 - eval/success_count']
min_values7 = data7['n_bits: 30 - eval/success_count__MIN']
max_values7 = data7['n_bits: 30 - eval/success_count__MAX']

steps8 = data8['Step']*50
mean_values8 = data8['n_bits: 35 - eval/success_count']
min_values8 = data8['n_bits: 35 - eval/success_count__MIN']
max_values8 = data8['n_bits: 35 - eval/success_count__MAX']

steps9 = data9['Step']*50
mean_values9 = data9['n_bits: 40 - eval/success_count']
min_values9 = data9['n_bits: 40 - eval/success_count__MIN']
max_values9 = data9['n_bits: 40 - eval/success_count__MAX']

steps10 = data10['Step']*50
mean_values10 = data10['n_bits: 45 - eval/success_count']
min_values10 = data10['n_bits: 45 - eval/success_count__MIN']
max_values10 = data10['n_bits: 45 - eval/success_count__MAX']


# Plot the mean success count
plt.plot(steps7, mean_values7, label='n_bits = 30', color='#78b7c9')
plt.plot(steps8, mean_values8, label='n_bits = 35', color='#97b319')
plt.plot(steps9, mean_values9, label='n_bits = 40', color='#f6e093')
plt.plot(steps10, mean_values10, label='n_bits = 45', color='#e58b7b')

# Plot the min and max as a filled area
plt.fill_between(steps7, min_values7, max_values7, color='#78b7c9', alpha=0.3)
plt.fill_between(steps8, min_values8, max_values8, color='#97b319', alpha=0.3)
plt.fill_between(steps9, min_values9, max_values9, color='#f6e093', alpha=0.3)
plt.fill_between(steps10, min_values10, max_values10, color='#e58b7b', alpha=0.3)

# Add labels and title
plt.xlabel('Steps')
plt.ylabel('Success Ratio')
plt.title("'Future' strategy's performance in varying n")
# plt.legend(loc='lower right')
plt.legend()
plt.tight_layout()

plt.savefig('future.pdf', format='pdf')
# Show the plot
plt.show()
