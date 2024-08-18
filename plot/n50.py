import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data7 = pd.read_csv('50her_future.csv')
data8 = pd.read_csv('50her_future_rs.csv')

# Extract data for plotting
steps7 = data7['Step']*50
mean_values7 = data7['n_bits: 50 - eval/success_count']

steps8 = data8['Step']*50
mean_values8 = data8['n_bits: 50 - eval/success_count']

# Plot the mean success count
plt.plot(steps7, mean_values7, label='future', color='#46788e')
plt.plot(steps8, mean_values8, label='future + reward shaping', color='#78b7c9')

# Add labels and title
plt.xlabel('Steps')
plt.ylabel('Success Ratio')
plt.title('n_bits = 50')
# plt.legend(loc='lower right')
plt.legend()
plt.tight_layout()

plt.savefig('n50.pdf', format='pdf')
# Show the plot
plt.show()
