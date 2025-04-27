import pandas as pd
import matplotlib.pyplot as plt
import os

from medhelm.utils.constants import MODEL_NAME_MAPPING

# Make sure plots directory exists
os.makedirs('./plots', exist_ok=True)

# Load the data from the CSV file
csv_file = './data/costs.csv'
data = pd.read_csv(csv_file)
data["Model"] = data["Model"].map(MODEL_NAME_MAPPING)

# Aggregate the data by model and sum the "Cost Input Output Tokens" column
aggregated_data = data.groupby('Model')['Cost Input Output Tokens'].sum()

# Load the leaderboard data from the CSV file
leaderboard_file = './data/leaderboard.csv'
leaderboard_data = pd.read_csv(leaderboard_file)

# Calculate the mean win rate for each model
mean_win_rate = leaderboard_data.groupby('Model')['Mean win rate'].mean()

# Merge the aggregated cost data with the mean win rate data
merged_data = pd.DataFrame({
    'Aggregated Cost': aggregated_data,
    'Mean Win Rate': mean_win_rate
}).dropna()

# Assign a unique color to each model
colors = plt.cm.tab10(range(len(merged_data)))

# Add a column for colors based on the model
merged_data['Color'] = [colors[i] for i in range(len(merged_data))]

# Plot the data
plt.figure(figsize=(12, 8))

for i, (model, row) in enumerate(merged_data.iterrows()):
    plt.scatter(
        row['Aggregated Cost'], 
        row['Mean Win Rate'], 
        color=row['Color'], 
        edgecolors='black', 
        s=200  # <-- Bigger circles (increase s from default ~20 to 200)
    )
    # Annotate each point with the model name slightly to the right
    plt.text(
        row['Aggregated Cost'] * 1.01,   # small shift to the right
        row['Mean Win Rate'],
        model,
        fontsize=10,
        verticalalignment='center'
    )

plt.title('Cost vs Mean Win Rate', fontsize=16)
plt.xlabel('Cost', fontsize=14)
plt.ylabel('Mean Win Rate', fontsize=14)
plt.grid(True)
plt.tight_layout()

# Set the x and y axis limits to ensure all data appears
plt.xlim(merged_data['Aggregated Cost'].min() * 0.9, merged_data['Aggregated Cost'].max() * 1.1)
plt.ylim(merged_data['Mean Win Rate'].min() * 0.9, merged_data['Mean Win Rate'].max() * 1.1)

# Save the scatter plot as an image
output_scatter_plot = './plots/cost_vs_win_rate.png'
plt.savefig(output_scatter_plot)
plt.show()
