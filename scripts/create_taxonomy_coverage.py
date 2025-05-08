import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Data structure for healthcare datasets by category and subcategory
data = [
    {
        "category": "Administration and Workflow",
        "subcategories": [
            {"subcategory": "Care Coordination and Planning", "count": 1},
            {"subcategory": "Organizing Workflow Processes", "count": 1},
            {"subcategory": "Overseeing Financial Activities", "count": 1},
            {"subcategory": "Providing Clinical Knowledge Support", "count": 1},
            {"subcategory": "Scheduling Resources and Staff", "count": 1}
        ]
    },
    {
        "category": "Clinical Decision Support",
        "subcategories": [
            {"subcategory": "Care Coordination and Planning", "count": 1},
            {"subcategory": "Planning Treatments", "count": 2},
            {"subcategory": "Predicting Patient Risks and Outcomes", "count": 1},
            {"subcategory": "Providing Clinical Knowledge and Support", "count": 2},
            {"subcategory": "Providing Clinical Knowledge Support", "count": 2},
            {"subcategory": "Supporting Diagnostic Decisions", "count": 2}
        ]
    },
    {
        "category": "Clinical Note Generation",
        "subcategories": [
            {"subcategory": "Documenting care plans", "count": 1},
            {"subcategory": "Documenting Diagnositc Reports", "count": 1},
            {"subcategory": "Documenting Patient Visits", "count": 3},
            {"subcategory": "Recording Procedures", "count": 1}
        ]
    },
    {
        "category": "Medical Research Assistance",
        "subcategories": [
            {"subcategory": "Analyzing Clinical Research Data", "count": 1},
            {"subcategory": "Conducting Literature Research", "count": 2},
            {"subcategory": "Ensuring Clinical Research Quality", "count": 1},
            {"subcategory": "Managing Research Enrollment", "count": 1},
            {"subcategory": "Recording Research Processes", "count": 1}
        ]
    },
    {
        "category": "Patient Communication and Education",
        "subcategories": [
            {"subcategory": "Delivering Personalized Care Instructions", "count": 1},
            {"subcategory": "Enhancing Patient Understanding and Accessibility in Health Communication", "count": 1},
            {"subcategory": "Facilitating Patient Engagement and Support", "count": 1},
            {"subcategory": "Patient Provider Messaging", "count": 1},
            {"subcategory": "Patient-Provider Messaging", "count": 3},
            {"subcategory": "Providing Patient Education Resources", "count": 1}
        ]
    }
]

# Flatten the data for plotting
categories = []
subcategories = []
counts = []
colors = []

# Define a color palette for the categories (in the same order as category_order)
color_palette = {
    'Clinical Decision Support': '#ff7f0e',       # Orange
    'Clinical Note Generation': '#2ca02c',        # Green
    'Patient Communication and Education': '#9467bd',  # Purple
    'Medical Research Assistance': '#d62728',     # Red
    'Administration and Workflow': '#1f77b4'      # Blue
}

for category_data in data:
    category = category_data['category']
    color = color_palette[category]  # Get color from dictionary using category name
    
    for item in category_data['subcategories']:
        categories.append(category)
        subcategories.append(item['subcategory'])
        counts.append(item['count'])
        colors.append(color)

# Create a DataFrame
df = pd.DataFrame({
    'Category': categories,
    'Subcategory': subcategories,
    'Count': counts,
    'Color': colors
})

# Sort by Category to ensure they're grouped together
# Define the custom category order
category_order = ['Clinical Decision Support', 'Clinical Note Generation', 
                 'Patient Communication and Education', 'Medical Research Assistance', 
                 'Administration and Workflow']

# Create a categorical type with our custom ordering
df['Category'] = pd.Categorical(df['Category'], categories=category_order, ordered=True)

# Sort first by our custom category order, then by count within each category
df = df.sort_values(['Category', 'Count'], ascending=[True, False])

# Create figure with appropriate size based on number of subcategories
fig, ax = plt.subplots(figsize=(14, 10))

# Create the horizontal bar chart
bars = ax.barh(np.arange(len(df)), df['Count'], color=df['Color'])

# Set y-tick labels to be the subcategories
ax.set_yticks(np.arange(len(df)))
ax.set_yticklabels(df['Subcategory'])

# Add grid lines for better readability
ax.grid(axis='x', linestyle='--', alpha=0.7)

# Add counts as labels on the bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
            str(int(width)), ha='left', va='center')

# Add a legend for categories
legend_elements = []
# Use the category_order to ensure legend entries are in the right order
for category in category_order:
    legend_elements.append(Patch(facecolor=color_palette[category], 
                                 label=category))
ax.legend(handles=legend_elements, loc='upper right')

ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.set_xlabel('Count')
ax.set_title('Count of Datasets per Subcategory')

plt.tight_layout()
plt.savefig('healthcare_dataset_counts.png', dpi=300, bbox_inches='tight')
plt.show()
