import matplotlib.pyplot as plt
import pandas as pd
import os

# DATA  (winrate number of the five categories on leaderboard) 
data = {
    "Clinical Decision Support": [
        {"Model": "o3-mini",          "Score": 0.800},
        {"Model": "DeepSeek R1",      "Score": 0.663},
        {"Model": "GPT-4o",           "Score": 0.631},
        {"Model": "Claude 3.7",       "Score": 0.600},
        {"Model": "Claude 3.5",       "Score": 0.544},
        {"Model": "Gemini 2.0",       "Score": 0.425},
        {"Model": "GPT-4o mini",      "Score": 0.363},
        {"Model": "Llama 3.3",        "Score": 0.313},
        {"Model": "Gemini 1.5",       "Score": 0.163},
    ],

    "Clinical Note Generation": [
        {"Model": "Claude 3.7",       "Score": 0.813},
        {"Model": "DeepSeek R1",      "Score": 0.708},
        {"Model": "Claude 3.5",       "Score": 0.688},
        {"Model": "o3-mini",          "Score": 0.667},
        {"Model": "GPT-4o",           "Score": 0.500},
        {"Model": "Gemini 2.0",       "Score": 0.375},
        {"Model": "GPT-4o mini",      "Score": 0.375},
        {"Model": "Gemini 1.5",       "Score": 0.229},
        {"Model": "Llama 3.3",        "Score": 0.146},
    ],

    "Communication & Education": [
        {"Model": "DeepSeek R1",      "Score": 0.797},
        {"Model": "Claude 3.7",       "Score": 0.719},
        {"Model": "Claude 3.5",       "Score": 0.594},
        {"Model": "GPT-4o",           "Score": 0.531},
        {"Model": "Gemini 2.0",       "Score": 0.500},
        {"Model": "o3-mini",          "Score": 0.438},
        {"Model": "Llama 3.3",        "Score": 0.406},
        {"Model": "GPT-4o mini",      "Score": 0.344},
        {"Model": "Gemini 1.5",       "Score": 0.172},
    ],

    "Medical Research Assistance": [
        {"Model": "o3-mini",          "Score": 0.781},
        {"Model": "Claude 3.5",       "Score": 0.719},
        {"Model": "GPT-4o",           "Score": 0.542},
        {"Model": "DeepSeek R1",      "Score": 0.531},
        {"Model": "Gemini 2.0",       "Score": 0.479},
        {"Model": "Gemini 1.5",       "Score": 0.438},
        {"Model": "Claude 3.7",       "Score": 0.354},
        {"Model": "Llama 3.3",        "Score": 0.354},
        {"Model": "GPT-4o mini",      "Score": 0.302},
    ],

    "Administration and Workflow": [
        {"Model": "Claude 3.5",       "Score": 0.725},
        {"Model": "Claude 3.7",       "Score": 0.650},
        {"Model": "GPT-4o",           "Score": 0.650},
        {"Model": "GPT-4o mini",      "Score": 0.613},
        {"Model": "DeepSeek R1",      "Score": 0.525},
        {"Model": "o3-mini",          "Score": 0.500},
        {"Model": "Gemini 2.0",       "Score": 0.313},
        {"Model": "Gemini 1.5",       "Score": 0.275},
        {"Model": "Llama 3.3",        "Score": 0.250},
    ],
}

# model-colour palette 
model_colors = {
    "o3-mini"      : "#6EE7B7",
    "GPT-4o mini"  : "#2DD4BF",
    "GPT-4o"       : "#059669",
    "Claude 3.7"   : "#A78BFA",
    "Claude 3.5"   : "#8B5CF6",
    "DeepSeek R1"  : "#2563EB", 
    "Gemini 2.0"   : "#F87171",
    "Gemini 1.5"   : "#DC2626",
    "Llama 3.3"    : "#FBBF24",
}


task_order = [
    "Clinical Decision Support",
    "Clinical Note Generation",
    "Communication & Education",
    "Medical Research Assistance",
    "Administration and Workflow",
]

flat = []
for t_idx, task in enumerate(task_order):      
    rows = data[task]
    for rank, entry in enumerate(sorted(rows,
                                        key=lambda d: d["Score"],
                                        reverse=True), start=1):
        flat.append(
            {"Task": task,
             "TaskIdx": t_idx,              
             "Model": entry["Model"],
             "Score": entry["Score"],
             "Rank": rank}
        )
df = pd.DataFrame(flat)

fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')

max_rank = df["Rank"].max()
ax.set_xlim(0.5, max_rank + 0.5)
ax.set_xticks(range(1, max_rank + 1))
ax.set_xticklabels([f"Rank {i}" for i in range(1, max_rank + 1)],
                   fontsize=11, fontweight='bold')

ax.set_ylim(-0.5, len(task_order) - 0.5)  
ax.invert_yaxis()                         

# zebra stripes
for y in range(len(task_order)):                      
    ax.axhspan(y - 0.5, y + 0.5,
               color='whitesmoke' if y % 2 else 'white',
               zorder=0)

# bubbles
for _, row in df.iterrows():
    clr = model_colors.get(row["Model"], "#9CA3AF")   
    ax.scatter(row["Rank"], row["TaskIdx"],
               s=row["Score"] * 2500,
               color=clr,
               edgecolors='white',
               linewidths=1.4,
               zorder=4)
    ax.text(row["Rank"], row["TaskIdx"],
            f'{row["Model"]}\n{row["Score"]:.3f}',
            ha='center', va='center',
            fontsize=8, fontweight='bold',
            color='black',                
            zorder=5)

ax.set_yticks(range(len(task_order)))
ax.set_yticklabels([])                                
for y, task in enumerate(task_order):                 
    ax.text(-0.06, y, task,
            ha='right', va='center',
            transform=ax.get_yaxis_transform(),
            fontsize=11, fontweight='bold')

handles = [plt.Line2D([0], [0], marker='o', markersize=9,
                      linestyle='',
                      markerfacecolor=model_colors.get(m, "#9CA3AF"),
                      label=m)
           for m in sorted(df["Model"].unique())]
n_cols = min(5, len(handles))         
ax.legend(handles=handles, title='Models',
          bbox_to_anchor=(0.5, -0.08),
          loc='upper center',
          ncol=n_cols,
          columnspacing=0.8,
          handletextpad=0.4,
          frameon=False,
          fontsize=9)

plt.tight_layout(rect=[0.15, 0.05, 1, 0.95])


save_dir = "./plots/"
os.makedirs(save_dir, exist_ok=True)
out_file = os.path.join(save_dir, "model_performance_ranking.png")
plt.savefig(out_file, dpi=300, bbox_inches='tight')
print("figure saved →", out_file)
