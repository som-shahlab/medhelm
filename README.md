# MEDHELM

## 🚀 Getting Started

1. Create your environment

```bash
conda env create -f env.yaml
```

2. Activate the environment

```bash
conda activate medhelm
```

3. Install medhelm

```bash
pip install -e .
```

## 📊 Generate Plots

1. Heatmap

```bash
bash scripts/plots/1_create_heatmaps.sh
```

2. Cost vs Win Rate

```bash
bash scripts/plots/2_create_cost_plot.sh
```

3. LLM-Judge Stats

```bash
bash scripts/plots/3_create_stats_plot.sh
```

## 📋 Generate Tables

1. Win rate table

```bash
bash scripts/tables/1_win_rate_table.sh
```
