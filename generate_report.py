import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'ggplot')
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Inter', 'Arial'], 'figure.titlesize': 14, 'figure.titleweight': 'bold'})

def _save_plot(ax, fname, labels):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set(**labels)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()

def create_prediction_graph(value):
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Predicted (kWh)"], [value / 1000], color="#38bdf8", width=0.4)
    for b in bars: ax.annotate(f'{b.get_height():.2f}', (b.get_x() + 0.2, b.get_height()), xytext=(0,3), textcoords="offset points", ha='center')
    _save_plot(ax, "prediction_graph.png", {'ylabel': "Energy (kWh)", 'title': "Predicted Energy Consumption"})

def create_training_graph(actual, predicted):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([x/1000 for x in actual], label="Actual (kWh)", color="#4ade80", marker='o')
    ax.plot([x/1000 for x in predicted], label="Predicted (kWh)", color="#f87171", marker='s', ls='--')
    ax.legend()
    _save_plot(ax, "training_graph.png", {'xlabel': "Samples", 'ylabel': "Energy (kWh)", 'title': "Actual vs Predicted (kWh)"})