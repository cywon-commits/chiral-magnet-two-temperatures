import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, json, numpy as np
# categorical slots (fixed order) from the dataviz reference palette; markers give secondary encoding
DCOL = {0.0: '#2a78d6', 0.2: '#eb6834', 0.3: '#1baf7a', 0.4: '#eda100', 0.5: '#e87ba4'}
DMRK = {0.0: 'o', 0.2: 's', 0.3: '^', 0.4: 'D', 0.5: 'v'}
INK, MUTED, GRID = '#222222', '#6b6b66', '#e4e3df'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'lines.linewidth': 1.6, 'lines.markersize': 4.5, 'legend.frameon': False, 'savefig.dpi': 160, 'savefig.bbox': 'tight'})
def summary(): return json.load(open('data/task1_summary.json'))
def get(S, proto, D, L, dirn):
    for e in S:
        if e['proto'] == proto and abs(e['D'] - D) < 1e-9 and e['L'] == L and e['dirn'] == dirn: return e
