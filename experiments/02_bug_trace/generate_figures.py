"""Generate probe figures for 02_bug_trace experiment."""
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

BASE = '/proj/assert-berzelius/users/x_andaf/cwm'
FIG_DIR = os.path.join(BASE, 'experiments/02_bug_trace/figures')
os.makedirs(FIG_DIR, exist_ok=True)

LAYERS = [16, 32, 48, 63]
COLORS = {16: 'blue', 32: 'orange', 48: 'green', 63: 'red'}
LAYER_NAMES = {16: 'Layer 16', 32: 'Layer 32', 48: 'Layer 48', 63: 'Layer 63'}

# ===========================================================================
# Figure 1: mutation_type probe, 16k (50 bins, solid) vs 4k (50 bins, dashed)
# ===========================================================================

d16k = torch.load(os.path.join(BASE, 'interp-bug-trajectories-hard-16k/probe_content_mutation_type.pt'), map_location='cpu')
d4k  = torch.load(os.path.join(BASE, 'interp-bug-trajectories-hard/probe_content_mutation_type.pt'), map_location='cpu')

hm16k = d16k['heatmap']
hm4k  = d4k['heatmap']

fig1, ax1 = plt.subplots(figsize=(12, 6))

bins50 = list(range(50))

for layer in LAYERS:
    color = COLORS[layer]

    # 16k: solid
    y16k = [hm16k[layer][b]['val_acc'] for b in bins50]
    ax1.plot(bins50, y16k, color=color, linestyle='-', linewidth=1.8, label=f'{LAYER_NAMES[layer]} (16k)')

    # 4k: dashed (also 50 bins, same x range)
    y4k = [hm4k[layer][b]['val_acc'] for b in bins50]
    ax1.plot(bins50, y4k, color=color, linestyle='--', linewidth=1.5, label=f'{LAYER_NAMES[layer]} (4k)')

    # perm baseline per layer: dotted line in layer color
    perm_bl = hm16k[layer]['perm_baseline']
    ax1.axhline(perm_bl, color=color, linestyle=':', linewidth=1.0, alpha=0.6)

# Random and majority baselines
random_bl = d16k['random_baseline']   # 0.333...
majority_bl = d16k['majority_baseline']  # 0.577...

ax1.axhline(random_bl,   color='gray', linestyle='--', linewidth=1.2, alpha=0.7, label='Random baseline (33.3%)')
ax1.axhline(majority_bl, color='gray', linestyle='-',  linewidth=1.2, alpha=0.7, label='Majority baseline (57.7%)')

ax1.set_xlim(0, 49)
ax1.set_ylim(0.3, 1.0)
ax1.set_xlabel('Relative time bin (0=start of generation, 49=end)', fontsize=12)
ax1.set_ylabel('Probe accuracy (val)', fontsize=12)
ax1.set_title('mutation_type probe: hard mutations (3-class)\nSolid=16k context, Dashed=4k context', fontsize=13)
ax1.grid(True, alpha=0.3)

# Build legend: layer entries (solid/dashed per layer) + baselines
legend_handles = []
for layer in LAYERS:
    color = COLORS[layer]
    solid_line  = mlines.Line2D([], [], color=color, linestyle='-',  linewidth=1.8, label=f'{LAYER_NAMES[layer]} (16k)')
    dashed_line = mlines.Line2D([], [], color=color, linestyle='--', linewidth=1.5, label=f'{LAYER_NAMES[layer]} (4k)')
    legend_handles.extend([solid_line, dashed_line])

legend_handles.append(mlines.Line2D([], [], color='gray', linestyle='--', linewidth=1.2, label='Random baseline (33.3%)'))
legend_handles.append(mlines.Line2D([], [], color='gray', linestyle='-',  linewidth=1.2, label='Majority baseline (57.7%)'))
legend_handles.append(mlines.Line2D([], [], color='gray', linestyle=':',  linewidth=1.0, label='Perm baseline (per layer, dotted)'))

ax1.legend(handles=legend_handles, loc='upper left', fontsize=9, ncol=2)

out1 = os.path.join(FIG_DIR, 'probe_hard_50bins.png')
fig1.tight_layout()
fig1.savefig(out1, dpi=150)
plt.close(fig1)
print(f'Saved: {out1}')

# ===========================================================================
# Figure 2: will_be_correct probe — hard 16k solid, easy bugonly dashed
# ===========================================================================

d16k_wbc   = torch.load(os.path.join(BASE, 'interp-bug-trajectories-hard-16k/probe_temporal_will_be_correct.pt'), map_location='cpu')
d_bugonly   = torch.load(os.path.join(BASE, 'interp-bug-trajectories-track_a-bugonly/probe_temporal_will_be_correct.pt'), map_location='cpu')

hm16k_wbc  = d16k_wbc['heatmap']
hm_bugonly  = d_bugonly['heatmap']

bugonly_bins = sorted([k for k in hm_bugonly[16].keys() if isinstance(k, int)])  # 0-9
n_bugonly = len(bugonly_bins)

fig2, ax2 = plt.subplots(figsize=(12, 6))

for layer in LAYERS:
    color = COLORS[layer]

    # 16k hard: solid, 50 bins
    y16k_wbc = [hm16k_wbc[layer][b]['val_acc'] for b in range(50)]
    ax2.plot(range(50), y16k_wbc, color=color, linestyle='-', linewidth=1.8, label=f'{LAYER_NAMES[layer]} (hard 16k)')

    # easy bugonly: dashed, 10 bins scaled to 0..49 (multiply by 5)
    x_bo = [b * 5 for b in bugonly_bins]
    y_bo = [hm_bugonly[layer][b]['val_acc'] for b in bugonly_bins]
    ax2.plot(x_bo, y_bo, color=color, linestyle='--', linewidth=1.5, marker='o', markersize=3,
             label=f'{LAYER_NAMES[layer]} (easy bugonly)')

# chance line
ax2.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.8, label='Chance (50%)')

ax2.set_xlim(0, 49)
ax2.set_ylim(0.5, 1.0)
ax2.set_xlabel('Relative time bin', fontsize=12)
ax2.set_ylabel('Probe accuracy (val)', fontsize=12)
ax2.set_title('will_be_correct probe: hard 16k mutations (buggy-only)\nSolid=hard 16k, Dashed=easy bugonly (10 bins scaled)', fontsize=13)
ax2.grid(True, alpha=0.3)

legend_handles2 = []
for layer in LAYERS:
    color = COLORS[layer]
    legend_handles2.append(mlines.Line2D([], [], color=color, linestyle='-',  linewidth=1.8, label=f'{LAYER_NAMES[layer]} (hard 16k)'))
    legend_handles2.append(mlines.Line2D([], [], color=color, linestyle='--', linewidth=1.5, label=f'{LAYER_NAMES[layer]} (easy bugonly)'))

legend_handles2.append(mlines.Line2D([], [], color='gray', linestyle=':', linewidth=1.5, label='Chance (50%)'))
ax2.legend(handles=legend_handles2, loc='upper left', fontsize=9, ncol=2)

out2 = os.path.join(FIG_DIR, 'probe_hard_wbc_50bins.png')
fig2.tight_layout()
fig2.savefig(out2, dpi=150)
plt.close(fig2)
print(f'Saved: {out2}')
