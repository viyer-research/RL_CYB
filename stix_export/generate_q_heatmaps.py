import json
import matplotlib.pyplot as plt
import numpy as np

with open('q_metadata.json') as f:
    qm = json.load(f)

# Use 'bline' data (identical to meander in this run)
data = qm['bline']
patch_vals = data['per_host_q']['Patch']
restore_vals = data['per_host_q']['Restore']

# Use canonical ALL_HOSTS ordering from train_dqn_3x to match per_host_q indexing
import sys
sys.path.insert(0, r'D:\Vasanth\RL_CYB')
from train_dqn_3x import ALL_HOSTS as hosts
# hosts is the canonical order used when computing per_host_q (Defender, Enterprise0, ..., Research2)

# Plot side-by-side heatmaps (1 x 20 horizontal)
fig, axes = plt.subplots(2, 1, figsize=(14, 4))

im1 = axes[0].bar(range(len(hosts)), patch_vals, color='#1565c0')
axes[0].set_title('Patch Q-values per host (initial state)')
axes[0].set_xticks(range(len(hosts)))
axes[0].set_xticklabels(hosts, rotation=90, fontsize=8)
axes[0].set_ylabel('Q-value')

im2 = axes[1].bar(range(len(hosts)), restore_vals, color='#43a047')
axes[1].set_title('Restore Q-values per host (initial state)')
axes[1].set_xticks(range(len(hosts)))
axes[1].set_xticklabels(hosts, rotation=90, fontsize=8)
axes[1].set_ylabel('Q-value')

plt.tight_layout()
plt.savefig('q_values_patch_restore.png', dpi=200)
print('Saved q_values_patch_restore.png')
