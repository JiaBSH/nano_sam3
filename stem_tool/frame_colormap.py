import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mark = [0, 14, 43, 121, 175, 242, 262, 297, 382, 409, 500, 647, 763, 767, 923, 1085, 1204, 1231]
data_len = 1232

# Read the filenames in the frames directory and extract numeric indices
frames_dir = r'E:\nanojc\nanoframes'
list_name = os.listdir(frames_dir)
existing_indices = set()
for name in list_name:
    try:
        idx = int(name[-8:-4])
    except Exception:
        # skip files that don't match expected pattern
        continue
    existing_indices.add(idx)

# Compute removed (anomalous) frames as those indices in full range not present in existing list
all_indices = set(range(data_len))
removed_indices = sorted(list(all_indices - existing_indices))

# Build color map for all frames: black for removed, red for marked, yellow for others
colors = []
mark_set = set(mark)
for i in range(data_len):
    if i in removed_indices:
        colors.append('k')
    elif i in mark_set:
        colors.append('r')
    else:
        colors.append('y')

# Plot a thin bar for each frame (height 1) to show categories along the x-axis
fig, ax = plt.subplots(figsize=(12, 3))
ax.bar(range(data_len), [1] * data_len, color=colors, width=1.0)
ax.set_xlim(-1, data_len)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xlabel('Frame index')
ax.set_title('Frame categories: black=removed, red=labelled, yellow=unlabelled')

# Save the figure to the workspace
out_path = os.path.join(os.path.dirname(__file__), 'frame_plot.png')
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f'Saved plot to: {out_path}')