"""
   Sample script for loading and visualizing data from the FIC dataset.
   After a recording is visualized, close all figure windows to continue to the next one.
   Tested with Python 3.6.4
   Author: Kyritsis Konstantinos
   Mail: kokirits@mug.ee.auth.gr
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pickle as pkl


# load the dataset pickle
with open('../Datasets/FIC/FIC.pkl', 'rb') as fh:
    dataset = pkl.load(fh)

# Extract all information
raw_data_vec = dataset['signals_raw']
proc_data_vec = dataset['signals_proc']
subject_id_vec = dataset['subject_id']
session_id_vec = dataset['session_id']
bite_gt_vec = dataset['bite_gt']
mm_gt_vec = dataset['mm_gt']

# Iterate and visualize the processed data and food intake cycle GT from each recording
# for i in range(0, len(raw_data_vec)):

#     this_sub_id = subject_id_vec[i]
#     this_session_id = session_id_vec[i]
#     this_proc = proc_data_vec[i]
#     this_bite_gt = bite_gt_vec[i]

#     t = this_proc[:, 0]
#     acc = this_proc[:, 1:4]
#     gyr = this_proc[:, 4:]

#     # for plotting purposes
#     max_acc = np.max(np.abs(acc))
#     max_gyr = np.max(np.abs(gyr))

#     f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#     f.suptitle('Subject: [' + str(this_sub_id) +
#                '] - Session: [' + str(this_session_id) + ']', fontsize=16)
#     ax1.plot(t, acc[:, 0], label='x')
#     ax1.plot(t, acc[:, 1], label='y')
#     ax1.plot(t, acc[:, 2], label='z')
#     ax1.set_ylabel('Accelerometer (g)')
#     ax1.set_ylim(-max_acc, max_acc)
#     ax1.grid()

#     ax2.plot(t, gyr[:, 0], label='x')
#     ax2.plot(t, gyr[:, 1], label='y')
#     ax2.plot(t, gyr[:, 2], label='z')

#     ax2.set_ylabel('Gyroscope ' + r'$(\frac{deg}{sec})$')
#     ax2.set_xlabel('Time (sec)')
#     ax2.set_ylim(-max_gyr, max_gyr)
#     ax2.grid()

#     # plot bite GT as rectangles
#     for i, gt in enumerate(this_bite_gt):
#         width = gt[1] - gt[0]
#         pos_acc = (gt[0], -max_acc)
#         pos_gyr = (gt[0], -max_gyr)
#         height_acc = max_acc * 2
#         height_gyr = max_gyr * 2
#         if i == 0:
#             ax1.add_patch(patches.Rectangle(pos_acc, width,
#                                             height_acc, edgecolor='black', facecolor='grey', alpha=0.5, label='food intake cycle'))
#             ax2.add_patch(patches.Rectangle(pos_gyr, width,
#                                             height_gyr, edgecolor='black', facecolor='grey', alpha=0.5, label='food intake cycle'))
#         else:
#             ax1.add_patch(patches.Rectangle(pos_acc, width,
#                                             height_acc, edgecolor='black', facecolor='grey', alpha=0.5))
#             ax2.add_patch(patches.Rectangle(pos_gyr, width,
#                                             height_gyr, edgecolor='black', facecolor='grey', alpha=0.5))

#     ax1.legend()
#     ax2.legend()

#     plt.show()
#     plt.close()
#     plt.clf()
i = 0

this_sub_id = subject_id_vec[i]
this_session_id = session_id_vec[i]
this_proc = proc_data_vec[i]
this_bite_gt = bite_gt_vec[i]

t = this_proc[:, 0]
acc = this_proc[:, 1:4]
gyr = this_proc[:, 4:]

# for plotting purposes
max_acc = np.max(np.abs(acc))
max_gyr = np.max(np.abs(gyr))

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Subject: [' + str(this_sub_id) +
            '] - Session: [' + str(this_session_id) + ']', fontsize=16)
ax1.plot(t, acc[:, 0], label='x')
ax1.plot(t, acc[:, 1], label='y')
ax1.plot(t, acc[:, 2], label='z')
ax1.set_ylabel('Accelerometer (g)')
ax1.set_ylim(-max_acc, max_acc)
ax1.grid()

ax2.plot(t, gyr[:, 0], label='x')
ax2.plot(t, gyr[:, 1], label='y')
ax2.plot(t, gyr[:, 2], label='z')

ax2.set_ylabel('Gyroscope ' + r'$(\frac{deg}{sec})$')
ax2.set_xlabel('Time (sec)')
ax2.set_ylim(-max_gyr, max_gyr)
ax2.grid()

# plt.xticks(np.arange(0, 1200, 2))

# plot bite GT as rectangles
for i, gt in enumerate(this_bite_gt):
    width = gt[1] - gt[0]
    pos_acc = (gt[0], -max_acc)
    pos_gyr = (gt[0], -max_gyr)
    height_acc = max_acc * 2
    height_gyr = max_gyr * 2
    if i == 0:
        ax1.add_patch(patches.Rectangle(pos_acc, width,
                                        height_acc, edgecolor='black', facecolor='grey', alpha=0.5, label='food intake cycle'))
        ax2.add_patch(patches.Rectangle(pos_gyr, width,
                                        height_gyr, edgecolor='black', facecolor='grey', alpha=0.5, label='food intake cycle'))
    else:
        ax1.add_patch(patches.Rectangle(pos_acc, width,
                                        height_acc, edgecolor='black', facecolor='grey', alpha=0.5))
        ax2.add_patch(patches.Rectangle(pos_gyr, width,
                                        height_gyr, edgecolor='black', facecolor='grey', alpha=0.5))

ax1.legend()
ax2.legend()

plt.show()
plt.close()
plt.clf()


