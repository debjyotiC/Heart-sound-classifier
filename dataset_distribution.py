import matplotlib.pyplot as plt
import numpy as np

y = np.array([60, 60, 60, 20, 60, 60, 60, 20])
labels = ["Normal (PhysioNet)", "Normal (Pascal)", "Noisy Normal", "Gathered Normal",
          "Murmur (PhysioNet)", "Murmur (Pascal)", "Noisy Murmur", "Gathered Murmur"]

fig, ax = plt.subplots()

# Capture each of the return elements.
patches, texts, pcts = ax.pie(y, labels=labels, startangle=90, autopct='%.1f%%', radius=1.2, pctdistance=0.85,
                              wedgeprops={'linewidth': 2.0, 'edgecolor': 'white'}, textprops={'size': 12})
# Style just the percent values.
plt.setp(pcts, color='white')
# plt.savefig("images/dataset_dist.png", dpi=600)
plt.show()
