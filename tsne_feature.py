import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

data_type = 'mfcc'
data = np.load(f"data/{data_type}.npz", allow_pickle=True)
x_data, y_data = data['out_x'], data['out_y']

tsne = TSNE(n_components=2, random_state=0)
tsne_data = tsne.fit_transform(x_data)

tsne_data = np.vstack((tsne_data.T, y_data)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
fg = sns.FacetGrid(tsne_df, hue="label", size=7)
fg.map(plt.scatter, "Dim_1", "Dim_2")
fg.add_legend()
plt.show()

