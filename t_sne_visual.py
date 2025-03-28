from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

arch = 'resnet50'
arch_name = 'ResNet-50'
embeddings = np.load(f'npys/top10_{arch}_embeddings.npy')
target = np.load(f'npys/top10_{arch}_target.npy')

target_value = list(set(target))
color_dict = {}
colors = ['black', 'red', 'yellow', 'green', 'orange', 'blue', 'magenta', 'slategray', 'cyan', 'aquamarine']
for i, t in enumerate(target_value):
    color_dict[t] = colors[i]

    
X_tsne = TSNE(learning_rate=400.0, perplexity=20).fit_transform(embeddings)
plt.figure(figsize=(5, 5))

for i in range(len(target_value)):
    tmp_X = X_tsne[target==target_value[i]]
    plt.scatter(tmp_X[:, 0], tmp_X[:, 1], c=color_dict[target_value[i]])

plt.title(f'{arch_name}')
plt.xticks([])
plt.yticks([])
plt.show()
plt.savefig(f't_sne/top10_{arch}_tsne.pdf')
