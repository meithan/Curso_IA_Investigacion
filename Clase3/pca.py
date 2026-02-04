#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# ## Cargar datos (iris dataset, incluido en scikit-learn)

# In[2]:


iris = datasets.load_iris()
X = iris.data
y = iris.target
keys = iris.keys()
target_names = iris.target_names
print(iris.feature_names)


# ## Graficar variables originales: ¿cuál combinación muestra mejor la estructura?
# - En 3D, 4 combinaciones
# - En 2D, 6 combinaciones
# - ¿Qué hacemos si hay 30 dimensiones en los datos?

# In[3]:


fig = plt.figure(figsize=(12,12))
for p, (i,j,k) in enumerate([(1,2,3), (0,2,3), (0,1,2), (0,1,3)]):
    ax = fig.add_subplot(2, 2, p+1, projection="3d", elev=-150, azim=110)
    scatter = ax.scatter(X[:,i], X[:,j], X[:,k], c=iris.target, s=40)
    ax.set(xlabel=iris.feature_names[i], ylabel=iris.feature_names[i], zlabel=iris.feature_names[i])
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    # Add a legend
    legend1 = ax.legend(
        scatter.legend_elements()[0],
        iris.target_names.tolist(),
        loc="upper right",
        title="Classes",
    )
    ax.add_artist(legend1)
plt.show()


# ## Con PCA, generamos nuevos ejes que hacen resaltar la estructura de los datos (en este caso, las diferencias entre grupos)

# In[4]:


pca = PCA(n_components=3).fit(iris.data)
X_projected = pca.transform(iris.data)


# In[5]:


fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)
scatter = ax.scatter(X_projected[:, 0], X_projected[:, 1], X_projected[:, 2], c=iris.target, s=40)
ax.set(title="First three principal components", xlabel="1st Principal Component", ylabel="2nd Principal Component", zlabel="3rd Principal Component")
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
# Add a legend
legend1 = ax.legend(
    scatter.legend_elements()[0],
    iris.target_names.tolist(),
    loc="upper right",
    title="Classes",
)
ax.add_artist(legend1)
plt.show()


# ## Graficar la varianza explicada de los 4 componentes

# In[6]:


pca = PCA(n_components=4).fit(iris.data)
plt.figure(figsize=(8, 5))
ax = plt.gca()
nums = range(1, len(pca.explained_variance_ratio_) + 1)
plt.plot(nums, pca.explained_variance_ratio_, "o-", label="Varianza explicada por componente")
for i, v in enumerate(pca.explained_variance_ratio_):
    ax.text(nums[i], v + 0.05, f"{100*v:.1f}%", ha='center', va='bottom')
plt.step(nums, np.cumsum(pca.explained_variance_ratio_), where='mid', label='Varianza explicada acumulada', color="red")
plt.xticks([1, 2, 3, 4], ["PC1", "PC2", "PC3", "PC4"])
plt.ylabel('Razón de varianza explicada')
plt.xlabel('Componente principal')
plt.legend(loc="center right")
plt.title('Varianza explicada de las PCs en iris dataset')
plt.show()


# ### La PC1 explica la mayoría de la varianza; veamos de qué combinación está hecha

# In[7]:


plt.figure(figsize=(8, 5))
ax = plt.gca()
nums = range(1, len(pca.explained_variance_ratio_) + 1)
bars = plt.bar(nums, pca.components_[0])
ax.bar_label(bars, padding=5)
plt.title("Pesos (loadings) de la PC1")
plt.xticks([1, 2, 3, 4], iris.feature_names)
plt.ylabel("Loading")
plt.tight_layout()
plt.show()


# In[ ]:




