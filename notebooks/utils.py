import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap

def elbow_and_silhouette(X, random_state=42, k_range=(2, 11)):

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), tight_layout=True)

    elbow = {}
    silhouette_scores = [] 

    k_range = range(*k_range)

    for i in k_range:
        kmeans = KMeans(n_clusters=i, random_state=random_state, n_init=10)
        kmeans.fit(X)
        elbow[i] = kmeans.inertia_ 

        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(X, labels))  

    # Corrigindo o chamado ao sns.lineplot
    sns.lineplot(x=list(elbow.keys()), y=list(elbow.values()), ax=axs[0])
    axs[0].set_title('Elbow Method')
    axs[0].set_xlabel('K')
    axs[0].set_xlabel('Inertia')

    sns.lineplot(x=list(k_range), y=silhouette_scores, ax=axs[1])
    axs[1].set_title('Silhouette Method')
    axs[1].set_xlabel('K')
    axs[1].set_xlabel('Silhouette Score')

    plt.show()
 

def show_clusters(
        dataframe,
        columns,
        color_amount,
        centroids, 
        show_centroids=False, 
        show_points=True,
        cluster_column=None):

    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.tab10.colors[:6]
    colors = ListedColormap(colors)

    x = dataframe[columns[0]]
    y = dataframe[columns[1]]
    z = dataframe[columns[2]]

    catch_centroids = show_centroids
    catch_points = show_points

    for i, centroid, in enumerate(centroids):
        if catch_centroids:
            ax.scatter(*centroid, s=500, alpha=0.5)
            ax.text(*centroid, f"{i}", fontsize=20, horizontalalignment="center", verticalalignment="center")

        if catch_points:
            s = ax.scatter(x, y, z, c=cluster_column, cmap=colors)
            ax.legend(*s.legend_elements(), bbox_to_anchor=(1.3, 0.5))

    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    ax.set_xlabel(columns[2])
    ax.set_title("Customer Segmentation")


    plt.show()
