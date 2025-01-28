import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
 
