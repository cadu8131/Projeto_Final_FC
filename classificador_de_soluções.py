import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Carregar dataset
dataset = np.load("phi4_dataset.npz")
X = dataset['data']
velocities = dataset['velocities']
t = dataset['time']

def analyze_phi4_results(filename="phi4_dataset.npz"):
    # Abre os dados
    try:
        data_dict = np.load(filename)
        # Extrai os arrays do arquivo .npz
        X = data_dict['data']           # Matriz (Amostras x Tempo)
        velocities = data_dict['velocities']
        time_steps = data_dict['time']

        print(f"Dataset carregado: {filename}")
    except FileNotFoundError:
        print(f"Erro: O arquivo '{filename}' não foi encontrado.")
        return

    # Pré Processamento dos dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Aplicando PCA
    pca = PCA(n_components = 2)
    X_pca = pca.fit_transform(X_scaled)

    # Quanto da física foi capturada nessas 2 variáveis?
    variance = pca.explained_variance_ratio_
    print(f"Variância explicada: PC1 = {variance[0]*100:.1f}%, PC2 = {variance[1]*100:.1f}%")

    # Escolhi k = 3 pois temos 3 cenários possíveis: aniquilação, reflexão e rebote.
    k = 3
    kmeans = KMeans(n_clusters = k, random_state = 42, n_init = 10)

    # Treinamos o K-Means nos dados
    clusters = kmeans.fit_predict(X_scaled)

    # Visualizando
    fig = plt.figure(figsize=(15, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=60, edgecolors='k', alpha=0.8)
    ax1.set_xlabel('PCA 1')
    ax1.set_ylabel('PCA 2')
    ax1.set_title('PCA Map')
    plt.colorbar(scatter, ax=ax1, label='Cluster ID')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(velocities, clusters, c=clusters, cmap='viridis', s=40)
    ax2.set_xlabel(r'$v_0$')
    ax2.set_ylabel('Cluster')
    ax2.set_yticks(range(k))
    ax2.set_title('Final State')

    # Destaque visual para a região de ressonância
    ax2.axvspan(0.18, 0.26, color='red', alpha=0.05, label='Resonance')
    ax2.legend()

    # Centróides
    ax3 = fig.add_subplot(2, 1, 2)
    colors = plt.cm.viridis(np.linspace(0, 1, k))

    for i in range(k):
        # Pega a média de todas as simulações classificadas como cluster 'i'
        cluster_mean = np.mean(X[clusters == i], axis=0)
        ax3.plot(time_steps, cluster_mean, label=f'Cluster {i} (Average)', color=colors[i], linewidth=2)

    ax3.set_title('Cluster Dynamics')
    ax3.set_xlabel('Time')
    ax3.set_ylabel(r'$\phi(0,t)$')
    ax3.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_phi4_results()