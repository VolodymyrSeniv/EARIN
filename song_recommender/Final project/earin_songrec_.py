import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, pairwise_distances
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from typing import List
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def preprocessing_the_data(df: pd.DataFrame) -> np.ndarray:
    # Preprocess the data by normalizing or scaling the song attribute values
    scaler = StandardScaler()
    attributes_scaled = scaler.fit_transform(df.iloc[:, :-2].values)
    return attributes_scaled

def kmeans_alg(attributes_scaled: np.ndarray) -> np.ndarray:
    # Apply K-means clustering on the dataset
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)  # Set n_init explicitly
    kmeans_labels = kmeans.fit_predict(attributes_scaled)
    return kmeans_labels, kmeans.cluster_centers_

def hierarchical_alg(attributes_scaled: np.ndarray) -> np.ndarray:
    # Apply Hierarchical clustering on the dataset
    hierarchical = AgglomerativeClustering(n_clusters=5)
    hierarchical_labels = hierarchical.fit_predict(attributes_scaled)
    return hierarchical_labels

def calculate_silhouette_score(attributes_scaled: np.ndarray, labels: np.ndarray) -> float:
    # Calculate silhouette score for clustering
    silhouette = silhouette_score(attributes_scaled, labels)
    return silhouette

def calculate_euclidean_distance(attributes_scaled: np.ndarray, labels: np.ndarray, cluster_centers: np.ndarray) -> float:
    # Calculate the average Euclidean distance between samples and their cluster centers
    distances = pairwise_distances(attributes_scaled, cluster_centers)
    avg_distance = np.mean(distances[np.arange(len(distances)), labels])
    return avg_distance

def recommend_songs(input_song: str, labels: np.ndarray, algorithm: str, num_recommendations: int) -> List[str]:
    # Find the cluster label for the input song
    input_song_index = df[df['song_title'] == input_song].index[0]
    input_song_label = labels[input_song_index]

    # Filter songs with the same cluster label as the input song
    cluster_songs = df[labels == input_song_label]

    # Exclude the input song from the recommendations
    cluster_songs = cluster_songs[cluster_songs['song_title'] != input_song]

    # Sort songs based on some criteria (e.g., popularity, similarity)
    # and return the top 'num_recommendations' songs
    recommended_songs = cluster_songs['song_title'].head(num_recommendations).tolist()

    return recommended_songs

def plot_kmeans(attributes_scaled: np.ndarray, kmeans_labels: np.ndarray, attribute_names: List[str]):
    pca = PCA(n_components=2)
    reduced_attributes = pca.fit_transform(attributes_scaled)

    num_clusters = len(np.unique(kmeans_labels))

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('K-means Clustering', fontsize=14)

    for cluster in range(num_clusters):
        cluster_points = reduced_attributes[kmeans_labels == cluster]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', alpha=0.7)


    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_hierarchical(attributes_scaled: np.ndarray, hierarchical_labels: np.ndarray):
    plt.figure(figsize=(10, 6))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    linkage = hierarchy.linkage(attributes_scaled, method='ward')
    dendrogram = hierarchy.dendrogram(linkage, truncate_mode='level', p=3)
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    attributes_scaled = preprocessing_the_data(df)
    kmeans_labels, kmeans_centers = kmeans_alg(attributes_scaled)
    hierarchical_labels = hierarchical_alg(attributes_scaled)

    attribute_names = df.columns[:-2].tolist()  # Exclude the last two columns (song_title and artist)

    plot_kmeans(attributes_scaled, kmeans_labels, attribute_names)
    plot_hierarchical(attributes_scaled, hierarchical_labels)

    kmeans_silhouette = calculate_silhouette_score(attributes_scaled, kmeans_labels)
    hierarchical_silhouette = calculate_silhouette_score(attributes_scaled, hierarchical_labels)
    kmeans_distance = calculate_euclidean_distance(attributes_scaled, kmeans_labels, kmeans_centers)
    hierarchical_distance = calculate_euclidean_distance(attributes_scaled, hierarchical_labels, None)

    print(f"Silhouette score (K-means): {kmeans_silhouette}")
    print(f"Silhouette score (Hierarchical): {hierarchical_silhouette}")
    print(f"Euclidean distance (K-means): {kmeans_distance}")
    print(f"Euclidean distance (Hierarchical): {hierarchical_distance}")

    while True:
        input_song = input("Enter a song name: ")
        recommended_songs_kmeans = recommend_songs(input_song, kmeans_labels, "K-means", 10)
        recommended_songs_hierarchical = recommend_songs(input_song, hierarchical_labels, "Hierarchical", 10)

        if recommended_songs_kmeans:
            print("Recommendations using K-means clustering:")
            print(recommended_songs_kmeans)
            kmeans_rating = int(input("Rate the recommendations from K-means (1-10): "))
            print(f"K-means rating (subjective): {kmeans_rating}")

        if recommended_songs_hierarchical:
            print("Recommendations using Hierarchical clustering:")
            print(recommended_songs_hierarchical)
            hierarchical_rating = int(input("Rate the recommendations from Hierarchical (1-10): "))
            print(f"Hierarchical rating (subjective): {hierarchical_rating}")

        if recommended_songs_kmeans or recommended_songs_hierarchical:
            break
        else:
            print("Input song not found in the dataset. Please enter a valid song name.")
