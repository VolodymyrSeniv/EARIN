import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def prerpocessing_the_data(df : pd.DataFrame) -> np.ndarray:
    # Preprocess the data by normalizing or scaling the song attribute values
    scaler = StandardScaler()
    attributes_scaled = scaler.fit_transform(df.iloc[:, 0:13].values)
    return attributes_scaled

def kmeans_alg(attributes_scaled: np.ndarray) -> np.ndarray:
    # Apply K-means clustering on the dataset
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans_labels = kmeans.fit_predict(attributes_scaled)
    return kmeans_labels

def hierarchical_alg(attributes_scaled: np.ndarray) -> np.ndarray:
    # Apply Hierarchical clustering on the dataset
    hierarchical = AgglomerativeClustering(n_clusters=5)
    hierarchical_labels = hierarchical.fit_predict(attributes_scaled)
    return hierarchical_labels

def scores_kmeans(attributes_scaled: np.ndarray, kmeans_labels: np.ndarray):
    # Calculate silhouette scores for K-means clustering
    silhouette = silhouette_score(attributes_scaled, kmeans_labels)
    print(f"Silhouette score (K-means): {silhouette}")

def scores_hierarhical(attributes_scaled: np.ndarray, hiearchical_labels: np.ndarray):
    silhouette = silhouette_score(attributes_scaled, hiearchical_labels)
    print(f"Silhouette score (Hierarchical): {silhouette}")


def plots_kmeans(attributes_scaled: np.ndarray, kmeans_labels: np.ndarray, hierarchical_lables: np.ndarray):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    # Plot for K-means Clustering
    scatter1 = ax1.scatter(attributes_scaled[:, 0], attributes_scaled[:, 1], c=kmeans_labels, cmap='viridis')
    ax1.set_xlabel('Attribute 1')
    ax1.set_ylabel('Attribute 2')
    ax1.set_title('K-means Clustering')
    plt.colorbar(scatter1, ax=ax1)
    # Plot for Hierarchical Clustering
    scatter2 = ax2.scatter(attributes_scaled[:, 0], attributes_scaled[:, 1], c=hierarchical_lables, cmap='viridis')
    ax2.set_xlabel('Attribute 1')
    ax2.set_ylabel('Attribute 2')
    ax2.set_title('Hierarchical Clustering')
    plt.colorbar(scatter2, ax=ax2)
    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()


# Function to recommend songs from the same cluster as the input song
def recommend_songs(song_title : str, cluster_labels : np.ndarray, algorithm_name : str, num_recommendations : int) -> List[str]:
    cluster_label = cluster_labels[df[df['song_title'] == song_title].index[0]]
    recommendations = df[(df['song_title'] != song_title) & (cluster_labels == cluster_label)]['song_title'].tolist()
    #print(f"Songs similar to '{song_title}' (Algorithm: {algorithm_name}):")
    return recommendations[:num_recommendations]

if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    kmeans_labels = kmeans_alg(prerpocessing_the_data(df))
    hierarchical_labels = hierarchical_alg(prerpocessing_the_data(df))
    plots_kmeans(prerpocessing_the_data(df), kmeans_labels, hierarchical_labels)
    #plots_hierarchical_alg(prerpocessing_the_data(df), hierarchical_labels)
    scores_kmeans(prerpocessing_the_data(df), kmeans_labels)
    scores_hierarhical(prerpocessing_the_data(df), hierarchical_labels)
    while True:
        input_song = input("Enter a song name: ")
        recommended_songs_kmeans = recommend_songs(input_song, kmeans_labels, "K-means", 10)
        recommended_songs_hierarchical = recommend_songs(input_song, hierarchical_labels, "Hierarchical", 10)

        if recommended_songs_kmeans and recommended_songs_hierarchical:
            print("Recommendations using K-means clustering:")
            print(recommended_songs_kmeans)
            print("Recommendations using Hierarchical clustering:")
            print(recommended_songs_hierarchical)
            break
        else:
            print("Input song not found in the dataset. Please enter a valid song name.")
