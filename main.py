import json
from compare_clustering_solutions import evaluate_clustering
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.feature_extraction.text import CountVectorizer
from time import time

MODEL_NAME = 'all-MiniLM-L6-v2'
model = SentenceTransformer(MODEL_NAME)


def read_data(file_path: str) -> pd.DataFrame:
    """
    reads data from csv file and returns it as a pandas
    :param file_path: path to csv file
    :return:
        data frame
    """

    df = pd.read_csv(file_path)
    return df


# This function creates clusters from data according to given epsilon and minimum size of cluster
# returns a list of DataFrames where each DataFrames represents a cluster. The last element of the list is a DataFrame
# which contains all the unclusterd sentences
def make_clusters(data: pd.DataFrame, eps: float, min_size: int, max_iter: int) -> list[pd.DataFrame]:
    """
    creates clusters from data frame and returns it as a list of dataframes. The last element of the list contains
    unclustered sentences
    :param data: dataframe with unclustered sentences and their embeddings
    :param eps: minimum distance to between sentence and cluster
    :param min_size: minimum size of cluster
    :param max_iter: maximum number of iterations to run the clustering algorithm
    :return:
        list of dataframe representing clusters of sentences. The last cluster is the unclustered sentences
    """

    clusters = []
    data_size = len(data)
    old_data_size = -1
    iteration = 0

    # Iterate until reached max_iter or until reached convergence (data was not changed)
    while data_size != old_data_size and iteration < max_iter:
        old_data_size = data_size
        for i, row in data.iterrows():
            # find the cluster which is closest to the sentence
            cluster_ind = find_cluster(row['vector'], clusters, eps)

            # add the sentence to the cluster
            if cluster_ind != -1:
                clusters[cluster_ind] = pd.concat([clusters[cluster_ind], pd.DataFrame([row])])

            # there is no such cluster. create a new cluster and add it to clusters
            else:
                cluster = pd.DataFrame([row])
                clusters.append(cluster)

        # create new data which contains sentences in small clusters
        data = extract_small_clusters(clusters, min_size)

        data_size = len(data)
        iteration += 1

    if iteration == max_iter:
        print('Maximum number of iterations reached')

    # add the unclustered sentences to the clusters
    clusters.append(data)
    return clusters


def extract_small_clusters(clusters: list[pd.DataFrame], min_size: int) -> pd.DataFrame:
    """
    creates a new dataframe containing all sentences from clusters with size smaller than min_size
    :param clusters: list of clusters dataframe
    :param min_size: minimum size of the clusters
    :return:
        a dataframe containing the sentences from the small clusters
    """

    small_clusters = []
    clusters_to_delete = []
    for cluster_ind, cluster in enumerate(clusters):
        if cluster.shape[0] < min_size:
            # add the cluster to small_clusters and its index to cluster_to_delete
            small_clusters.append(cluster)
            clusters_to_delete.append(cluster_ind)

    # delete the small clusters from clusters
    while clusters_to_delete:
        del clusters[clusters_to_delete.pop()]

    # create new dataset with small clusters
    data = pd.concat(small_clusters)
    return data



def find_cluster(vector: np.ndarray, clusters: list[pd.DataFrame], eps: float) -> int:
    """
    finds the closest cluster to given vector
    :param vector: embedding vector of a sentence
    :param clusters: list of clusters dataframe
    :param eps: minimum distance between the vector and the closest cluster
    :return:
        the index of the closest cluster to given vector. if no cluster is found, return -1
    """

    # edge case where there are no clusters
    if not clusters:
        return -1

    # calculates the centroid of each cluster
    centroids = np.array([calculate_centroid(cluster) for cluster in clusters])

    # create distance matrix
    distance_mat = distance.cdist(centroids, vector.reshape(1, -1), 'cosine')

    # find the closest cluster to the vector
    min_distance = distance_mat.min()

    if min_distance < eps:
        return int(distance_mat.argmin())
    else:
        return -1


def calculate_centroid(cluster: pd.DataFrame) -> np.ndarray:
    """
    calculates the centroid of a cluster
    :param cluster: dataframe containing the sentences and their embeddings
    :return:
        the centroid of the cluster
    """

    return cluster['vector'].mean()


def name_clusters(clusters: list[pd.DataFrame]) -> list[str]:
    """
    names the clusters
    :param clusters: list of clusters dataframe
    :return:
        a list of names for the clusters
    """

    cluster_names = [name_cluster(cluster) for cluster in clusters]
    return cluster_names


def name_cluster(cluster: pd.DataFrame) -> str:
    """
    this function is used to name a cluster by calculating the closest sentence to the centroid and then
    finding the closest n-gram of the sentence to the centroid
    :param cluster: a dataframe containing the sentences
    :return:
        a string representing the name of the cluster
    """

    # Calculate centroid
    centroid = calculate_centroid(cluster).reshape(1, -1)

    # Calculate distance of each sentence in the cluster to the centroid
    distances = cluster['vector'].apply(lambda vector: distance.cdist(vector.reshape(1, -1), centroid)[0][0])

    # find the closest sentence to the centroid
    closest_sentence_index = distances.argmin()
    closest_sentence = cluster.iloc[closest_sentence_index]

    # Find ngrams
    count_vectorizer = CountVectorizer(ngram_range=(3, 6))
    count_vectorizer.fit([closest_sentence['text']])
    ngrams = count_vectorizer.get_feature_names_out()
    ngrams_encoding = model.encode(ngrams)

    # Find closest ngram to centroid
    closest_ngram = distance.cdist(ngrams_encoding, centroid).argmin()
    return ngrams[closest_ngram]


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    this function is used to add a new column to the dataframe which contains the embeddings of the sentences and
    cleans the sentences
    :param df: a dataframe containing the sentences
    :return:
        a dataframe containing the embeddings of the sentences and the cleaned sentences
    """

    df['vector'] = df['text'].apply(model.encode)
    df['text'] = df['text'].apply(lambda x: x.strip().lower())
    df = df.sample(frac=1)
    return df


def analyze_unrecognized_requests(data_file, output_file, min_size):
    """
    This function is used to analyze the unrecognized requests and save the results to a csv file in the output directory
    :param data_file: path to the file containing the unrecognized requests
    :param output_file: path to the output file
    :param min_size: the minimum size of the clusters
    :return:
    """
    start_time = time()

    df = read_data(data_file)
    df = preprocess_data(df)

    clusters = make_clusters(df, 0.30, int(min_size), 10)
    names = name_clusters(clusters)
    save_clusters(clusters, names, output_file)

    print("Total runtime: ", time() - start_time)


def save_clusters(clusters, names, output_file):
    """
    This function is used to save the clusters to a csv file in the output directory
    :param clusters: the clusters to save
    :param names: the names of the clusters
    :param output_file: a path to the output file
    :return:
    """

    output = {
        "cluster_list": [],
        "unclustered": clusters[-1]['text'].to_list()
    }
    for cluster, name in zip(clusters[:-1], names[:-1]):
        cluster_info = {
            "cluster_name": name,
            "requests": cluster['text'].to_list()
        }
        output["cluster_list"].append(cluster_info)
    json.dump(output, open(output_file, 'w'), indent=4)


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['min_cluster_size'])

    evaluate_clustering(config['example_solution_file'], config['output_file'])