import sys
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import v_measure_score
from sklearn.preprocessing import StandardScaler

def preprocess_train(df: pd.DataFrame):
    """Preprocess train data: filter, map categoricals, impute, scale"""
    data = df.copy()
    labels = data["kategorija"]
    data = data.drop(columns=["kategorija", "vannastavne_aktivnosti"])
    
    mask = data["izostanci"] <= 20
    data = data[mask]
    labels = labels[mask]

    mappings = {
        "pol": {"M": 0, "Z": 1},
        "dodatni_casovi": {"ne": 0, "da": 1},
        "internet": {"ne": 0, "da": 1},
    }
    for col, map_dict in mappings.items():
        if col in data.columns:
            data[col] = data[col].map(map_dict)

    imputer = SimpleImputer()
    data[data.columns] = imputer.fit_transform(data)

    scaler = StandardScaler()
    data[["godine", "izostanci"]] = scaler.fit_transform(data[["godine", "izostanci"]])

    return data, labels, scaler


def preprocess_test(df: pd.DataFrame, scaler: StandardScaler):
    """Preprocess test data using train imputer and scaler"""
    data = df.copy()
    labels = data["kategorija"]
    data = data.drop(columns=["kategorija", "vannastavne_aktivnosti"])

    mappings = {
        "pol": {"M": 0, "Z": 1},
        "dodatni_casovi": {"ne": 0, "da": 1},
        "internet": {"ne": 0, "da": 1},
    }
    for col, map_dict in mappings.items():
        if col in data.columns:
            data[col] = data[col].map(map_dict)


    data[["godine", "izostanci"]] = scaler.transform(data[["godine", "izostanci"]])

    return data, labels


# def cluster_em(train_data, test_data, n_clusters=3, true_labels=None):
#     gmm = GaussianMixture(n_components=n_clusters, random_state=42)
#     gmm.fit(train_data)

#     train_clusters = gmm.predict(train_data)
#     test_clusters = gmm.predict(test_data)

#     v_train, v_test = None, None
#     if true_labels is not None:
#         v_train = v_measure_score(true_labels[0], train_clusters)
#         v_test = v_measure_score(true_labels[1], test_clusters)

#     return train_clusters, test_clusters, v_train, v_test


# def cluster_kmeans(train_data, test_data, n_clusters=3, true_labels=None):
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     kmeans.fit(train_data)

#     train_clusters = kmeans.predict(train_data)
#     test_clusters = kmeans.predict(test_data)

#     v_train, v_test = None, None
#     if true_labels is not None:
#         v_train = v_measure_score(true_labels[0], train_clusters)
#         v_test = v_measure_score(true_labels[1], test_clusters)

#     return train_clusters, test_clusters, v_train, v_test


# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import v_measure_score
# from sklearn.cluster import DBSCAN

# def cluster_dbscan(train_data, test_data, eps=0.5, min_samples=5, true_labels=None):
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     dbscan.fit(train_data)

#     train_clusters = dbscan.labels_

#     if len(set(train_clusters)) > 1:
#         knn = KNeighborsClassifier(n_neighbors=3)
#         knn.fit(train_data, train_clusters)
#         test_clusters = knn.predict(test_data)
#     else:
#         test_clusters = [-1] * len(test_data)

#     v_train, v_test = None, None
#     if true_labels is not None:
#         v_train = v_measure_score(true_labels[0], train_clusters)
#         v_test = v_measure_score(true_labels[1], test_clusters)

#     return train_clusters, test_clusters, v_train, v_test



if __name__ == "__main__":

    train_file = sys.argv[1]
    test_file = sys.argv[2]

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    # Preprocess train
    X_train, y_train, scaler = preprocess_train(df_train)

    # Preprocess test using train transformations
    X_test, y_test = preprocess_test(df_test, scaler)

    # Run clustering (primer sa GMM)
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X_train)
    test_clusters = gmm.predict(X_test)
    print(v_measure_score(y_test, test_clusters))

    # # train/test split
    # X_train, a, y_train, b = train_test_split(
    #     X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    # )

    # # === GMM ===
    # train_clusters, test_clusters, v_train, v_test = cluster_em(
    #     X_train, X_test, n_clusters=3, true_labels=(y_train, y_test)
    # )
    # print("\n=== GMM ===")
    # print("Train V-measure:", v_train)
    # print("Test V-measure:", v_test)

    # # === KMeans ===
    # train_clusters, test_clusters, v_train, v_test = cluster_kmeans(
    #     X_train, X_test, n_clusters=3, true_labels=(y_train, y_test)
    # )
    # print("\n=== KMeans ===")
    # print("Train V-measure:", v_train)
    # print("Test V-measure:", v_test)

    # # === DBSCAN ===
    # train_clusters, test_clusters, v_train, v_test = cluster_dbscan(
    #     X_train, X_test, eps=0.7, min_samples=5, true_labels=(y_train, y_test)
    # )
    # print("\n=== DBSCAN ===")
    # print("Train V-measure:", v_train)
    # print("Broj klastera u train setu:", len(set(train_clusters)))
    # print("Test V-measure:", v_test)
    # print("Broj klastera u test setu:", len(set(test_clusters)))


    # VIZUALIZACIJA BROJA NEDOSTAJUCIH POLJA
    # columns = ["dodatni_casovi", "vannastavne_aktivnosti", "internet"]

    # for col in columns:
    #     total = len(df)
    #     ones = (df[col] == "da").sum()
    #     zeros = (df[col] == "ne").sum()
    #     missing = total - (ones + zeros)
        
    #     print(f"Kolona: {col}")
    #     print(f"  Broj 'da' (1): {ones}")
    #     print(f"  Broj 'ne' (0): {zeros}")
    #     print(f"  Broj nedostajućih: {missing}\n")

    # # ODRADJENE SU VIZUALIZACIJE ODNOSA SVAKOG ATRIBUTA I USPEHA STUDENATA
    # # Vizualizacija
    # plt.figure(figsize=(8, 6))
    # sns.countplot(
    #     data=df,
    #     x="izostanci",
    #     hue="kategorija",
    #     palette="Set2"
    # )
    # plt.title("Odnos uspeha")
    # plt.xlabel("")
    # plt.ylabel("Broj učenika")
    # plt.legend(title="Kategorija")
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(8, 6))

    # # Scatter plot: izostanci po kategoriji
    # sns.stripplot(
    #     data=df,
    #     x="kategorija",
    #     y="izostanci",
    #     jitter=True,       # raspršuje tačke da se ne preklapaju
    #     palette="Set2",
    #     size=6             # veličina tačaka
    # )

    # plt.title("Scatter plot izostanaka po kategoriji uspeha")
    # plt.xlabel("Kategorija")
    # plt.ylabel("Broj izostanaka")
    # plt.tight_layout()
    # plt.show()
