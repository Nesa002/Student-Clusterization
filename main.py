import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import v_measure_score
from sklearn.preprocessing import StandardScaler

def clean_data(df: pd.DataFrame):
    data = df.copy()

    labels = None
    if "kategorija" in data.columns:
        labels = data["kategorija"]
        data = data.drop(columns=["kategorija"])


    mappings = {
        "pol": {"M": 0, "Z": 1},
        "dodatni_casovi": {"ne": 0, "da": 1},
        "vannastavne_aktivnosti": {"ne": 0, "da": 1},
        "internet": {"ne": 0, "da": 1},
    }
    for col, map_dict in mappings.items():
        if col in data.columns:
            data[col] = data[col].map(map_dict)

    # Imputer za nedostajuće vrednosti
    imputer = SimpleImputer(strategy="median")
    data[data.columns] = imputer.fit_transform(data)


    # Skaliranje svih kolona
    scaler = StandardScaler()
    data[data.columns] = scaler.fit_transform(data)

    return data, labels


def cluster_em(data: pd.DataFrame, n_clusters: int = 3, true_labels=None):
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(data)
    clusters = gmm.predict(data)

    v_measure = None
    if true_labels is not None:
        v_measure = v_measure_score(true_labels, clusters)

    return clusters, gmm, v_measure


if __name__ == "__main__":
    import sys
    train_file = sys.argv[1]
    df = pd.read_csv(train_file, sep=",")

    cleaned, labels = clean_data(df)

    clusters, model, v_score = cluster_em(cleaned, n_clusters=7, true_labels=labels)
    print("Predikcije klastera:", clusters[:10])
    print("V-measure score:", v_score)


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