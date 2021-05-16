import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_data(data_path):
    train_data = pd.read_csv(data_path, parse_dates=['date_confirmation'])

    # Based on https://www.kaggle.com/getting-started/27270 and https://stackoverflow.com/questions/16453644
    train_data = pd.concat([train_data, pd.get_dummies(train_data['sex'], prefix='sex')], axis=1)
    train_data['confirmation_day'] = (train_data.date_confirmation - train_data.date_confirmation.min()).dt.days

    X = train_data[['age', 'sex_unknown', 'sex_female', 'sex_male', 'latitude', 'longitude', 'confirmation_day', 'Lat',
                    'Long_', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Incidence_Rate', 'Case-Fatality_Ratio']]

    y = train_data['outcome']
    return train_test_split(X, y, test_size=0.2)


def get_knn_model(k, X_train, y_train):
    model_knn = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-2)
    )
    model_knn.fit(X_train, y_train)
    return model_knn


def plot(range_, train, valid, name, xlab):
    fig = plt.figure(figsize=(8, 5))
    sns.set_theme(style="darkgrid", palette='Paired')
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(range_, train, label="train", c='navy')
    plt.plot(range_, valid, label="validation", c='red')
    plt.legend(loc='right', fontsize=15)
    ax.set_xlabel(xlab, fontsize=15)
    ax.set_ylabel('Precision', fontsize=15)
    ax.set_title(name.title(), fontsize=15)
    plt.savefig(f"../plots/metric_{name}")


def main():
    X_train, X_valid, y_train, y_valid = prepare_data('../results/cases_train_processed.csv')
    knn_train = []
    knn_valid = []

    knn_range = [25, 50, 100, 200, 300, 400, 500, 750, 1000]
    for k in knn_range:
        print(k)
        knn_model = get_knn_model(k, X_train, y_train)
        knn_train.append(knn_model.score(X_train, y_train))
        knn_valid.append(knn_model.score(X_valid, y_valid))

    print(knn_train)
    print(knn_valid)

    plot(knn_range, knn_train, knn_valid, 'knn', 'Number of nearest neighbours')

    gbc_train = []
    gbc_valid = []
    gbc_range = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]

    for k in gbc_range:
        print(k)
        gbc_model = GradientBoostingClassifier(n_estimators=k)
        gbc_model.fit(X_train, y_train)
        gbc_train.append(gbc_model.score(X_train, y_train))
        gbc_valid.append(gbc_model.score(X_valid, y_valid))

    plot(gbc_range, gbc_train, gbc_valid, 'gbc', 'Number of boosting stages')


if __name__ == '__main__':
    main()


