import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle


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


def main():
    X_train, X_valid, y_train, y_valid = prepare_data('../results/cases_train_processed.csv')
    knn_model = get_knn_model(200, X_train, y_train)
    with open('../models/model_knn.pkl', 'wb') as f:
        pickle.dump(knn_model, f)
    print(knn_model.score(X_train, y_train))
    print(knn_model.score(X_valid, y_valid))

    y_pred = knn_model.predict(X_valid)
    print(confusion_matrix(y_valid, y_pred))
    print(classification_report(y_valid, y_pred))

    gbc_model = GradientBoostingClassifier(n_estimators=1000)
    gbc_model.fit(X_train, y_train)
    with open('../models/model_gbc.pkl', 'wb') as f:
        pickle.dump(gbc_model, f)
    print(gbc_model.score(X_train, y_train))
    print(gbc_model.score(X_valid, y_valid))

    y_pred = gbc_model.predict(X_valid)
    print(confusion_matrix(y_valid, y_pred))
    print(classification_report(y_valid, y_pred))


if __name__ == '__main__':
    main()
