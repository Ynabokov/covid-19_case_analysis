import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

RESULTS_DIR = '../results'

def clean_balance(data_path):
	train_data = pd.read_csv(data_path, parse_dates=['date_confirmation'])

	# Based on https://www.kaggle.com/getting-started/27270 and https://stackoverflow.com/questions/16453644
	train_data = pd.concat([train_data, pd.get_dummies(train_data['sex'], prefix='sex')], axis=1)
	train_data['confirmation_day'] = (train_data.date_confirmation - train_data.date_confirmation.min()).dt.days
	dec_description = ['died.*', 'death.*', 'Death.*', 'passed away', 'dead', 'Passed', 'deceased']
	train_data['deceased_info'] =(pd.concat([train_data['additional_information'].str.contains(word,regex=True) for word in dec_description],axis=1).sum(1) > 0).astype(int)
	
	return train_data

def prepare_data(data_path):
	train_data = clean_balance(data_path)
	balanced_data = train_data[train_data['outcome'] == 'deceased']
	dec_len = balanced_data.shape[0]

	for outcome in ['nonhospitalized', 'hospitalized', 'recovered']:
		balanced_data = balanced_data.append(train_data[train_data['outcome'] == outcome].sample(dec_len), ignore_index = True)

	X = balanced_data[['age', 'sex_unknown', 'sex_female', 'sex_male', 'latitude', 'longitude', 'confirmation_day', 'Lat', 
	'Long_', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Incidence_Rate', 'Case-Fatality_Ratio', 'deceased_info']]

	y = balanced_data['outcome']
	return X, y


def grid_search_knn(X,y):
	# Based on https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html and https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html
	scoring = {'F1_deceased':make_scorer(f1_score, labels = ['deceased'], average= 'weighted'),
	'Recall_deceased':make_scorer(recall_score, labels = ['deceased'], average= 'weighted'),
	'Accuracy': make_scorer(accuracy_score), 'Recall':make_scorer(recall_score,  average = 'weighted')}
	pipeline = Pipeline(steps=[('st_scal', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=-2))])

	param_grid = {
	    'knn__n_neighbors': range(50, 501, 25),
	    'knn__weights': ['uniform', 'distance']
	}
	gs = GridSearchCV(pipeline, param_grid=param_grid, scoring=scoring, refit='F1_deceased', return_train_score=True, n_jobs = -2, verbose=1)
	gs.fit(X, y)
	return gs


def fscore_estimator(label, predlabel):
	report = classification_report(label, predlabel, output_dict=True)
	print (classification_report(label, predlabel))
	return report['deceased']['f1-score']


def grid_search_gbc(X,y):
	params = {
	    'n_estimators': [50, 100, 250, 500],
	    'learning_rate': [0.1, 0.3, 0.5, 0.7, 0.9, 1],
	    'max_depth': [3, 7, 11, 15]
	}
	gbc = GradientBoostingClassifier()
	gridsearch = GridSearchCV(gbc, params, scoring=make_scorer(fscore_estimator), verbose=100, return_train_score=True)
	gridsearch.fit(X_train, y_train)
	best_model = gridsearch.best_estimator_
	print(gridsearch.best_params_)
	print(gridsearch.best_score_)
	return gridsearch


def predict_test(testfile, X, y):
	balanced_data = clean_balance(testfile)
	test = balanced_data[['age', 'sex_unknown', 'sex_female', 'sex_male', 'latitude', 'longitude', 'confirmation_day', 'Lat',
	'Long_', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Incidence_Rate', 'Case-Fatality_Ratio', 'deceased_info']]
	model_knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=175, weights='distance', n_jobs=-2))

	model_knn.fit(X, y)
	pred = model_knn.predict(test)
	print(y_pred[:20])
	np.savetxt(RESULTS_DIR + '/predictions.txt', pred, delimiter=',', fmt = '%s') 


def main():
    X, y = prepare_data(RESULTS_DIR + '/cases_train_processed.csv')
    # gs = grid_search_knn(X,y)
    # res = gs.cv_results_
    # par_res = {"F1-Score on ‘deceased’" : res["mean_test_F1_deceased"], "Recall on ‘deceased’" : res["mean_test_Recall_deceased"], "Overall Accuracy" : res["mean_test_Accuracy"], "Overall Recall" : res["mean_test_Recall"],  "rank" : res["rank_test_F1_deceased"]}
    # d = pd.concat([pd.DataFrame(res["params"]),pd.DataFrame(par_res)],axis=1)

    # d['Hyperparameters'] = "n_neighbors=" + d['knn__n_neighbors'].map(str) + ", weights=" + d['knn__weights']
    # d[['Hyperparameters', "F1-Score on ‘deceased’","Recall on ‘deceased’", "Overall Accuracy",  "Overall Recall"]].to_csv(RESULTS_DIR + 'knn_tuning.txt', index=None, sep=' ')
    # predict_test(RESULTS_DIR + '/cases_test_processed.csv', X, y)


if __name__ == '__main__':
    main()