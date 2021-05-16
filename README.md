# COVID 19 Project

Please consult with README_1 and README_2 to understand pre-processing, EDA and ML models creation.

### Required Libraries (also can be found in src/requirements.txt)

- os
- pandas
- numpy
- matplotlib
- seaborn
- pickle
- sklearn


### To actually run the project use the following command (main.py is situated in src directory):

> python3 main.py


***Note***

Beware that it may take a lot of time to find the best hyperparameters for both used classifiers.


### Code Structure

    .
    ├── README.md
    ├── plots
    ├── results
    │   ├── knn_tuning.txt  
    │   ├── gbc_tuning.txt  
    │   └── predictions.txt      
    └── src      
        └── main.py
