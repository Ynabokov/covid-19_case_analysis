# COVID 19 Project - part 1


### Data Sources *(saved in data directory)*

- Cases and countries data: https://github.com/MadanKrishnan97/CMPT459CourseProjectSpring2021

### Required Libraries (also can be found in src/requirements.txt)

- os
- pandas
- numpy
- matplotlib
- haversine
- tqdm
- contextily
- fiona
- shapely
- geopandas

***Note***

Last 4 libraries are used only to generate some plots in part 1.1. However, their installation can be very tricky, so please consult with https://geopandas.org/install.html. Also, please make sure that you have recent GDAL version (https://gdal.org/index.html) installed on your device, especially if you want to install geopandas with pip3.


### To actually run the project use the following command (main.py is situated in src directory):

> python3 main.py

***Note***

Beware that cleaning, transformation and join of datasets (especially, cases_train_transformed.csv) may take some decent time. The total running time  to process all three datasets may take up to 20-30 minutes. Therefore, the code uses cached files where possible. If you want to regenerate files in results directory, you will need to remove them first.


### Code Structure

    .
    ├── data                    # Raw data
    ├── plots
    ├── results                     # Cleaned and preprocessed data
    │   ├── cases_train_transformed.csv      
    │   ├── cases_test_transformed.csv      
    │   └── location_transformed.csv      
    ├── src
    │   ├── eda.ipynb      
    │   ├── main.py  
    │   ├── helper .py files    
    │   └── requirements.txt
    └── README.md
