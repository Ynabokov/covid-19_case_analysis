import numpy as np
import pandas as pd
import os

RESULTS_DIR = '../results/'
LOCATION_TR_FILENAME = RESULTS_DIR + 'location_transformed.csv'


def most_popular(col):
	return col.value_counts().index[0]


# Based on Geographic midpoint algorithm from http://www.geomidpoint.com/calculation.html
def calc_midpoint(orig_lat, orig_lon):
	lat = (orig_lat * np.pi)/180
	lon = (orig_lon * np.pi)/180

	x = np.mean(np.cos(lat) * np.cos(lon))
	y = np.mean(np.cos(lat) * np.sin(lon))
	z = np.mean(np.sin(lat))
  
	Lon = np.arctan2(y, x)
	Hyp = np.sqrt(x * x + y * y)
	Lat = np.arctan2(z, Hyp)

	final_lat = (Lat * 180)/np.pi
	final_lon = (Lon * 180)/np.pi
	return[final_lat, final_lon]


def find_state_centres(us):
	centres = []  
	for state in us['Province_State'].unique():
	  state_loc = us[us['Province_State'] == state]
	  if(len(state_loc.index) == 1):
	    centres.append([state_loc['Lat'].iloc[0], state_loc['Long_'].iloc[0]])
	    continue
	    
	  centres.append(calc_midpoint(state_loc['Lat'], state_loc['Long_']))
	return centres



def preprocess_locations(location):
	us = location[location.Country_Region == 'US'].copy()
	non_us = location[~(location.Country_Region == 'US')]

	# Population = Confirmed/Incidence_Rate * 100,000
	us["Population"] = (us["Confirmed"]/us["Incidence_Rate"]) * 100000
	preliminary = us.groupby('Province_State', as_index=False).agg(Country_Region = ('Country_Region', lambda x: 'United States'),
		Last_Update = ('Last_Update', most_popular), Confirmed = ('Confirmed', sum), Deaths = ('Deaths', sum),
		Recovered = ('Recovered', sum), Active = ('Active', sum), Population = ('Population', sum))

	# Case-Fatality Ratio (%) = Number of deaths / Number of confirmed cases.
	preliminary["Case-Fatality_Ratio"] = preliminary["Deaths"]/preliminary["Confirmed"]
	preliminary["Combined_Key"] = preliminary["Province_State"] + ', ' + preliminary["Country_Region"]
	# Incidence_Rate = Confirmed/Population * 100,000
	preliminary["Incidence_Rate"] = (preliminary["Confirmed"]/preliminary["Population"]) * 100000
	
	final_us = preliminary.join(pd.DataFrame(find_state_centres(us), columns = ['Lat','Long_']))
	# Make sure that cruise ships statistics is consistent
	final_us = final_us.replace(np.inf, np.nan)

	final_us = final_us[['Province_State', 'Country_Region', 'Last_Update', 'Lat', 'Long_',
	 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Combined_Key', 'Incidence_Rate', 'Case-Fatality_Ratio']]

	location_transformed = pd.concat([final_us, non_us], ignore_index=True)
	location_transformed = location_transformed.sort_values(by = 'Country_Region', ignore_index=True)

	return location_transformed


def transfrom_locations(location):
	if os.path.exists(LOCATION_TR_FILENAME):
		print('Using cached transformed locations from ' + LOCATION_TR_FILENAME)
		return pd.read_csv(LOCATION_TR_FILENAME)

	location_transformed = preprocess_locations(location)
	location_transformed.to_csv(LOCATION_TR_FILENAME, index=False)
	return location_transformed
