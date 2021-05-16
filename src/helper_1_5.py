import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import tqdm.notebook as tqnb
from haversine import haversine_vector, Unit

RESULTS_DIR = '../results/'
DOUBLE_NAME_DICT = {'Czech Republic' : 'Czechia', 'South Korea' : 'Korea, South',
 'Republic of Congo' : 'Congo (Brazzaville)', 'Democratic Republic of the Congo' : 'Congo (Kinshasa)'}

def get_location_index(case, location, country_stat):
	no_prov, prov, coordinates_list = country_stat
	
	if case['country'] in DOUBLE_NAME_DICT.keys():
	  return location[location['Country_Region'] == DOUBLE_NAME_DICT[case['country']]].first_valid_index()
	elif case['country'] in no_prov:
	  return location[location['Country_Region'] == case['country']].first_valid_index()
	elif case['country'] in prov:
	  base_country = location[location['Country_Region'] == case['country']]
	  base_province = base_country[base_country['Province_State'] == case['province']]
	  if len(base_province) == 0:
	    pr_loc = np.nanargmin(haversine_vector( (case['latitude'], case['longitude']), 
	    	list(zip(base_country['Lat'],base_country['Long_'])), Unit.METERS, comb=True))
	    return base_country.iloc[pr_loc].name
	  return base_province.first_valid_index()  
	else:
	  return np.nanargmin(haversine_vector( (case['latitude'], case['longitude']), coordinates_list, Unit.METERS, comb=True))


def merge(cases, locations):
	tqdm.pandas()

	country_counts = locations.groupby('Country_Region')
	no_prov = country_counts.filter(lambda x: len(x) == 1)['Country_Region'].unique()
	prov = country_counts.filter(lambda x: len(x) > 1)['Country_Region'].unique()
	coordinates_list = list(zip(locations['Lat'],locations['Long_']))

	cases_with_key = cases.copy()
	print('Finding optimal location for each case')
	cases_with_key['key'] = cases_with_key.progress_apply(get_location_index,
	 location=locations, country_stat=(no_prov, prov, coordinates_list), axis=1)
	print('Merging cases and locations datasets')
	# Progress bar based on https://stackoverflow.com/questions/56256861/is-it-possible-to-use-tqdm-for-pandas-merge-operation
	return pd.merge(cases_with_key, locations, left_on='key', right_index=True).progress_apply(lambda x: x) 




def join_cases_locations(cases, locations, filename):
	filepath = RESULTS_DIR + filename
	if os.path.exists(filepath):
		print('Using cached cases-locations dataset from ' + filepath)
		return pd.read_csv(filepath)

	merged = merge(cases, locations)
	merged.to_csv(filepath, index=False)
	return merged
