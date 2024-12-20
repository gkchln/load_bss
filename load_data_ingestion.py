# Imports
import pandas as pd
import logging
from datetime import datetime, timedelta
from datetime import time as dt_time
import requests
import time
from tqdm import tqdm
import calendar
from os import environ

# Logging paramneters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

# API access parameters
access_url = 'https://api.terna.it/transparency/oauth/accessToken'
API_KEY = environ['API_KEY']
SECRET = environ['API_SECRET']
payload = f'grant_type=client_credentials&client_id={API_KEY}&client_secret={SECRET}'
access_headers = {'Content-Type': 'application/x-www-form-urlencoded'}

# API load endpoint parameters
url = 'https://api.terna.it/transparency/v1.0/gettotalload'


### Functions ###
def get_access_token(url, payload, headers):
    """Authentication to Terna Developer API"""
    r_access = requests.post(url, headers=headers, data=payload)
    r_access.raise_for_status()
    return r_access.json()['access_token']

def fetch_total_load(url, headers, date_from, date_to):
    """Fetch total load from Terna API for a specific time period (which must not exceed 60 days).

    Args:
        url (str): endpoint url for total load
        headers (str: headers passed in the GET call
        date_from (str): start of requested period in DD/MM/YYYY format
        date_to (str): end of requested period in DD/MM/YYYY format

    Returns:
        pandas.DataFrame: Total load data for period requested
    """
    params = {'dateFrom': date_from, 'dateTo': date_to}
    r = requests.get(url, headers=headers, params=params)
    if r.status_code != 200:
        logging.error(r.text)
        r.raise_for_status()
    else:
        try:
            return pd.DataFrame(r.json()['totalLoad'])
        except:
            logging.error(r)
            raise

def get_load_data(url, access_token, date_from, date_to):
    """Get total load from Terna API for a time period spanning more than 60 days by making successive calls for monthly chunks.

    Args:
        url (str): endpoint url for total load
        access_token (str): access token obtained from authentication
        date_from (datetime.date or str): start of requested period (must be provided in YYYY-MM-DD format if str)
        date_to (datetime.date or str): end of requested period (must be provided in YYYY-MM-DD format if str)

    Returns:
        pandas.DataFrame: Total load data for period requested
    """
    headers = {'Authorization': 'Bearer {}'.format(access_token)}
    monthly_periods = pd.period_range(date_from, date_to, freq='M')
    dfs = []

    logging.info("Fetching {} monthly chunks of load data from {} to {}...".format(len(monthly_periods), date_from, date_to))
    for i, period in enumerate(tqdm(monthly_periods)):
        period_start = period.start_time.strftime('%d/%m/%Y')
        if i == len(monthly_periods) - 1:
            period_end = (period.end_time + timedelta(days=1)).strftime('%d/%m/%Y')
        else:
            period_end = period.end_time.strftime('%d/%m/%Y')
        df = fetch_total_load(url, headers, period_start, period_end)
        dfs = [df] + dfs
        time.sleep(1)
    output_df = pd.concat(dfs, ignore_index=True)
    output_df['Date']= output_df.Date.astype('datetime64[ns]')
    # We keep midnight of the last day
    output_df = output_df[output_df.Date <= datetime.combine(datetime.strptime(date_to, "%Y-%m-%d") + timedelta(days=1), dt_time.min)]

    return output_df


### Script parameters ###
# last_day_previous_month = (datetime.today().replace(day=1) - timedelta(days=1)).date()
# first_day_previous_month = last_day_previous_month.replace(day=1)

# By default we retrieve all data for last month
date_from = "2023-01-01"
date_to = "2023-12-31"
outfile = 'data/load_2023.csv'

### Main ###
if __name__ == "__main__":
    access_token = get_access_token(access_url, payload, access_headers)
    time.sleep(1) # We need to sleep one second otherwise we get blocked
    df = get_load_data(url, access_token, date_from, date_to)
    df.to_csv(outfile, index=False)
    logging.info("Load data saved at {}".format(outfile))
