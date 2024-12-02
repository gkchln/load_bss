import pandas as pd
import logging
import holidays
from utils import weekday_mapping, month_to_season, hour_loss_days, zones
from datetime import timedelta

# Logging paramneters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

def clean_load(input_df):
    df = input_df.copy()
    df = df[df.Date.dt.minute == 0] # The original granularity of the data is the hour and 15-min interval values are just interpolated
    df['Total_Load_MW'] = df['Total_Load_MW'].astype(float)
    df.drop('Forecast_Total_Load_MW', axis=1, inplace=True)
    df.sort_values('Date', inplace=True)
    # Drop eventual duplicates
    len_before = len(df)
    df.drop_duplicates(subset=['Date', 'Bidding_Zone'], inplace=True)
    len_after = len(df)
    if len_after < len_before:
        logging.warning("Dropped {} duplicate date/zone entries".format(len_before - len_after))
    df = df.pivot(columns='Bidding_Zone', values='Total_Load_MW', index='Date')
    return df


def add_calendar(input_df):
    df = input_df.copy()
    # year, day, weekday, hour
    df['year'] = df.index.year
    df['day'] = df.index.date
    df['weekday'] = df.index.weekday.map(weekday_mapping)
    df['hour'] = df.index.hour
    # Add day type in which we make the distinction between Mondays, Working days (From Tuesday to Friday), Saturdays and Holidays (including Sundays)
    holidays_it = holidays.IT(years=df.year.unique()) # Retrieve holidays in Italy
    df['daytype'] = 'Working day'
    df.loc[df.weekday == 'Saturday', 'daytype'] = 'Saturday'
    df.loc[df.weekday == 'Sunday', 'daytype'] = 'Holiday'
    df.loc[df.weekday == 'Monday', 'daytype'] = 'Monday'
    df.loc[df.day.apply(lambda day: day in holidays_it), 'daytype'] = 'Holiday'
    return df


def to_daily_curves(input_df, zonal=False):
    df = input_df.copy()
    if zonal:
        df.drop('Italy', axis=1, inplace=True)
    else:
        df.drop(zones, axis=1, inplace=True)

    # Duplicate rows of hour 3 for hour loss days to fill the missing hour
    rows_to_dup = df[(df['hour'] == 3) & (df['day'].astype(str).isin(hour_loss_days))]
    rows_to_dup_names = rows_to_dup.index
    rows_dup_names = pd.to_datetime(rows_to_dup_names) - pd.to_timedelta(1, unit='h')
    rows_to_dup.index = rows_dup_names
    df = pd.concat([df, rows_to_dup])
    df.sort_index(inplace=True)
    df.loc[rows_to_dup.index, 'hour'] = 2 # we also need to change hour column

    # Operation to duplicate the load at midnight to add it as point of previous day and next day
    rows_to_dup = df[df['hour'] == 0][1:] # First row concerns a day before the start of the period
    rows_to_dup_names = rows_to_dup.index
    rows_dup_names = pd.to_datetime(rows_to_dup_names) - timedelta(days=1) + timedelta(hours=23, minutes=59, seconds=59)
    rows_to_dup.index = rows_dup_names
    df = pd.concat([df, rows_to_dup])
    df.sort_index(inplace=True)

    cols_to_shift = [col for col in df.columns if col not in ['Italy', 'hour'] + zones]
    rows_to_shift = df.index[df.index.minute == 59]
    df.loc[rows_to_shift, "hour"] = 24
    # Shift columns based on rows_to_shift
    for col in cols_to_shift:
        for row in rows_to_shift:
            df.loc[row, col] = df.loc[row - timedelta(minutes=59, seconds=59), col]
    df = df.iloc[:-1]

    # Pivot table
    value_cols = zones if zonal else ['Italy']
    df = df.pivot(index='hour', columns=['day', 'daytype'], values=value_cols)
    df.columns = df.columns.map(lambda x: tuple(map(str, x)))
    df.columns = ["_".join(col) for col in df.columns.values]

    # Check for eventual NaN
    cols_with_na = df.columns[df.isna().any()]
    for col in cols_with_na:
        logging.warning(f"NA values found for {col}, using value of next hour to impute missing value")
    df.bfill(inplace=True)
    df.index.name = None

    # Express load in GW
    df = df / 1000

    return df.T

def preprocess_load(input_df, zonal=False):
    df = clean_load(input_df)
    logging.info("Load data cleant")
    df = add_calendar(df)
    logging.info("Calendar info added")
    df = to_daily_curves(df, zonal=zonal)
    logging.info("Daily load curves built")
    return df


### Script parameters ###
infile = 'data/load_2020_to_2022.csv'
outfile = 'data/daily_curves_2020_to_2022.csv'
zonal = False

if __name__ == "__main__":
    logging.info("Preprocessing load data located at {}...".format(infile))
    input_df = pd.read_csv(infile, parse_dates=[0])
    df = preprocess_load(input_df, zonal=zonal)
    df.to_csv(outfile)
    logging.info("Output saved at {}".format(outfile))
    