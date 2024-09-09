import pandas as pd
from utils import weekday_mapping, month_to_season
import holidays
import argparse

def add_calendar(df):
    '''
    Function to pre-process load data before subsequent FDA and NMF analysis. The example input file is "load.csv".
    '''
    # Add year, month, week, day, weekday, hour
    df['year'] = df.index.year
    df['monthofyear'] = df.index.month
    df['season'] = df.monthofyear.map(month_to_season)
    df['weekofyear'] = df.index.isocalendar().week
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

if __name__ == "__main__":
    # Parse arguments provided by the user
    parser = argparse.ArgumentParser(description="Add calendar data to the input load dataframe")
    parser.add_argument("--infile", dest="infile", type=str, help="Input file for the daily load curves")
    parser.add_argument("--outfile", dest="outfile", type=str, help="Output file for the NMF results")
    args = parser.parse_args()
    # Read the input file, add the calendar data and write to CSV
    df = pd.read_csv(args.infile, index_col=[0], parse_dates=[0])
    df = add_calendar(df)
    # HOTFIX: Retrieve data for Working days only
    # df = df[df.daytype == 'Working day']
    df.to_csv(args.outfile)