import os.path

import pandas as pd


def load_data(folder_name, output_dir="output", data_size=-1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    from pandas import DataFrame
    months = ["apr", "may", "jun", "jul", "aug", "sep"]
    file_format = "uber-raw-data-{}14.csv"
    df = DataFrame()
    for month in months:
        file_name = folder_name + "/" + file_format.format(month)
        df_sub = pd.read_csv(file_name)
        df = df.append(df_sub)
    if data_size > 0:
        df = df.sample(n=data_size)
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%m/%d/%Y %H:%M:%S')
    df['month'] = df['Date/Time'].dt.month
    df['weekday'] = df['Date/Time'].dt.day_name()
    df['day'] = df['Date/Time'].dt.day
    df['hour'] = df['Date/Time'].dt.hour
    df['minute'] = df['Date/Time'].dt.minute
    df['lat_short'] = round(df['Lat'], 2)
    df['lon_short'] = round(df['Lon'], 2)
    demand = (df.groupby(['lat_short', 'lon_short']).count()['Date/Time']).reset_index()
    demand.columns = ['Latitude', 'Longitude', 'Number of Trips']
    demand.to_csv(output_dir + "/demand.csv", index=False)
    demand_w = (df.groupby(['lat_short', 'lon_short', 'weekday']).count()['Date/Time']).reset_index()
    demand_w.columns = ['Latitude', 'Longitude', 'Weekday', 'Number of Trips']
    demand_w.to_csv(output_dir + "/demand_dow.csv", index=False)
    demand_h = (df.groupby(['lat_short', 'lon_short', 'hour']).count()['Date/Time']).reset_index()
    demand_h.columns = ['Latitude', 'Longitude', 'Hour', 'Number of Trips']
    demand_h.to_csv(output_dir + "/demand_h.csv", index=False)
    demand_wh = (df.groupby(['lat_short', 'lon_short', 'weekday', 'hour']).count()['Date/Time']).reset_index()
    demand_wh.columns = ['Latitude', 'Longitude','Weekday', 'Hour', 'Number of Trips']
    demand_wh.to_csv(output_dir + "/demand_h_dow.csv", index=False)

load_data("data", data_size=100)
