import numpy as np
import pandas as pd
import os
# pd.set_option('display.max_columns', None)

cwd = os.getcwd()

print("-"*50)
print("Read Data")
print("-"*50)

directory = cwd + "/data/"
# print(directory)

clinton_flow = pd.read_csv(directory + "clinton_flow.csv", header=0)
clinton_weather = pd.read_csv(directory + "clinton_weather_data.csv", header=0)

print("------------")
print("BEFORE MERGE")
print("------------")

print("Dataset No. of Rows Clinton Flow: ", clinton_flow.shape[0])
print("Dataset No. of Columns Clinton Flow: ", clinton_flow.shape[1])
print("Columns of Clinton Flow: ", list(clinton_flow.columns.values))
print("-" * 50)
print("Dataset No. of Rows Clinton Weather: ", clinton_weather.shape[0])
print("Dataset No. of Columns Clinton Weather: ", clinton_weather.shape[1])
print("Columns of Clinton Weather: ", list(clinton_weather.columns.values))

# Remove everything but relevant variables

clinton_weather = clinton_weather[["DATE", "PRCP", "SNOW", "TMAX", "TMIN"]]
clinton_flow = clinton_flow[["Date", "Flow"]]


clinton_weather_data = clinton_weather.rename(columns={"DATE": "Date"})
clinton = pd.merge(clinton_flow, clinton_weather_data, on=["Date"])


print("------------")
print("AFTER MERGE")
print("------------")

print("Dataset No. of Rows Clinton: ", clinton.shape[0])
print("Dataset No. of Columns Clinton: ", clinton.shape[1])
print("Columns of Clinton: ", list(clinton.columns.values))
print("-" * 50)
print(clinton.head(3))
print(clinton.tail(3))



print("-" * 50)
print("Dataset No. of Rows Clinton: ", clinton.shape[0])
print("Dataset No. of Columns Clinton: ", clinton.shape[1])
print("Columns of Clinton: ", list(clinton.columns.values))
print("-" * 50)
print(clinton)
print(clinton.describe())

# PRCP = Precipitation (tenths of mm)
# SNOW = Snowfall (mm)
# SNWD = Snow depth (mm)
# TMAX = Maximum temperature (tenths of degrees C)
# TMIN = Minimum temperature (tenths of degrees C)







