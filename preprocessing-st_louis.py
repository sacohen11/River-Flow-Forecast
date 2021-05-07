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

st_louis_flow = pd.read_csv(directory + "st_louis_flow.csv", header=0)
st_louis_weather1 = pd.read_csv(directory + "st_louis_weather_data_1.csv", header=0)
st_louis_weather2 = pd.read_csv(directory + "st_louis_weather_data_2.csv", header=0)



print("Dataset No. of Rows St. Louis Flow: ", st_louis_flow.shape[0])
print("Dataset No. of Columns St. Louis Flow: ", st_louis_flow.shape[1])
print("Columns of St. Louis Flow: ", list(st_louis_flow.columns.values))
print("-" * 50)
print("Dataset No. of Rows St. Louis Weather 1: ", st_louis_weather1.shape[0])
print("Dataset No. of Columns St. Louis Weather 1: ", st_louis_weather1.shape[1])
print("Columns of St. Louis Weather 1: ", list(st_louis_weather1.columns.values))
print("-" * 50)
print("Dataset No. of Rows St. Louis Weather 2: ", st_louis_weather2.shape[0])
print("Dataset No. of Columns St. Louis Weather 2: ", st_louis_weather2.shape[1])
print("Columns of St. Louis Weather 2: ", list(st_louis_weather2.columns.values))
print("-" * 50)


#Repeated Columns
weather1 = list(st_louis_weather1.columns.values)
weather2 = list(st_louis_weather2.columns.values)
repeated = []
for i in weather1:
    for j in weather2:
        if i == j:
            repeated.append(i)
print("Repeated Columns: ", repeated)
print("-" * 50)


#Droping useless columns
st_louis_weather1 = st_louis_weather1[["DATE", "PRCP", "SNOW", "TMAX", "TMIN"]]
st_louis_weather2 = st_louis_weather2[["DATE", "PRCP", "SNOW", "TMAX", "TMIN"]]
st_louis_flow = st_louis_flow[['Date', "Flow"]]


#Droping everything before 1969
st_louis_weather2 = st_louis_weather2[11229:]

##Concatenating Weather rows datasets
weather = pd.concat([st_louis_weather1, st_louis_weather2], axis=0, ignore_index=True)

# print(weather)
# print(list(weather.columns.values))

#Merging weather and flow datasets
weather = weather.rename(columns={"DATE": "Date"})
st_louis = pd.merge(st_louis_flow, weather, on=['Date'])

# Delete Unwanted columns
st_louis = st_louis[["Date", "Flow", "PRCP", "SNOW", "TMAX", "TMIN"]]

# print(st_louis)

print("-" * 50)
print("Dataset No. of Rows St. Louis: ", st_louis.shape[0])
print("Dataset No. of Columns St. Louis: ", st_louis.shape[1])
print("Columns of St. Louis: ", list(st_louis.columns.values))
print("-" * 50)
print(st_louis)
print(st_louis.describe())

# PRCP = Precipitation (tenths of mm)
# SNOW = Snowfall (mm)
# SNWD = Snow depth (mm)
# TMAX = Maximum temperature (tenths of degrees C)
# TMIN = Minimum temperature (tenths of degrees C)

# Output
st_louis.to_csv("./data/st_louis_preprocessed.csv")






