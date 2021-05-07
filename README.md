# River-Flow-Forecast

## Data

There are two locations at which we will be forecasting river flow: St. Louis, MO and Clinton, IA. Luis will work with Clinton, IA, while Sam will work with St. Louis, MO.

For each location, there is a CSV file with flow data from USGS and 1-2 CSV files with weather data from NOAA. The flow data was downloaded through the R Markdown file 'flow_data.RMD', while the NOAA data was downloaded manually from NOAA's website (https://www.ncdc.noaa.gov/cdo-web/search). St. Louis has two weather data CSV's because one of the sites where weather data was recorded only went until 1968. So a supplemental site from a few miles away was chosed to fill in the data from 1969 - 2021.

We are using daily data from 1900-01-01 until 2021-04-06 for both locations. The mean daily flow is our target variable, while daily percipitation, daily snowfall, the maximum daily temperature, and the minimum daily temperature are our features. To reduce the size of the data, Luis and I both aggregate the data up to the weekly level.

## Code
Sam and Luis will write the pre-processing funcitons together in the same script to ensure that the variables are processed in the same way. Then, they will create individual files to run forecasting on their unique datasets.
