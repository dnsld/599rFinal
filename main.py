import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
import seaborn as sns
import plotly.express as px
import datetime as dt

sns.set()

# -------------- #
### FEMA DATASET ###
# loading dataset from Kaggle
orig_data = "~/Desktop/us_disaster_declarations.csv"   # UPDATE WITH FILEPATH
orig_df = pd.read_csv(orig_data,sep=',')

# transforming important field into date
orig_df['declaration_date'] = orig_df['declaration_date'].str[:10]
orig_df['declaration_date'] = orig_df['declaration_date'].astype('|S')
orig_df['declaration_date'] = orig_df['declaration_date'].apply(lambda x: pd.to_datetime(x[0]))

# confirming results
### print(orig_df['declaration_date'].head())

# analyzing only unique FEMA declared natural disasters
n_fema_unique = len(pd.unique(orig_df['fema_declaration_string']))
df2 = orig_df.drop_duplicates(subset=['fema_declaration_string'])
new_df_len = df2.shape[0]
### print(f"The original number of unique FEMA natural disasters was {n_fema_unique} and the new number of rows in the updated data frame are {new_df_len}")

# eliminating unnecessary columns
df2 = df2[['fema_declaration_string','disaster_number','state','incident_type','fy_declared','declaration_date','designated_area']].copy()
df2 = df2[~(df2['fy_declared'] <= 1981)]
### print(df2.dtypes)
# -------------- #


### ZILLOW DATA ###
# loading data from Zillow on single family home prices
housing_data_csv = "~/Desktop/Metro_mlp_uc_sfrcondo_sm_month.csv"
housing_data = pd.read_csv(housing_data_csv,sep=',')

# creating map to abbreviate full state names to abbreviations
us_state_to_abbrev = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "American Samoa": "AS",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR",
    "United States Minor Outlying Islands": "UM",
    "U.S. Virgin Islands": "VI",
}

# invert the dictionary
abbrev_to_us_state = dict(map(reversed, us_state_to_abbrev.items()))

# creating a dataframe with the abbreviations dictionary
lookup = pd.DataFrame([abbrev_to_us_state])
lookup = pd.Series(abbrev_to_us_state, name='State')
lookup.index.name = 'StateName'
lookup.reset_index()

# check for NANs in dataframe
# calculating percent missing values of rows for each column @@@@@@@@
percent_missing = round(housing_data.isnull().sum() * 100 / len(housing_data) , 2)
### print(percent_missing)

# descriptive Stats
# sub-setting month on month price columns
pricing_cols = housing_data.iloc[: ,5:-1]

# replacing missing values with 0 to make data ready for computation
pricing_cols_final = pricing_cols.fillna(0)

# verification for NANs in dataframe -final check     @@@@@@@@
percent_missing = round(pricing_cols_final.isnull().sum() * 100 / len(pricing_cols_final), 2)
### print(percent_missing)

state_cols = housing_data.iloc[:,0:5]
state_cols_final = state_cols[['RegionID','RegionName','StateName']]

# calculating the median of month on month house prices
pricing_cols_final['median'] = round(pricing_cols_final.median(axis=1), 0)

# concatenating the final dataframe to integrate state columns and median prices
zillow_dataset = result = pd.concat([state_cols_final, pricing_cols_final], axis=1)
zillow_dataset = zillow_dataset[['RegionID','RegionName','StateName','median']]
zillow_dataset = zillow_dataset.merge(lookup,on='StateName',how='left')

# final data set is ready to be used for further analysis
zillow_dataset = zillow_dataset [['RegionName','StateName','State','median']]
zillow_dataset = zillow_dataset.dropna()

# print(zillow_dataset.dtypes)
### ZILLOW DATA END ###


### GBS EMPLOYMENT DATA ###

#Uploading Employment Data from CSV exported from sql
#location_file_name = "Employment.csv" #name of exported SQL file    # UPDATE WITH FILEPATH
#location_df = pd.read_csv(location_file_name, sep=',') #import CSV as dataframe

#bar chart to show percentages of regions
#fig = px.bar(location_df,x='region', y='percent_of_class').update_xaxes(categoryorder='total descending')
#fig.show()


# -------------- #

### MERGING DATA ###

# creating FEMA reports by state
series_FEMA = df2.groupby('state', as_index=True)['state'].count()
state_FEMA = pd.DataFrame({'state':series_FEMA.index, 'count':series_FEMA.values})


# grouping housing prices by state
zs = zillow_dataset.groupby('StateName', as_index=True)['median'].mean()
state_housing = pd.DataFrame({'state':zs.index, 'median':zs.values})


# merging state housing price data and FEMA disasters and renaming columns
state_all = state_FEMA.merge(state_housing, on='state', how='outer')
state_all = state_all.dropna()
state_all = state_all.rename(columns={"count":"disasters","median":"median_home_price"})


# creating a dataframe for regression on incidents over time
reg_series = df2.groupby(['fy_declared', 'state']).size()
reg_df = reg_series.reset_index()
reg_df = reg_df.rename(columns={0:"disasters"})


# regression
X = reg_df[['fy_declared']]
Y = reg_df['disasters']

regr = linear_model.LinearRegression()
regr.fit(X, Y)
Y_pred = regr.predict(X)


# regions
# northwest, southwest, midwest, southeast, northeast, east_coast
# https://stackoverflow.com/questions/41189392/new-column-in-pandas-dataframe-based-on-existing-column-values

northwest = ['WA','OR','CA','NV','ID','MT','WY','UT']
southwest = ['AZ','NM','TX','OK','CO']
midwest = ['ND','SD','NE','KS','MN','IA','MO','IL','WI','MI','IN','OH']
southeast = ['AR','LA','MS','TN','KY','AL','GA','FL','SC','NC']
mid_atlantic = ['VA','WV','MD','DE','NJ','PA']
northeast = ['NY','PA','NH','VT','MA','RI','CT']

# -------------- #


### GRAPHS ###

# Housing EDA WORK

# graph to show median price by state
price = px.bar(state_housing,x='state', y='median').update_xaxes(categoryorder='total descending')
# price.show()

# FEMA EDA graphs
fig, axes = plt.subplots(2,2, sharex=False, sharey=False,  figsize=(15,15) )
fig.suptitle('EDA on FEMA Data')

# graph 1, disasters by year
sns.countplot(ax=axes[0,0],x='fy_declared',data=df2,palette="rocket")
axes[0,0].set_title('Disasters by Year')
axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(),rotation = 90)

# graph 2, disasters by state
sns.countplot(ax=axes[0,1],x='state',data=df2, palette="crest", order=df2['state'].value_counts().iloc[:15].index)
axes[0,1].set_title('Top 15 States w. Most Disasters')

# graph 3
sns.barplot(ax=axes[1, 0], x='state', y='median', order=state_housing.sort_values('median', ascending=False).state.iloc[:15], data=state_housing, palette="rocket")
axes[1, 0].set_title('Median Price by State')

# graph 4
sns.scatterplot(ax=axes[1, 1], x='disasters', y='median_home_price', data=state_all,legend=True,palette="light_palette")
axes[1, 1].set_title('Disasters x Home Price')

plt.show()
plt.clf()

# Disaster X Time regression
plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()
plt.clf()

# Employment by Region Graph
region = sns.barplot(x=location_df['region'], y=location_df['percent_of_class'], palette="rocket") #box plot
region.set_title('Percent of Class by Region')
plt.show()
