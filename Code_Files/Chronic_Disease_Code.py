#!/usr/bin/env python
# coding: utf-8

# ### Load Libraries

# In[1]:


import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime as dt
import re
import itertools
from dateutil.parser import parse
from datetime import datetime
import requests
import xml.etree.ElementTree as et
import io
import urllib.request
import json
import pymongo
import seaborn as sns


# ### JSON Loading and Formating

# In[2]:


'''

Please be aware that the following lines require significant computational resources due to the dataset 
containing over 1 million observations.

Time on Mac (RAM-64GB, 10-core CPU, 16-core GPU): 107.8347499370575/sec or ~1.8min

'''


# In[3]:


#%% get the data.gov link and connect to it
cdi_url = 'https://data.cdc.gov/api/views/g4ie-h725/rows.json?accessType=DOWNLOAD'

try:
    response = urllib.request.urlopen(cdi_url)
except urllib.error.URLError as e:
    if hasattr(e, 'reason'):
        print('Failed to establish a connection to the server.')
        print('Reason: ', e.reason)
    elif hasattr(e, 'code'):
        print('The server couldn\'t fulfill the request.')
        print('Error code: ', e.code)
else:
    # convert to string if successful
    json_string = response.read().decode('utf-8')

# the json package loads() converts the string to python dictionaries and lists
eq_json = json.loads(json_string)


# In[4]:


#%% function that takes json data and prints out the hierarchy of the json structure
def json_layers(data, prefix=""):
    if isinstance(data, dict):  # Check if the data is a dict
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            print(new_prefix)
            json_layers(value, prefix=new_prefix)
    elif isinstance(data, list):  # Check if the data is a list
        for item in data:
            json_layers(item, prefix=prefix)
# takes in json.loads() object
json_layers(eq_json)


# In[5]:


#%% create columns list of columns collection
columns = eq_json['meta']['view']['columns']

# loop through columns to get index values for fieldnames
for index, column in enumerate(columns, start=0):
    print(f'{index}. {column["fieldName"]}')


# In[6]:


#%% store only relevant fieldnames in a list
field_names = [col['fieldName'] for col in columns[8:42]]
len(field_names)


# In[7]:


#%% create list from data nested key 'data'
data = eq_json['data']
#json.dumps(data, indent=2)

# use list comprehension to extract only the elements needed to create df
data_rows = [list[8:-2] for list in data]

# create df from json data
cdi_df = pd.DataFrame(data_rows, columns=field_names)


# In[8]:


cdi_df.head()


# ### Exploratory Data Analysis (EDA)

# In[9]:


cdi_df.shape


# In[10]:


cdi_df.info()


# In[11]:


# check for inf values
# returns false if none found - remove second .any() to see results per attribute
cdi_df.isin([np.inf, -np.inf]).any().any() 


# In[12]:


# check for nans
na_cnt = cdi_df.agg(lambda x: (int(x.isna().sum()), '{:.2f}'.format((x.isna().sum() /
                        len(x))*100)))
# use transpose to move row to header and header to row
na_df = na_cnt.transpose().sort_values(by=0, ascending=False)

# add column names
na_df.columns = ['Total NAs', 'Percent']
na_df


# In[13]:


# remove attributes that contain all nan values
cdi_df.drop(cdi_df.columns[[7,18,19,20,21,23,30,31,32,33]], axis=1, inplace=True)


# In[14]:


# drop rows where no data is available
cdi_df.drop(cdi_df[cdi_df['datavaluefootnote'] == 'No data available'].index, inplace=True)


# In[15]:


# confirm removal
cdi_df.info()


# In[16]:


# convert yearstart/end to numeric
cdi_df[['yearstart', 'yearend']] = cdi_df[['yearstart', 'yearend']].apply(
    lambda x: pd.to_numeric(x))


# ### Feature Engineering

# In[17]:


#%% create new column with year range bins
cdi_df['year_bins']= cdi_df.apply(lambda row: '5yr' if row['yearend'] - row['yearstart'] == 4
                          else '3yr' if row['yearend'] - row['yearstart'] == 2
                          else '2yr' if row['yearend'] - row['yearstart'] == 1
                          else '1yr', axis=1)

# create bin column object
col = cdi_df['year_bins']
# remove col from cdi_df
cdi_df.pop('year_bins')
# add bin column back to cdi_df in 3rd col position
cdi_df.insert(2, col.name, col)


# In[18]:


cdi_df.sample(n=10)


# In[19]:


# %% create us regions attribute
regions_to_states = {
    'South': ['West Virginia', 'District of Columbia', 'Maryland', 'Virginia',
              'Kentucky', 'Tennessee', 'North Carolina', 'Mississippi',
              'Arkansas', 'Louisiana', 'Alabama', 'Georgia', 'South Carolina',
              'Florida', 'Delaware'],
    'Southwest': ['Arizona', 'New Mexico', 'Oklahoma', 'Texas'],
    'West': ['Washington', 'Oregon', 'California', 'Nevada', 'Idaho', 'Montana',
             'Wyoming', 'Utah', 'Colorado', 'Alaska', 'Hawaii'],
    'Midwest': ['North Dakota', 'South Dakota', 'Nebraska', 'Kansas', 'Minnesota',
                'Iowa', 'Missouri', 'Wisconsin', 'Illinois', 'Michigan', 'Indiana',
                'Ohio'],
    'Northeast': ['Maine', 'Vermont', 'New York', 'New Hampshire', 'Massachusetts',
                  'Rhode Island', 'Connecticut', 'New Jersey', 'Pennsylvania']
}

# map regions dict to df
cdi_df['us_regions'] = cdi_df['locationdesc'].map({state: region for region, states in regions_to_states.items()
                                                   for state in states})


# ### Analysis

# In[20]:


# %% ETHNICITY - Mortality
# 2010 - 2021 for all US based on ethnicity for cancer mortality's
cancer_df = cdi_df.loc[(~cdi_df['stratification1'].isin(['Overall', 'Male', 'Female'])) &
                       (cdi_df['datavaluetype'] == 'Average Annual Number') &
                       (cdi_df['datavaluealt'].notna()) &
                       (cdi_df['locationabbr'] == 'US') &
                       # switch between mortality and incidence
                       (cdi_df['question'] == 'Invasive cancer (all sites combined), mortality')]

# average annual number of cancer (all sites combined,mortality) by ethnicity
# Invasive cancer (all sites combined), mortality
pivot_df = cancer_df.pivot(index='stratification1',
                           columns='yearend',
                           values='datavaluealt')

pivot_df = pivot_df.astype('int')
#pivot_df = pivot_df.applymap('{:,.0f}'.format)


# In[21]:


# %% graph mortality rates based on ethnicity

sns.set_style("darkgrid")

# transpose the pivoted data
df_t = pivot_df.transpose()

# List of columns to plot
# col_plot = cancer_df['stratification1'].unique().tolist()

# Create a figure and subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))

# removes last subplot
fig.delaxes(axes[2, 1])

# flatten axes - unpacks the subplot arrays
axes = axes.flatten()[:5]


# Create a formatter for the y-axis values
def y_format(x, pos):
    if x >= 1000000:
        return f'{x / 1000000:.2f}M'
    else:
        return f'{x / 1000:.1f}k'


y_formatter = ticker.FuncFormatter(y_format)

# Iterate over the columns and plot each series in a separate subplot
for i, column in enumerate(df_t.columns):
    ax = axes[i]  # Get the current subplot
    df_t[column].plot(marker='o', ax=ax, color='blue')  # Plot the specific column

    # Customize the subplot title and labels
    ax.set_title(f'Cancer Mortality Rates - {column}')
    ax.set_xlabel('Year')
    ax.set_ylabel('Rolling 5yr Avg')

    # Apply the y-axis formatter
    ax.yaxis.set_major_formatter(y_format)

# Adjust spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()


# In[22]:


# %% GENDER
# 2010 - 2021 for all US based on gender for cancer mortality's
gender_df = cdi_df.loc[(cdi_df['stratification1'].isin(['Male', 'Female'])) &
                       (cdi_df['datavaluetype'] == 'Average Annual Number') &
                       (cdi_df['datavaluealt'].notna()) &
                       (cdi_df['locationabbr'] == 'US') &
                       # switch between mortality and incidence
                       (cdi_df['question'] == 'Invasive cancer (all sites combined), mortality')]

# average annual number of cancer (all sites combined,mortality) by ethnicity
# Invasive cancer (all sites combined), mortality
pivot_df = gender_df.pivot(index='yearend',
                           columns='stratification1',
                           values='datavaluealt')

pivot_df = pivot_df.astype('int')


# In[23]:


# %% GENDER W/SUBPLOTS
sns.set_style("darkgrid")

# transpose the pivoted data
df_t = pivot_df.transpose()

# Create a figure and subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

# removes last subplot
# fig.delaxes(axes[2,1])

# flatten axes - unpacks the subplot arrays
axes = axes.flatten()[:2]


def y_format(x, pos):
    if x >= 1000000:
        return f'{x / 1000000:.2f}M'
    else:
        return f'{x / 1000:.1f}k'


y_formatter = ticker.FuncFormatter(y_format)

# Iterate over the columns and plot each series in a separate subplot
for i, column in enumerate(pivot_df.columns):
    ax = axes[i]  # Get the current subplot
    pivot_df[column].plot(marker='o', ax=ax, color='blue')  # Plot the specific column

    # Customize the subplot title and labels
    ax.set_title(f'Cancer Mortality Rates - {column}')
    ax.set_xlabel('Year')
    ax.set_ylabel('Rolling 5yr Avg')

    # Apply the y-axis formatter
    ax.yaxis.set_major_formatter(y_format)

# Adjust spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()


# #### Andrew's Portion of the Project 

# In[24]:


#Answering question: What percentage of current US smokers have attempted quitting?

#Narrowing down the data frame specific to the question.
quit_attempts = cdi_df[cdi_df['question'].str.contains('Quit attempts in the past year among current smokers')]
quit_attempts = quit_attempts[quit_attempts['stratification1'].str.contains('Overall')] 
quit_attempts.loc[:, 'datavalue'] = pd.to_numeric(quit_attempts['datavalue'], errors='coerce')
quit_attempts = quit_attempts[quit_attempts['locationabbr'] == 'US']
quit_attempts


# In[25]:


#Graph showing how many current smokers attempt quitting every year. 

quitting_by_year = quit_attempts.groupby('yearstart')['datavalue'].mean()
fig, ax = plt.subplots()
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.bar(quitting_by_year.index, quitting_by_year.values, color='red')
ax.set_xlabel('Year')
ax.set_ylabel('Percentage Who Attempted Quitting Smoking')
ax.set_title('Quit Attempt Percentages by Year Among Current Smokers in the US')
plt.show()


# In[26]:


#Answering question: What percentage of US adults smoke?

#Narrowing down the data frame specific to the question.
adult_smokers = cdi_df[cdi_df['question'].str.contains('Current smoking among adults aged >= 18 years')]
adult_smokers = adult_smokers[adult_smokers['datavaluetype'].str.contains('Crude Prevalence')]
adult_smokers = adult_smokers[adult_smokers['stratification1'].str.contains('Overall')]
adult_smokers = adult_smokers[adult_smokers['locationabbr'] == 'US']
adult_smokers


# In[27]:


#Shows the change in smoking rates for the whole US year-by-year.

adult_smokers.loc[:, 'datavalue'] = pd.to_numeric(adult_smokers['datavalue'], errors='coerce')
smokers_by_year = adult_smokers.groupby('yearstart')['datavalue'].mean()
fig, ax = plt.subplots()
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.bar(smokers_by_year.index, smokers_by_year.values, color='blue')
ax.set_xlabel('Year')
ax.set_ylabel('Percentage of US Smoking Population')
ax.set_title('Crude Prevalence of US Smoking Population')
plt.show()


# In[28]:


#Answering question: What percentage of US states have strong anti-smoking laws?

#Narrowing down the data frame specific to the question.
states_with_control_laws = cdi_df[cdi_df['question'].str.contains('States that allow stronger local tobacco control and prevention laws')]
states_with_control_laws = states_with_control_laws[states_with_control_laws['datavaluetype'].str.contains('Yes/No')]
states_with_control_laws = states_with_control_laws[states_with_control_laws['stratification1'].str.contains('Overall')]
states_with_control_laws


# In[29]:


#Pie chart that shows percentage of states with strong tobacco control laws.

value_counts = states_with_control_laws['datavalue'].value_counts()
plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
plt.title('States That Allow Stronger Local Tobacco Control and Prevention Laws')
plt.axis('equal')
plt.show()


# In[30]:


#Answering question: Which US states smoke the most?

#Narrowing down the data frame specific to the question.
cig_sales = cdi_df[cdi_df['question'].str.contains('Sale of cigarette packs')]
cig_sales = cig_sales[cig_sales['datavaluetype'].str.contains('Number')]
cig_sales = cig_sales[cig_sales['stratification1'].str.contains('Overall')]
cig_sales


# In[31]:


#Bar chart that shows the states with the heaviest smokers.

cig_sales['datavalue'] = cig_sales['datavalue'].str.replace(',', '').str.replace('.', '').astype(float)
grouped_data = cig_sales.groupby('locationabbr')['datavalue'].mean().reset_index()
fig, ax = plt.subplots(figsize=(18, 9))
ax.grid(axis='y', linestyle='--', alpha=1.0)
ax.bar(grouped_data['locationabbr'], grouped_data['datavalue'], color='#2ca02c')
ax.set_title('Total Average Per Capita Cigarette Pack Sales by State')
ax.set_xlabel('State')
ax.set_ylabel('Average Per Capita Sales')


# In[32]:


#Answering question: What percentage of young people in the US currently smoke?

#Narrowing down the data frame specific to the question.
young_smokers = cdi_df[cdi_df['question'].str.contains('Current cigarette smoking among youth')]
young_smokers = young_smokers[young_smokers['datavaluetype'].str.contains('Crude Prevalence')]
young_smokers = young_smokers[young_smokers['stratification1'].str.contains('Overall')]
young_smokers = young_smokers[young_smokers['locationabbr'] == 'US']
young_smokers


# In[33]:


#Bar chart that shows how smoking percentages have changed among youth every other year.

youth_smoking_by_year = young_smokers.groupby('yearstart')['datavalue'].mean()
fig, ax = plt.subplots()
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.bar(youth_smoking_by_year.index, youth_smoking_by_year.values, color='#ff7f0e')
ax.set_xlabel('Year')
ax.set_ylabel('Percent of Youth that Smoke')
ax.set_title('Crude Prevalence of Cigarette Smoking Percentage Among Youth')
plt.show()


# In[34]:


# Bill Steel.  Date: 5/16/2023-5/26/2023
# Home Work 2

#PLEASE NOTE:  MONGODB MUST BE RUNNING FOR THE LAST PORTION OF ANALYSIS

'''The following portion of our HW2 program reads in a JSON data set that looks at the causes of death, number of deaths,
and the number of deaths per 100K people from 1999 through 2017 by State (including Washington DC and the Unites States).

The program itself performs the following:
1.  Reads in the JSON data set
2.  Leverages a function written by Ryan Summers that shows the various levels within the dataset
3.  Selects the relevant data from the JSON data (colnames and data values are in 2 different places)
4.  Cleanses the data (colname cleanup, conversion of strings to numbers).  This leaves 10868 rows by 5 columns.
5.  Analysis based on the list of questions below
6.  The last part of the program loads the data into MongoDB and reads from it for the final analysis.

During the analyses the questions being asked in this program:
Question 1: How many Deaths by Year/State are there?  How about deaths per 100K people by state?
Question 2: What are the leading causes of death over this timeframe?
Question 3: What states have the highest deaths per 100K?
Question 4: How have US deaths and deaths_per_100K changed over this period?
Question 5: Is there anything different in the cause profiles between the states with the highest/lowest deaths per 100K?

'''

#Library Imports
import urllib.request
import json
import pymongo
import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt


# In[35]:


# getting json_string from the cdc url
cdc_url = "https://data.cdc.gov/api/views/bi63-dtpu/rows.json"

try:
    response = urllib.request.urlopen(cdc_url)
except urllib.error.URLError as e:
    if hasattr(e, 'reason'):
        print('We failed to reach a server.')
        print('Reason: ', e.reason)
    elif hasattr(e, 'code'):
        print('The server couldn\'t fulfill the request.')
        print('Error code: ', e.code)
else:
    # the url request was successful - convert the response to a string
    json_string = response.read().decode('utf-8')


# In[36]:


#%% Ryan's function that takes json data and prints out the levels within a json structure to help understand it

def json_layers(data, prefix=""):
    if isinstance(data, dict):  # Check if the data is a dict
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            print(new_prefix)
            json_layers(value, prefix=new_prefix)
    elif isinstance(data, list):  # Check if the data is a list
        for item in data:
            json_layers(item, prefix=prefix)
cdc_json = json.loads(json_string)
print(json_layers(cdc_json))          


# In[37]:


# the json package loads() converts the string to python dictionaries and lists

#getting the column names from the json metadata
col_names = pd.json_normalize(cdc_json, record_path =['meta','view','columns'])
col_names=col_names['name']
#print(col_names)

#grabbing the actual data into a pandas dataframe
df = pd.json_normalize(cdc_json,'data')

#selecting the columns of interest
cols = [8,10,11,12,13] 
df = df[df.columns[cols]]

#adding the column headers to the updated dataframe (df) and changing columns to numbers for grouping/analysis
df.columns=[col_names[8],col_names[10],col_names[11],col_names[12],col_names[13]]
df.Deaths = pd.to_numeric(df.Deaths, errors='coerce')
df = df.rename(columns={'Age-adjusted Death Rate': 'Deaths_per_100K'})
df.Deaths_per_100K=pd.to_numeric(df.Deaths_per_100K,errors='coerce')

#The cleansed data frame against which analyses will be performed
df


# In[38]:


#Question 1:  How many Deaths by Year/State are there?  How about deaths per 100K people by state?
grouped_df=df.groupby(['Year','State'], as_index=False)[['Deaths_per_100K','Deaths']].sum()
grouped_df


# In[39]:


#Question 2: What are the leading causes of death over this timeframe?
grouped_df_cause=df.groupby(['Cause Name'],as_index=False)[['Deaths']].sum()
grouped_df_cause=grouped_df_cause.loc[1:10]
grouped_df_cause['Deaths']=grouped_df_cause['Deaths']/1000000
grouped_df_cause.columns=['Cause Name','Deaths (in millions)']
grouped_df_cause.set_index('Cause Name', inplace=True)
grouped_df_cause


# In[40]:


#visualition of deaths by cause
grouped_df_cause.plot(kind='bar')
plt.title("US deaths by cause from 1999-2017")


# In[41]:


#Question 3: What states have the highest deaths per 100K?
grouped_df_state=df.groupby(['State'],as_index=False)[['Deaths_per_100K']].sum()
grouped_df_state['Deaths_per_100K']=grouped_df_state['Deaths_per_100K']/19 #average over 19 years
grouped_df_state.set_index('State', inplace=True)
grouped_df_state=grouped_df_state.sort_values(by=['Deaths_per_100K'],ascending=False)
grouped_df_state


# In[42]:


#visualition of deaths per 100K people by state (sorted)

values = grouped_df_state['Deaths_per_100K']
idx = grouped_df_state.index
fig = plt.figure(figsize=(10, 5))
clrs = ['orange' if (x == grouped_df_state.at['United States','Deaths_per_100K']) else 'cornflowerblue' for x in values ]
plt.bar(idx, values, color=clrs, width=0.4) 
plt.tick_params("x",rotation=90)
plt.rc('xtick', labelsize=10) 
plt.title("Deaths per 100,000 people by State")
plt.show()


# In[43]:


grouped_df_US=grouped_df[(grouped_df['State'] == 'United States')]
grouped_df_US


# In[44]:


grouped_df_US=grouped_df[(grouped_df['State'] == 'United States')]
grouped_df_US.set_index('Year', inplace=True)
grouped_df_US


# In[45]:


#Question 4: How have US deaths and deaths_per_100K changed over this period?

#get the percent differenct from 1999 to 2017
percent_decrease=((grouped_df_US.at['1999','Deaths_per_100K']
                  -grouped_df_US.at['2017','Deaths_per_100K'])/grouped_df_US.at['1999','Deaths_per_100K'])*100
print('In the US, Deaths per 100K people have decreased {:.2f}% from 1999-2017'.format(percent_decrease))
print('This has happened as the population has grown and raw numbers of deaths have increased')

#plot 1 (grouped by deaths)
grouped_df_US=grouped_df[(grouped_df['State'] == 'United States')]
grouped_df_US.set_index('Year', inplace=True)
grouped_df_US_plot=grouped_df_US.drop(['Deaths'], axis=1)

#plot 2 (grouped by deaths_per_100K)
grouped_df_US=grouped_df[(grouped_df['State'] == 'United States')]
grouped_df_US.set_index('Year', inplace=True)
grouped_df_US_plot1=grouped_df_US.drop(['Deaths_per_100K'],axis=1)

fig, ax1 = plt.subplots()
t=grouped_df_US_plot.index
s1=grouped_df_US_plot1[:]['Deaths']/1000000
ax1.plot(t, s1, 'b-')
ax1.set_xlabel('Year')

# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('US deaths (in millions)', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
s2=grouped_df_US_plot[:]['Deaths_per_100K']
ax2.plot(t, s2, 'r-')
ax2.set_ylabel('US Deaths per 100K people', color='r')
ax2.tick_params('y', colors='r')

ax1.tick_params("x", rotation=75)
plt.show()


# In[46]:


# Connection to Mongo DB
try:
    client=pymongo.MongoClient('localhost', 27017)
    print ("Connected successfully!!!")
except pymongo.errors.ConnectionFailure as e:
   print ("Could not connect to MongoDB: %s" % e )
else:
    # use database named cdc_db or create it if not there already
    cdc_db = client.cdc
    # create collection named causes or create it if not there already
    try:
        cdc_coll.drop() #dropping to ensure I don't duplicate the same documents
    except:
        cdc_coll = cdc_db.causes
        # add all the entries from the panda dataframe to the cdc collection
        cdc_coll.insert_many(df.to_dict('records'))
        print("Added", len(df), "to causes collection in cdc database")
    else:
        cdc_coll = cdc_db.causes
        # add all the entries from the panda dataframe to the cdc collection
        cdc_coll.insert_many(df.to_dict('records'))
        print("Added", len(df), "to causes collection in cdc database")


# In[47]:


# Grab back the collection of CDC records from MongoDB
cdc_docs = cdc_coll.find()

# Loop through and print out California documents
cal_cdc=[]
haw_cdc=[]
mis_cdc=[]
for doc in cdc_docs:
    if doc["State"]=="California":
        cal_cdc.append(doc)
    if doc['State']=='Hawaii':
        haw_cdc.append(doc)
    if doc['State']=='Mississippi':
        mis_cdc.append(doc)

california_df=pd.DataFrame(cal_cdc)
hawaii_df=pd.DataFrame(cal_cdc)
mississippi_df=pd.DataFrame(cal_cdc)

# close the database connection
client.close()


# In[48]:


#Question 5: What are the leading causes of death in california over this timeframe (most populous state)?

california_df=pd.DataFrame(cal_cdc)
california_df=california_df.drop(['_id'],axis=1)

california_df.Deaths = pd.to_numeric(california_df.Deaths, errors='coerce')
california_df = california_df.rename(columns={'Age-adjusted Death Rate': 'Deaths_per_100K'})
california_df.Deaths_per_100K=pd.to_numeric(california_df.Deaths_per_100K,errors='coerce')

grouped_cal_df_cause=california_df.groupby(['Cause Name'],as_index=False)[['Deaths']].sum()
grouped_cal_df_cause=grouped_cal_df_cause.loc[1:10]
grouped_cal_df_cause['Deaths']=grouped_cal_df_cause['Deaths']/1000000
grouped_cal_df_cause.columns=['Cause Name','Deaths (in millions)']
grouped_cal_df_cause.set_index('Cause Name', inplace=True)
grouped_cal_df_cause


# In[49]:


#visualition of deaths by cause
grouped_cal_df_cause.plot(kind='bar')
plt.title("California deaths by cause from 1999-2017")


# In[50]:


#Question 5: What are the leading causes of death in hawaii over this timeframe (least deaths/100K state)?

hawaii_df=pd.DataFrame(haw_cdc)
hawaii_df=hawaii_df.drop(['_id'],axis=1)

hawaii_df.Deaths = pd.to_numeric(hawaii_df.Deaths, errors='coerce')
hawaii_df = hawaii_df.rename(columns={'Age-adjusted Death Rate': 'Deaths_per_100K'})
hawaii_df.Deaths_per_100K=pd.to_numeric(hawaii_df.Deaths_per_100K,errors='coerce')

grouped_haw_df_cause=hawaii_df.groupby(['Cause Name'],as_index=False)[['Deaths']].sum()
grouped_haw_df_cause=grouped_haw_df_cause.loc[1:10]
grouped_haw_df_cause['Deaths']=grouped_haw_df_cause['Deaths']/1000000
grouped_haw_df_cause.columns=['Cause Name','Deaths (in millions)']
grouped_haw_df_cause.set_index('Cause Name', inplace=True)
grouped_haw_df_cause


# In[51]:


#visualition of deaths by cause
grouped_haw_df_cause.plot(kind='bar')
plt.title("Hawaii deaths by cause from 1999-2017")


# In[52]:


#Question 5: What are the leading causes of death in Mississippi over this timeframe (most deaths/100K state)?

mississippi_df=pd.DataFrame(mis_cdc)
mississippi_df=mississippi_df.drop(['_id'],axis=1)

mississippi_df.Deaths = pd.to_numeric(mississippi_df.Deaths, errors='coerce')
mississippi_df = mississippi_df.rename(columns={'Age-adjusted Death Rate': 'Deaths_per_100K'})
mississippi_df.Deaths_per_100K=pd.to_numeric(mississippi_df.Deaths_per_100K,errors='coerce')

grouped_mis_df_cause=mississippi_df.groupby(['Cause Name'],as_index=False)[['Deaths']].sum()
grouped_mis_df_cause=grouped_mis_df_cause.loc[1:10]
grouped_mis_df_cause['Deaths']=grouped_mis_df_cause['Deaths']/1000000
grouped_mis_df_cause.columns=['Cause Name','Deaths (in millions)']
grouped_mis_df_cause.set_index('Cause Name', inplace=True)
grouped_mis_df_cause


# In[53]:


#visualition of deaths by cause
grouped_mis_df_cause.plot(kind='bar')
plt.title("Mississippi deaths by cause from 1999-2017")


# In[ ]:




