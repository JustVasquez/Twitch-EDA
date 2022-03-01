# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 15:52:47 2021

@author: jvasq
"""
# https://www.kaggle.com/aayushmishra1512/twitchdata


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from os import path 

# After importing our packages, we set the path from where we're calling our data

DATA_DIR = '/Users/jvasq/PythonProjects/Kaggle/Twitch Data'

tdf = pd.read_csv(path.join(DATA_DIR, 'twitchdata-update.csv'))

# We quickly examine the data we're working with 

tdf.head()
tdf.columns

tdf.columns = tdf.columns.map(lambda x: x.replace(' ','_'))

stats = tdf.describe()

cols = tdf.columns.tolist()

# Since 'Channel' names can have both upper and lowercase letters, we are 
# creating a new column with all letters lowercase, that way we don't have to 
# worry about entering the channel name correctly when calling it in our code

tdf['Channel_lower'] = tdf['Channel'].str.lower()

# Since the steam time is in minutes (and is a really high number for each channel)
# I converted them to hours to be a bit more comprehensible 

tdf['Stream_Time_hours'] = (tdf['Stream_time(minutes)']/60).astype(float)
tdf['Watch_Time_hours'] = (tdf['Watch_time(Minutes)']/60).astype(int)

tdf['Stream_Time_Monthly_Avg_Hours'] = tdf['Stream_Time_hours']/12
#tdf.drop(columns=['Stream Time (Monthly Average)'], inplace=True)

# I added these columns to calculate how many followers each channel gained 
# per views gained and hours streaming, respectively, to see quantify the
# growth between those relationships 

tdf['FG/VG'] = (tdf['Followers_gained']/tdf['Views_gained']).astype(float)
tdf['FG/STh'] = (tdf['Followers_gained']/tdf['Stream_Time_hours']).astype(float)

# Just listed a few of the channels I watch or am familiar with to see if any 
# of them are in the dataset 

#otv = tdf.loc[(tdf['Channel_lower'] == 'pokimane') | 
#              (tdf['Channel_lower'] == 'scarra') | 
#              (tdf['Channel_lower'] == 'lilypichu') | 
#              (tdf['Channel_lower'] == 'yvonnie') | 
#              (tdf['Channel_lower'] == 'disguisedtoast') | 
#              (tdf['Channel_lower'] == 'quarterjade') | 
#              (tdf['Channel_lower'] == 'masayoshi') | 
#              (tdf['Channel_lower'] == 'boxbox') | 
#              (tdf['Channel_lower'] == 'ludwig') | 
#              (tdf['Channel_lower'] == 'shiphtur') | 
#              (tdf['Channel_lower'] == 'natsumiii') | 
#              (tdf['Channel_lower'] == 'angelskimi') |
#              (tdf['Channel_lower'] == 'baboabe') |
#              (tdf['Channel_lower'] == 'fuslie') |
#              (tdf['Channel_lower'] == 'edisonparklive')]

# Instead of repeating the code for each channel I wanted to add, I realized 
# I could use the .isin method to pass a list to .loc to get the same information
# which also makes it easier to update when I want to add a new channel to the list

otvf = ['pokimane','scarra','lilypichu','yvonnie','disguisedtoast','quarterjade',
       'masayoshi','boxbox','ludwig','shiphtur','natsumiii','angelskimi',
       'baboabe','fuslie','edisonparklive','starsmitten','xchocobars','sykkuno']

otv = tdf.loc[tdf['Channel_lower'].isin(otvf)]

# Taking a look at the otv group's stream time in a bar plot

g = sns.barplot(x='Channel', y='Stream_Time_hours', data=otv, palette='Blues_d')
g.set_title('Stream Hours')

# I wanted to adjust the aspect (width) as the names were too close to each 
# other, so I changed it to .catplot which allows us to adjust the aspect.
# Might go back and look at how to adjust the barplot to reflect the same 

g = sns.catplot(kind='bar', x='Channel', y='Stream_Time_hours', data=otv, palette='Blues_d',
                aspect = 1.7)

# Created a scatter plot visualizing the relationship between the number of 
# views gained and the number of followers gained, separated by whether the 
# channel is partnered or not to see if more views leads to more followers. 
# A correlation plot may better quantify the relationship between the two.

f = sns.relplot(x='Views_gained', y='Followers_gained', col='Partnered', 
                data=tdf, aspect=0.7)

# This scatter plot shows the relationship between stream time and watch time, 
# separated by whether the channel is partnered, mainly to see that channels
# streaming more does or doesn't lead to more watch time 

s = sns.relplot(x='Stream_Time_hours', y='Watch_Time_hours', col='Partnered',
                hue='Mature', data=tdf, aspect=0.7)

#

r = sns.relplot(x='Stream_Time_hours', y='FG/VG', data=tdf, aspect=1.7, size='Followers')

# The corr plot/heat map visualizes the correlation between the different 
# variables. Instead of just passing through "tdf", we specify which columns 
# to calculate because it is not necessary to pass all columns, such as 
# 'Watch time(minutes)' and 'Watch Time (hours)' because they represent the 
# same information, just at different levels of granularity 

corr = tdf[['Watch_Time_hours','Stream_Time_hours','Peak_viewers',
            'Average_viewers','Followers','Followers_gained','Views_gained','Partnered','Mature']].corr()
cmap = sns.color_palette('dark:salmon_r')
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
c = sns.heatmap(corr, cmap=cmap, mask=mask, square=True, annot=True)

# Looking at the top 50 streamers below
# .iloc can take two parameters to index both axes, first for the index values
# of the observations we want to examine, the second is to call the indices of
# the columns we want 

df = tdf.iloc[:50, :] 
#df = tdf.iloc[:50]

# This line plot shows the relationship between stream time and followers 
# gained for the top 50 streamers 

plt.figure(figsize=(12,6))
sns.lineplot(df['Stream_Time_hours'], df['Followers_gained'], palette='Set1')
plt.title('Top 50: Streaming Time vs Followers Gained')
plt.show()

# This bar plot visualizes the number of channels per language within the 
# top 1000 Twitch channels, with the number of channels displayed above each bar

ax = plt.figure(figsize=(20,7))
tdf['Language'].value_counts().plot.bar()
for i, v in enumerate(tdf['Language'].value_counts()):
    plt.text(i-.12, v+3, str(v), color='gray',fontweight='bold')
plt.title('Channel Counts by Language')
plt.xlabel('Languages')
plt.ylabel('Count')
plt.show()

# column_max will return the max info for whichever channel has the max value
# for the column passed into the function

def column_max(x):
    return tdf.loc[tdf[x] == tdf[x].max()]
    
column_max('Followers')

column_max('Peak_viewers')

column_max('Watch_Time_hours')

# We import statsmodels ols to run a linear regression for some of the 
# variables in this dataset. Also we adjusted the column names in the dataset 
# because ols would not accept column names with spaces or with parentheses 

import statsmodels.formula.api as smf

# Here we set up the model and list the variables to be used in this regression
# and then fit the model and save as results

model = smf.ols(formula=
                """
                Followers ~ Peak_viewers + Average_viewers + Views_gained + Mature + 
                Partnered + Stream_Time_hours + Watch_Time_hours
                """, data=tdf)
results = model.fit()
results.summary2()

# Misc plots 
# Inspired by https://www.kaggle.com/bibiuwun/twitcheda

sns.catplot(data=tdf, x='Partnered', y='Followers_gained')

sns.catplot(data=tdf, x='Partnered', y='Followers')

mf = tdf['Followers_gained'].min()

tdf.loc[tdf['Followers_gained'] == mf]

sns.boxplot(x='Partnered', y='Followers_gained', data=tdf)

sns.violinplot(x='Mature', y='Followers', data=tdf)


sns.violinplot(x='Mature', y='Followers', data=df)





