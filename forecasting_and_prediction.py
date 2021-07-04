#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install seaborn --upgrade


# # Project: Forecast and Predictions
# 

# In[2]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math


# Imported all required libraries

# ## Project Goal 

# The gym chain Model Fitness is developing a customer interaction strategy based on analytical data. One of the most common problems gyms and other services face is customer churn. How do you know if a customer is no longer with you? For a gym, it makes sense to say a customer has left if they don't come for a month. In order to fight churn, Model Fitness has digitized a number of its customer profiles. Your task is to analyze them and come up with a customer retention strategy.

# ## Data Pre-processing and EDA 

# ###  Look at the dataset: does it contain any missing features? Study the mean values and standard deviation. 

# In[3]:


df = pd.read_csv('/datasets/gym_churn_us.csv')
print(df.info())
print(df.describe())
df.head()


# Read the data file the table contents are:
# •	'Churn' — the fact of churn for the month in question
# •	Current dataset fields:
# User data for the preceding month
# 'gender'
# 'Near_Location' — whether the user lives or works in the neighborhood where the gym is located
# 'Partner' — whether the user is an employee of a partner company (the gym has partner companies whose employees get discounts; in those cases the gym stores information on customers' employers)
# Promo_friends — whether the user originally signed up through a "bring a friend" offer (they used a friend's promo code when paying for their first membership)
# 'Phone' — whether the user provided their phone number
# 'Age'
# 'Lifetime' — the time (in months) since the customer first came to the gym
# •	Data from the log of visits and purchases and data on current membership status
# 'Contract_period' — 1 month, 3 months, 6 months, or 1 year
# 'Month_to_end_contract' — the months remaining until the contract expires
# 'Group_visits' — whether the user takes part in group sessions
# 'Avg_class_frequency_total' — average frequency of visits per week over the customer's lifetime
# 'Avg_class_frequency_current_month' — average frequency of visits per week over the preceding month
# 'Avg_additional_charges_total' — the total amount of money spent on other gym services: cafe, athletic goods, cosmetics, massages, etc.
# 
# Data Info:
# It shows the data type and column name
# 
# Data Describe:
# Studying the descibe method showed that there were no missing values (like NaN).
# 
# It also showed that there were more men than women in the dataset, most customers lived near a gym location, an almost equal amount of customers are employees of a partner company and are not employees, most customers were not recommended with a promotional code from a friend, the dataset contains almost all of the phone numbers for customers, more customers do not attend group sessions, the average number of months a customer has had a gym contract is almost 5 months, most customers are around 30 years old, most customers have about 4 months remaining on their contract, most customers have only had a contract for the past 4 months, the average umber of customers come almost twice a week and the churn rate leans towards 0 which means that most customers did not churn (did not stop subscribing).

# In[4]:


df.columns = ['gender', 'near_location', 'employer_partner', 'friend_promo', 'phone_exists', 'total_contract_period_months', 
               'group_sessions', 'age', 'other_services_total_dollars', 'contract_remaining_months', 'lifetime_months',
               'avg_visits_per_week_total', 'avg_visits_per_week_last_month', 'churn']


# Here we changed columns names in lower case to avoid further confusion.

# In[5]:


df.head()


# ### Look at the mean feature values in two groups: for those who left (churn) and for those who stayed 

# In[6]:


churn_data = df.groupby('churn').mean()
churn_data


# 1. To look at the mean feature values in the groups of those who left (did not renew membership) and those who stayed (did renew membership), I grouped the data by churn rate (where a churn rate of 0 means did not churn and a churn rate of 1 means did churn) and calculated the mean value for each feature.
# 2. This data grouping shows that an equal amount of male and female customers decide to renew or not renew.
# 3. It shows that most of the people who do not renew do not live near a gym location, are not employees of a partner company and were not given a promotional code by a friend.
# 4. Those who decide not to renew their membership had a 6 month long contract, did attend group sessions, were almost 30 years old, spent over 150 dollars on additional gym services, had about 5 months remaining on the contract, had been a member for almost 5 months and visit about twice a week.
# 5. Those who decided to renew their membership had an almost 2 month long contract, did not attend group sessions, were about 27 years old, spent less than 150 dollars on additional gym services, had about 1.5 months remaining in the contract, had been a member for 1 month and visit the gym an average of once a week.

# ###  Plot bar histograms and feature distributions for those who left (churn) and those who stayed 

# In[7]:


user_churned = df[df['churn'] == 1]
user_not_churned = df[df['churn'] == 0]


# Here we sliced data for churn(subscription stop) and not churn(continue to churn)

# In[8]:


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# Here we defined funcion a method to label bar graphs below

# In[9]:


binary_columns = ['gender','near_location', 'employer_partner', 'friend_promo', 'phone_exists', 'group_sessions']


# Here we took column names which has information in binary form such as 0 and 1

# In[10]:


for col in binary_columns:
    
    # create subplit
    fig = plt.figure(figsize=(5,5))
    ax = fig.subplots()
    
    # define x axis labels based on feature
    if col == 'gender':
        x = ['Male', 'Female']
        labels = ['Male', 'Female']

    if col in range(1,5):
        x = ['Yes', 'No']
        labels = ['Yes', 'No']
        
     
    # create array of churned data
    y_churned = [user_churned[user_churned[col] == 1][col].count(),
           user_churned[user_churned[col] == 0][col].count()]

    # create array of unchurned data
    y_not_churned = [user_not_churned[user_not_churned[col] == 1][col].count(),
           user_not_churned[user_not_churned[col] == 0][col].count()]
    
    # the label locations
    x = np.arange(len(labels)) 
    
    # the width of the bars
    width = 0.35 

    # plot bar graphs
    rects1 = ax.bar(x - width/2, y_churned, width, color = 'red', label='Churn')
    rects2 = ax.bar(x + width/2, y_not_churned, width, color='green',label='No Churn')

    # edit labels and formalities
    ax.set_ylabel('Count')
    ax.set_title(col)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)

    # add number
    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()


# 1. To plot the features by those who churned (stopped subscribing) and those who did not churn (continued subscribing), I started by plotting the features with binary values on bar graphs. For each feature, I graphed the number of values for both churn and non churn.
# 
# 2. For gender, the graph indicates that a very similar number of males and females churned and did not churn. Meaning there is no correlation between gender and churning.
# 
# 3. For near_location (whether the customer lives near a gym location or not), there is a large amount customers who did not churn and who live near a gym location. This makes sense: if you live nearby a gym, it is easier to get to, it is more of an incentive to stay. It is aso interesting to note that for those customers who do not live near a gym location, the churn and non churn count is low which might just mean that few people sign up for a gym membership who do not live near a gym location.
# 
# 4. For employer_partner (whether the customer is an employee of a company that the gym partners with or not), there is a large amount of customers who do not churn for both customers who are and are not employees. This could indicate that being an employee of a partner company does not have much of an affect on customers. However, almost double of users who churned were not employees of a partner company which shows that this feature has an impact on customer churn.
# 
# 5. For friend_promo (whether the customer joined the gym because of a promotion they got from a friend), where there is a very large number of users who did not churn and did not get a promo code from a friend. But there is also a significant number of users who had a promo code and did not churn - meaning a promo code is an incentive to stay. Also, there is about 5 times the number of customers who churned without a promo code than there is customers who churned with a promo code. All in all, having a promo code from a friend does have an impact on churn.
# 
# 6. For phone_exists (whether there is a customer phone number on file), it shows that they have the phone number of almost all of customers who subscribe to the gym and that more than double of customers whose phone nuber is on record did not churn. This is probably because the gym reached out to them in an effort to remind them to renew their subscription or convince them to do so.
# 
# 7. For group_sessions (whether the customer participated in group sessions), it shows that there is an equal number of those who did not churn and those who did and did not attend group sessions. It is notable that customers are more likely to churn (by double) if they do not attend group sessions.

# In[11]:


# columns with numerical values = FEATURE DISTRIBUTIONS
cols = ['total_contract_period_months', 'age', 'other_services_total_dollars', 'contract_remaining_months', 
        'lifetime_months', 'avg_visits_per_week_total', 'avg_visits_per_week_last_month']

fig = make_subplots(rows=7, cols=1, subplot_titles=('Age',
                                                    'Total Contract Period (Months)',
                                                    'Contract Remaining (Months)',
                                                    'Lifetime (Months)', 
                                                    'Other Services Total (Dollars)',
                                                    'Average Visits Per Week (Total)',
                                                    'Average Visits Per Week (Last Month)'
                                                   ))
# age
fig.add_trace(go.Histogram(
    x = user_churned['age'],
    name = 'Churn',
    histnorm = 'density',
    marker_color='red',
    showlegend=True),row=1, col=1)
fig.add_trace(go.Histogram(
    x = user_not_churned['age'],
    name= 'No churn',
    histnorm = 'density',
    marker_color='green',
    showlegend=True),row=1, col=1)

# total_contract_period_months
fig.add_trace(go.Bar(
    x=[1,6,12],
    y=[user_churned[user_churned['total_contract_period_months'] == 1]['total_contract_period_months'].count(),
       user_churned[user_churned['total_contract_period_months'] == 6]['total_contract_period_months'].count(),
       user_churned[user_churned['total_contract_period_months'] == 12]['total_contract_period_months'].count()],
    name='Churn',
    marker_color='red', showlegend=False), row=2, col=1)
fig.add_trace(go.Bar(
    x=[1,6,12],
    y=[user_not_churned[user_not_churned['total_contract_period_months'] == 1]['total_contract_period_months'].count(),
       user_not_churned[user_not_churned['total_contract_period_months'] == 6]['total_contract_period_months'].count(),
       user_not_churned[user_not_churned['total_contract_period_months'] == 12]['total_contract_period_months'].count()],
    name='No churn',
    marker_color='green', showlegend=False), row=2, col=1)

# contract_remaining_months
a=[1,2,3,4,5,6,7,8,9,10,11,12]
for i in a:
    fig.add_trace(go.Bar(
        x=a,
        y=[user_churned[user_churned['contract_remaining_months'] == i]['contract_remaining_months'].count()],
    name='Churn',
    marker_color='red', showlegend=False), row=3, col=1)
    fig.add_trace(go.Bar(
        x=a,
        y=[user_not_churned[user_not_churned['contract_remaining_months'] == i]['contract_remaining_months'].count()],
    name='No churn',
    marker_color='green', showlegend=False), row=3, col=1)

# lifetime_months
fig.add_trace(go.Histogram(
    x = user_churned['lifetime_months'],
    name = 'Churn',
    histnorm = 'density',
    marker_color='red',
    showlegend=False),row=4, col=1)
fig.add_trace(go.Histogram(
    x = user_not_churned['lifetime_months'],
    name= 'No churn',
    histnorm = 'density',
    marker_color='green',
    showlegend=False),row=4, col=1)

# other_services_total_dollars
fig.add_trace(go.Histogram(
    x = user_churned['other_services_total_dollars'],
    name = 'Churn',
    histnorm = 'density',
    marker_color='red',
    showlegend=False),row=5, col=1)
fig.add_trace(go.Histogram(
    x = user_not_churned['other_services_total_dollars'],
    name= 'No churn',
    histnorm = 'density',
    marker_color='green',
    showlegend=False),row=5, col=1)

# avg_visits_per_week_total
fig.add_trace(go.Histogram(
    x = user_churned['avg_visits_per_week_total'],
    name = 'Churn',
    histnorm = 'density',
    marker_color='red',
    showlegend=False),row=6, col=1)
fig.add_trace(go.Histogram(
    x = user_not_churned['avg_visits_per_week_total'],
    name= 'No churn',
    histnorm = 'density',
    marker_color='green',
    showlegend=False),row=6, col=1)

# avg_visits_per_week_last_month
fig.add_trace(go.Histogram(
    x = user_churned['avg_visits_per_week_last_month'],
    name = 'Churn',
    histnorm = 'density',
    marker_color='red',
    showlegend=False),row=7, col=1)
fig.add_trace(go.Histogram(
    x = user_not_churned['avg_visits_per_week_last_month'],
    name= 'No churn',
    histnorm = 'density',
    marker_color='green',
    showlegend=False),row=7, col=1)

# plot all of them
fig.update_layout(height=1700, width=1000,
              title_text="Destribution of features")
fig.show()


# 1. To plot the features by those who churned (stopped subscribing) and those who did not churn (continued subscribing), here plotted continuous numerical values on histograms. For each feature, plotted the number of values for both churn and non churn.
# 
# 2. For age, it shows that younger customers (between the ages of 20 and 25) tended to churn. After the age of 25, customers tended not to churn.
# 
# 3. For total_contract_period_months (the number of months the customers entire contract is), it shows that most customers who had a 1 month contract (possible a trial period), ended up churning. However, many more of the customers who had the same 1 month contract did not churn so perhaps this trial period worked for some customers.
# 
# 4. For contract_remaining_months (the number of months remaining on a customers contract is0, it shows that customers are more likely to churn when they have to choose whether to renew their contract or not (at 1 month remaining on the contract). Customers who do not churn renew their contract at either 1 month, 6 months or 12 months.
# 
# 5. For lifetime_months (the number of months a customer has been a member of the gym for the entire lifetime), it shows that most users who churn have been members for less than 3 months. However, many users who do not churn fall under 2-5 months of lifetime. This 2 month lifetime mark seems to be a major turning point for customers.
# 
# 6. For other_services_total_dollars (the amount of money customers spent on other services at the gym), it shows that customers who churned did not end up spending a lot of money on other services, while customers who did not churn did spend a considerable amount more on other gym services.
# 
# 7. For average_visits_per_week_total (for the customers lifetime, the averge number of visits per week), it shows that customers who did not churn visited the gym must more frequently on a weekly basis than customers who churned. The highest point for customers who did not churn is between 1.8 and 2.5 weekly visits, while the highest point for customers who did churn is 1 to 1.8.
# 
# 8. For average_visits_per_week_month (for the last month, the average number of visits per week), it shows that a huge amount of customers who churn did not visit the gym last month or visited the gym less than once a week.

# ### Build a correlation matrix and display it 

# In[12]:


cm = df.corr()
plt.figure(figsize=(15, 9))
sns.heatmap(cm, annot=True, square=True)
plt.show()


# This correlation matric shows the features that have the strongest correlation with target variable and the features that strongly correlate with each other. The features who have a strong positive correlation with other features are total_contract_period_months and contract_remaining_months (the number of months remaining on the contract compared with the total number of months that the contract is for) and avg_visits_per_week_last_month and avg_visits_per_week_total (the average number of weekly visits in the past month compared to the average number of weekly visits during the total time). It makes sense that these are stronly related since the monthly calculation is a portion of the total calculation. The features that have a strong negative correlation with the target variable (churn) are total_contract_period_months, age, contract_remaining_months, lifetime_months, avg_visits_per_week_total, avg_visits_per_week_last_month.

# ## Build a model to predict user churn 

# ### Build a binary classification model for customers where the target feature is the user's leaving next month 

# In[13]:


# features (X matrix)
X = df.drop(['churn'], axis=1)

# target variable (y)
y = df['churn']

# divide data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify=y)

# Create scaler object and apply it to train set
scaler = StandardScaler()

# Train scaler and transform the matric for train set
X_train_st = scaler.fit_transform(X_train)

# apply standardization of feature matric for test set
X_test_st = scaler.transform(X_test)

# define the models to compare
models = [LogisticRegression(random_state=0), RandomForestClassifier(random_state=0)]

# function that predicts model by taking data as input and outputting metrics
def make_prediction(model, X_train, y_train, X_test, y_test):
    model = model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Model: ', model)
    print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
    print('Precision: {:.2f}'.format(precision_score(y_test, y_pred)))
    print('Recall: {:.2f}'.format(recall_score(y_test, y_pred)))
    print('\n')

# output metric for both models
for i in models:
    make_prediction(i, X_train, y_train, X_test, y_test)


# 1. compared two models on the train set where the target feature is the customer train set: Logistic Regression and Random Forest Classifier.
# 
# 2. Accuracy is the share of accurate predictions among all predictions where the close to 1, the better. For this metric, Logistic Regression has a better score.
# 
# 3. Precision tells us what share of predictions in class 1 are true by looking at the share of correct answers only in the target class, where the locse to 1 the better. For this metric, Logistic Regression has a better score.
# 
# 4. Recall aims at minimizing the opposite risks by demonstrating the number of real class 1 objects you were able to discover with your model where the closer to 1, the better. For this metric, Logistic Regression has a better score.
# 
# 5. All in all, the Logistic Regression model gave better results.

# ## Create user clusters 

# ### Identify object (user) clusters 

# In[14]:


sc = StandardScaler()
x_sc = sc.fit_transform(X)

linked = linkage(x_sc, method='ward')

plt.figure(figsize=(15,10))
dendrogram(linked, orientation='top')
plt.show()
 
km = KMeans(n_clusters = 5, random_state=0)
labels = km.fit_predict(x_sc)

sil_score = silhouette_score(x_sc, labels)

df['cluster'] = labels

cluster_data = df.groupby('cluster').mean()
display(cluster_data)


# Process followed to build cluster:
# 1. Standardized the data
# 2. Build a matrix of distances based on the standardized feature matrix
# 3. Plotted a dendrogram
# 4. Trained the clustering model with the K-means algorithm and predict customer clusters.
# 5. calculated silhouette score
# 6. found the mean feature values for clusters
# 
# Review:
# In order to use K-Means clustering (which groups objects step by step under the assumption that the number of user clusters is already known), we must determine the number of user clusters that can be identified. This can be done using a dendrogram which shows different clusters and suggests an optimal number. The dendrogram for this data has suggested that the optimal number of clusters is 4.
# 
# After adding the cluster number to each data entry, I grouped data by cluster and calculated the mean for each feature values. Some interesting things: cluster 1 contains users who do not have a phone number on file and cluster 1 contains users who do not live near a location. As we saw in feature graphs, these two features greatly impact churn and furthermore, the mean churn value for these clusters are close to 1 which means most users in these clusters churned / stopped subscribing. The highest churn mean value is cluster 3 where all users live near a gym location: this cluster also had the smallest total contract length and we saw previously that most people who have a 1-2 month contract churn. The clusters with the smallest churn are 0 and 4: gender is split equally, most users live near a gym, almost all phone number exist and they spend the most on additional gym services.

# ### Plot distributions of features for the clusters 

# In[15]:


numerical_col = ['age','other_services_total_dollars','lifetime_months','avg_visits_per_week_total','avg_visits_per_week_last_month']

fig = make_subplots(rows=3, cols=2,subplot_titles=numerical_col)

r = 1
c = 1
idx = 1
legend = True

for i in numerical_col:
    # add cluster data
    fig.add_trace(go.Histogram(x=df.query('cluster == 0')[i], name='cluster0', legendgroup='cluster0',
                               marker = {'color':'Red'},showlegend=legend), row=r, col=c)
    fig.add_trace(go.Histogram(x=df.query('cluster == 1')[i],name='cluster1', legendgroup='cluster1',
                               marker = {'color':'Orange'},showlegend=legend), row=r, col=c)
    fig.add_trace(go.Histogram(x=df.query('cluster == 2')[i],name='cluster2', legendgroup='cluster2',
                               marker = {'color':'Yellow'},showlegend=legend), row=r, col=c)
    fig.add_trace(go.Histogram(x=df.query('cluster == 3')[i],name='cluster3', legendgroup='cluster3',
                               marker = {'color':'Green'},showlegend=legend),row=r, col=c)
    fig.add_trace(go.Histogram(x=df.query('cluster == 4')[i],name='cluster4', legendgroup='cluster4',
                               marker = {'color':'Blue'},showlegend=legend),row=r, col=c)
    # rotate to next row col
    legend = False
    r = (math.floor(idx/2) + 1)
    c = (idx%2 + 1)
    idx = idx+1
    
fig.update_layout(barmode='overlay', height=1000)
fig.update_traces(opacity=0.65)
fig.show()


# Here we have created histogram for continuos values:
# For age variable: cluster 3 has the youngest population of cluters, when compared to other clusters.
# 
# For other_services_total_dollars: most customers spent less than 200 dollars on other services provided at the gym but those in clusters 4 and 0 show that they have spent more.
# 
# For lifetime_months, cluster 3 spent the least amount of months being a member to the gym.
# 
# For avg_visits_per_week_total, cluster 3 visited the gym the least amount of times when compared to cluster 0 which visited about 2 times a week of cluster 4 which visits more than 2 times a week and clusters 1 and 2 which visited about once a week.
# 
# For avg_visits_per_week_last_month, again cluster 3 visited the gym the least amount of times when compared to cluster 0 which visited about twice weekly, cluster 4 visited more than twice weekly and clusters 1 and 2 visited about once a week.

# In[16]:


group_columns = ['gender','near_location','employer_partner','friend_promo','phone_exists','total_contract_period_months',
             'group_sessions','contract_remaining_months']

fig = make_subplots(rows=4, cols=2,subplot_titles=group_columns)

idx = 0
r = (math.floor(idx/2) + 1)
c = (idx%2 + 1)
legend = True

for i in group_columns:
    fig.add_trace(go.Histogram(x=df.query('cluster == 0')[i],name='cluster0', legendgroup='cluster0',
                               marker = {'color':'Red'},showlegend=legend), row=r, col=c)
    fig.add_trace(go.Histogram(x=df.query('cluster == 1')[i],name='cluster1', legendgroup='cluster1',
                               marker = {'color':'Orange'},showlegend=legend), row=r, col=c)
    fig.add_trace(go.Histogram(x=df.query('cluster == 2')[i], name='cluster2', legendgroup='cluster2',
                               marker = {'color':'Yellow'},showlegend=legend), row=r, col=c)
    fig.add_trace(go.Histogram(x=df.query('cluster == 3')[i],name='cluster3', legendgroup='cluster3',
                               marker = {'color':'Green'},showlegend=legend),row=r, col=c)
    fig.add_trace(go.Histogram(x=df.query('cluster == 4')[i],name='cluster4', legendgroup='cluster4',
                               marker = {'color':'Blue'},showlegend=legend),row=r, col=c)
    # rotate to next row col
    idx = idx+1
    r = (math.floor(idx/2) + 1)
    c = (idx%2 + 1)
    legend = False
    
fig.update_xaxes(type="category", row=3, col=2)
fig.update_layout(barmode='group', height=1200)
fig.show()


# To plot the distribution for each cluster for each feature, plotted bar graph for binary value columns:
# For gender, it shows an equal amount of male and female in each cluster although clusters 3 and 4 have a little more females than males.
# 
# For near_location, it shows that cluster 2 are the customers who did not live near a location.
# 
# For employer_partner, it shows that cluster 1 are the customers who are empoyees of a partner company, while cluster 3 are the ones who are not employees of a partner company.
# 
# For friend_promo, it shows that cluster 1 are the customers who received a promotional code from their friend, while cluster 3 are not.
# 
# For phone_exists, cluster 2 was the custoemrs with no phones on file while all the other clusters had some sort of phone on file for the customers.
# 
# For total_contract_period_months, cluster 0 was the ones who had a 12 month contract and clsuter 3 was the customers with a one month contract.
# 
# For group_sessions, it shows that cluster 3 mainly did not attend group sessions and cluster 0 did mostly attend group sessions.
# 
# For contract_remaining_months, cluster 3 had 0 months remaining, while cluster 1 has 12 months remaining.

# ### Calculate the churn rate for each cluster 

# In[17]:


churn_pivot = df.pivot_table(index='cluster', values='churn', 
                               aggfunc=['count', 'sum', lambda x: abs(round(((x == 0).sum() / x.count()-1)*100,2))]).reset_index()
churn_pivot.columns = ['cluster', 'total number of entries', 'churn sum', 'churn rate (%)']
display(churn_pivot)


# create pivot table where it is separated by cluster and calculated based on churn:
# This pivot table shows that the clusters with the highest churn rate is (in order), cluster 3, cluster 2 and cluster 1. These clusters are phrone to leaving. The clusters with the lowest churn rate (in order), are cluster 0 and cluster 4. These clusters are more loyal to the gym.
# 

# ## Conclusion and Analysis: 

# To improve their customer retention strategy:
# 
# For acquiring new customers, focus on customers that live near a gym location, that are employees at a partner company and that are between the ages of 27 and 34 years. The data shows that customers who fall under one or (ideally) all of these categories are less likely to churn.
# 
# For already existing customers, focus on attaining their phone numbers, encourage them to spend more money on other gym services and increase their amount of weekly visits. These are all factors that make a customer less likely to churn. For example, the gym can encourage weekly visits by having one night a week where there are discounted prices on additional services such as food, spa, etc. This encourages customers to come in while also spending money on additional services. Additionally, focus on customers who have 1-2 months before their current contract ends. This is when customers are more likely to decide whether to continue their subscription or not.
# 
# The most loyal customers are those who are between the ages of 27 and 34, who spend the most on other gym services, who live near the location, who are employees at partner companies, who spend money on additional gym services and who visit the gym about 2 times on a weekly basis.
