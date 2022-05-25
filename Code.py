# -*- coding: utf-8 -*-
###############################################################################
# FINANCIAL DASHBOARD 2 - v1
###############################################################################

#==============================================================================
# Initiating
#==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import plotly.graph_objects as go
import math
import statsmodels.api as sm
from time import time
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
from pandas_datareader import data as wb
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import shap
shap.initjs()

#===================================================
# GLOBAL VARIABLES
#===================================================

   # Load Data Churn_Train

Churn_Train = pd.read_csv("C:/Users/omantilla/OneDrive - IESEG/Documents/IESEG/Communications Skills/Group Project/Data-20211105/churn_train.csv")
 

    # Load Data Churn_Test
Churn_Test = pd.read_csv("C:/Users/omantilla/OneDrive - IESEG/Documents/IESEG/Communications Skills/Group Project/Data-20211105/churn_test.csv")
  
# Label Encoder Churn
le_churn = LabelEncoder()
Churn_Train["churn"] = le_churn.fit_transform(Churn_Train["churn"])

# Label Encoder international_plan
le_international_plan = LabelEncoder()
Churn_Train["international_plan"] = le_international_plan.fit_transform(Churn_Train["international_plan"])

# Label Encoder voice_mail_plan
le_voice_mail_plan = LabelEncoder()
Churn_Train["voice_mail_plan"] = le_voice_mail_plan.fit_transform(Churn_Train["voice_mail_plan"])

# Label Encoder area_code
le_area_code = LabelEncoder()
Churn_Train["area_code"] = le_area_code.fit_transform(Churn_Train["area_code"])

# Label Encoder area_code
le_state = LabelEncoder()
Churn_Train["state"] = le_state.fit_transform(Churn_Train["state"])
    
# Label Encoder international_plan
le_international_plan = LabelEncoder()
Churn_Test["international_plan"] = le_international_plan.fit_transform(Churn_Test["international_plan"])

# Label Encoder voice_mail_plan
le_voice_mail_plan = LabelEncoder()
Churn_Test["voice_mail_plan"] = le_voice_mail_plan.fit_transform(Churn_Test["voice_mail_plan"])

# Label Encoder area_code
le_area_code = LabelEncoder()
Churn_Test["area_code"] = le_area_code.fit_transform(Churn_Test["area_code"])

# Label Encoder area_code
le_state = LabelEncoder()
Churn_Test["state"] = le_state.fit_transform(Churn_Test["state"])
    
# Feature Variables Creation
features = ['state', 
                'account_length', 
                'area_code', 
                'international_plan', 
                'voice_mail_plan', 
                'number_vmail_messages', 
                'total_day_minutes', 
                'total_day_calls', 
                'total_day_charge', 
                'total_eve_minutes', 
                'total_eve_calls', 
                'total_eve_charge', 
                'total_night_minutes', 
                'total_night_calls', 
                'total_night_charge', 
                'total_intl_minutes', 
                'total_intl_calls', 
                'total_intl_charge', 
                'number_customer_service_calls']

X, y = Churn_Train[features], Churn_Train["churn"]
X = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size=0.2, random_state=42)
#==============================================================================
# Predictive Models
#==============================================================================

logreg = sm.Logit(y_train,X_train).fit()

    # Predict
pred_trainLG = logreg.predict(X_train)
pred_testLG = logreg.predict(X_test)

# Evaluate predictions
acc_trainlg = accuracy_score(y_train, np.round(pred_trainLG))
acc_testlg = accuracy_score(y_test, np.round(pred_testLG))

# split data in train and test (stratify y)
X, y = Churn_Train[features], Churn_Train["churn"]
X = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size=0.2, random_state=42)
    
# define model
rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X_train,y_train) 
    
# predict probabilities
pred_train = rf.predict(X_train)
pred_test = rf.predict_proba(X_test)

# evaluate predictions
acc_train = accuracy_score(y_train, pred_train)
acc_test = accuracy_score(y_test, np.argmax(pred_test, axis=1))

# normalize data
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), batch_size=32, early_stopping=False, random_state=42)
mlp = mlp.fit(X_train, y_train)
    
# predict probabilities
pred_trainNR = mlp.predict_proba(X_train)
pred_testNR = mlp.predict_proba(X_test)

# evaluate predictions
acc_train = accuracy_score(y_train, np.argmax(pred_trainNR, axis=1))
acc_test = accuracy_score(y_test, np.argmax(pred_testNR, axis=1))
#==============================================================================
# Tab 1
#==============================================================================

def tab1():
    
# Add dashboard title and description
    st.title("Customer Churn Behaviour")
    st.header('Abstract')
    
    st.markdown("""  Nowadays, companies are focused in maintain and improve the their profits. One of the biggest challenges for the retaintion  departments is to identy churners and prepare retaintion packages. Our target with this report is to present insights based on the Logistict Regresion, Random Forest and Neural Networks models to improve the Linear Regresion Model that is being used for the company. This way the company can offer a better retantion programs for customers that are churn prospects and keep their profits and improve the services that might be triggering the users to churn""")
    
    

    

#     # This draws the candelstick chart
#     st.plotly_chart(fig, use_container_width=True)
    
#     # This adds a footnote for the Candlesticks
#     st.caption('Candlestick (Price Movement and Volume)')
    
    
    

#==============================================================================
# Tab 2
#==============================================================================

def tab2():
    
# Add dashboard title and description
    st.title("Current Insights:eyeglasses:")
    
# Print Churn_Train Data Set     
    st.write(Churn_Train)
    st.caption('Churn Data Set')
    
# Add a Dashboard description
    st.markdown("""To date, the telecommunications company has 4250 customers. Of these customers 598 have unsubscribed from the service.  The churn rate is a very important measure because it is much more profitable to retain existing customers than to acquire new ones. This is because it saves marketing costs and sales costs. You will get a return on retention because you will gain the trust and loyalty of the customer. We will identify through different models why these customers churn and quantify the features that influence churn.  """)
    


    
 
    
#     # Draws the Plot Line 
#     fig.add_trace(go.Scatter(x=ticker_data["date"], y=ticker_data["open"],
#                     mode='lines', name= 'Line'))
    
#     # Moving Average
#     def moving_average(x, w):
#         return np.convolve(x, np.ones(w), 'valid') / w

#     data = np.array(ticker_data['close'])

#     fig.add_trace(go.Scatter(x=ticker_data["date"], y=moving_average(data,4),
#                     mode='lines', name= 'MA'))
    
#     # This draws the candelstick chart
#     st.plotly_chart(fig, use_container_width=True)
    

    
    
    
#==============================================================================
# Tab 3
#==============================================================================

def tab3():
    
# Load Data Churn_Train

# Load Data Churn_Train

    
# Add dashboard title and description
    st.title("Logistic Regression")
    

    
# split data in train and test (stratify y)

    
# Logistic Regression

    
    st.write(logreg.summary())



# Print Accuracy
    st.markdown("""
    ### Logistic Regresessio Accuracy""")
    st.write(f"Train:\tACC={acc_trainlg:.4f}")
    st.write(f"Test:\tACC={acc_testlg:.4f}")
    
#==============================================================================
# Tab 4
#==============================================================================

def tab4():
    
    # Add dashboard title and description
    st.title("Neural Network")
    
    st.markdown("""
    ### Neural Network Accuracy""")
    st.write(f"Train:\tACC={acc_train:.4f}")
    st.write(f"Test:\tACC={acc_test:.4f}")
    
    
    
#     # Get the financial information from each ticker
#     financials = si.get_financials(ticker, yearly = True, quarterly = True)
    
#     # Creates the select option: Income Statement, Balance Sheet and Cash Flow
#     option = st.selectbox(
#        'Financials',
#        ('Income Statement', 'Balance Sheet', 'Cash Flow'))

#     st.write('You selected:', option)
  
#     # Creates the frequency options: Annual and Quarterly
#     freq = st.selectbox(
#        'Financials Frequency',
#        ('Annual', 'Quarterly'))
    
#     st.write('You selected:', freq)
    
#     # The Following IFS allows to mix up the different options and frecuencies
#     if  option == 'Income Statement':
#         if freq == 'Annual':
#             st.table(financials['yearly_income_statement'])
            
#     if  option == 'Income Statement':
#         if freq == 'Quarterly':
#             st.table(financials['quarterly_income_statement'])
    
#     if  option == 'Balance Sheet':
#         if freq == 'Annual':
#             st.table(financials['yearly_balance_sheet'])
            
#     if  option == 'Balance Sheet':
#         if freq == 'Quarterly':
#             st.table(financials['quarterly_balance_sheet'])
            
#     if  option == 'Cash Flow':
#         if freq == 'Annual':
#             st.table(financials['yearly_cash_flow'])
            
#     if  option == 'Cash Flow':
#         if freq == 'Quarterly':
#             st.table(financials['quarterly_cash_flow'])
    
#==============================================================================
# Tab 5
#==============================================================================

def tab5():
    
    # Add dashboard title and description
    st.title("Random Forest")

# Important features well formatted
    rfdf = pd.DataFrame(rf.feature_importances_, index=X.columns, columns=["Feature importance"]).sort_values(by= "Feature importance", ascending = False)
    
    st.write(rfdf)
    
    st.markdown("""
    ### Random Forest Accuracy""")
        
    st.write(f"Train:\tACC={acc_train:.4f}")
    st.write(f"Test:\tACC={acc_test:.4f}")
    
    
    st.markdown("""
    ### Feature Selection""")
    
# Feature Selection
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)

# Concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)

# Naming the dataframe columns
    featureScores.columns = ['Features','Score']  

# This prints the 10 best features
    st.write(featureScores.nlargest(10,'Score'))  
    
    # Gets the analysis for each ticker
#     analyst_info = si.get_analysts_info(ticker)
    
#     # Gets the keys from dictionary 'analyst_info'
#     keys = analyst_info.keys()
    
    
#     # This for loop gets the keys already extracted from keys = analyst_info.keys() and asks for each key value to build the table
#     for key in keys:
#         st.write(key)
#         st.table(analyst_info[key])
    
#==============================================================================
# Tab 6
#==============================================================================

def tab6():
    
# Add dashboard title and description
    st.title("Interpretability Techniques: Partial Dependence")
    


    fig, ax = plt.subplots(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(estimator=mlp, X=X_train, features=[9, 7, (9, 7),4, 19, (4,19), 6, 10, (6,10)], ax=ax)
    fig.tight_layout(pad=2.0)
    
    st.pyplot(fig)
    
    st.mardown("""These graphs make it possible to study how the variable weighs on the model's prediction. This graph measures the intensity of the overall impact of the variable on churn. This model allows you to use 2 features maximum.""")
    
#     # This is the selectbox creation for the number of simulations
#     num_sim = st.selectbox(
#        'Number of Simulations',
#        (200, 500, 1000))
    
#     # This is the selectbox creation for the time horizon
#     t_horizon = st.selectbox(
#        'Time Horizon',
#        (30, 60, 90))
    
#     # This is to get the stock or ticker prices from yahoo
#     stock_price = wb.DataReader(ticker, 'yahoo', start_date, end_date)
   
    
#     # Take the close price
#     close_price = stock_price['Close']
    
#     # Plot close stock price
#     fig, ax = plt.subplots()
#     fig.set_size_inches(15, 5, forward=True)
#     plt.plot(close_price)
    
    
#     # Take the last close price
#     last_price = close_price[-1]

#     # Generate the stock price dinamically
#     time_horizon = t_horizon
#     next_price = []
    
#     # The returns ((today price - yesterday price) / yesterday price)
#     daily_return = close_price.pct_change()
#     daily_return.head()
    
#     # The volatility (high value, high risk)
#     daily_volatility = np.std(daily_return)

#     for n in range(time_horizon):

#     # Generate the random percentage change around the mean (0) and std (daily_volatility)
#         future_return = np.random.normal(0, daily_volatility)
    
#     # Generate the random future price
#         future_price = last_price * (1 + future_return)
    
#     # Save the price and go next
#         next_price.append(future_price)
#         last_price = future_price
        
#     # Setup the Monte Carlo simulation
#     np.random.seed(123)
#     simulations = num_sim
#     time_horizone = t_horizon

#     # Run the simulation
#     simulation_df = pd.DataFrame()

#     for i in range(simulations):

#         # The list to store the next stock price
#         next_price = []

#         # Create the next stock price
#         last_price = close_price[-1]

#         for j in range(time_horizone):
#             # Generate the random percentage change around the mean (0) and std (daily_volatility)
#             future_return = np.random.normal(0, daily_volatility)

#             # Generate the random future price
#             future_price = last_price * (1 + future_return)

#             # Save the price and go next
#             next_price.append(future_price)
#             last_price = future_price

#         # Store the result of the simulation
#         simulation_df[i] = next_price
    
#     # Plot the simulation stock price in the future
#     fig, ax = plt.subplots()
#     fig.set_size_inches(15, 10, forward=True)

#     plt.plot(simulation_df)
#     plt.title('Monte Carlo simulation for AAPL stock price in next 200 days')
#     plt.xlabel('Day')
#     plt.ylabel('Price')

#     plt.axhline(y=close_price[-1], color='red')
#     plt.legend(['Current stock price is: ' + str(np.round(close_price[-1], 2))])
#     ax.get_legend().legendHandles[0].set_color('red')
#     st.pyplot(plt)
    
#     # Get the ending price of the 200th day
#     ending_price = simulation_df.iloc[-1:, :].values[0, ]

#     # Price at 95% confidence interval
#     future_price_95ci = np.percentile(ending_price, 5)

#     # Value at Risk
#     # 95% of the time, the losses will not be more than 16.35 USD
#     VaR = close_price[-1] - future_price_95ci
#     st.write('VaR at 95% confidence interval is: ' + str(np.round(VaR, 2)) + ' USD')

#==============================================================================
# Tab 7
#==============================================================================

def tab7():
    
    # Add dashboard title and description
    st.title("Interpretability Techniques: ICE")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.tight_layout(pad=0.1)
    ice = PartialDependenceDisplay.from_estimator(estimator=mlp,
                        X=Churn_Test,
                        features=[7,6,9,10,4,19],
                        target=0,
                        kind="both",
                        ice_lines_kw={"color":"#808080","alpha": 0.3, "linewidth": 0.5},
                        pd_line_kw={"color": "#ffa500", "linewidth": 4, "alpha":1},
                        # centered=True, # will be added in the future
                        ax=ax)
    st.pyplot(fig)                                                
                                                    
                                                    
                                                    
                                                    
                        # centered=True, # will be added in the future
                                                  

#==============================================================================
# Tab 8
#==============================================================================

def tab8():
    
    # Add dashboard title and description
    st.title("Interpretability Techniques: Shapley values")
    
#==============================================================================
# Main body
#==============================================================================


def run():
     
 
    
    # Add a radio box
    select_tab = st.sidebar.radio("Select tab", ['Abstract', 'Insight', 'Logistic Regression', 'Neural Network', 'Random Forest', 'Interpretability Techniques: PDP', 'Interpretability Techniques: ICE', 'Interpretability Techniques: Shapley values',])
    
    # Show the selected tab
    if select_tab == 'Abstract':
        # Run tab 1
        tab1()
    elif select_tab == 'Insight':
        # Run tab 2
        tab2()
    elif select_tab == 'Logistic Regression':
        # Run tab 3
        tab3()
    elif select_tab == 'Neural Network':
        # Run tab 4
        tab4()
    elif select_tab == 'Random Forest':
        # Run tab 5
        tab5()
    elif select_tab == 'Interpretability Techniques: PDP':
        # Run tab 6
        tab6()
    elif select_tab == 'Interpretability Techniques: ICE':
        # Run tab 7
        tab7()
    elif select_tab == 'Interpretability Techniques: Shapley values':
        # Run tab 8
        tab8()  
    
        
        
if __name__ == "__main__":
    run()
    
###############################################################################
# END
###############################################################################