import py_compile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC,SVR
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, ElasticNet
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler,PowerTransformer
from sklearn.model_selection import train_test_split
import plotly.express as px
import streamlit as st


st.set_page_config("machine learning app",":chart_with_upwards_trend:")#,layout="wide",initial_sidebar_state="expanded")

st.title('Simple machine learner app')
st.header('=================================')

tab1, tab2, tab3, tab4 = st.tabs(["How to run the app", "Definitions","About machine learning","How to choose the algorithm"])

with tab1:
   st.header("How to run the app")
   st.markdown("* ##### all inputs or commands from the right sidebar")
   st.markdown("* ##### all output is shown  in the main page")
   st.markdown("1- upload csv file containing numeric values, with column names in the first row")
   st.markdown("2- Choose one of the columns as the target (Y), be sure that Y has no Nan values")
   st.markdown("3- All other columns will be automatically selected as independent variable (X1,X2,...ect)")
   st.markdown("4- all data that is not numbers will be converted to Nan automatically ")
   st.markdown("5- Choose the Maximum allowed percent of Nan values per columns, columns which has higher percentage will be removed ")
   st.markdown("6- Choose the algorithm")
   st.markdown("7- the algorithm will run to find the best relation that describes target from independent variables")
   st.markdown("8- output is correlation of determination for training, testing sets and predicted target")
   st.markdown("9- output can be downloaded as CSV file")


with tab2:
   st.header("Definitions")
   list=['Target','Independent Variables','Features','Features importance', 'Training score','Testing score','coeffecient of determination (R^2)','Training set/Testing set','mean absolute error','Nan']
   selection_tab2=st.selectbox('',list)
   if selection_tab2==list[0]: 
       st.markdown("* **Target** : this is the column that you want to predict based on independent variables, donated as Y, also known as dependent variable")
   elif selection_tab2==list[1]:
       st.markdown("* **Independent variables** : these are the columns that you predict the target from them , donated as X1,X2,..ect, also known as features")
   elif selection_tab2==list[2]:
       st.markdown("* **Features** : the columns that you predict the target from them , donated as X1,X2,..ect, also known as independent variables")
   elif selection_tab2==list[3]:
       st.markdown("* **Features importance** : The importance of each feature in predicting the target, higher number means higher importance")
   elif selection_tab2==list[4]:
       st.markdown("* **Training score** : The coeffecient of determination for the training set")
   elif selection_tab2==list[5]:
       st.markdown("* **Testing score** : The coeffecient of determination for the testing set")
   elif selection_tab2==list[6]:
       st.markdown("* **coeffecient of determination** : is a number between 0 and 1 that measures how well a statistical model predicts an outcome, also known by R square")
       st.markdown("* if the number is 1 , the model is perfectly predicts the target")
       st.markdown("* if the number is 0 , the model is prediction is poor")
       st.markdown("* the higher the number, the better the model is")
   elif selection_tab2==list[7]:
       st.markdown("* **Training set/Testing set** : usually the data are split into two subsets, training set and testing set")
       st.markdown("* training set is the data that fed into the machine learning model")
       st.markdown('* testing set is the data we test the model on, in this set we predect the target `Y_predect` and compare it with `Y_actual`')
   elif selection_tab2==list[8]:
       st.markdown("* **mean absolute error** : Absolute Error is the amount of error in your measurements. It is the difference between the predicted value and “actual” value. For example, if a model predict weight to be 90 Kgs but you know  true weight is 89 pounds, then the model has an absolute error of `90 Kgs – 89 lbs = 1 lbs.` and that is for one data point")
       st.markdown("* * mean absolute error is the mean of all absolute errors in all data")
   elif selection_tab2==list[9]:
       st.markdown("* **Nan** : Not a number, it mainly means missing values, so a column have 70 % Nan, means 70 % of this column has missing values")

            

with tab3:
   st.header("About machine learning")
   st.markdown("* Simply speaking, it is a way to know the relation between dependent variable and indepentent variables")
   st.markdown("* there are several methods to build this relation, when the relation is built, this building is called machine learning algorithm")
   st.markdown("* Machine learning algorithms vary where some depend on linear relationship, others depend on non linear relationship and others combine both")
   st.markdown("* machine learning algorithm can predict the target based on dependent variables")
   st.markdown("* machine learning algorithm can specify which features are important to influence target change` features importance`")

with tab4:
   st.header("How to choose the algorithm")
   st.markdown("* the ultimate way is to test each and all algorithms, and choose the one that achieves the best score in both training and testing sets")
   st.markdown("* in reality, this rarely happens as randomness plays a role here, also it is rarely found that one algorithms scores higher tha alle others in both training and testing sets")
   st.markdown("* Here a methedology that can help in choosing the best model [SCI-KIT LEARN METHODOLOGY](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)")

st.write('******************************************************************')
data=st.sidebar.file_uploader("Choose csv file to upload",type='csv')

if data is not None:
    df_raw = pd.read_csv(data)
else:
    st.sidebar.write('*Kindly upload valid csv data')
    
if data is not None: 
    st.write(df_raw)
    st.write(' Kindly note that all not number cells  will be converted to Nan')
    df=df_raw.copy()
    for column in df.columns:
        df[column]=pd.to_numeric(df[column],errors='coerce')
    st.sidebar.write('======================================')
    yy=st.sidebar.selectbox('Choose target or dependent variable (y)',df.columns)
    st.sidebar.write('======================================')
    st.sidebar.write('## Note')
    st.sidebar.write('All other columns will be chosen as independent variables (X1, X2,...etc)')
    st.sidebar.write('======================================')
    st.write('**you choosed**', yy,'***to be the target**')
    st.write('******************************************************************')
    y=df[[yy]]
    st.write('Y has', y.isnull().sum(),' Nan values')
    st.write('******************************************************************')
    st.write('y description :', y.describe())
    st.write('******************************************************************')
    X=df.drop(yy,axis=1)
    X=X[X.describe().columns]
    st.write('******************************************************************')
    st.write('******************************************************************')
    st.write('******************************************************************')
    st.write('X description before Nan removal:', X.describe())
    st.write('******************************************************************')
    st.write(f'X has {X.isnull().sum().sum()} missing values and {X.shape[1]} columns `before` Nan removal:')
    st.write('******************************************************************')
    st.write(' kindly note that all nan values will be substituted either by most frequent,median or mean value for each column')
    st.write('******************************************************************')
    st.sidebar.write('Choose the percentage of NaN present in each column, any column having more than this percent will be removed from the dataset')
    zz=st.sidebar.selectbox('columns to be removed from data having NAN percentage more than :',reversed(range(10,110,10)))
    X=X.dropna(axis='columns', how='any', thresh=X.shape[0]*(1-(zz/100)))
    st.sidebar.write('======================================')
    substitution=st.sidebar.radio("**replace Nan values or delete them**",('Replace by Median','Replace by Most Frequent','Replace by Mean','Delete Nan rows'))
    if substitution =='Median':

        X=X.fillna(X.median())
    elif substitution =='Most Frequent':
        X=X.fillna(X.mode().iloc[0])

    elif substitution =='Delete Nan rows':
        X.dropna(inplace=True)
        y=y.iloc[X.index]
    else:
        X=X.fillna(X.mean())
    st.write('X description after Nan removal:', X.describe())
    st.write('******************************************************************')
    st.write(f'X has {X.isnull().sum().sum()} missing values and {X.shape[1]} columns `after` Nan removal:')
    st.write('******************************************************************')
    st.write('X shape:',X.shape,'Y shape:',y.shape)


    models=[DecisionTreeRegressor(),RandomForestRegressor(),AdaBoostRegressor(),LinearRegression(),GradientBoostingRegressor(),SGDRegressor(),ElasticNet(),Lasso()]
    st.sidebar.write('======================================')
    model=st.sidebar.selectbox('Choose algorithm model',models)
    st.write(model)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    try:
        model.fit(X_train,y_train,random_state=42)
    except:
        model.fit(X_train,y_train)
    y_train_pred=model.predict(X_train)
    y_test_pred=model.predict(X_test)

    training_accuracy=r2_score(y_train,y_train_pred)
    training_mean_error=mean_absolute_error(y_train, y_train_pred)

    testing_accuracy=r2_score(y_test,y_test_pred)
    testing_mean_error=mean_absolute_error(y_test,y_test_pred)
    feature_importance=pd.DataFrame()
    feature_importance['Name'] = X.columns
    try:   
        feature_importance['importance in the model']=model.feature_importances_
    except:
        pass
    try:
        feature_importance['column coeffecient']=np.abs(model.coef_)
    except:
        pass


    st.title('Machine learning model results:')
    st.header('training score:' )
    st.write(training_accuracy)
    st.header('testing score:')
    st.write(testing_accuracy )
    st.header('training mean absolute error:')
    st.write(training_mean_error)
    st.header('testing mean absolute error:')
    st.write(testing_mean_error)
    st.write('******************************************************************')
    st.header('feature importance:')
    try:
        st.dataframe(feature_importance.sort_values(by='importance in the model',ascending=False))
    except:
        st.dataframe(feature_importance.sort_values(by='column coeffecient',ascending=False))
    st.sidebar.write('======================================')
    st.write('******************************************************************')
    if st.sidebar.button('Compare prediction, with actual data ?'):
        comparing=pd.DataFrame()
        comparing['Actual']=y
        comparing['prediction']=model.predict(X)
        comparing['difference']=np.abs(comparing['Actual']-comparing['prediction'])
        st.write(comparing)
        st.write('******************************************************************')
        st.write(comparing.describe())
        #fig,ax=plt.subplots()
        #plt.scatter(comparing['Actual'],comparing['prediction'])
        #ax.set_xlabel('Actual Y')
        #ax.set_ylabel('Predected Y')
        #st.pyplot(fig)
        figure = px.scatter(comparing,x='Actual', y='prediction')
        st.plotly_chart(figure)


    st.sidebar.write('======================================')        
    if st.sidebar.button('Export data to excel file ?'):
        comparing=pd.DataFrame()
        comparing['Actual']=y
        comparing['prediction']=model.predict(X)
        comparing['difference']=np.abs(comparing['Actual']-comparing['prediction'])
        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv(index=False).encode('utf-8')

        csv_1 = convert_df(feature_importance)
        csv_2 = convert_df(comparing)
        st.download_button(label="Download feature importance as CSV", data=csv_1, file_name='features_importance.csv', mime='text/csv')
        st.download_button(label="Download actual/predicted Y as CSV", data=csv_2, file_name='Actual-predicted Y.csv', mime='text/csv')












       
        







