import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC,SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler,PowerTransformer
from sklearn.model_selection import train_test_split
import streamlit as st
import io

def convert_to_number(data,number=np.nan):
    '''
    This function takes two arguments, the dataframe and the value that all non-numbers needs to be converted to
    '''
    df=data.copy()
    
    def is_not_number(x):
        try:
            float(x)
            return False
        except:
            return True
        
    def is_number(x):
        try:
            float(x)
            return True
        except:
            return False
    
    for column in df.columns:
        df.loc[df[column].apply(is_not_number),column]=number
    return df

def to_float(data,x=0):
    '''
    converting all columns to float starting from x column, where x is the position of the columns
    Nat values doesn't allow the column to be converted to float, be sure to remove all NAT values
    
    - inputs :  dataframe and the first column position to start converting from
      syntax to_float(data,x=0)
      
    - output : dataframe that all of its columns are float, if possible
    '''
    df=data.copy()
    columns = df.columns[x:]
    for column in columns:
        try:
            df[column]=df[column].astype(float)
        except:
            pass
    return df    

st.title('Simple machine learner app')
st.header('=================================')

tab1, tab2 = st.tabs(["How to run the app", "Definitions"])

with tab1:
   st.header("How to run the app")
   st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

with tab2:
   st.header("Definitions")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

st.header('This app allows a csv file to be uploaded and apply basic machine learning algorithms')
data=st.sidebar.file_uploader("Choose csv file to upload",type='csv')

if data is not None:
    df_raw = pd.read_csv(data)
else:
    st.sidebar.write('*Kindly upload valid csv data')
    
if data is not None: 
    st.write(df_raw)
    st.write(' Kindly note that all not number cells  will be converted to Nan')
    df=to_float(convert_to_number(df_raw))
    st.sidebar.write('======================================')
    yy=st.sidebar.selectbox('Choose target or dependent variable (y)',df.columns)
    st.sidebar.write('======================================')
    st.sidebar.write('## Note')
    st.sidebar.write('All other columns that contains numbers only will be chosen as independent variables (X1, X2,...etc)')
    st.sidebar.write('======================================')
    st.write('**you choosed**', yy,'***to be the target**')
    st.write('******************************************************************')
    y=df[yy]
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
    buffer = io.StringIO()
    X.info(buf=buffer)
    s = buffer.getvalue()
    st.write('X information before Nan removal:')
    st.write(s)
    st.write('******************************************************************')
    X_nan_values=X.isnull().sum().sum()
    st.write('**X has null values of**', X_nan_values)
    st.write(' kindly note that all nan values will be substituted either by most frequent,median or mean value for each column')
    st.write('******************************************************************')
    st.sidebar.write('Choose the percentage of NaN present in each column, any column having more than this percent will be removed from the dataset')
    zz=st.sidebar.selectbox('columns to be removed from data having NAN percentage more than :',reversed(range(10,110,10)))
    X.dropna(axis='columns', how='any', thresh=X.shape[0]*(zz/100), inplace=True)
    st.sidebar.write('======================================')
    substitution=st.sidebar.radio("**replace Nan values by median or most frequent**",('Median','Most Frequent','Mean'))
    if substitution =='Median':
        X=X.fillna(X.median())
    elif substitution =='Most Frequent':
        X=X.fillna(X.mode().iloc[0])
    else:
        X=X.fillna(X.mean())
    st.write('X description after Nan removal:', X.describe())
    st.write('******************************************************************')
    buffer_ = io.StringIO()
    X.info(buf=buffer)
    s_ = buffer.getvalue()
    st.write('X information after Nan removal:')
    st.write(s_)
    st.write('******************************************************************')
    X_nan_values_=X.isnull().sum().sum()
    st.write('**After processing, X has  null values of **', X_nan_values_)
    st.write('******************************************************************')
    st.write(X.shape,y.shape)


    models=[DecisionTreeRegressor(),RandomForestRegressor(),AdaBoostRegressor(),LinearRegression(),GradientBoostingRegressor()]
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
        feature_importance['column coeffecient']=model.coef_
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












       
        







