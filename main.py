import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC,SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler,PowerTransformer
from sklearn.model_selection import train_test_split
import streamlit as st
import io 

st.title('Simple machine learner')
st.header('This app let you upload csv file and apply basic machine learning algorithms')
data=st.sidebar.file_uploader("Choose csv file to upload",type='csv')

if data is None:
    st.sidebar.write('*Kindly upload valid csv data')
else:
    df = pd.read_csv(data)
 
st.write(df)
yy=st.sidebar.selectbox('Choose target or dependent variable (y)',df.columns)
st.sidebar.write('#### note: All other columns that contains numbers only will be chosen as independent variables (X1, X2,...etc)')
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
st.write(' kindly note that all columns that has more than 70 % nan values will be removed')
st.write(' kindly note that all nan values will be substituted either by most frequent, or mean value for each column')
st.write('******************************************************************')
X.dropna(axis='columns', how='any', thresh=X.shape[0]*0.7, inplace=True)
substitution=st.sidebar.radio("**replace Nan values by median or most frequent**",('Median','most frequent'))
if substitution =='Median':
    X=X.fillna(X.median())
else:
    X=X.fillna(X.mode().iloc[0])
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


models=[DecisionTreeRegressor(),RandomForestRegressor(),AdaBoostRegressor(),LinearRegression()]
model=st.sidebar.selectbox('Choose algorithm model',models)
st.write(model)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

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
st.header('feature importance:')
try:
    st.dataframe(feature_importance.sort_values(by='importance in the model',ascending=False))
except:
    st.dataframe(feature_importance.sort_values(by='column coeffecient',ascending=False))

if st.button('Compare prediction, with actual data ?'):
    comparing=pd.DataFrame()
    comparing['Actual']=y
    comparing['prediction']=model.predict(X)
    comparing['difference']=np.abs(comparing['Actual']-comparing['prediction'])
    st.write(comparing)
    st.write(comparing.describe())










       
        







