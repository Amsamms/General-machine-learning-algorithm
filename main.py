import py_compile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC,SVR
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, ElasticNet, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler,PowerTransformer, PolynomialFeatures,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
import plotly.express as px
import plotly.graph_objects as go
import sklearn
#import shap #temorary
import streamlit as st
import scipy.stats as stats


st.set_page_config("machine learning app",":chart_with_upwards_trend:")#,layout="wide",initial_sidebar_state="expanded")

st.title('Simple machine learner app')
st.header('=================================')

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Main app","How to run the app", "Definitions","About machine learning","How to choose the algorithm", "Example"])

with tab1:
    st.markdown("## Main display area")
    
with tab2:
    st.markdown(
        """
        ## How to use this app:
        
        1. Start by uploading a CSV or Excel file using the 'Choose csv file to upload for preprocessing and modeling' button on the left sidebar. This file should contain the data you want to train the model on. Make sure that your file contains a single sheet and the column names are defined in the first row.
        2. From the 'Choose target variable' dropdown on the left sidebar, select the column that you want the model to predict.
        3. Indicate whether the problem is a regression or classification problem by selecting the appropriate option from the 'Problem nature' radio buttons on the left sidebar.
        4. Choose the machine learning model you want to use from the 'Choose algorithm model' dropdown on the left sidebar. The available options will change depending on whether you indicated a regression or classification problem.
        5. After choosing your model, the application will train the model on your data. You can view the training and testing scores, as well as the mean absolute error for the training and testing data, on the left sidebar under the 'Machine learning model results' section.
        6. If you want to understand the importance of each feature in your data, check the 'feature importance' checkbox on the left sidebar. You can choose between a 'Fast (not accurate enough)' calculation and a 'Shaply (slow but accurate)' calculation. The feature importance will be displayed in the main panel.
        7. If you want to compare the model's predictions with the actual target values, check the 'Compare prediction, with actual data?' checkbox on the left sidebar. A table comparing the predictions and actual values will be displayed in the main panel.
        8. If you have new data that you want to predict the target variable for, check the 'predict target from input data?' checkbox on the left sidebar and upload the new data using the 'Choose csv file to upload for prediction' button. Make sure the new data has the same structure as the original data. The predicted values will be displayed in the main panel, and you can download the predictions as a CSV file.
        
        Please ensure that your data is prepared correctly and does not contain any non-numeric values. If any errors occur, they will be displayed on the screen.
        """
    )


with tab3:
   st.header("Definitions")
   list=['Target','Independent Variables','Dependent variable','Features','Features importance', 'Training score','Testing score','coeffecient of determination (R^2)','Training set/Testing set','mean absolute error','Nan','CSV','Continuos','Classification','Backward fill','Forward fill','polynomial']
   selection_tab2=st.selectbox('',list)
   if selection_tab2=='Target': 
       st.markdown("* **Target** : this is the column that you want to predict based on independent variables, donated as Y, also known as dependent variable")
   elif selection_tab2=='Independent Variables':
       st.markdown("* **Independent variables** : these are the columns that you predict the target from them , donated as X1,X2,..ect, also known as features")
   elif selection_tab2=='Dependent variable':
       st.markdown("* **Dependent variable** : this is the column that you want to predict based on independent variables, donated as Y, also known as Target")
   elif selection_tab2=='Features':
       st.markdown("* **Features** : the columns that you predict the target from them , donated as X1,X2,..ect, also known as independent variables")
   elif selection_tab2=='Features importance':
       st.markdown("* **Features importance** : The importance of each feature in predicting the target, higher number means higher importance")
   elif selection_tab2=='Training score':
       st.markdown("* **Training score** : The coeffecient of determination for the training set")
   elif selection_tab2=='Testing score':
       st.markdown("* **Testing score** : The coeffecient of determination for the testing set")
   elif selection_tab2=='coeffecient of determination (R^2)':
       st.markdown("* **coeffecient of determination** : is a number between 0 and 1 that measures how well a statistical model predicts an outcome, also known by R square")
       st.markdown("* if the number is 1 , the model is perfectly predicts the target")
       st.markdown("* if the number is 0 , the model is prediction is poor")
       st.markdown("* the higher the number, the better the model is")
   elif selection_tab2=='Training set/Testing set':
       st.markdown("* **Training set/Testing set** : usually the data are split into two subsets, training set and testing set")
       st.markdown("* training set is the data that fed into the machine learning model")
       st.markdown('* testing set is the data we test the model on, in this set we predect the target `Y_predect` and compare it with `Y_actual`')
   elif selection_tab2=='mean absolute error':
       st.markdown("* **mean absolute error** : Absolute Error is the amount of error in your measurements. It is the difference between the predicted value and “actual” value. For example, if a model predict weight to be 90 Kgs but you know  true weight is 89 pounds, then the model has an absolute error of `90 Kgs – 89 lbs = 1 lbs.` and that is for one data point")
       st.markdown("* * mean absolute error is the mean of all absolute errors in all data")
   elif selection_tab2=='Nan':
       st.markdown("* **Nan** : Not a number, it mainly means missing values, so a column have 70 % Nan, means 70 % of this column has missing values")
   elif selection_tab2=='CSV':
       st.markdown("* **CSV** : Comma Separated Values, this is extension can be thought of as simplified xlsx file, Microsoft office can export any excel file to be CSV ")
   elif selection_tab2=='Continuos':
       st.markdown("* **Continuos** : if the target is a continuous value, like age or temperature. Then the algorithm should be of continuos nature ")
   elif selection_tab2=='Classification':
       st.markdown("* **Classification** : if the target is a not-continuous value, like 0 or 1, good or bad, type-1 or type-2 or type-3. Then the algorithm should be classification algorithm ")
   elif selection_tab2=='Backward fill':
       st.markdown("* **Backward fill** : One of the methods of filling nan values where each nan value is replaced by previous value in the same column ")
   elif selection_tab2=='Forward fill':
       st.markdown("* **Forward fill** : One of the methods of filling nan values where each nan value is replaced by next value in the same column ")
   elif selection_tab2=='polynomial':
       st.markdown("* **polynomial** :  a polynomial is an equation consisting of variables and coefficients, that involves only the operations of addition, subtraction, multiplication, and positive-integer powers of variables. An example of a polynomial of a single variable x is *x^2 − 4x + 7*. An example with three variables is *x^3 + 2xyz2 − yz + 1* ")   

        

with tab4:
    st.markdown(
        """
        ## About Machine Learning
        
        Machine Learning is a subset of artificial intelligence (AI) that provides systems the ability to learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.

        The learning process is based on feeding data to the system and allowing it to learn patterns and make decisions. For example, a machine learning model can be trained to recognize cats in a picture by showing it thousands of pictures of cats.

        There are two types of problems in machine learning: 

        - **Regression problems** involve predicting a continuous value. For example, predicting the price of a house based on its features is a regression problem.
        - **Classification problems** involve predicting a category or class. For example, predicting whether an email is spam or not is a classification problem.
        
        Different machine learning models can be used depending on the type of problem and the data. This application allows you to choose from several models, train them on your data, and make predictions with the trained model.
        """
    )

with tab5:
   st.header("How to choose the algorithm")
   st.markdown("* the ultimate way is to test each and all algorithms, and choose the one that achieves the best score in both training and testing sets")
   st.markdown("* in reality, this rarely happens as randomness plays a role here, also it is rarely found that one algorithms scores higher than all others in both training and testing sets")
   st.markdown("* Here a methedology that can help in choosing the best model [SCI-KIT LEARN METHODOLOGY](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)")

with tab6:
    st.markdown(" under preparation")
    #st.video('https://www.youtube.com/watch?v=hdLL5jjEOXM')

st.write('******************************************************************')

# Uploading data and converting it to numbers only 
data= st.sidebar.file_uploader("Choose excel or csv file to upload",type=['csv','xls','xlsx'],key='1')
if data is not None:
    try:
        df_raw = pd.read_csv(data,encoding_errors='ignore')
    except:
        pass
    try:
        df_raw = pd.read_csv(data)
    except:
        pass
    try:
        df_raw = pd.read_excel(data)
    except:
        pass
    try:
        df_raw = pd.read_excel(data, engine='openpyxl')
    except:
        pass
else:
    st.sidebar.write('*Kindly upload valid csv data')
    
if data is not None:
    try:
        st.write('Raw dataset') 
        st.dataframe(df_raw)
        st.write('******************************************************************')
    except:
        pass
    df=df_raw.copy()
    for column in df.columns:
        df[column]=pd.to_numeric(df[column],errors='coerce')
    # drop columns with all missing values
    df = df.dropna(axis=1, how='all')

    
    #Initial Nan processing
    if st.sidebar.checkbox(" Initial processing of Nan values"):
        st.sidebar.write('Choose the percentage of NaN present in each column, any column having more than this percent will be removed from the dataset')
        zz=st.sidebar.selectbox('columns to be removed from data having NAN percentage more than :',reversed(range(10,100,10)))        
        df=df.dropna(axis='columns', thresh=df.shape[0]*(1-(zz/100)))

    df_whole_numbers=df.copy()

    st.sidebar.write('====================================')
    
    # Choosing Target and modeling configuration
    yy = st.sidebar.selectbox('Choose target or dependent variable (y)', df.columns)

    cols = df.columns.tolist()
    cols.remove(yy)
    if st.sidebar.checkbox('remove some columns before modeling'):
        try:
            removed_x = st.sidebar.multiselect('choose columns to be removed', cols)
            cols = [col for col in cols if col not in removed_x]
        except:
            st.write("The selected columns could not be removed from modeling")
            pass
    if st.sidebar.checkbox('use only some columns in modeling'):
        try:
            used_x = st.sidebar.multiselect('choose columns to be used in modeling', cols)
            cols = [col for col in used_x if col != yy]
        except:
            st.write("the selected columns could not be used in modeling")
            pass

    df = df[cols + [yy]]
    st.sidebar.write('====================================')
    
    #Saving dataframe before preprocessing
    df_before_preprocessing=df.copy()

      
    # preprocessing options
    if st.sidebar.checkbox(" additional processing of Nan values"):
        substitution=st.sidebar.radio("**replace Nan values or delete them**",('Replace by Median','Replace by Most Frequent','Replace by Mean','Filling with Forward, Backward and Column Mean','Delete Nan rows'),index=4)
        if substitution =='Replace by Median':
            df=df.fillna(df.median())
        elif substitution =='Replace by Most Frequent':
            df=df.fillna(df.mode().iloc[0])
        elif substitution =='Delete Nan rows':
            df.dropna(inplace=True)
        elif substitution=='Filling with Forward, Backward and Column Mean':
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(df.mean())
        else:
            df = df.fillna(method='bfill').fillna(method='ffill').fillna(df.mean())
        st.sidebar.write('====================================')
        st.write('dataset after nan processing')
        st.write(df.describe())
        st.write('******************************************************************')        
    if st.sidebar.checkbox("Remove outliers from the data set"):
        try:
            outlier_limit=st.sidebar.slider('Number of Standard deviations data will be filtered upon',1.0,10.0,4.0,0.2)
            def df_without_outliers (data,a=4.0):
                df=data.copy()    
                z_scores = stats.zscore(df[df.describe().columns],nan_policy='omit')
                z_scores.fillna(0,inplace=True)   # in case one column is filled with nan values
                abs_z_scores = np.abs(z_scores)
                filtered_entries = (abs_z_scores < a).all(axis=1)
                df_without_outliers = df[filtered_entries]
                return df_without_outliers
            df = df_without_outliers(df, a= outlier_limit)
            st.write('dataset after outlier removal')
            st.write(df.describe())
        except:
            st.write('dataset could not be outliers removed')
            pass
        st.write('******************************************************************')
        st.sidebar.write('====================================')
    # Saving dataframe after outlier removal and nan processing   
    df_after_outlierremov_and_nanprocess=df.copy()
    
    if st.sidebar.checkbox(" Normalize dataset",key='n'):
        try:
            min_limit=st.sidebar.number_input('all columns will have minimum of:',value=1)
            max_limit=st.sidebar.number_input('all columns will have maximum of:',value=100)
            minmax_scaler=MinMaxScaler((min_limit,max_limit))
            df=pd.DataFrame(minmax_scaler.fit_transform(df),columns=df.columns)
            st.write('dataset after Normalization')
            st.write(df.describe())
        except:
            st.write('dataset could not be normalized')
            pass
        st.write('******************************************************************')
        st.sidebar.write('====================================')   
    if st.sidebar.checkbox("Make dataset has normal distribution-(Normalize should be checked)",key='n_d'):
        try:
            power_transformer=PowerTransformer(standardize=False)
            df=pd.DataFrame(power_transformer.fit_transform(df),columns=df.columns)
            st.write('dataset after normal distribution transformation')
            st.write(df.describe())
        except:
            st.write('dataset could not be transformed to normal distribution')
            pass
        st.write('******************************************************************')
    if st.sidebar.checkbox("Standarize dataset",key='s'):
        try:
            standard_scaler= StandardScaler()
            df=pd.DataFrame(standard_scaler.fit_transform(df),columns=df.columns)
            st.write('dataset after Standarization')
            st.write(df.describe())
        except:
            st.write('dataset could not be standarized')
            pass
        st.write('******************************************************************')            
      
    st.sidebar.write('====================================')



    #Saving df after processing
    Whole_df_after_preprocessing=df.copy()



    # Assigning X and y
    st.write('**you choosed**', yy,'***to be the target**')
    st.write('******************************************************************')
    y=df[[yy]]
    st.write('y description :', y.describe())
    st.write('******************************************************************')
    X=df.drop(yy,axis=1)
    st.write('******************************************************************')
    st.write('******************************************************************')
    st.write('******************************************************************')
    st.write('X shape:',X.shape,'Y shape:',y.shape)
      
    # Modeling and choosing algorithm
    problem_nature= st.sidebar.radio('Problem nature',['Continuos','Classification'],key='problem_nature')
    if problem_nature=='Continuos': 
        models=[DecisionTreeRegressor(),RandomForestRegressor(),AdaBoostRegressor(),GradientBoostingRegressor(),SGDRegressor(),ElasticNet(),Lasso(),LinearRegression(),SVR(kernel='linear'),SVR(kernel="rbf"),'polynomial regression',KNeighborsRegressor()]
        raw_model=st.sidebar.selectbox('Choose algorithm model',models)
        st.write(raw_model)
        #st.write(type(raw_model))
        #if isinstance(raw_model, sklearn.linear_model._base.LinearRegression):
        if raw_model=='polynomial regression':
            degree=st.sidebar.slider('choose polynomial degree',2,6,3,1)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        @st.cache(allow_output_mutation=True)
        def model_select_fit(raw_model):
            # IMPORTANT: Cache model_select_fit to prevent computation on every rerun
            if raw_model=='polynomial regression':
                #scaler=MinMaxScaler()
                #global X_scaled
                #X_scaled=scaler.fit_transform(X_train)
                #poly=PolynomialFeatures(degree=4)
                #global X_poly
                #X_poly=poly.fit_transform(X_scaled)
                #X_poly=poly.fit_transform(X_train)
                raw_model=Pipeline([('polynomial',PolynomialFeatures(degree=degree)),('linear regression',LinearRegression())])
                raw_model.fit(X_train,y_train)
            else:
                try:
                    raw_model.fit(X_train,y_train,random_state=42)
                except:
                    raw_model.fit(X_train,y_train)
            return raw_model
        model= model_select_fit(raw_model)
        y_train_pred=model.predict(X_train)
        y_test_pred=model.predict(X_test)

        training_accuracy=r2_score(y_train,y_train_pred)
        training_mean_error=mean_absolute_error(y_train, y_train_pred)

        testing_accuracy=r2_score(y_test,y_test_pred)
        testing_mean_error=mean_absolute_error(y_test,y_test_pred)

    if problem_nature=='Classification': 
        models=[DecisionTreeClassifier(),RandomForestClassifier(),AdaBoostClassifier(),GradientBoostingClassifier(),SVC(kernel='linear')]
        raw_model=st.sidebar.selectbox('Choose algorithm model',models)
        st.write(raw_model)
        #st.write(type(raw_model))
        #if isinstance(raw_model, sklearn.linear_model._base.LinearRegression):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        @st.cache(allow_output_mutation=True)
        def model_select_fit(raw_model):     
            try:
                raw_model.fit(X_train,y_train,random_state=42)
            except:
                raw_model.fit(X_train,y_train)
            return raw_model
        model= model_select_fit(raw_model)
        y_train_pred=model.predict(X_train)
        y_test_pred=model.predict(X_test)

        training_accuracy=accuracy_score(y_train,y_train_pred)
        training_mean_error=mean_absolute_error(y_train, y_train_pred)

        testing_accuracy=accuracy_score(y_test,y_test_pred)
        testing_mean_error=mean_absolute_error(y_test,y_test_pred)

    #Displaying results
    st.sidebar.write('training score = ',training_accuracy)
    st.sidebar.write('testing score = ',testing_accuracy)
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
    
    #Features importance calculations
    feature_importance=pd.DataFrame()
    if raw_model=='polynomial regression':
        feature_importance['Name'] = model.steps[0][1].get_feature_names_out(input_features=X.columns)
        feature_importance['column coeffecient'] = np.abs(model.steps[1][1].coef_[0])
        # if st.sidebar.checkbox('polynomial feature importance'):
        #     st.header('polynomial features importance')
        #     st.dataframe(feature_importance.sort_values(by='column coeffecient',ascending=False))
    else:
        feature_importance['Name'] = X.columns
        try:   
            feature_importance['importance in the model']=model.feature_importances_
        except:
            pass 
        try:
            feature_importance['column coeffecient']=np.abs(model.coef_[0])
        except:
            pass
        if len(feature_importance.columns)<2:
            shap_features=1

    #Features importance display
    st.sidebar.write('====================================')
    if st.sidebar.checkbox('feature importance',value=True):
        effect=st.sidebar.radio('',options=['Fast(not accurate enough)','Shaply( slow but accurate)'])
        if effect=='Fast(not accurate enough)':
            st.header(' Fast feature importance:')
            try:
                st.dataframe(feature_importance.sort_values(by='importance in the model',ascending=False))
            except:
                pass
            try:
                st.dataframe(feature_importance.sort_values(by='column coeffecient',ascending=False))
            except:
                pass
            if len(feature_importance.columns)<2:
                st.write(' Fast feature importance can not be made for this algorithm, try Shaply option')
        elif effect=='Shaply( slow but accurate)':
            st.header(' Shaply feature importance:')
            @st.cache
            def detailed_importance():
            # IMPORTANT: Cache Shaply features to prevent computation on every rerun
                # Fits the explainer
                explainer = shap.Explainer(model.predict, X_test)
                # Calculates the SHAP values - It takes some time
                shap_values = explainer(X_test,max_evals="auto")
                return shap_values            
            shap_values=detailed_importance()
            feature_importance=pd.DataFrame()
            feature_importance['Name']=shap_values.feature_names
            feature_importance['importance']=np.mean(np.abs(shap_values.values),axis=0)
            st.dataframe(feature_importance.sort_values(by='importance',ascending=False))
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values)
            st.pyplot(fig)
            #st.pyplot(fig=shap.summary_plot(shap_values),clear_figure=False)
    st.write('******************************************************************')

    # after predicting converting the values to the initial form
    Whole_df_after_preprocessing_contains_y_predict = Whole_df_after_preprocessing.copy()
    Whole_df_after_preprocessing_contains_y_predict[yy]=model.predict(X)
    Whole_df_after_preprocessing_contains_y_predict_transformed_to_original=Whole_df_after_preprocessing_contains_y_predict.copy()  
    # reverse order for preprocessing steps
    if st.session_state['s']:
        Whole_df_after_preprocessing_contains_y_predict_transformed_to_original = pd.DataFrame(standard_scaler.inverse_transform(Whole_df_after_preprocessing_contains_y_predict_transformed_to_original),columns=Whole_df_after_preprocessing_contains_y_predict_transformed_to_original.columns)
    if st.session_state['n_d']:
        Whole_df_after_preprocessing_contains_y_predict_transformed_to_original = pd.DataFrame(power_transformer.inverse_transform(Whole_df_after_preprocessing_contains_y_predict_transformed_to_original),columns=Whole_df_after_preprocessing_contains_y_predict_transformed_to_original.columns)
    if st.session_state['n']:
        Whole_df_after_preprocessing_contains_y_predict_transformed_to_original = pd.DataFrame(minmax_scaler.inverse_transform(Whole_df_after_preprocessing_contains_y_predict_transformed_to_original),columns=Whole_df_after_preprocessing_contains_y_predict_transformed_to_original.columns)     
    st.sidebar.write('====================================')
    st.write('******************************************************************')
    
    # Display comparing data
    if st.sidebar.checkbox('Compare prediction, with actual data ?'):
        comparing=pd.DataFrame()
        comparing['Actual']= df_after_outlierremov_and_nanprocess[yy].values
        comparing['prediction']=Whole_df_after_preprocessing_contains_y_predict_transformed_to_original[yy].values
        comparing['difference']=np.abs(comparing['Actual']-comparing['prediction'])
        st.subheader('actual Vs prediction')
        st.write(comparing)
        st.write('******************************************************************')
        st.write(comparing.describe())
        if problem_nature=='Continuos':
            figure = px.scatter(comparing,x='Actual', y='prediction')
            st.plotly_chart(figure)



    # Export data to excel files
    st.sidebar.write('====================================')        
    if st.sidebar.checkbox('Export data to excel file ?'):
        comparing=pd.DataFrame()
        comparing['Actual']=df_after_outlierremov_and_nanprocess[yy].values
        comparing['prediction']=Whole_df_after_preprocessing_contains_y_predict_transformed_to_original[yy].values
        comparing['difference']=np.abs(comparing['Actual']-comparing['prediction'])
        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv(index=False).encode('utf-8')

        # csv_1 = convert_df(feature_importance)
        csv_1= convert_df(feature_importance)
        csv_2 = convert_df(comparing)
        csv_3 = convert_df(Whole_df_after_preprocessing)
        st.download_button(label="Download feature importance as CSV", data=csv_1, file_name='features_importance.csv', mime='text/csv')
        st.download_button(label="Download actual/predicted Y as CSV", data=csv_2, file_name='Actual-predicted Y.csv', mime='text/csv')
        st.download_button(label="Download dataset after processing steps", data=csv_3, file_name='Processed_dataset.csv', mime='text/csv')
        st.write('******************************************************************')
    
    # display feature effect on target
    st.sidebar.write('====================================')
    if st.sidebar.checkbox('feature importance effect on target'):
        if effect =='Fast(not accurate enough)':
            try:
                st.subheader('feature effect on target')
                st.write(' all other features will be averaged and the chosen feature will be left as it is, then the model will be run and to measure its effect on target a plot is drawn')
                sorted_features= feature_importance.sort_values(by=feature_importance.columns[1],ascending=False)['Name'].values
                feature=st.sidebar.selectbox('Choose feature to evaluate its effect, based on the model',sorted_features)
                if raw_model !='polynomial regression':
                    evaluation_df=X.copy()
                    for column in evaluation_df.drop(feature, axis=1).columns:
                        evaluation_df.loc[:,column]=evaluation_df[column].mean()
                    st.write(evaluation_df)
                    y_evaluation=model.predict(evaluation_df)
                    if len(y_evaluation.shape)==1:
                        pass
                    else:
                        y_evaluation=y_evaluation.reshape(-1)
                    figure_1 = go.Figure()
                    figure_1.add_trace(go.Scatter(x=evaluation_df[feature], y=y_evaluation,mode='markers'))
                    figure_1.update_layout(xaxis_title=feature, yaxis_title=yy,title= f'effect of changing {feature} on {yy} ')
                    st.plotly_chart(figure_1)
                else:
                    st.write(" * if algorith is polynomial regression, feature importance will be hard to plot ")
            except:
                st.write('Fast feature importance can not be made for this algorithm, try Shaply option')
        else:
            st.subheader('shaply feature effect on target')
            st.write(' all other features will be averaged and the choosed feature will be left as it is, then the model will be run and to measure its effect on target a plot is drawn')
            sorted_features= feature_importance.sort_values(by=feature_importance.columns[1],ascending=False)['Name'].values
            feature=st.sidebar.selectbox('Choose feature to evaluate its effect, based on the model',sorted_features)
            # if raw_model !='polynomial regression':
            evaluation_df=X.copy()
            for column in evaluation_df.drop(feature, axis=1).columns:
                evaluation_df.loc[:,column]=evaluation_df[column].mean()
            for column in evaluation_df.drop(feature, axis=1).columns:
                evaluation_df[column]=pd.to_numeric(evaluation_df[column],errors='coerce')
            st.write(evaluation_df)
            y_evaluation=model.predict(evaluation_df)
            if len(y_evaluation.shape)==1:
                pass
            else:
                y_evaluation=y_evaluation.reshape(-1)
            figure_1 = go.Figure()
            figure_1.add_trace(go.Scatter(x=evaluation_df[feature], y=y_evaluation,mode='markers'))
            figure_1.update_layout(xaxis_title=feature, yaxis_title=yy,title= f'effect of changing {feature} on {yy} ')
            st.plotly_chart(figure_1)
        # else:
        #     st.write(" * if algorith is polynomial regression, feature importance will be hard to plot ")

st.sidebar.write('====================================') 
st.sidebar.write('====================================') 
st.sidebar.write('====================================')

 
# Prediction Button 
if st.sidebar.checkbox('predict target from input data?'):
    st.write('******************************************************************')    
    st.sidebar.markdown("### upload files in the second upload bottom only when you want to predict ")
    data_predict= st.sidebar.file_uploader("Choose csv file to upload for predection",type=['csv','xls','xlsx'],key='2')  
    st.sidebar.write('*Kindly upload valid excel or csv data for predection, with the same column names as the original one including the target, all data should be numbers with no NaN values')
    if data_predict is not None:  
        try:
            df_predict = pd.read_csv(data_predict,encoding_errors='ignore')
        except:
            pass
        try:
            df_predict = pd.read_csv(data_predict)
        except:
            pass
        try:
            df_predict = pd.read_excel(data_predict)
        except:
            pass
        try:
            df_predict = pd.read_excel(data_predict,engine='openpyxl')
        except:
            pass

        for column in df_predict.columns:
            try:
                df_predict[column]=pd.to_numeric(df_predict[column],errors='coerce')
            except:
                pass
        df_predict = df_predict[cols + [yy]]
        if st.session_state['n']:
            df_predict = pd.DataFrame(minmax_scaler.transform(df_predict),columns=df_predict.columns)
        if st.session_state['n_d']:
            df_predict = pd.DataFrame(power_transformer.transform(df_predict),columns=df_predict.columns)        
        if st.session_state['s']:
            df_predict = pd.DataFrame(standard_scaler.transform(df_predict),columns=df_predict.columns)         

        X_for_predection=df_predict[X.columns]
        st.header('Predection results')
        st.subheader('    input raw data for predection   ')
        st.dataframe(X_for_predection)
        for column in X_for_predection:
            try:
                X_for_predection[column]= X_for_predection[column].fillna(X_for_predection[column].mean())
            except:
                X_for_predection[column]= X_for_predection.fillna(X[column].mean())

        predection_data=pd.DataFrame()
        predics=model.predict(X_for_predection)
        if len(predics)==1:
            pass
        else:
            predics=predics.reshape(-1)
        predection_data[yy]= predics

        data_for_download= pd.concat([X_for_predection,predection_data],axis=1)
        st.write('******************************************************************')
        tab11, tab12, tab13, tab14 = st.columns(4)
        with tab11:
            st.markdown(" ###### Processed input data for predection")
            st.dataframe(X_for_predection)

        with tab12:
            st.markdown(" ###### Predected values from processed input")
            st.dataframe(predection_data)
        # after predicting converting the values to the initial form
        X_inverse_containing_y_predicted = X_for_predection.copy()
        X_inverse_containing_y_predicted[yy]=predics
        X_inverse_containing_y_predicted_to_original=X_inverse_containing_y_predicted.copy()  
        # reverse order for preprocessing steps
        if st.session_state['s']:
            X_inverse_containing_y_predicted_to_original = pd.DataFrame(standard_scaler.inverse_transform(X_inverse_containing_y_predicted_to_original),columns=X_inverse_containing_y_predicted_to_original.columns)
        if st.session_state['n_d']:
            X_inverse_containing_y_predicted_to_original = pd.DataFrame(power_transformer.inverse_transform(X_inverse_containing_y_predicted_to_original),columns=X_inverse_containing_y_predicted_to_original.columns)
        if st.session_state['n']:
            X_inverse_containing_y_predicted_to_original = pd.DataFrame(minmax_scaler.inverse_transform(X_inverse_containing_y_predicted_to_original),columns=X_inverse_containing_y_predicted_to_original.columns)               
        with tab13:
            st.markdown('###### Returned X-values after inversing prprocessing')
            st.dataframe(X_inverse_containing_y_predicted_to_original[X.columns])
        with tab14:
            st.markdown('###### y-Values after inversing preprocessing')
            st.dataframe(X_inverse_containing_y_predicted_to_original[yy])
            
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv(index=False).encode('utf-8')
        
        csv_file = convert_df(X_inverse_containing_y_predicted_to_original)
        st.download_button(label="Download data as CSV", data=csv_file, file_name='predection_data.csv', mime='text/csv')
st.sidebar.write('====================================') 
st.sidebar.write('====================================') 


