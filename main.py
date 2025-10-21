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


st.set_page_config("machine learning app",":chart_with_upwards_trend:",layout="wide",initial_sidebar_state="expanded")

# Initialize session state for preprocessing flags
if 'n' not in st.session_state:
    st.session_state['n'] = False
if 'n_d' not in st.session_state:
    st.session_state['n_d'] = False
if 's' not in st.session_state:
    st.session_state['s'] = False

st.title('🤖 Machine Learning Application')
st.markdown("""
<style>
    .main-header {
        font-size: 1.2rem;
        color: #4A90E2;
        padding: 10px 0;
        border-bottom: 2px solid #4A90E2;
        margin-bottom: 20px;
    }
    .step-box {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4A90E2;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .info-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("### Welcome! Train machine learning models in 4 easy steps - no coding required!")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Main App","📖 Quick Start Guide", "📚 Definitions","🧠 About ML","💡 Algorithm Selection", "🎯 Examples"])

with tab1:
    st.markdown('<div class="info-box">👈 Use the sidebar to get started! Follow the steps below.</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("## 🚀 Quick Start Guide")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Step 1️⃣")
        st.info("**Upload Data**")
    with col2:
        st.markdown("""
        - Click the **'Choose excel or csv file to upload'** button in the sidebar
        - Select your CSV or Excel file
        - Ensure column names are in the first row
        - Data should be mostly numeric
        """)

    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Step 2️⃣")
        st.info("**Configure Your Model**")
    with col2:
        st.markdown("""
        - **Choose target variable**: Select the column you want to predict
        - **Problem nature**: Select 'Continuous' for numeric predictions or 'Classification' for categories
        - **Choose algorithm**: Pick a machine learning model from the dropdown
        """)

    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Step 3️⃣")
        st.info("**Optional: Preprocess**")
    with col2:
        st.markdown("""
        - Handle missing values (NaN)
        - Remove outliers
        - Normalize or standardize data
        - These steps can improve model performance
        """)

    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Step 4️⃣")
        st.info("**View Results**")
    with col2:
        st.markdown("""
        - See training and testing scores
        - Check feature importance
        - Compare predictions vs actual values
        - Export results to CSV
        - Make predictions on new data
        """)

    st.success("💡 **Tip**: Work through the sidebar options from top to bottom for the best experience!")


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

st.markdown("---")

# Uploading data and converting it to numbers only
st.sidebar.markdown("## 🔹 Step 1: Upload Your Data")
data= st.sidebar.file_uploader("Choose excel or csv file to upload",type=['csv','xls','xlsx'],key='1',help="Upload a CSV or Excel file with your training data")
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
    st.sidebar.info('📤 Please upload a CSV or Excel file to begin')
    st.info("👈 **Get Started**: Upload your data file using the sidebar to begin training your model!")

if data is not None:
    st.sidebar.success('✅ Data uploaded successfully!')
    try:
        with st.expander("📋 View Raw Dataset", expanded=True):
            st.dataframe(df_raw)
        st.markdown("---")
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

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🔹 Step 2: Configure Model")

    # Choosing Target and modeling configuration
    yy = st.sidebar.selectbox('Choose target or dependent variable (y)', df.columns, help="Select the column you want to predict")

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

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🔹 Step 3: Data Preprocessing (Optional)")

    #Saving dataframe before preprocessing
    df_before_preprocessing=df.copy()


    # preprocessing options
    with st.sidebar.expander("🧹 Handle Missing Values (NaN)", expanded=False):
     if st.checkbox(" Process Nan values", key="process_nan"):
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
        with st.expander("📊 Dataset After NaN Processing"):
            st.write(df.describe())

    with st.sidebar.expander("🎯 Remove Outliers", expanded=False):
     if st.checkbox("Remove outliers from the data set", key="remove_outliers"):
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
            with st.expander("📊 Dataset After Outlier Removal"):
                st.write(df.describe())
        except:
            st.error('Dataset could not have outliers removed')
            pass

    # Saving dataframe after outlier removal and nan processing
    df_after_outlierremov_and_nanprocess=df.copy()

    with st.sidebar.expander("📏 Normalize Data", expanded=False):
     if st.checkbox("Normalize dataset",key='n'):
        try:
            min_limit=st.sidebar.number_input('all columns will have minimum of:',value=1)
            max_limit=st.sidebar.number_input('all columns will have maximum of:',value=100)
            minmax_scaler=MinMaxScaler((min_limit,max_limit))
            df=pd.DataFrame(minmax_scaler.fit_transform(df),columns=df.columns)
            with st.expander("📊 Dataset After Normalization"):
                st.write(df.describe())
        except:
            st.error('Dataset could not be normalized')
            pass

    with st.sidebar.expander("📊 Normal Distribution Transform", expanded=False):
     if st.checkbox("Make dataset has normal distribution",key='n_d',help="Apply this after normalization for better results"):
        try:
            power_transformer=PowerTransformer(standardize=False)
            df=pd.DataFrame(power_transformer.fit_transform(df),columns=df.columns)
            with st.expander("📊 Dataset After Normal Distribution Transform"):
                st.write(df.describe())
        except:
            st.error('Dataset could not be transformed to normal distribution')
            pass

    with st.sidebar.expander("⚖️ Standardize Data", expanded=False):
     if st.checkbox("Standardize dataset",key='s'):
        try:
            standard_scaler= StandardScaler()
            df=pd.DataFrame(standard_scaler.fit_transform(df),columns=df.columns)
            with st.expander("📊 Dataset After Standardization"):
                st.write(df.describe())
        except:
            st.error('Dataset could not be standardized')
            pass

    st.sidebar.markdown("---")

    #Saving df after processing
    Whole_df_after_preprocessing=df.copy()

    # Assigning X and y
    st.success(f'✅ **Target Variable Selected**: {yy}')

    with st.expander("📊 Target Variable (y) Statistics", expanded=False):
        y=df[[yy]]
        st.write(y.describe())

    y=df[[yy]]
    X=df.drop(yy,axis=1)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Features (X) Shape", f"{X.shape[0]} rows × {X.shape[1]} columns")
    with col2:
        st.metric("Target (y) Shape", f"{y.shape[0]} rows × {y.shape[1]} column")

    st.markdown("---")

    # Modeling and choosing algorithm
    st.sidebar.markdown("## 🔹 Step 4: Select Algorithm")
    problem_nature= st.sidebar.radio('Problem nature',['Continuos','Classification'],key='problem_nature',help="Continuous for numeric predictions, Classification for categories")
    if problem_nature=='Continuos':
        models=[DecisionTreeRegressor(),RandomForestRegressor(),AdaBoostRegressor(),GradientBoostingRegressor(),SGDRegressor(),ElasticNet(),Lasso(),LinearRegression(),SVR(kernel='linear'),SVR(kernel="rbf"),'polynomial regression',KNeighborsRegressor()]
        raw_model=st.sidebar.selectbox('Choose algorithm model',models)
        st.info(f"🤖 **Selected Model**: {raw_model}")
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
        st.info(f"🤖 **Selected Model**: {raw_model}")
        #st.write(type(raw_model))
        #if isinstance(raw_model, sklearn.linear_model._base.LinearRegression):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        @st.cache(allow_output_mutation=True)
        def model_select_fit(raw_model):
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
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📊 Model Performance")
    st.sidebar.metric("Training Score", f"{training_accuracy:.4f}")
    st.sidebar.metric("Testing Score", f"{testing_accuracy:.4f}")

    st.markdown("---")
    st.header('🎯 Machine Learning Model Results')

    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📈 Training Score",
            value=f"{training_accuracy:.4f}",
            help="Higher is better (max 1.0)"
        )

    with col2:
        st.metric(
            label="📉 Testing Score",
            value=f"{testing_accuracy:.4f}",
            help="Higher is better (max 1.0)"
        )

    with col3:
        st.metric(
            label="📊 Train MAE",
            value=f"{training_mean_error:.4f}",
            help="Mean Absolute Error - Lower is better"
        )

    with col4:
        st.metric(
            label="📊 Test MAE",
            value=f"{testing_mean_error:.4f}",
            help="Mean Absolute Error - Lower is better"
        )

    # Add interpretation
    if testing_accuracy > 0.9:
        st.success("🎉 Excellent model performance!")
    elif testing_accuracy > 0.7:
        st.info("👍 Good model performance!")
    elif testing_accuracy > 0.5:
        st.warning("⚠️ Moderate model performance. Consider trying different algorithms or preprocessing.")
    else:
        st.error("❌ Poor model performance. Try different preprocessing or algorithms.")

    st.markdown("---")
    
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
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🔹 Step 5: Analysis & Export")
    if st.sidebar.checkbox('📊 View Feature Importance',value=False):
        effect=st.sidebar.radio('Calculation Method:',options=['Fast(not accurate enough)','Shaply( slow but accurate)'])
        if effect=='Fast(not accurate enough)':
            st.subheader('⚡ Fast Feature Importance')
            try:
                st.dataframe(feature_importance.sort_values(by='importance in the model',ascending=False), use_container_width=True)
            except:
                pass
            try:
                st.dataframe(feature_importance.sort_values(by='column coeffecient',ascending=False), use_container_width=True)
            except:
                pass
            if len(feature_importance.columns)<2:
                st.warning('⚠️ Fast feature importance cannot be calculated for this algorithm. Try the Shaply option.')
        elif effect=='Shaply( slow but accurate)':
            st.subheader('🎯 SHAP Feature Importance')
            st.info('⏳ This may take a few moments to calculate...')
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
            st.dataframe(feature_importance.sort_values(by='importance',ascending=False), use_container_width=True)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values)
            st.pyplot(fig)
            #st.pyplot(fig=shap.summary_plot(shap_values),clear_figure=False)
    st.markdown("---")

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

    # Display comparing data
    if st.sidebar.checkbox('📈 Compare Predictions vs Actual'):
        comparing=pd.DataFrame()
        comparing['Actual']= df_after_outlierremov_and_nanprocess[yy].values
        comparing['prediction']=Whole_df_after_preprocessing_contains_y_predict_transformed_to_original[yy].values
        comparing['difference']=np.abs(comparing['Actual']-comparing['prediction'])

        st.subheader('📊 Actual vs Prediction Comparison')

        col1, col2 = st.columns([2, 1])

        with col1:
            st.dataframe(comparing, use_container_width=True)

        with col2:
            st.write("**Statistics:**")
            st.write(comparing.describe())

        if problem_nature=='Continuos':
            st.subheader('📈 Prediction Scatter Plot')
            figure = px.scatter(comparing,x='Actual', y='prediction',
                              title='Actual vs Predicted Values',
                              labels={'Actual': 'Actual Values', 'prediction': 'Predicted Values'})
            # Add perfect prediction line
            figure.add_trace(go.Scatter(x=comparing['Actual'], y=comparing['Actual'],
                                       mode='lines', name='Perfect Prediction',
                                       line=dict(color='red', dash='dash')))
            st.plotly_chart(figure, use_container_width=True)

    # Export data to excel files
    if st.sidebar.checkbox('💾 Export Results to CSV'):
        comparing=pd.DataFrame()
        comparing['Actual']=df_after_outlierremov_and_nanprocess[yy].values
        comparing['prediction']=Whole_df_after_preprocessing_contains_y_predict_transformed_to_original[yy].values
        comparing['difference']=np.abs(comparing['Actual']-comparing['prediction'])
        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv(index=False).encode('utf-8')

        st.subheader('💾 Export Data Files')

        col1, col2, col3 = st.columns(3)

        # csv_1 = convert_df(feature_importance)
        csv_1= convert_df(feature_importance)
        csv_2 = convert_df(comparing)
        csv_3 = convert_df(Whole_df_after_preprocessing)

        with col1:
            st.download_button(
                label="📊 Feature Importance",
                data=csv_1,
                file_name='features_importance.csv',
                mime='text/csv',
                use_container_width=True
            )

        with col2:
            st.download_button(
                label="📈 Predictions vs Actual",
                data=csv_2,
                file_name='Actual-predicted Y.csv',
                mime='text/csv',
                use_container_width=True
            )

        with col3:
            st.download_button(
                label="📋 Processed Dataset",
                data=csv_3,
                file_name='Processed_dataset.csv',
                mime='text/csv',
                use_container_width=True
            )

        st.success('✅ Click the buttons above to download your files!')
        st.markdown("---")
    
    # display feature effect on target
    if st.sidebar.checkbox('🔬 Analyze Feature Effects on Target'):
        if effect =='Fast(not accurate enough)':
            try:
                st.subheader('🔬 Feature Effect on Target')
                st.info('ℹ️ This analysis averages all other features while varying the selected feature to measure its individual impact on predictions.')
                sorted_features= feature_importance.sort_values(by=feature_importance.columns[1],ascending=False)['Name'].values
                feature=st.sidebar.selectbox('Select feature to analyze:',sorted_features)
                if raw_model !='polynomial regression':
                    evaluation_df=X.copy()
                    for column in evaluation_df.drop(feature, axis=1).columns:
                        evaluation_df.loc[:,column]=evaluation_df[column].mean()

                    with st.expander("📊 View Evaluation DataFrame", expanded=False):
                        st.write(evaluation_df)

                    y_evaluation=model.predict(evaluation_df)
                    if len(y_evaluation.shape)==1:
                        pass
                    else:
                        y_evaluation=y_evaluation.reshape(-1)
                    figure_1 = go.Figure()
                    figure_1.add_trace(go.Scatter(x=evaluation_df[feature], y=y_evaluation,mode='markers',
                                                  marker=dict(size=8, color='#4A90E2')))
                    figure_1.update_layout(
                        xaxis_title=feature,
                        yaxis_title=yy,
                        title=f'Effect of {feature} on {yy}',
                        template='plotly_white'
                    )
                    st.plotly_chart(figure_1, use_container_width=True)
                else:
                    st.warning("⚠️ For polynomial regression, individual feature effects are complex to visualize due to interaction terms.")
            except:
                st.error('❌ Fast feature importance cannot be calculated for this algorithm. Try the Shaply option.')
        else:
            st.subheader('🎯 SHAP Feature Effect on Target')
            st.info('ℹ️ This analysis averages all other features while varying the selected feature to measure its individual impact on predictions.')
            sorted_features= feature_importance.sort_values(by=feature_importance.columns[1],ascending=False)['Name'].values
            feature=st.sidebar.selectbox('Select feature to analyze:',sorted_features)
            # if raw_model !='polynomial regression':
            evaluation_df=X.copy()
            for column in evaluation_df.drop(feature, axis=1).columns:
                evaluation_df.loc[:,column]=evaluation_df[column].mean()
            for column in evaluation_df.drop(feature, axis=1).columns:
                evaluation_df[column]=pd.to_numeric(evaluation_df[column],errors='coerce')

            with st.expander("📊 View Evaluation DataFrame", expanded=False):
                st.write(evaluation_df)

            y_evaluation=model.predict(evaluation_df)
            if len(y_evaluation.shape)==1:
                pass
            else:
                y_evaluation=y_evaluation.reshape(-1)
            figure_1 = go.Figure()
            figure_1.add_trace(go.Scatter(x=evaluation_df[feature], y=y_evaluation,mode='markers',
                                         marker=dict(size=8, color='#28a745')))
            figure_1.update_layout(
                xaxis_title=feature,
                yaxis_title=yy,
                title=f'Effect of {feature} on {yy}',
                template='plotly_white'
            )
            st.plotly_chart(figure_1, use_container_width=True)
        # else:
        #     st.write(" * if algorith is polynomial regression, feature importance will be hard to plot ")

st.sidebar.markdown("---")
st.sidebar.markdown("---")

# Prediction Button
if st.sidebar.checkbox('🔮 Predict on New Data'):
    st.markdown("---")
    st.header('🔮 Make Predictions on New Data')
    st.sidebar.markdown("#### 📤 Upload New Data")
    data_predict= st.sidebar.file_uploader("Choose csv file to upload for prediction",type=['csv','xls','xlsx'],key='2',help="Upload new data with same structure as training data")
    st.info('📋 **Important**: Upload data with the same column names as your training data. The target column should be included (values can be dummy/placeholder).')
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
        st.success('✅ New data uploaded successfully!')
        st.subheader('📊 Prediction Results')

        with st.expander('📋 View Input Data for Prediction', expanded=False):
            st.dataframe(X_for_predection, use_container_width=True)
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

        st.markdown("---")

        tab11, tab12, tab13, tab14 = st.tabs(["📊 Processed Input", "🎯 Predictions", "🔄 Original Scale Input", "✅ Final Predictions"])

        with tab11:
            st.markdown("**Processed Input Data**")
            st.dataframe(X_for_predection, use_container_width=True)

        with tab12:
            st.markdown("**Predicted Values (Processed Scale)**")
            st.dataframe(predection_data, use_container_width=True)
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
            st.markdown("**Input Features (Original Scale)**")
            st.dataframe(X_inverse_containing_y_predicted_to_original[X.columns], use_container_width=True)

        with tab14:
            st.markdown("**Final Predictions (Original Scale)**")
            st.dataframe(X_inverse_containing_y_predicted_to_original[[yy]], use_container_width=True)

        st.markdown("---")

        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv(index=False).encode('utf-8')

        csv_file = convert_df(X_inverse_containing_y_predicted_to_original)

        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            st.download_button(
                label="💾 Download Predictions as CSV",
                data=csv_file,
                file_name='prediction_data.csv',
                mime='text/csv',
                use_container_width=True
            )

        st.success('✅ Predictions complete! Download your results above.')

st.sidebar.markdown("---") 


