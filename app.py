import streamlit as st
import pandas as pd
import seaborn as sns
import pickle 
import joblib

##Models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# st.title('Automatic EDA')
html_temp = """
<div style="background-color:tomato; padding:10px">
<h2 style ="color:white; text-align:center;">Automatic EDA Project</h2></div>
"""
st.markdown(html_temp, unsafe_allow_html=True)
st.write("")
# st.write("Built with Streamlit")

lists = ['EDA', 'Plots', 'ML Models', 'About']
choice = st.sidebar.selectbox("Select", lists)

if choice == 'EDA':
    # st.subheader('EDA')
    uploaded_file = st.file_uploader('Upload dataset', type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)

        if st.checkbox('Shape'):
            st.write(df.shape)
        
        if st.checkbox('Columns'):
            column = df.columns.to_list()
            st.write(column)

        if st.checkbox('Summary'):
            st.write(df.describe())
        
        if st.checkbox('Show Selected Columns'):
            column = df.columns.to_list()
            selected_columns = st.multiselect("Select Columns", column)
            df1 = df[selected_columns]
            st.dataframe(df1)
            btn1 = st.button('Save new Dataframe')
            if btn1:
                df1.to_csv('Selected Column Dataframe.csv')

        if st.checkbox('Value Counts'):
            st.write(df.iloc[:,-1].value_counts())

        if st.checkbox('Unique Values'):
            all_columns_1 = df.columns.to_list()
            selected_columns_1 = st.multiselect("Select Columns", all_columns_1)
            st.write(df[selected_columns_1].nunique())
        
        if st.checkbox('Delete Columns'):
            all_columns_2 = df.columns.to_list()
            selected_columns_2 = st.selectbox("Select Columns", all_columns_2)
            df2 = df.drop([selected_columns_2], axis=1)
            st.dataframe(df2)
            btn = st.button('Save new Dataframe')
            if btn:
                df2.to_csv('Deleted Column Dataframe.csv')
        
        if st.checkbox('Null Values in Column'):
            all_columns_3 = df.columns.to_list()
            selected_columns_3 = st.multiselect("Select Columns", all_columns_3)
            df3 = df[selected_columns_3].isnull()
            st.dataframe(df3)
        
        if st.checkbox('Handling NaN values'):
            x = st.text_input('Enter value with which you want to replace NaN')
            if st.button('Handle NaN'):
                df4 = df.fillna(x)
                st.dataframe(df4)

                if st.button('Save Dataframe'):
                    df4.to_csv('Handeled NaN.csv')

        if st.checkbox('Heatmap'):
            st.write(sns.heatmap(df.corr(), annot=True))
            st.pyplot()

elif choice == 'Plots':
    st.subheader('Data Visulalization')
    uploaded_file_1 = st.file_uploader('Upload dataset', type=['csv'])
    if uploaded_file_1 is not None:
        data = pd.read_csv(uploaded_file_1)
        st.dataframe(data)

        plots = ['area', 'bar', 'line', 'hist', 'box', 'kde']
        type_of_plot = st.selectbox('Type of Plot', plots)

        all_column_names = data.columns.to_list()
        selected_columns_names = st.multiselect('Select Columns to Plot', all_column_names)

        generate_plot = st.button('Generate Plot')
        if generate_plot:
            st.success("Generating {} plot for {}".format(type_of_plot, selected_columns_names))

            if type_of_plot == 'area':
                custom_data_1 = data[selected_columns_names]
                st.area_chart(custom_data_1)
            
            elif type_of_plot == 'bar':
                custom_data_2 = data[selected_columns_names]
                st.bar_chart(custom_data_2)
            
            elif type_of_plot == 'line':
                custom_data_3 = data[selected_columns_names]
                st.line_chart(custom_data_3)
            
            # Custom Plot 
            elif type_of_plot:
                cust_plot= data[selected_columns_names].plot(kind=type_of_plot)
                st.write(cust_plot)
                st.pyplot()

elif choice == 'ML Models':
    st.subheader('ML Models')
    uploaded_file_2 = st.file_uploader('Upload dataset', type=['csv'])
    if uploaded_file_2 is not None:
        data1 = pd.read_csv(uploaded_file_2)
        st.dataframe(data1)

        models = ['Linear Regression', 'KNN', 'SVM', 'Random Forest']
        model_selection = st.selectbox('Choose ML Model', models)

        if model_selection == 'Linear Regression':
            all_column_names1 = data1.columns.to_list()
            selected_features = st.multiselect('Select features', all_column_names1)
            x = data1[selected_features].values 
            selected_target = st.selectbox('Select target variable', all_column_names1)
            y = data1[selected_target].values
            if st.button('Evaluate Model'):
                #Classification
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
                regressor = LinearRegression()
                regressor.fit(x_train, y_train)
                y_pred = regressor.predict(x_test)
                # st.write(y_pred)
                acc_score = accuracy_score(y_test, y_pred.round(), normalize=False)
                st.success('{} model gives an accurcay of {}'.format(model_selection, acc_score))
                if st.button('Save model'):
                    pickle.dump(regressor, open('Linear_Regression.pkl', 'wb'))


        elif model_selection == 'KNN':
            all_column_names1 = data1.columns.to_list()
            selected_features = st.multiselect('Select features', all_column_names1)
            x = data1[selected_features].values 
            selected_target = st.selectbox('Select target variable', all_column_names1)
            st.write('Select non-continuous values')
            y = data1[selected_target].values

            neighbors = st.slider("Neighbors", 1, 15)
            if st.button('Evaluate Model'):         
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)       
                neigh = KNeighborsClassifier(neighbors)
                neigh.fit(x_train, y_train)
                y_pred = neigh.predict(x_test)
                acc_score = accuracy_score(y_test, y_pred, normalize=False)
                st.success('{} model gives an accurcay of {}'.format(model_selection, acc_score))
                if st.button('Save model'):
                    pickle.dump(neigh, open('KNN.pkl', 'wb'))

        elif model_selection == 'SVM':
            all_column_names1 = data1.columns.to_list()
            selected_features = st.multiselect('Select features', all_column_names1)
            x = data1[selected_features].values 
            selected_target = st.selectbox('Select target variable', all_column_names1)
            st.write('Select non-continuous values')
            y = data1[selected_target].values

            C = st.slider("Neighbors", 0.1, 10.0)
            if st.button('Evaluate Model'):         
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)       
                vector = SVC(C)
                vector.fit(x_train, y_train)
                y_pred = vector.predict(x_test)
                acc_score = accuracy_score(y_test, y_pred, normalize=False)
                st.success('{} model gives an accurcay of {}'.format(model_selection, acc_score))
                if st.button('Save model'):
                    pickle.dump(vector, open('SVM.pkl', 'wb'))
        
        elif model_selection == 'Random Forest':
            all_column_names1 = data1.columns.to_list()
            selected_features = st.multiselect('Select features', all_column_names1)
            x = data1[selected_features].values 
            selected_target = st.selectbox('Select target variable', all_column_names1)
            st.write('Select non-continuous values')
            y = data1[selected_target].values

            if st.button('Evaluate Model'):         
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)       
                clf = RandomForestClassifier()
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
                acc_score = accuracy_score(y_test, y_pred, normalize=False)
                st.success('{} model gives an accurcay of {}'.format(model_selection, acc_score))
                if st.button('Save model'):
                    pickle.dump(clf, open('Random_Forest.pkl', 'wb'))

elif choice == 'About':
    st.subheader('About this App')
    html_temp1 = """
    <p> Automatic EDA Web App is a platform for Data Science and Machine Learning Enthusiasts where in they can perform Exploratory Data Analysis to its most advanced concepts with just a click. People with no prior knowledge of Data Science can see and perform data analytics on their preferred datasets. 
<br>This Web App consists of extensive tools for data analytics from basics of data science to data visualization. Data Visualisation tools of Streamlit makes it easier for the user to plot beautiful plots and visualize their data in their preferred manner. Automatic EDA Web App contains lot of tools for data analysis from Summary of a dataset to handling missing values in a data frame. Data frames after preferred analysis can be saved into their local machines.
This Web App also provides user the flexibility to build models and save them for later use. Models like Linear Regression, Random Forest are provided in order to get the best results for the desired dataset. 
<br><br>
Explore. Dive in already!
<br>
Code in <a href="https://github.com/killerrings">Github</a>
</p>
    """ 
    # st.write('Automatic EDA Web App is a platform for Data Science and Machine Learning Enthusiasts where in they can perform Exploratory Data Analysis to its most advanced concepts with just a click. People with no prior knowledge of Data Science can see and perform data analytics on their preferred datasets. This web App consists of extensive tools for data analytics from basics of data science to data visualization. Data Visualisation tools of Streamlit makes it easier for the user to plot beautiful plots and visualize their data in their preferred manner. Automatic EDA Web App contains lot of tools for data analysis from Summary of a dataset to handling missing values in a data frame. Data frames after preferred analysis can be saved into their local machines. This Web App also provides user the flexibility to build models and save them for later use. Models like Linear Regression, Random Forest are provided in order to get the best results for the desired dataset. Explore. Dive in already!')
    st.markdown(html_temp1,unsafe_allow_html=True)

          




            







