import streamlit as st
import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np
import cx_Oracle
import xlrd
import xlsxwriter
import base64
from io import BytesIO
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import types, create_engine

st.title('Data Preprocessing app')

temp = '\\temp.csv'
path = os.getcwd()

path = path+temp










# File Uploading

def file_upload():
    try:

        file_options = ('.XLSX', '.CSV', 'Oracle')
        file_select = st.sidebar.radio('Select type of file', file_options)

        if file_select == '.XLSX':
            uploaded_file = st.sidebar.file_uploader('select a file', type='xlsx')

            if uploaded_file:

                if st.sidebar.button('Upload File'):
                    df = pd.read_excel(uploaded_file)
                    st.dataframe(df)
                    df.to_csv(path, index=False)
                    return df


        elif file_select == '.CSV':
            uploaded_file = st.sidebar.file_uploader('Select a file', type='.csv')

            if uploaded_file:
                if st.sidebar.button('Upload File'):
                    df = pd.read_csv(uploaded_file)
                    st.dataframe(df)

                    df.to_csv(path, index=False)
                    return df

        else:

            st.info('Enter Oracle Database information')
            user = st.text_input('Enter the user name')
            Password = st.text_input('Enter the Password', type='password')
            host = st.text_input('Enter the host Address')
            port_number = st.text_input('Enter the port number')
            query = st.text_input('Enter the required query')

            if st.button('connect'):
                connection_query = '{}/{}@{}:{}/ORCL'.format(user, Password, host, port_number)
                connection = cx_Oracle.connect(connection_query)

                if connection is not None:
                    st.info('Connection Established successfully')
                    df = pd.read_sql(query, connection)
                    st.dataframe(df)
                    df.to_csv(path, index=False)

    except Exception as e:
        st.write("Error", e.__class__, "occured")
        return df


# IQR Function

def iqr(df, column_name):
    try:
        if column_name:
            q1 = df[column_name].quantile(0.25)
            q3 = df[column_name].quantile(0.75)
            iqr = q3 - q1
            lower_value = q1 - 1.5 * iqr
            upper_value = q3 + 1.5 * iqr
            df_rem_out = df[~((df < lower_value) | (df > upper_value))]
            return df_rem_out
    except Exception as e:
        st.write("Error", e.__class__, "occurred")
        return df


# Outlier Function

def outlier_Treatment():
    try:

        df = pd.read_csv(path)
        column_name = st.text_input('Enter the column where outliers treatment required:')
        st.info('Find the list of columns below:')
        st.write(df.columns)
        st.sidebar.button('process iqr')
        df = pd.read_csv(path)
        if column_name in df.columns:
            df = iqr(df, column_name)
            df.to_csv(path, index=False)
            return df
        else:
            st.info('Unknown Column')

    except Exception as e:
        st.write("Error", e.__class__, "occured")
        return df















# Missing value treatment

def mvt_mean(df):
    try:

        clean_df = (df.fillna(df.mean()))
        clean_df.fillna(clean_df.select_dtypes(include=object).mode().iloc[0], inplace=True)
        st.dataframe(clean_df)
        st.write(clean_df.dtypes)
        st.write('data description', df.describe())
        st.info('only numerical values will be treated using Mean')
        return clean_df


    except Exception as e:
        st.write('error', e.__class__, 'occured')
        return df


def mvt_median(df):
    try:

        clean_df = (df.fill_na(df.median()))
        clean_df.fill_na(clean_df.select_dtypes(include='object').mode().iloc[0], inplace=True)
        st.dataframe(clean_df)
        st.write(df.dtypes)
        st.info('The total missing value is:{:.2f}%'.format(((df.isna().sum().sum()) / (df.count().sum()) * 100)))
        st.info('Median is being treated:{}'.format(list(dict(df.median()).keys())))
        st.info('Dataframe shape is:'.format(df.shape))
        st.write('Data description:', df.describe())
        st.info('only numerical values will be treated here')
        st.info('categorical data will be treated using Mode')
        st.line_chart(clean_df)
        return clean_df

    except Exception as e:
        st.write('Error', e.__class__, "occured")
        return df


def mvt_mode(df):
    try:

        clean_df = (df.fillna(df.select_dtypes(include='object').mode().iloc[0]))
        st.dataframe(clean_df)
        return clean_df
    except Exception as e:
        st.write('Error', e.__class__, 'occured.')
        return df


def mvt_knn(df):
    try:

        num_col = list(df.select_dtypes(include='float64').columns)
        knn = KNNImputer(n_neighbors=1, add_indicator=True)
        knn.fit(df[num_col])
        knn_imputer = pd.DataFrame(knn.transform(df[num_col]))
        df[num_col] = knn_imputer.iloc[:, :df[num_col].shape[1]]
        clean_df = df
        clean_df = df.fillna(df.mode().iloc[0])
        st.dataframe(clean_df)
        return df

    except Exception as e:
        st.write('Error', e.__class__, 'Occured')
        return df


def mvt_options(df):
    try:

        m_options = ('Mean', 'Mode', 'Median', 'KNN Imputer')
        mvt_selection = st.sidebar.radio('Select the method for missing value treatment', m_options)
        if mvt_selection == 'Mean':
            st.sidebar.write('You selected Mean')
            if st.sidebar.button('Process Mean'):
                df = pd.read_csv(path)
                df = mvt_mean(df)
                df.to_csv(path, index=False)
                return df
        elif mvt_selection == 'Mode':
            st.sidebar.write('you selected Mode')
            if st.sidebar.button('Process Mode'):
                df = pd.read_csv(path)
                df = mvt_mode(df)
                df.to_csv(path, index=False)
                return df
        elif mvt_selection == 'Median':
            st.sidebar.write('you selected Median')
            if st.sidebar.button('Process Median'):
                df = pd.read_csv(path)
                df = mvt_median(df)
                df.to_csv(path, index=False)
                return df

        else:
            st.sidebar.write('You Selected KNN Imputer')
            if st.sidebar.button('Process KNN Imputer'):
                df = pd.read_csv(path)
                df = mvt_knn(df)
                df.to_csv(path, index=False)
                return df

    except Exception as e:
        st.write('Error', e.__class__, 'Occured')
        return df


# Feature Scaling Function


def f_ss(df):
    try:

        X = df.select_dtypes(include=np.number)
        X_mean = np.mean(X)
        std_X = np.std(X)
        X_std = (X - np.mean(X)) / np.std(X)
        st.dataframe(X_std)
        return X_std

    except Exception as e:
        st.write('Error', e.__class__, 'Occured')
        return df


def f_rs(df):
    try:

        X = df.select_dtypes(include=np.number)
        median_X = np.median(X)
        q3 = X.quantile(0.75) - X.quantile(0.25)
        rs_X = (X - np.median(X)) / q3
        st.dataframe(rs_X)
        return rs_X


    except Exception as e:
        st.write('Error', e.__class__, 'occured')
        return df


def f_mm(df):
    try:

        X = df.select_dtypes(include=np.number)
        min_X = np.min(X)
        max_X = np.max(X)
        X_minmax = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        st.dataframe(X_minmax)
        return X_minmax


    except Exception as e:
        st.write('Error', e.__class__, 'occured')
        return df


def f_maxabs(df):
    try:

        X = df.select_dtypes(include=np.number)
        max_abs_X = np.max(abs(X))
        X_maxabs = X / np.max(abs(X))
        st.dataframe(X_maxabs)
        return X_maxabs


    except Exception as e:
        st.write('Error', e.__class__, 'Occured')
        return df


def fso(df):
    try:

        f_options = ('Standard Scaler', 'Robust Scaler', 'Min Max Scaler', 'Max Absolute Scaler')
        fso_selection = st.sidebar.radio('Select the method for feature scaling:', f_options)

        if fso_selection == 'Standard Scaler':
            st.sidebar.write('you selected Standard Scaler')
            if st.sidebar.button('Process Standard Scaler'):
                df = pd.read_csv(path)
                df = f_ss(df)
                return df

        elif fso_selection == 'Robust Scaler':
            st.sidebar.write('You selected Robust Scaler')
            if st.sidebar.button('Process Robust Scaler'):
                df = pd.read_csv(path)
                df = f_rs(df)
                return df

        elif fso_selection == 'Min Max Scaler':
            st.sidebar.write('You selected Min Max scaler')
            if st.sidebar.button('Process Min Max scaler'):
                df = pd.read_csv(path)
                df = f_mm(df)
                return df

        elif fso_selection == 'Max Absolute Scaler':
            st.sidebar.write('You selected Max Absolute Scaler')
            if st.sidebar.button('Process Max Absolute Scaler'):
                df = pd.read_csv(path)
                df = f_maxabs(df)
                return df

    except Exception as e:
        st.write('Error', e.__class__, 'Occured')
        return df


# Data Export:

def data_export(df):
    try:
        st.sidebar.markdown("<h3 style='text-align: left; color: black;'>Dta Export</h3>",unsafe_allow_html=True)
        File_Download_Options = ('.XLSX', '.CSV', 'Oracle')
        File_Download_Select = st.sidebar.radio('Select type of file to download', File_Download_Options)
        if File_Download_Select == '.CSV':
            if st.sidebar.button('Download CSV'):
                df = pd.read_csv(path)
                st.sidebar.markdown(get_table_download_link_csv(df), unsafe_allow_html=True)
                return 0
        elif File_Download_Select == '.XLSX':
            if st.sidebar.button('Download XLSX'):
                df = pd.read_csv(path)
                st.sidebar.markdown(get_table_download_link_xlsx(df), unsafe_allow_html=True)
                return 0
        else:
            st.info('Enter Oracle Database info')
            users = st.text_input('Enter the user name:')
            password = st.text_input('Enter the Password:', type='password')
            host = st.text_input('Enter the host address:')
            port = st.text_input('Enter the port number:')
            table = st.text_input('Enter the name of the table to create,if table already exists it will be replaced')
            if st.button('connect'):
                df = pd.read_csv(path)
                conn = create_engine('oracle+cx_oracle://{}:{}@{}:{}/ORCL'.format(users, password, host, port))
                df.to_sql('{}'.format(table), conn, if_exists='replace')
                # con_query='{}/{}@{}:{}/ORCL'.format(user,password,host,port)
                # con=cx_Oracle.connect(con_query)
                if conn is not None:
                    st.info('connection Established successfully and table inserted')

    except Exception as e:
        st.write("Error", e.__class__, "occured")
        return df

def get_table_download_link_csv(df):
    try:

        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        return f'<a href ="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'

    except Exception as e:
        st.write('Oops!', e.__class__, 'Occured')
        return df


def get_table_download_link_xlsx(df):
    try:

        val = to_excel(df)
        b64 = base64.b64encode(val)
        return f'<a href ="data:application/octet-stream;base64,{b64.decode()} "download="myfilename.xlsx">Download xlsx file</a>'

    except Exception as e:
        st.write('Oops!', e.__class__, 'Occured')
        return df


def to_excel(df):

    try:

        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer)
        writer.save()
        processed_data = output.getvalue()
        return processed_data

    except Exception as e:
        st.write('Oops!', e.__class__, 'Occured')
        return df


# Main Function

def main_options():
    try:

        options = ('Missing value treatment', 'Outlier Treatment', 'Feature Scaling')
        select_option = st.sidebar.radio('Which action you want to perform?', options)
        return select_option

    except Exception as e:
        st.write('Error', e.__class__, 'occured')
        return df


def main():
    try:

        df = file_upload()

        m_option = main_options()
        if m_option == 'Missing value treatment':
            df = mvt_options(df)
            df

        elif m_option == 'Outlier Treatment':
            outlier_Treatment()


        else:

            fso(df)

        data_export(df)

    except Exception as e:
        st.write("Error", e.__class__, "occured")
        return df


main()
