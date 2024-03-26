import geopy
import streamlit as st
import pandas as pd
import httpx
import asyncio
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time
import zipfile
import altair as alt
import geopandas as gpd
import folium
from geopy.geocoders import Nominatim
from branca.colormap import linear
import requests
import certifi
import ssl




async def fetch_data_async(title):
    url = "https://data.cms.gov/data.json"
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.get(url)
        response_data = response.json()

        dataset = response_data['dataset']
        for dataset_entry in dataset:
            if title == dataset_entry['title']:
                for distro in dataset_entry['distribution']:
                    if 'format' in distro and 'description' in distro:
                        if distro['format'] == "API" and distro['description'] == "latest":
                            latest_distro = distro['accessURL']

        stats_endpoint = latest_distro + "/stats"
        stats_response = await client.get(stats_endpoint)
        stats_data = stats_response.json()
        total_rows = stats_data['total_rows']

        all_data = []
        size = 5000
        i = 0
        while i < total_rows:
            offset_url = f"{latest_distro}?size={size}&offset={i}"
            offset_response = await client.get(offset_url)
            print(f"Made request for {size} results at offset {i}")
            data = offset_response.json()
            all_data.extend(data)
            i += size

    return all_data

def remove_periods(df, columns_with_periods):
    modified_df = df.copy()
    for column in columns_with_periods:
        modified_df.rename(columns={column: column.replace(".", "")}, inplace=True)
    return modified_df

async def main():
    st.sidebar.title("Fetch CMS Dataset")
    dataset_title = st.sidebar.text_input("Enter the dataset title")

    if dataset_title:
        all_data = await fetch_data_async(dataset_title)
        df = pd.DataFrame(all_data)

        st.subheader("DataFrame")
        st.write(df.head())

        st.sidebar.title("Column Cleanup")
        columns_with_periods = st.sidebar.multiselect("Select columns containing periods", df.columns)

        if columns_with_periods:
            df_clean = remove_periods(df, columns_with_periods)
            st.subheader("Modified DataFrame")
            st.write(df_clean.head())

            st.sidebar.title("Select Model Data")
            excluded_columns = st.sidebar.multiselect("Select columns to exclude from subset", df_clean.columns)

            if excluded_columns:
                subset_df = df_clean.drop(columns=excluded_columns)
                st.subheader("Subset DataFrame")
                st.write(subset_df.head())

                le = LabelEncoder()
                for col in subset_df.select_dtypes(include=['object']):
                    subset_df[col] = le.fit_transform(subset_df[col])

                st.subheader("Encoded Subset DataFrame")
                st.write(subset_df.head())

                return subset_df, df

    return None, None

if __name__ == "__main__":
    subset_df, df = asyncio.run(main())
   # subset_df, df = main()
    
    if df is not None:
        print(df.head())  # Example usage, replace this with your actual code
    if subset_df is not None:
        print(subset_df.head())  # Example usage, replace this with your actual code
        # Rest of your Streamlit app code

        # User selects target variable (y)
        target_variable = st.sidebar.selectbox('Select target variable (y)', subset_df.columns)

        st.sidebar.header('Set Parameters')

        parameter_split_size = st.sidebar.slider('Data split ratio (percent for Training Set)', 10, 90, 80, 5)

        st.sidebar.subheader('Learning Parameters')
        with st.sidebar.expander('See parameters'):
            parameter_n_estimators = st.slider('Number of boosting rounds (n_estimators)', 0, 1000, 100, 100)
            parameter_max_depth = st.slider('Maximum tree depth (max_depth)', 1, 10, 3, 1)
            parameter_learning_rate = st.slider('Learning rate (eta)', 0.01, 1.0, 0.1, 0.01)

        sleep_time = st.slider('Sleep time', 0, 3, 0)

        # Initiate the model building process
        with st.status("Running ...", expanded=True) as status:
        
            st.write("Loading data ...")
            time.sleep(sleep_time)

            st.write("Preparing data ...")
            time.sleep(sleep_time)
            X = subset_df.drop(columns=[target_variable])
            y = subset_df[target_variable]
                
            st.write("Splitting data ...")
            time.sleep(sleep_time)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-parameter_split_size)/100, random_state=42)
        
            st.write("Model training ...")
            time.sleep(sleep_time)

            xgb = XGBRegressor(
                n_estimators=parameter_n_estimators,
                max_depth=parameter_max_depth,
                learning_rate=parameter_learning_rate
            )
            xgb.fit(X_train, y_train)
        
        st.write("Applying model to make predictions ...")
        time.sleep(sleep_time)
        y_train_pred = xgb.predict(X_train)
        y_test_pred = xgb.predict(X_test)
            
        st.write("Evaluating performance metrics ...")
        time.sleep(sleep_time)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        st.write("Displaying performance metrics ...")
        time.sleep(sleep_time)
        rf_results = pd.DataFrame(['XGBoost', train_mse, train_r2, test_mse, test_r2]).transpose()
        rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
        rf_results = rf_results.round(3)
        
        status.update(label="Status", state="complete", expanded=False)

        # Display data info
        st.header('Input data', divider='rainbow')
        col = st.columns(4)
        col[0].metric(label="No. of samples", value=X.shape[0], delta="")
        col[1].metric(label="No. of X variables", value=X.shape[1], delta="")
        col[2].metric(label="No. of Training samples", value=X_train.shape[0], delta="")
        col[3].metric(label="No. of Test samples", value=X_test.shape[0], delta="")
        
        with st.expander('Initial dataset', expanded=True):
            st.dataframe(subset_df, height=210, use_container_width=True)
        with st.expander('Train split', expanded=False):
            train_col = st.columns((3,1))
            with train_col[0]:
                st.markdown('**X**')
                st.dataframe(X_train, height=210, hide_index=True, use_container_width=True)
            with train_col[1]:
                st.markdown('**y**')
                st.dataframe(y_train, height=210, hide_index=True, use_container_width=True)
        with st.expander('Test split', expanded=False):
            test_col = st.columns((3,1))
            with test_col[0]:
                st.markdown('**X**')
                st.dataframe(X_test, height=210, hide_index=True, use_container_width=True)
            with test_col[1]:
                st.markdown('**y**')
                st.dataframe(y_test, height=210, hide_index=True, use_container_width=True)

        # Zip dataset files
        subset_df.to_csv('dataset.csv', index=False)
        X_train.to_csv('X_train.csv', index=False)
        y_train.to_csv('y_train.csv', index=False)
        X_test.to_csv('X_test.csv', index=False)
        y_test.to_csv('y_test.csv', index=False)

        list_files = ['dataset.csv', 'X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv']
        with zipfile.ZipFile('dataset.zip', 'w') as zipF:
            for file in list_files:
                zipF.write(file, compress_type=zipfile.ZIP_DEFLATED)

        with open('dataset.zip', 'rb') as datazip:
            btn = st.download_button(
                label='Download ZIP',
                data=datazip,
                file_name="dataset.zip",
                mime="application/octet-stream"
            )

        # Display model parameters
        st.header('Model parameters', divider='rainbow')
        parameters_col = st.columns(3)
        # parameters_col[0].metric(label="Data split ratio (% for Training Set)", value=parameter_split_size, delta="")
        parameters_col[0].metric(label="Number of boosting rounds (n_estimators)", value=parameter_n_estimators,
                                 delta="")
        parameters_col[1].metric(label="Max tree depth (max_depth)", value=parameter_max_depth, delta="")
        parameters_col[2].metric(label="Learning rate (eta)", value=parameter_learning_rate, delta="")

        # Display feature importance plot
        st.header('Feature importance', divider='rainbow')
        feature_importance = xgb.feature_importances_
        feature_names = list(X.columns)
        df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
        df_importance = df_importance.sort_values(by='Importance', ascending=False)

        bars = alt.Chart(df_importance).mark_bar().encode(
            x=alt.X('Importance:Q', axis=alt.Axis(title='Importance')),
            y=alt.Y('Feature:N', sort='-x', axis=alt.Axis(title='Feature')),
            color=alt.Color('Importance:Q', scale=alt.Scale(scheme='turbo'), legend=None)
        ).properties(height=400)

        st.altair_chart(bars, use_container_width=True)

        # Prediction results
        st.header('Prediction results', divider='rainbow')
        s_y_train = pd.Series(y_train, name='actual').reset_index(drop=True)
        s_y_train_pred = pd.Series(y_train_pred, name='predicted').reset_index(drop=True)
        df_train = pd.DataFrame(data=[s_y_train, s_y_train_pred], index=None).T
        df_train['class'] = 'train'

        s_y_test = pd.Series(y_test, name='actual').reset_index(drop=True)
        s_y_test_pred = pd.Series(y_test_pred, name='predicted').reset_index(drop=True)
        df_test = pd.DataFrame(data=[s_y_test, s_y_test_pred], index=None).T
        df_test['class'] = 'test'

        df_prediction = pd.concat([df_train, df_test], axis=0)

        prediction_col = st.columns((2, 0.2, 3))

        # Display dataframe
        with prediction_col[0]:
            st.dataframe(df_prediction, height=320, use_container_width=True)

        # Display scatter plot of actual vs predicted values
        with prediction_col[2]:
            scatter = alt.Chart(df_prediction).mark_circle(size=60).encode(
                x='actual',
                y='predicted',
                color='class'
            )
            st.altair_chart(scatter, use_container_width=True)

        # Make predictions on the entire dataset
        subset_df['Predicted'] = xgb.predict(X)

        # Calculate residuals
        subset_df['Residual'] = y - subset_df['Predicted']

        df_with_predictions = pd.concat([df, subset_df[['Predicted', 'Residual']]], axis=1)

        # Sort entities based on the residuals and select the top 50
        top_residuals = df_with_predictions.nlargest(50, 'Residual')

        # Display table containing entities with the largest residuals
        st.header('Entities with the Largest Residuals')
        st.table(top_residuals)

        # Plot residuals vs predicted values in a scatter plot
        scatter_pred_residuals = alt.Chart(subset_df).mark_circle(size=60).encode(
            x='Predicted',
            y='Residual',
            tooltip=['Predicted', 'Residual']
        ).properties(
            width=800,
            height=400
        )
        st.header('Predicted vs Residuals Scatter Plot')
        st.altair_chart(scatter_pred_residuals, use_container_width=True)

            # Select city and state columns for longitude and latitude
        selected_columns = st.multiselect("Select columns for longitude and latitude", df.columns)
        if selected_columns:
            # Concatenate city and state columns to derive longitude and latitude
            location_df = df_with_predictions[selected_columns]
            # Perform any necessary data cleaning or transformation to derive longitude and latitude

            ctx = ssl.create_default_context(cafile=certifi.where())
            geopy.geocoders.options.default_ssl_context = ctx

            # Create a geocoder object with the custom session
            geolocator = Nominatim(user_agent="my_geocoder", timeout=10)

            # Geocode addresses to get latitude and longitude coordinates
            location_df['Location'] = location_df.apply(geolocator.geocode)

            # Extract latitude and longitude coordinates from the Location column
            location_df['Latitude'] = location_df['Location'].apply(lambda loc: loc.latitude if loc else None)
            location_df['Longitude'] = location_df['Location'].apply(lambda loc: loc.longitude if loc else None)

            # Plot map with top residuals
            #st.subheader("Map with Top Residuals")
            #top_residuals = df_with_predictions.nlargest(50, 'Residual')  # Get top 50 residuals
            st.map(location_df, latitude='Latitude',longitude='Longitude',color='Residuals')  # Plot top residuals on a map

        # Display the DataFrame with predictions and residuals
        #st.subheader("DataFrame with Predictions and Residuals")
        #st.write(df_with_predictions)

