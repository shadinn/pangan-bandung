import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error,mean_squared_error

# Load Data
data = pd.read_excel(r'datapds (1).xlsx', parse_dates=True, index_col='Bulan')
data['Index'] = data.index
data.index.freq = 'MS'
datafevd = pd.read_excel(r'datafevd.xlsx')
datairf = pd.read_excel(r'datairf.xlsx')
datavecm1 = pd.read_excel(r'vecmpanjang.xlsx', sheet_name='Sheet2')
datavecm2 = pd.read_excel(r'vecmpanjang.xlsx')
top3 = pd.read_excel(r'top3.xlsx')
#warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Layout 
st.set_page_config(
    page_title = 'Pangan Bandung',
    page_icon = '✅',
    layout = 'wide'
)

#Sidebar
st.sidebar.title("Pangan Pedia")
menu = st.sidebar.radio("Navigation Menu",["Forecasting","VECM"])

def emmse(df):
    train_size = 0.7
    split_idx = round(len(df)* train_size)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    fitted_model = ExponentialSmoothing(train['variabel'],trend='mul',seasonal_periods=12).fit()
    test_predictions = fitted_model.forecast(12)
    mse = mean_squared_error(test, test_predictions)
    return round(mse, 2)

def emrmse(df):
    train_size = 0.7
    split_idx = round(len(df)* train_size)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    fitted_model = ExponentialSmoothing(train['variabel'],trend='mul',seasonal_periods=12).fit()
    test_predictions = fitted_model.forecast(12)
    rmse = np.sqrt(mean_squared_error(test, test_predictions))
    return round(rmse, 2)

def emmae(df):
    train_size = 0.7
    split_idx = round(len(df)* train_size)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    fitted_model = ExponentialSmoothing(train['variabel'],trend='mul',seasonal_periods=12).fit()
    test_predictions = fitted_model.forecast(12)
    mae = mean_absolute_error(test, test_predictions)
    return round(mae, 2)

def trend(df):
    result = sm.tsa.seasonal_decompose(df, model='additive', period=12)
    return result.trend

def seasonal(df):
    result = sm.tsa.seasonal_decompose(df, model='additive', period=12)
    return result.seasonal


if menu == "Forecasting":
    st.title('FORECASTING')
    pilihan = ['Beras', 'Daging Ayam', 'Daging Sapi', 'Telur Ayam', 'Bawang Putih', 'Bawang Merah', 'Cabai Merah', 'Cabai Rawit', 'Minyak Goreng', 'Gula Pasir']
    pilihan = st.selectbox('Pilih Data', pilihan)
    if pilihan == 'Beras' :
        df = data['Beras']
        df = {'variabel':df}
        df = pd.DataFrame(df)
    if pilihan == 'Minyak Goreng' :
        df = data['Minyak_Goreng']
        df = {'variabel':df}
        df = pd.DataFrame(df)
    if pilihan == 'Daging Ayam' :
        df = data['Daging_Ayam']
        df = {'variabel':df}
        df = pd.DataFrame(df)
    if pilihan == 'Daging Sapi' :
        df = data['Daging_Sapi']
        df = {'variabel':df}
        df = pd.DataFrame(df)
    if pilihan == 'Bawang Merah' :
        df = data['Bawang_Merah']
        df = {'variabel':df}
        df = pd.DataFrame(df)
    if pilihan == 'Bawang Putih' :
        df = data['Bawang_Putih']
        df = {'variabel':df}
        df = pd.DataFrame(df)
    if pilihan == 'Telur Ayam' :
        df = data['Telur_Ayam']
        df = {'variabel':df}
        df = pd.DataFrame(df)
    if pilihan == 'Gula Pasir' :
        df = data['Gula_Pasir']
        df = {'variabel':df}
        df = pd.DataFrame(df)
    if pilihan == 'Cabai Merah' :
        df = data['Cabai_Merah']
        df = {'variabel':df}
        df = pd.DataFrame(df)
    if pilihan == 'Cabai Rawit' :
        df = data['Cabai_Rawit']
        df = {'variabel':df}
        df = pd.DataFrame(df)
        
    mse,rmse,mae = st.columns(3)
    hmse = emmse(df)
    hmae = emmae(df)
    hrmse = emrmse(df)
    mse.metric('MSE', hmse )
    mae.metric('MAE', hmae)
    rmse.metric('RMSE', hrmse)

    coltren, colsea = st.columns(2)
    with coltren :
        tren = st.checkbox('Tren')
        if tren:
                st.line_chart(trend(df))

    with colsea :
        season = st.checkbox('Seasonal')
        if season :
            st.line_chart(seasonal(df))


    col1, col2, col3 = st.columns(3)


    with col1:
        st.subheader('Holts Winters Single Exponential Smoothing ')
        m = 12
        alpha = 1/(2*m)
        if pilihan == 'Beras' :
            data['HWES1'] = SimpleExpSmoothing(data['Beras']).fit(smoothing_level=alpha,optimized=False,use_brute=True).fittedvalues
            st.line_chart(data=data, x="Index", y=['Beras','HWES1'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Daging Ayam' :
            data['HWES1'] = SimpleExpSmoothing(data['Daging_Ayam']).fit(smoothing_level=alpha,optimized=False,use_brute=True).fittedvalues
            st.line_chart(data=data, x="Index", y=['Daging_Ayam','HWES1'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Daging Sapi' :
            data['HWES1'] = SimpleExpSmoothing(data['Daging_Sapi']).fit(smoothing_level=alpha,optimized=False,use_brute=True).fittedvalues
            st.line_chart(data=data, x="Index", y=['Daging_Sapi','HWES1'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Telur Ayam' :
            data['HWES1'] = SimpleExpSmoothing(data['Telur_Ayam']).fit(smoothing_level=alpha,optimized=False,use_brute=True).fittedvalues
            st.line_chart(data=data, x="Index", y=['Telur_Ayam','HWES1'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Bawang Putih' :
            data['HWES1'] = SimpleExpSmoothing(data['Bawang_Putih']).fit(smoothing_level=alpha,optimized=False,use_brute=True).fittedvalues
            st.line_chart(data=data, x="Index", y=['Bawang_Putih','HWES1'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Bawang Merah' :
            data['HWES1'] = SimpleExpSmoothing(data['Bawang_Merah']).fit(smoothing_level=alpha,optimized=False,use_brute=True).fittedvalues
            st.line_chart(data=data, x="Index", y=['Bawang_Merah','HWES1'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Cabai Merah' :
            data['HWES1'] = SimpleExpSmoothing(data['Cabai_Merah']).fit(smoothing_level=alpha,optimized=False,use_brute=True).fittedvalues
            st.line_chart(data=data, x="Index", y=['Cabai_Merah','HWES1'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Cabai Rawit' :
            data['HWES1'] = SimpleExpSmoothing(data['Cabai_Rawit']).fit(smoothing_level=alpha,optimized=False,use_brute=True).fittedvalues
            st.line_chart(data=data, x="Index", y=['Cabai_Rawit','HWES1'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Minyak Goreng' :
            data['HWES1'] = SimpleExpSmoothing(data['Minyak_Goreng']).fit(smoothing_level=alpha,optimized=False,use_brute=True).fittedvalues
            st.line_chart(data=data, x="Index", y=['Minyak_Goreng','HWES1'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Gula Pasir' :
            data['HWES1'] = SimpleExpSmoothing(data['Gula_Pasir']).fit(smoothing_level=alpha,optimized=False,use_brute=True).fittedvalues
            st.line_chart(data=data, x="Index", y=['Gula_Pasir','HWES1'] ,width=0, height=0, use_container_width=True)
    with col2:
        st.subheader('Holts Winters Double Exponential Smoothing')
        if pilihan == 'Beras' :
            data['HWES2_ADD'] = ExponentialSmoothing(data['Beras'],trend='add').fit().fittedvalues
            data['HWES2_MUL'] = ExponentialSmoothing(data['Beras'],trend='mul').fit().fittedvalues
            st.line_chart(data=data, x="Index", y=['Beras','HWES2_ADD', 'HWES2_MUL'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Telur Ayam' :
            data['HWES2_ADD'] = ExponentialSmoothing(data['Telur_Ayam'],trend='add').fit().fittedvalues
            data['HWES2_MUL'] = ExponentialSmoothing(data['Telur_Ayam'],trend='mul').fit().fittedvalues
            st.line_chart(data=data, x="Index", y=['Telur_Ayam','HWES2_ADD', 'HWES2_MUL'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Bawang Merah' :
            data['HWES2_ADD'] = ExponentialSmoothing(data['Bawang_Merah'],trend='add').fit().fittedvalues
            data['HWES2_MUL'] = ExponentialSmoothing(data['Bawang_Merah'],trend='mul').fit().fittedvalues
            st.line_chart(data=data, x="Index", y=['Bawang_Merah','HWES2_ADD', 'HWES2_MUL'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Bawang Putih' :
            data['HWES2_ADD'] = ExponentialSmoothing(data['Bawang_Putih'],trend='add').fit().fittedvalues
            data['HWES2_MUL'] = ExponentialSmoothing(data['Bawang_Putih'],trend='mul').fit().fittedvalues
            st.line_chart(data=data, x="Index", y=['Bawang_Putih','HWES2_ADD', 'HWES2_MUL'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Cabai Rawit' :
            data['HWES2_ADD'] = ExponentialSmoothing(data['Cabai_Rawit'],trend='add').fit().fittedvalues
            data['HWES2_MUL'] = ExponentialSmoothing(data['Cabai_Rawit'],trend='mul').fit().fittedvalues
            st.line_chart(data=data, x="Index", y=['Cabai_Rawit','HWES2_ADD', 'HWES2_MUL'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Cabai Merah' :
            data['HWES2_ADD'] = ExponentialSmoothing(data['Cabai_Merah'],trend='add').fit().fittedvalues
            data['HWES2_MUL'] = ExponentialSmoothing(data['Cabai_Merah'],trend='mul').fit().fittedvalues
            st.line_chart(data=data, x="Index", y=['Cabai_Merah','HWES2_ADD', 'HWES2_MUL'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Daging Sapi' :
            data['HWES2_ADD'] = ExponentialSmoothing(data['Daging_Sapi'],trend='add').fit().fittedvalues
            data['HWES2_MUL'] = ExponentialSmoothing(data['Daging_Sapi'],trend='mul').fit().fittedvalues
            st.line_chart(data=data, x="Index", y=['Daging_Sapi','HWES2_ADD', 'HWES2_MUL'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Daging Ayam' :
            data['HWES2_ADD'] = ExponentialSmoothing(data['Daging_Ayam'],trend='add').fit().fittedvalues
            data['HWES2_MUL'] = ExponentialSmoothing(data['Daging_Ayam'],trend='mul').fit().fittedvalues
            st.line_chart(data=data, x="Index", y=['Daging_Ayam','HWES2_ADD', 'HWES2_MUL'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Minyak Goreng' :
            data['HWES2_ADD'] = ExponentialSmoothing(data['Minyak_Goreng'],trend='add').fit().fittedvalues
            data['HWES2_MUL'] = ExponentialSmoothing(data['Minyak_Goreng'],trend='mul').fit().fittedvalues
            st.line_chart(data=data, x="Index", y=['Minyak_Goreng','HWES2_ADD', 'HWES2_MUL'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Gula Pasir' :
            data['HWES2_ADD'] = ExponentialSmoothing(data['Gula_Pasir'],trend='add').fit().fittedvalues
            data['HWES2_MUL'] = ExponentialSmoothing(data['Gula_Pasir'],trend='mul').fit().fittedvalues
            st.line_chart(data=data, x="Index", y=['Gula_Pasir','HWES2_ADD', 'HWES2_MUL'] ,width=0, height=0, use_container_width=True)

    with col3:
        st.subheader('Holts Winters Triple Exponential Smoothing')
        if pilihan == "Beras":
            data['HWES3_ADD'] = ExponentialSmoothing(data['Beras'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
            data['HWES3_MUL'] = ExponentialSmoothing(data['Beras'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues
            st.line_chart(data=data, x="Index", y=['Beras','HWES3_ADD', 'HWES3_MUL'] ,width=0, height=0, use_container_width=True)
        if pilihan == "Minyak Goreng":
            data['HWES3_ADD'] = ExponentialSmoothing(data['Minyak_Goreng'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
            data['HWES3_MUL'] = ExponentialSmoothing(data['Minyak_Goreng'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues
            st.line_chart(data=data, x="Index", y=['Minyak_Goreng','HWES3_ADD', 'HWES3_MUL'] ,width=0, height=0, use_container_width=True)
        if pilihan == "Gula Pasir":
            data['HWES3_ADD'] = ExponentialSmoothing(data['Gula_Pasir'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
            data['HWES3_MUL'] = ExponentialSmoothing(data['Gula_Pasir'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues
            st.line_chart(data=data, x="Index", y=['Gula_Pasir','HWES3_ADD', 'HWES3_MUL'] ,width=0, height=0, use_container_width=True)
        if pilihan == "Daging Ayam":
            data['HWES3_ADD'] = ExponentialSmoothing(data['Daging_Ayam'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
            data['HWES3_MUL'] = ExponentialSmoothing(data['Daging_Ayam'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues
            st.line_chart(data=data, x="Index", y=['Daging_Ayam','HWES3_ADD', 'HWES3_MUL'] ,width=0, height=0, use_container_width=True)
        if pilihan == "Daging Sapi":
            data['HWES3_ADD'] = ExponentialSmoothing(data['Daging_Sapi'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
            data['HWES3_MUL'] = ExponentialSmoothing(data['Daging_Sapi'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues
            st.line_chart(data=data, x="Index", y=['Daging_Sapi','HWES3_ADD', 'HWES3_MUL'] ,width=0, height=0, use_container_width=True)
        if pilihan == "Telur Ayam":
            data['HWES3_ADD'] = ExponentialSmoothing(data['Telur_Ayam'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
            data['HWES3_MUL'] = ExponentialSmoothing(data['Telur_Ayam'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues
            st.line_chart(data=data, x="Index", y=['Telur_Ayam','HWES3_ADD', 'HWES3_MUL'] ,width=0, height=0, use_container_width=True)
        if pilihan == "Bawang Putih":
            data['HWES3_ADD'] = ExponentialSmoothing(data['Bawang_Putih'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
            data['HWES3_MUL'] = ExponentialSmoothing(data['Bawang_Putih'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues
            st.line_chart(data=data, x="Index", y=['Bawang_Putih','HWES3_ADD', 'HWES3_MUL'] ,width=0, height=0, use_container_width=True)
        if pilihan == "Bawang Merah":
            data['HWES3_ADD'] = ExponentialSmoothing(data['Bawang_Merah'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
            data['HWES3_MUL'] = ExponentialSmoothing(data['Bawang_Merah'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues
            st.line_chart(data=data, x="Index", y=['Bawang_Merah','HWES3_ADD', 'HWES3_MUL'] ,width=0, height=0, use_container_width=True)
        if pilihan == "Cabai Merah":
            data['HWES3_ADD'] = ExponentialSmoothing(data['Cabai_Merah'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
            data['HWES3_MUL'] = ExponentialSmoothing(data['Cabai_Merah'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues
            st.line_chart(data=data, x="Index", y=['Cabai_Merah','HWES3_ADD', 'HWES3_MUL'] ,width=0, height=0, use_container_width=True)
        if pilihan == "Cabai Rawit":
            data['HWES3_ADD'] = ExponentialSmoothing(data['Cabai_Rawit'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
            data['HWES3_MUL'] = ExponentialSmoothing(data['Cabai_Rawit'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues
            st.line_chart(data=data, x="Index", y=['Cabai_Rawit','HWES3_ADD', 'HWES3_MUL'] ,width=0, height=0, use_container_width=True)


    st.subheader('Forecasting')
    forecast, tabel = st.columns(2)
    with forecast:
        if pilihan == 'Minyak Goreng':
            final_model = ExponentialSmoothing(data['Minyak_Goreng'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
            forecast_predictions = final_model.forecast(steps=12)
            df = pd.DataFrame({'Forecast': forecast_predictions})
            df = pd.concat([data, df], axis=1)
            df['Bulan'] = df.index
            st.line_chart(data=df, x="Bulan", y=['Minyak_Goreng','Forecast'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Gula Pasir':
            final_model = ExponentialSmoothing(data['Gula_Pasir'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
            forecast_predictions = final_model.forecast(steps=12)
            df = pd.DataFrame({'Forecast': forecast_predictions})
            df = pd.concat([data, df], axis=1)
            df['Bulan'] = df.index
            st.line_chart(data=df, x="Bulan", y=['Gula_Pasir','Forecast'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Bawang Putih':
            final_model = ExponentialSmoothing(data['Bawang_Putih'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
            forecast_predictions = final_model.forecast(steps=12)
            df = pd.DataFrame({'Forecast': forecast_predictions})
            df = pd.concat([data, df], axis=1)
            df['Bulan'] = df.index
            st.line_chart(data=df, x="Bulan", y=['Bawang_Putih','Forecast'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Bawang Merah':
            final_model = ExponentialSmoothing(data['Bawang_Merah'],trend='mul',seasonal_periods=12).fit()
            forecast_predictions = final_model.forecast(steps=12)
            df = pd.DataFrame({'Forecast': forecast_predictions})
            df = pd.concat([data, df], axis=1)
            df['Bulan'] = df.index
            st.line_chart(data=df, x="Bulan", y=['Bawang_Merah','Forecast'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Daging Ayam':
            final_model = ExponentialSmoothing(data['Daging_Ayam'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
            forecast_predictions = final_model.forecast(steps=12)
            df = pd.DataFrame({'Forecast': forecast_predictions})
            df = pd.concat([data, df], axis=1)
            df['Bulan'] = df.index
            st.line_chart(data=df, x="Bulan", y=['Daging_Ayam','Forecast'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Daging Sapi':
            final_model = ExponentialSmoothing(data['Daging_Sapi'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
            forecast_predictions = final_model.forecast(steps=12)
            df = pd.DataFrame({'Forecast': forecast_predictions})
            df = pd.concat([data, df], axis=1)
            df['Bulan'] = df.index
            st.line_chart(data=df, x="Bulan", y=['Daging_Sapi','Forecast'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Telur Ayam':
            final_model = ExponentialSmoothing(data['Telur_Ayam'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
            forecast_predictions = final_model.forecast(steps=12)
            df = pd.DataFrame({'Forecast': forecast_predictions})
            df = pd.concat([data, df], axis=1)
            df['Bulan'] = df.index
            st.line_chart(data=df, x="Bulan", y=['Telur_Ayam','Forecast'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Cabai Merah':
            final_model = ExponentialSmoothing(data['Cabai_Merah'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
            forecast_predictions = final_model.forecast(steps=12)
            df = pd.DataFrame({'Forecast': forecast_predictions})
            df = pd.concat([data, df], axis=1)
            df['Bulan'] = df.index
            st.line_chart(data=df, x="Bulan", y=['Cabai_Merah','Forecast'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Cabai Rawit':
            final_model = ExponentialSmoothing(data['Cabai_Rawit'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
            forecast_predictions = final_model.forecast(steps=12)
            df = pd.DataFrame({'Forecast': forecast_predictions})
            df = pd.concat([data, df], axis=1)
            df['Bulan'] = df.index
            st.line_chart(data=df, x="Bulan", y=['Cabai_Rawit','Forecast'] ,width=0, height=0, use_container_width=True)
        if pilihan == 'Beras':
            final_model = ExponentialSmoothing(data['Beras'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
            forecast_predictions = final_model.forecast(steps=12)
            df = pd.DataFrame({'Forecast': forecast_predictions})
            df = pd.concat([data, df], axis=1)
            df['Bulan'] = df.index
            st.line_chart(data=df, x="Bulan", y=['Beras','Forecast'] ,width=0, height=50, use_container_width=True)
    with tabel :
        if pilihan == 'Minyak Goreng':
            final_model = ExponentialSmoothing(data['Minyak_Goreng'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
            forecast_predictions = final_model.forecast(steps=12)
            df = pd.DataFrame({'Forecast': forecast_predictions})
            st.write(df)
        if pilihan == 'Gula Pasir':
            final_model = ExponentialSmoothing(data['Gula_Pasir'],trend='mul',seasonal_periods=12).fit()
            forecast_predictions = final_model.forecast(steps=12)
            df = pd.DataFrame({'Forecast': forecast_predictions})
            st.write(df)
        if pilihan == 'Bawang Putih':
            final_model = ExponentialSmoothing(data['Bawang_Putih'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
            forecast_predictions = final_model.forecast(steps=12)
            df = pd.DataFrame({'Forecast': forecast_predictions})
            st.write(df)
        if pilihan == 'Bawang Merah':
            final_model = ExponentialSmoothing(data['Bawang_Merah'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
            forecast_predictions = final_model.forecast(steps=12)
            df = pd.DataFrame({'Forecast': forecast_predictions})
            st.write(df)
        if pilihan == 'Daging Ayam':
            final_model = ExponentialSmoothing(data['Daging_Ayam'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
            forecast_predictions = final_model.forecast(steps=12)
            df = pd.DataFrame({'Forecast': forecast_predictions})
            st.write(df)
        if pilihan == 'Daging_Sapi':
            final_model = ExponentialSmoothing(data['Daging_Sapi'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
            forecast_predictions = final_model.forecast(steps=12)
            df = pd.DataFrame({'Forecast': forecast_predictions})
            st.write(df)
        if pilihan == 'Telur Ayam':
            final_model = ExponentialSmoothing(data['Telur_Ayam'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
            forecast_predictions = final_model.forecast(steps=12)
            df = pd.DataFrame({'Forecast': forecast_predictions})
            st.write(df)
        if pilihan == 'Cabai Merah':
            final_model = ExponentialSmoothing(data['Cabai_Merah'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
            forecast_predictions = final_model.forecast(steps=12)
            df = pd.DataFrame({'Forecast': forecast_predictions})
            st.write(df)
        if pilihan == 'Cabai_Rawit':
            final_model = ExponentialSmoothing(data['Cabai_Rawit'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
            forecast_predictions = final_model.forecast(steps=12)
            df = pd.DataFrame({'Forecast': forecast_predictions})
            st.write(df)
        if pilihan == 'Beras':
            final_model = ExponentialSmoothing(data['Beras'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
            forecast_predictions = final_model.forecast(steps=12)
            df = pd.DataFrame({'Forecast': forecast_predictions})
            st.write(df)

if menu == 'VECM':
    st.title('Analisis Fluktuasi Harga Komoditas Pangan Terhadap Inflasi di Kota Bandung')

    st.header(' Analisis VECM')
    vecm1, vecm2 = st.columns(2)
    with vecm1 :
        st.caption('Jangka Pendek')
        fig = px.bar(datavecm1, x= 'variabel', y='koefisien', width= 525, height=400)
        st.plotly_chart(fig)
        st.caption('Pada jangka pendek hanya variabel telur ayam yang mempengaruhi IHK sebesar 0.00124%')
    with vecm2 :
        st.caption('Jangka Panjang')
        fig = px.bar(datavecm2, x= 'Variabel', y='Koefisien', width= 525, height=400)
        st.plotly_chart(fig)
        st.caption('Pada hasil VECM Jangka Panjang keseluruhan variabel mempengaruhi IHK')
    
    st.header('Analisis IRF')
    irf1, irf2 = st.columns(2)
    with irf1:
        pilihan = ['Beras', 'Daging Ayam', 'Daging Sapi', 'Telur Ayam', 'Bawang Putih', 'Bawang Merah', 'Cabai Merah', 'Cabai Rawit', 'Minyak Goreng', 'Gula Pasir']
        pilihandata = st.selectbox('Pilih Data', pilihan)
        if pilihandata == 'Beras':
            st.line_chart(data=datairf, x='Periode', y='Beras', width=0, height=0, use_container_width=True)
        elif pilihandata == 'Daging Ayam':
            st.line_chart(data=datairf, x='Periode', y='D Ayam', width=0, height=0, use_container_width=True)
        elif pilihandata == 'Daging Sapi':
            st.line_chart(data=datairf, x='Periode', y='D Sapi', width=0, height=0, use_container_width=True)
        elif pilihandata == 'Telur Ayam':
            st.line_chart(data=datairf, x='Periode', y='Telur ayam', width=0, height=0, use_container_width=True)
        elif pilihandata == 'Bawang Merah':
            st.line_chart(data=datairf, x='Periode', y='B Merah', width=0, height=0, use_container_width=True)
        elif pilihandata == 'Bawang Putih':
            st.line_chart(data=datairf, x='Periode', y='B Putih', width=0, height=0, use_container_width=True)
        elif pilihandata == 'Cabai Merah':
            st.line_chart(data=datairf, x='Periode', y='C Merah', width=0, height=0, use_container_width=True)
        elif pilihandata == 'Cabai Rawit':
            st.line_chart(data=datairf, x='Periode', y='C Rawit', width=0, height=0, use_container_width=True)
        elif pilihandata == 'Gula Pasir':
            st.line_chart(data=datairf, x='Periode', y='Gula', width=0, height=0, use_container_width=True)
        elif pilihandata == 'Minyak Goreng':
            st.line_chart(data=datairf, x='Periode', y='Minyak', width=0, height=0, use_container_width=True)
    with irf2:
        st.caption('Intepretasi IRF')
        st.caption('Analisis respon IHK terhadap guncangan harga masing-masing komoditas pangan ini diproyeksikan dalam jangka waktu 10 bulan ke depan dari bulan April 2023.')
        st.caption('Dari hasil analisis IRF dapat disimpulkan bahwa pada 10 periode ke depan dari periode penelitian, fluktuasi harga komoditas beras, daging sapi, telur ayam, cabai merah, cabai rawit, bawang merah, bawang putih, minyak yang terjadi pada rentang periode 7 s.d 10 akan berdampak pada peningkatan IHK Kota Bandung. Sedangkan, fluktuasi harga komoditas gula pasir akan berdampak pada penurunan IHK kota Bandung.')
        st. caption('Hal ini kemungkinan terjadi dikarenakan tingginya permintaan komoditas yang seringkali tidak dapat diimbangi dengan ketersediaan pasokan, sehingga terjadinya kelangkaan pada periode-periode tersebut yang dapat menyebabkan harga di tingkat konsumen meningkat. Peristiwa tersebut merupakan penyebab Inflasi dari sisi demand pull inflation. ')
    st.header('Analisis FEVD')
    col1, col2 = st.columns(2)
    with col1:
        st.caption('Grafik Forecast Error Vector Decomposition')
        fig = px.bar(datafevd, x= 'Periode', y=['IHK','Telur ayam', 'B Merah', "B Putih", "Beras", "C Merah", "C Rawit", "D Ayam", "D Sapi", "Gula", "Minyak"],barmode='group', width= 525, height=400)
        st.plotly_chart(fig)
    with col2:
        st.caption('Top 3 Komoditas')
        fig = px.bar(top3, x= 'Variabel', y='Nilai', width= 525, height=400)
        st.plotly_chart(fig)
    col3, col4 = st.columns(2)
    with col3:
        st.write(datafevd)
    with col4:
        st.caption('Intepretasi FEVD')
        st. caption('Analisis Forecast Error Variance Decomposition (FEVD) dapat diketahui komoditas pangan mana yang paling dominan dalam mempengaruhi IHK di Kota Bandung pada 10 periode kedepan dari periode penelitian. Berdasarkan hasil analisis FEVD menunjukkan bahwa pada periode pertama, keragaman IHK di Kota Bandung disebabkan oleh guncangan IHK Kota Bandung itu sendiri, yaitu sebesar 100%. Selanjutnya, pada periode ke-2 dan seterusnya variabel lain mulai mempengaruhi keragaman IHK. Tiga  komoditas pangan yang paling dominan dalam menjelaskan keragaman inflasi Kota Bandung yaitu daging sapi sebesar 18.6%, Bawang Putih 3.9%, dan Telur Ayam 2,2%. ')

    st.subheader('Conclusion')
    col1,col2 = st.columns (2)
    with col1:
        st.caption('Berdasarkan hasil penelitian yang telah dilakukan, diperoleh simpulan hasil VECM dan analisis IRF menunjukkan bahwa fluktuasi harga komoditas beras, daging sapi, telur ayam, cabai merah, cabai rawit, bawang merah, bawang putih minyak akan berdampak pada peningkatan IHK Kota Bandung. Sebaliknya, fluktuasi harga gula pasir akan berdampak pada penurunan IHK Kota Bandung. Hasil Analisis FEVD menunjukkan harga komoditas pangan yang memiliki kontribusi dalam menjelaskan keragaman inflasi di Kota Bandung dari yang paling besar pengaruhnya adalah Daging Sapi, bawang putih dan telur ayam.')
    with col2:
        st.caption('Inflasi Kota Bandung merespon fluktuasi harga pada komoditas pangan yang menjadi objek penelitian. Oleh karena itu, diperlukan upaya kebijakan pengendalian inflasi di Kota Bandung melalui Tim Pengendalian Inflasi Daerah (TPID). TPID perlu melakukan pemantauan atas perkembangan harga dan kondisi stok komoditas pangan di Kota Bandung khususnya pada waktu-waktu dimana terjadi lonjakan harga seperti musim paceklik ataupun menjelang Hari Besar Keagamaan nasional.')
