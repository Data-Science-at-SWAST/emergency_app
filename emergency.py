import streamlit as st
import pandas as pd
from modules import (ensemble, utility)
import warnings 
from PIL import Image
import plotly.figure_factory as ff

warnings.filterwarnings('ignore')

 
############################# Initial set up of sidebar and upload

st.set_page_config(page_title='Open Source Data Science',  layout='wide', page_icon=':chart_with_upwards_trend:') 

st.markdown('## :chart_with_upwards_trend: Open Source NHS Data Science Library')
st.markdown('#### Ensemble Regression Prediction Model')

# General page formatting and set-up
    
uploaded_file = st.sidebar.file_uploader('',type=('csv','xlsx'))

if uploaded_file is None:
    st.sidebar.error("Please upload either a csv or xlsx file")
    

else:

    file_ext = uploaded_file.name.split(".")[1] # file extension i.e .csv
       

    if file_ext == 'csv':
        
        df = pd.read_csv(uploaded_file)
        
    elif file_ext == 'xlsx':
        
        df = pd.read_excel(uploaded_file)
        
              
    target_ds = st.sidebar.selectbox('Select date column',df.select_dtypes(include=['object','datetime', 'datetime64']).columns.tolist(), help = 'The programme will automatically find columns that can be forecasted, just select from this list when you have imported a dataset')
    
    target = st.sidebar.selectbox('Choose column you would like to forecast',df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.tolist(), help = 'The programme will automatically find columns that can be forecasted, just select from this list when you have imported a dataset')
        
    #tp = st.sidebar.selectbox(label='Choose Time interval', options = ['Hours', 'Days','Week', 'Month', 'Year'], index = 1)
    # This doesnt seem to make any difference when changing the options
        
    hori = st.sidebar.number_input("Choose the forecast horizon",value = 54, min_value = 1,  help = 'The horizon refers to how many units you want to forecast, i.e. if you want to forecast 7 days this number would be 7')

    df = df.rename(columns={target_ds:'ds'})
    
    #for excel date
    try:
        df['ds'] = pd.to_datetime(df['ds'], format = '%d/%m/%Y')
        df.sort_values(by='ds',inplace = True) # sorting values oldest - newest i.e. ascending=True (im not 100% sure on this)
    except ValueError:
        pass
         
    # for excel time
    try:
        df['ds'] = pd.to_datetime(df['ds'], format = '%H:%M:%S')
        df.sort_values(by='ds', inplace = True) 
    except ValueError:
        pass
    # This is the only way i could get hours to work, not sure how good this is, but at least it works.    
    
    st.sidebar.info('Ready to forecast? Click below to execute model')

    r = st.sidebar.button('Upload Data & Parameters')   


    if r:
        

        with st.spinner('Processing forecast model'):
            
            df = df.set_index('ds') # this needs to be ordered!!!!!!!!!!!!! A-Z
    
            df2 = df.select_dtypes(include=['int16','int32','int64','float16','float32','float64'])
            
            st.markdown('**Training data**')
            
            
            #st.line_chart(df2,)
            st.line_chart(df2[target]) # I found if looking at mulitple columns with large variation
            # it can be hard without zooming in a lot to see your target variable, to compare with 
            # forecast etc..
            
        
            model = utility.default_ensemble()
        
            model.fit(df[target])
    
            forecast_frame = model.predict(horizon=hori, return_all_models=True)
        
            result = pd.concat([df[target], forecast_frame], axis=1)
            
            forecast_frame = forecast_frame[forecast_frame['arima_mean'].notnull()]
            result = result[result['arima_mean'].notnull()]
               
            line_frame = forecast_frame[['yhat_upper95', 'yhat', 'yhat_lower_95']]
                    
            st.markdown('**Forecast data**')

            st.line_chart(line_frame)   
    
            @st.cache
            def convert_df(df):
                return forecast_frame.to_csv().encode('utf-8')
            
            csv = convert_df(forecast_frame)
            
            st.download_button(label='Download Forecast', data = csv, file_name='forecast.csv')             
        
            with st.expander('View Forecast Table'):
            
                st.markdown('Output / Forecast Data')
            
                st.dataframe(result)
        
    
with st.expander('More information'):    

    st.markdown('Our chosen benchmark method, based on performance, is an ensemble (a simple average) of Facebooks Prophet and Regression with ARIMA errors. Both methods are flexible enough to add in special calendar events such as national holidays. In our model we chose to include new years day as this clearly stood out in the time series. In our regression model, we model the error process using the same ARIMA model- (1, 1, 3)(1, 0, 1, 7) - for each sub region. EMS providers in different regions may wish to experiment with alternative error processes.')
                
    st.markdown('Our cross-validation demonstrated that performance of the ensemble was superior to either method on its own, the other candidate models and a naive benchmark. However, we note that Prophet is also a reasonable choice for ambulance trusts new to forecasting (albeit they should recognise the shortcomings in terms of coverage). We emphasise the critical importance of a naive benchmark such as seasonal naïve in cross-validation to confirm that more complex models add value. We found that over our forecast horizon seasonal naive outperformed several state-of-the-art forecasting techniques.')
            
    st.markdown("Read the full study ""https://osf.io/a6nu5")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.header("**Andy Mayne**")
    col1.image(Image.open('images/am.jpg'), width = 210)
    col1.markdown("**Data Scientist & Forecasting Manager**  \n Emergency app developer & study collaborator  \n ""mailto:andrew.mayne@swast.nhs.uk")
    
    col2.header("**Dr Tom Monks**")    
    col2.image(Image.open('images/Thomas_Monks.jpg'), width = 200)
    col2.markdown("**Associate Professor of Health Data Science**  \n Study lead  \n ""mailto:t.m.w.monks@exeter.ac.uk")
    
    col3.header("**Mike Allen**")    
    col3.image(Image.open('images/Michael_Allen.jpg'), width = 180)
    col3.markdown("**Senior Modeller**  \n Study collaborator  \n ""mailto:m.allen@exeter.ac.uk")
    
    col4.header("**Lucy Collins**")    
    col4.image(Image.open('images/Photo.JPG'), width = 230)
    col4.markdown("**Capacity Planning Analyst - SWAST**  \n Study collaborator  \n ""mailto:lucy.collins@swast.nhs.uk")
    
    col5.header("**Alison Harper**")    
    col5.image(Image.open('images/Alison-Harper.jpg'), width = 230)
    col5.markdown("**Postdoctoral Research Associate**  \n Study collaborator   \n ""mailto:a.l.harper@exeter.ac.uk")
    
