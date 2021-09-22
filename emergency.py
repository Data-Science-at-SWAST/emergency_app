import streamlit as st
from PIL import Image
import time
import pandas as pd
from modules import (ensemble, utility)
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore')

############################# Initial set up of sidebar and upload

df = pd.DataFrame()

st.set_page_config(page_title='emergency predict',  layout='wide', initial_sidebar_state='auto', page_icon=':ambulance:') 

st.image(Image.open('images/emergency logo.png'), width = 400)

st.sidebar.markdown("Open Source NHS Data Science Library")

# General page formatting and set-up

    
uploaded_file = st.sidebar.file_uploader("Choose a file",type=('csv','xlsx'))

if uploaded_file is not None:
    
    try:
        df = pd.read_csv(uploaded_file)
    except:
        df = pd.read_excel(uploaded_file)
    
    if 'ds' not in df or 'dst' not in df:
        st.warning("Please name your target date column ds or dst within your uploaded data to continue")
        st.stop()
        
        
        
    elif 'ds' in df:
       
        df['ds'] = pd.to_datetime(df['ds'], format='%d/%m/%Y')
        df = df.set_index('ds')
        
    elif 'dst' in df:
        df['dst'] = pd.to_datetime(df['dst'], format='%d/%m/%Y %H:%M')
        df = df.set_index('dst')
        df.rename(columns={"dst": "ds")
    
target = st.sidebar.selectbox('Choose column you would like to forecast',df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.tolist(), help = 'The programme will automatically find columns that can be forecasted, just select from this list when you have imported a dataset')
st.sidebar.text(''' ''')
hori = st.sidebar.number_input("Choose your forecast horizon",value = 1, min_value = 1, max_value=999, step = 0, help = 'The horizon refers to how many units you want to forecast, i.e. if you want to forecast 7 days this number would be 7')
st.sidebar.text(''' ''')
ram = st.sidebar.selectbox('Choose forecast model',['ensemble','individualised'], help = 'Picking ensemble will combine the models to produce 1 forecast, individualised will split these out into 3 forecasts')
st.sidebar.text(''' ''')
pi = st.sidebar.selectbox('Pick Prediction Intervals',['90%','80%','60%'], help ='With a 90% prediction interval, the graph will plot an upper and lower line to represent where 90% of the data will fall')
st.sidebar.text(''' ''')
decom = st.sidebar.selectbox("Include the Seasonal Decomposition", ['No', 'Yes'], help ='Seasonal Decomposition will break down the dataset into its individual seasonal components, by default this is set to no')
st.sidebar.text(''' ''')
crossval = st.sidebar.selectbox("Include the Cross Validation results", ['No', 'Yes'], help ='Cross validation will show you the forecasts performance. However, it can be a lengthly process (typically 2-3 minutes), so if you are already happy with your model you may choose to choose No to avoid running it.')
st.sidebar.text(''' ''')
r = st.sidebar.button('Run Forecast Model')

st.markdown('The emergency forecast tool is an open source application to predict ambulance service demand.  Built in collaboration with the University of Exeter and referenced in the paper titled: Forecasting the daily demand for emergency medical ambulances in England and Wales: A benchmark model and external validation. By Thomas Monks, Michael Allen, Alison Harper, Andy Mayne and Lucy Collins.')
    
    ############################# beginning of the data input section   
    
if uploaded_file is None:
    st.warning("Please upload a .csv file to begin forecasting")
else:
    st.info("Data loaded! Begin forecasting by clicking Run Forecast Model in the Banner bar")


if r == True:
       
    if ram == 'ensemble':
        ram = False
    else:
        False
        
    if pi == '90%':
        pi = 0.1
    elif pi == '80%':
        pi = 0.2
    elif pi == '60%':
        pi = 0.4
    

    ############################# beginning of the data input section   
    with st.expander("Review input data"):
    
        st.markdown('''This is an opportunity to review your input data to ensure that NaN or egregious values do not contaminate your predictions. Clicking on the legend icons will allow you to filter out metrics. There is a date slider along the bottom should you wish to concentrate on a particular date period. Finally, in the top right corner you can choose to enter full screen mode, zoom, download, toggle tooltips and add sparklines. To restart your predictions click on the x symbol on the left hand pane.''')
           
        if uploaded_file is not None:
            
            with st.spinner('Uploading data'):
                
                df_input = df.reset_index(drop=False)
                
                n = df_input.select_dtypes(include=['int16','int32','int64','float16','float32','float64'])
                
                fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                specs=[[{"type": "table"}], [{"type": "scatter"}]]
                  
                )
            
                for i in n.columns:
                    fig.add_trace(
                        go.Scatter(x=df_input["ds"],y=df_input[i], mode="lines", name=i),row=2, col=1)
        
                fig.add_trace(
                    go.Table(
                        header=dict(
                            values=df_input.columns,
                            font=dict(size=10),
                            fill_color = 'rgba(19,24,31,0.5)',
                            line_color = 'rgba(255,255,255,0.2)',
                            font_color = '#F2F2F2',
                            align="left"
                        ),
                        cells=dict(
                            values=[df_input[k].tolist() for k in df_input.columns],
                            align = "left",
                            fill_color = 'rgba(47,58,75,0.5)',
                            line_color = 'rgba(255,255,255,0.2)',
                            font_color = '#F2F2F2',)
                    ),
                    row=1, col=1
                )
        
                fig.update_xaxes(color='#F2F2F2', gridcolor = 'rgba(255,255,255,0.2)')
                fig.update_yaxes(color='#F2F2F2', gridcolor = 'rgba(255,255,255,0.2)')
        
        
                fig.update_layout(
                    height=1000,
                    showlegend=True,
                    paper_bgcolor='rgba(34,42,55,1)',
                    title_font_color='rgba(255,255,255,1)',
                    modebar_color='rgba(255,255,255,1)',
                    plot_bgcolor='rgba(47,58,75,0.5)',
                    margin= dict(l=0,r=10,b=0,t=0),
                    legend_font_color='rgba(255,255,255,1)',
                    colorway=['#E29D89','#46AFF6','#096F64','#3993BA','#02DAC5','#FC5523','#CF6679'],
                    legend=dict(orientation="h",yanchor="bottom",y=0.47,xanchor="right",x=0.99))

                st.plotly_chart(fig, use_container_width=True)
                
                with st.spinner('Data Loading Complete!'):
                    time.sleep(1)
            

    ############################# Beginning of the forecast model section
        
    with st.spinner('Processing Forecast Model'):    
        
        with st.expander("Forecast Model"):
    
            st.markdown('Our chosen benchmark method, based on performance, is an ensemble (a simple average) of Facebooks Prophet and Regression with ARIMA errors. Both methods are flexible enough to add in special calendar events such as national holidays. In our model we chose to include new years day as this clearly stood out in the time series. In our regression model, we model the error process using the same ARIMA model- (1, 1, 3)(1, 0, 1, 7) - for each sub region. EMS providers in different regions may wish to experiment with alternative error processes.')
            
            st.markdown('Our cross-validation demonstrated that performance of the ensemble was superior to either method on its own, the other candidate models and a naive benchmark. However, we note that Prophet is also a reasonable choice for ambulance trusts new to forecasting (albeit they should recognise the shortcomings in terms of coverage). We emphasise the critical importance of a naive benchmark such as seasonal naïve in cross-validation to confirm that more complex models add value. We found that over our forecast horizon seasonal naive outperformed several state-of-the-art forecasting techniques.')
            
            st.markdown("Read the full study ""https://osf.io/a6nu5")
            
            if uploaded_file is not None:  
    
                    st.markdown("This section is the processed forecast model using the fbprophet and Linear Regression with ARIMA errors ensemble. You have similiar filter options to the first section. Within this view yhat represents the forecast and your target variable historical information is plotted adjecent. Null values will display where the outputs overlap.")
    
    
                    fcstdf = df
    
                    #fcstdf.index.freq = 'D'
                    model = utility.default_ensemble()
                    model.fit(fcstdf[target])
                    forecast_frame = model.predict(horizon=hori, return_all_models = ram, alpha = pi)          
    
                    result = pd.concat([fcstdf[target], forecast_frame], axis=1)
    
                    fig = make_subplots(rows=2, cols=1,shared_xaxes=True,vertical_spacing=0.06, specs=[[{"type": "table"}], [{"type": "scatter"}]])
    
                    result.reset_index(drop=False, inplace = True)
    
                    result['ds'] = pd.to_datetime(result['ds'])
                    result['ds'] = result['ds'].dt.date
    
                    resultnumerical = result.select_dtypes(include=['int16','int32','int64','float16','float32','float64'])
    
                    for i in resultnumerical.columns:
                        fig.add_trace(
                            go.Scatter(x=result['ds'],y=result[i], mode="lines", name=i),row=2, col=1)
    
                    fig.add_trace(
                        go.Table(
                            header=dict(
                                values=result.columns,
                                font=dict(size=10),
                                fill_color = 'rgba(19,24,31,0.5)',
                                line_color = 'rgba(255,255,255,0.2)',
                                font_color = '#F2F2F2',
                                align="left"
                            ),
                            cells=dict(
                                values=[result[k].tolist() for k in result.columns],
                                align = "left",
                                fill_color = 'rgba(47,58,75,0.5)',
                                line_color = 'rgba(255,255,255,0.2)',
                                font_color = '#F2F2F2',)
                        ),
                        row=1, col=1
                    )
                    fig.update_xaxes(color='#F2F2F2', gridcolor = 'rgba(255,255,255,0.2)')
                    fig.update_yaxes(color='#F2F2F2', gridcolor = 'rgba(255,255,255,0.2)')
                    fig.update_layout(
                    height=1000,
                    showlegend=True,
                    paper_bgcolor='rgba(34,42,55,1)',
                    title_font_color='rgba(255,255,255,1)',
                    modebar_color='rgba(255,255,255,1)',
                    plot_bgcolor='rgba(47,58,75,0.5)',
                    margin= dict(l=0,r=10,b=0,t=0),
                    legend_font_color='rgba(255,255,255,1)',
                    colorway=['#E29D89','#46AFF6','#096F64','#3993BA','#02DAC5','#FC5523','#CF6679'],
                    legend=dict(orientation="h",yanchor="bottom",y=0.47,xanchor="right",x=0.99))
                     
                    
                    st.plotly_chart(fig, use_container_width=True)
                        
                    with st.spinner('Forecast Model Built!'):
                        time.sleep(1)
    
    
    with st.expander("Raw Forecast Data table (for copying)"):
        rfdt, rfdt2 = st.columns((1,4))
        
        result = result[result['yhat'].notnull()]
        result = result[['ds','yhat']]
        
        rfdt.table(result.style.hide_index())
        
    ############################# beginning of the decomposition section       
    
    with st.spinner('Processing Decomposition'):    
            
        with st.expander("View Decomposition"):
            st.markdown('Decomposition shows a breakdown of the core components of the forecast model, in this application this is seasonality (a pattern in a time series that repeats in a regular way)i, trend (the growth either positive or negative over time).')
            st.markdown('https://www.oxfordreference.com/view/10.1093/acref/9780199541454.001.0001/acref-9780199541454')
            
            if uploaded_file is not None:
                
                
                    if decom == 'No':
                        st.markdown("Change Seaonal Decomposition to Yes to see results")
                    else:
                                
                        st.markdown('''Seasonal decomposition aims to break down the individualised components that made up the above forecasts. For more information view the statsmodels seasonal decomposition page.''')
        
        
                        sd = df
                        sd.reset_index(drop = False, inplace = True)
                        sd['ds'] = pd.to_datetime(sd['ds'])
                        sd = sd.set_index('ds')
                        sd.index.freq = 'D'
                
                        sdresult = seasonal_decompose(sd[target], model='additive')
                
                
                        it = ['resid','seasonal','trend']
                
                
                        fig = make_subplots(
                            rows=3, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.06,
                            specs=[[{"type": "scatter"}],[{"type": "scatter"}],[{"type": "scatter"}]] #need to enumerate to create the necessary row plots
                      
                            )
                
                
                        fig.add_trace(go.Scatter(x=sdresult.resid.index, y=sdresult.resid, mode="lines", name='residual'),row=1, col=1)
                        fig.add_trace(go.Scatter(x=sdresult.seasonal.index, y=sdresult.seasonal, mode="lines", name='seasonal'),row=2, col=1)
                        fig.add_trace(go.Scatter(x=sdresult.trend.index, y=sdresult.trend, mode="lines", name='trend'),row=3, col=1) 
                    
                        fig.update_xaxes(color='#F2F2F2', gridcolor = 'rgba(255,255,255,0.2)')
                        fig.update_yaxes(color='#F2F2F2', gridcolor = 'rgba(255,255,255,0.2)')
                
                
                        fig.update_layout(
                            height=1000,
                            showlegend=True,
                            paper_bgcolor='rgba(34,42,55,1)',
                            title_font_color='rgba(255,255,255,1)',
                            margin= dict(l=0,r=10,b=0,t=0),
                            modebar_color='rgba(255,255,255,1)',
                            plot_bgcolor='rgba(47,58,75,0.5)',
                            legend_font_color='rgba(255,255,255,1)',
                            colorway=['#E29D89','#46AFF6','#096F64','#3993BA','#02DAC5','#FC5523','#CF6679'],
                   
                            )
                        st.plotly_chart(fig, use_container_width=True)
              
                        with st.spinner('Decomposition Built!'):
                            time.sleep(1)    
    
    ############################# Beginning of the Cross Validation Section
    
    with st.expander("Cross Validation"):
        st.markdown('A method of assessing the accuracy and validity of a statistical model.  The available data are divided into two parts.  Modelling of the data uses one part only.  The model selected for this part is then used to predict the values in the other part of the data, a valid model should show good predictive accuracy.  In this cross validation the measure is mean absolute error (MAE)i')
        st.markdown('https://www.oxfordreference.com/view/10.1093/acref/9780199541454.001.0001/acref-9780199541454')
        
        if uploaded_file is not None:
            
            if crossval == 'No':
                st.info("Change Cross Validation to Yes to see results")
            else:
    
                with st.spinner("Processing Cross Validation"):
    
                    cvdf = df
                    cvdf.reset_index(drop = False, inplace = True)
                    cvdf['ds'] = pd.to_datetime(cvdf['ds'])
                    cvdf = cvdf.set_index('ds')
                    
                    
                    if cvdf.shape[0] - 168 <= 0:
                        st.warning("You need to provide more training data to enable cross validation")
                        
                    else:
                    
                        max_horizon = 84
                        min_train_size = cvdf.shape[0] - 168
                        horizons = [day for day in range(7, max_horizon+7,7)]
                
                        max_horizon = max(horizons)
                        
                        naive_df = ensemble.rolling_forecast_origin(cvdf[target], min_train_size = min_train_size, horizon = max_horizon, step=7)
                        
                        naive = ensemble.cross_validation_score(ensemble.SNaive(7),
                                                                naive_df,
                                                                horizons = horizons, 
                                                                metric = ensemble.mean_absolute_error, 
                                                                n_jobs=-1)
                
                        mae = ensemble.ensemble_cross_val_score(model = utility.default_ensemble(),
                                                           data = cvdf[target],
                                                           horizons = horizons, 
                                                           metric = ensemble.mean_absolute_error, 
                                                           min_train_size=min_train_size
                                                         )
                        
                        
                        
                        naiveresult = pd.DataFrame(naive, columns=horizons)
                        ensembleresult = pd.DataFrame(mae, columns=horizons)
                        
                        naiveresult.reset_index(drop = False, inplace = True)
                        ensembleresult.reset_index(drop = False, inplace = True)
                        
                        compare = pd.concat([naiveresult.describe().T['mean'], ensembleresult.describe().T['mean']],axis=1)
                        compare.columns = ['SNaive', 'Ensemble']
                    
                    
                        ############################# This is where it starts
                    
                        fig = make_subplots(
                        rows=3, cols=1,
                        shared_xaxes=True,
                        subplot_titles=("Seasonal Naive Performance", "Ensemble Performance", "Comparison"),
                        vertical_spacing=0.06,
                        specs=[[{"type": "table"}],[{"type": "table"}],[{"type": "scatter"}]]
                              
                        )
                        
                        
                        fig.add_trace(go.Table(header=dict(
                                        values=naiveresult.columns,
                                        font=dict(size=10),
                                        fill_color = 'rgba(19,24,31,0.5)',
                                        line_color = 'rgba(255,255,255,0.2)',
                                        font_color = '#F2F2F2',
                                        align="left"
                                    ),
                                    cells=dict(
                                        values=[naiveresult[k].tolist() for k in naiveresult.columns],
                                        align = "left",
                                        fill_color = 'rgba(47,58,75,0.5)',
                                        line_color = 'rgba(255,255,255,0.2)',
                                        font_color = '#F2F2F2',)
                                ),
                                row=1, col=1
                            )
                              
                        
                        fig.add_trace(go.Table(header=dict(
                                        values=ensembleresult.columns,
                                        font=dict(size=10),
                                        fill_color = 'rgba(19,24,31,0.5)',
                                        line_color = 'rgba(255,255,255,0.2)',
                                        font_color = '#F2F2F2',
                                        align="left"
                                    ),
                                    cells=dict(
                                        values=[ensembleresult[k].tolist() for k in ensembleresult.columns],
                                        align = "left",
                                        fill_color = 'rgba(47,58,75,0.5)',
                                        line_color = 'rgba(255,255,255,0.2)',
                                        font_color = '#F2F2F2',)
                                ),
                                row=2, col=1
                            )    
                
                        fig.add_trace(go.Scatter(x=compare.index, y=compare['Ensemble'], mode="lines", name='Ensemble'),row=3, col=1) 
                        fig.add_trace(go.Scatter(x=compare.index, y=compare['SNaive'], mode="lines", name='Seasonal Naive'),row=3, col=1) 
                            
                        fig.update_xaxes(color='#F2F2F2', gridcolor = 'rgba(255,255,255,0.2)')
                        fig.update_yaxes(color='#F2F2F2', gridcolor = 'rgba(255,255,255,0.2)')
                
                
                        fig.update_layout(
                            height=1000,
                            showlegend=True,
                            title_text="MAE Score",
                            paper_bgcolor='rgba(34,42,55,1)',
                            title_font_color='rgba(255,255,255,1)',
                            modebar_color='rgba(255,255,255,1)',
                            plot_bgcolor='rgba(47,58,75,0.5)',
                            margin= dict(l=0,r=10,b=10,t=30),
                            legend_font_color='rgba(255,255,255,1)',
                            colorway=['#E29D89','#46AFF6','#096F64','#3993BA','#02DAC5','#FC5523','#CF6679'])
                        
                        
                        fig.update_layout(
                            height=1000,
                            showlegend=True,
                            title_text="Cross Validation",
                            paper_bgcolor='rgba(34,42,55,1)',
                            title_font_color='rgba(255,255,255,1)',
                            modebar_color='rgba(255,255,255,1)',
                            margin= dict(l=0,r=10,b=10,t=30),
                            plot_bgcolor='rgba(47,58,75,0.5)',
                            legend_font_color='rgba(255,255,255,1)',
                            colorway=['#E29D89','#46AFF6','#096F64','#3993BA','#02DAC5','#FC5523','#CF6679'],
                           
                            )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        with st.spinner('Cross Validation Complete!'):
                            time.sleep(1)  

with st.expander("Forecast Study & Emergency App Development"):
    
    st.markdown("Read the full study here ""https://osf.io/a6nu5")          
    
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
