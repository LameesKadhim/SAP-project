# -*- coding: utf-8 -*-
import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_table
import pandas as pd
import dash_daq as daq
import cufflinks as cf
import plotly.express as px
import joblib
from dash.dependencies import Input, Output
import base64
import plotly.graph_objects as go
# Use Plotly locally
cf.go_offline()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',
                        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css']


df1 = pd.read_csv('../Dataset/admission_predict_V1.2.csv').head()
model = joblib.load('../Backend/model_RandF.sav')
#----------------------------------------------------------------------------------------

# Plotting regression result
df = pd.read_csv('../Dataset/admission_predict_V1.2.csv')
X = df.drop(['Chance of Admit','Serial No.'], axis=1)
y = df['Chance of Admit']
y_predicted = model.predict(X)

regression_fig = go.Figure()
regression_fig.add_trace(go.Scatter(x=y,
                                    y=y_predicted,
                                    mode='markers',
                                    name='actual vs. predicted'))

regression_fig.add_trace(go.Scatter(x=[y.min(), y.max()], 
                                    y=[y.min(), y.max()],
                                    mode='lines',
                                    name='regression line'))
regression_fig.update_layout(title='actual vs. predicted chance of admission',
                                xaxis_title='Actual output',
                                yaxis_title='Predicted output')
#----------------------------------------------------------------------------------
# Feature importance Visualization
importance_frame = pd.DataFrame()
importance_frame['Features'] = X.columns
importance_frame['Importance'] = model.feature_importances_
importance_frame = importance_frame.sort_values(by=['Importance'], ascending=True)

importance_fig = px.bar(importance_frame, y='Features', x='Importance', color='Features',orientation='h')

importance_fig.update_layout(title='The impact of the various features on the chance of admission',
                            xaxis_title='Importance',
                            yaxis_title='',
                            height=500, width = 700 )
#----------------------------------------------------------------------------------
gerVSadmit_fig = px.scatter(df, x="GRE Score", 
                                y="Chance of Admit",
                                log_x=True,
                                size_max=60,
                                title = 'GRE vs. Chance of admission')
#----------------------------------------------------------------------------------
toeflVSadmit_fig = px.scatter(df, x="TOEFL Score", 
                                y="Chance of Admit",
                                log_x=True,
                                size_max=60,
                                title = 'TOEFL vs. Chance of admission')
#-----------------------------------------------------------------------------------
cgpaVSadmit = px.scatter(df, x="CGPA", 
                            y="Chance of Admit",
                            log_x=True, size_max=60,
                            title = 'CGPA vs. Chance of admission')
#--------------------------------------------------------------------------
df_count = df.groupby('University Rating', as_index = False).agg('count')
df_count ['std_count'] = df_count['LOR']
lorVSadmit_fig = px.bar(df_count, 
                        y='std_count',
                        x='University Rating', 
                        title = ' Student Distriution across Universities')
#---------------------------------------------------------------------
df.sort_values(by=['University Rating'], inplace=True)
df_avg =df.groupby ('University Rating', as_index=False)['Chance of Admit'].mean()
rateVSadmit_fig=go.Figure()
rateVSadmit_fig.add_trace(go.Scatter(x=df_avg['University Rating'],
                                    y=df_avg['Chance of Admit'],
                                    mode='lines+markers'))

rateVSadmit_fig.update_layout(title='Effect of Uni Ratings on avg. admission chance',
                                xaxis_title='University Rating',
                                yaxis_title='Avg. Chance of Admit')
#----------------------------------------------------------------------
total = df_count['std_count'].sum()
df_count['percentage'] = df_count['std_count']/total

colors = ['#003f5c','#58508d','#bc5090','#ff6361','#ffa600']
pie_fig = px.pie(df_count, 
                values=df_count['percentage'], 
                names='University Rating',
                title=" Percentage of students across universities") 


pie_fig.update_traces(hoverinfo='label+percent', textfont_size=15,
                  textinfo='label+percent',
                  marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)))
#--------------------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div( children=[
    html.Div(children=[

    # Start Header ***********************************************
    html.Header(className='header-style',children=[
        html.Div(className= 'container ', children=[
            html.Div(className='ten columns', children=[
                #html.Img(src=app.get_asset_url('logo.png'), style={'width':'50%'}),
                html.H3("SAP", style={'font-family': 'fantasy'})
        ]),

            html.Div(className='two columns', style = {'text-align':'left','padding-top':'15px'} , children=[
                html.A(" Video ", href='https://github.com/LameesKadhim/SAP-project', className='fa fa-youtube-play header-links', target="_blank"),
                html.A(" Source code ", href='https://github.com/LameesKadhim/SAP-project', className='fa fa-github header-links', target="_blank")
                
        ])
      ])
    ]),
    # End Header **************************************************
    
    dcc.Tabs(parent_className='custom-tabs',
             className='custom-tabs-container', children=[

        # Start HOME  Tab*********************************************
        dcc.Tab(label=' HOME', className='tab-icon fa fa-home',selected_className='custom-tab--selected',  children=[

            # Start Overlay Section *********************************
            html.Section(className='row overlay-img', children=[
                html.Div(className='overlay', children=[
                    html.Div(className='banner', children=[
                        html.H1('Student Admission Predicion'),
                        html.P('SAP is the best place for the bachelor students to understand their chances of getting accepted into shortlisted universities')
                    ])
                ])
            ]),
            # End Overlay Section *************************************

            # INTRODUCTION ***************************************
            html.Div(className='container' , children=[
                html.Div(className='row', children=[
                    # LOGO
                    html.Div(className='three columns', children=[
                        html.Img(src=app.get_asset_url('logo.png'), style={'width': '220px'}),
                    ]),
                    # TEXT
                    html.Div(className='nine columns', children=[
                        html.H2('Objective of the project', className='block-caption'),

                        html.P('''Our university acceptance calculator can help you
                                 to find the probability of getting accepted into a 
                                 particular university based on your profile, and it’s 
                                 completely free. Enter your scores and GPA to see our 
                                 predicted output. This output will give you a fair 
                                 idea about your chances for a particular university.''', 
                                 className='text-content')
                    ])
                ]),
            ]),

            # Start About Us Section *****************************************
            html.Div(className='container', children=[
                html.Section(className='AboutUs', children=[
                        html.H3('Datology Team',className='block-caption'),
                        
                        html.Div(style={'overflow': 'hidden'}, children=[
                            html.Div(style={'float': 'left','width':'20%'}, children=[
                                html.H6("Saif Almaliki"),
                                html.Img(src=app.get_asset_url('saif.jpg'), className='avatar'),
                                html.Div(children=[
                                    html.A(href= 'https://github.com/SaifAlmaliki', className='fa fa-github social-link ', target="_blank"),
                                    html.A(href= 'https://www.linkedin.com/in/saif-almaliki-5a681376/', className='fa fa-linkedin social-link', target="_blank")
                                ])
                            ]),

                            
                            html.Div(style={'float': 'left','width':'20%'}, children=[
                                html.H6("Sepideh Hosseini"),
                                html.Img(src=app.get_asset_url('Sepideh.jpg'), className='avatar'),
                                html.Div(children=[
                                    html.A(href= 'https://github.com/Sepideh-hd', className='fa fa-github social-link ', target="_blank"),
                                    html.A(href= 'https://www.linkedin.com/in/sepideh-hosseini-dehkordi-16452610a/', className='fa fa-linkedin social-link', target="_blank")
                                    
                                ])
                            ]),

                            html.Div(style={'float': 'left','width':'20%'}, children=[
                                html.H6("Lamees Kadhim"),
                                html.Img(src=app.get_asset_url('lamees.png'), className='avatar'),
                                html.Div(children=[
                                    html.A(href= 'https://github.com/LameesKadhim', className='fa fa-github social-link ', target="_blank"),
                                    html.A(href= 'https://www.linkedin.com/in/lamees-mohammed-nazar-976587119/', className='fa fa-linkedin social-link', target="_blank")
                                    
                                ])
                            ]),


                            html.Div(style={'float': 'left','width':'20%'}, children=[
                                html.H6("Tamanna"),
                                html.Img(src=app.get_asset_url('girl.png'), className='avatar'),
                                html.Div(children=[
                                    html.A(href= 'https://github.com/tamanna18', className='fa fa-github social-link ', target="_blank"),
                                    html.A(href= 'https://www.linkedin.com/in/tamanna-724345189/', className='fa fa-linkedin social-link', target="_blank")
                                    
                                ])
                            ]),

                            html.Div(style={'float': 'left','width':'20%'}, children=[
                                html.H6("Kunal"),
                                html.Img(src=app.get_asset_url('boy.jpg'), className='avatar'),
                                html.Div(children=[
                                    html.A(href= 'https://github.com/kunalait', className='fa fa-github social-link ', target="_blank"),
                                    html.A(href= 'https://www.linkedin.com/in/kunal-2375b515a/', className='fa fa-linkedin social-link', target="_blank")
                                    
                                ])
                            ])
                        ])
                ])
            ])

        ]),

        # Start Dataset Tab **********************************
        dcc.Tab(label=' DATASET', className='tab-icon fa fa-database',selected_className='custom-tab--selected' , children=[
            # Start Overlay Section *********************************
            html.Section(className='row overlay-img', children=[
                html.Div(className='overlay', children=[
                    html.Div(className='banner', children=[
                        html.H2('Dataset Details')
                    ])
                ])
            ]),
            # End Overlay Section *************************************

             # About Dataset
             html.Div(className='container', children=[
                html.Label('DATASET', className='block-caption'),
                html.Div(className='row', style={'text-align':'center'}, children=[
                    html.P('''This dataset was built with the purpose of helping students in shortlisting 
                        universities with their profiles. The predicted output gives them a fair idea 
                        about their chances for a particular university. 
                        We use the dataset which is available in Link below: ''',
                        className='text-content')
                ]),

                html.Div(className='row', style={'text-align': 'left' , 'margin-down':'5px'} , children=[
                    html.A(
                            "View our dataset source link",
                             href='https://www.kaggle.com/mohansacharya/graduate-admissions?select=Admission_Predict.csv',
                            target="_blank")
                    ]),


                html.Div(className='row', children=[
                    html.H6('Attributes Of Dataset', className='block-caption'),
                    html.Ul(className='text-content', children=[
                        html.Li('Serial: Students serial number'),
                        html.Li('GRE Score (130-170)'),
                        html.Li('TOEFL Score (60-120)'),
                        html.Li('SOP ( Statement of Purpose)'),
                        html.Li('LOR(Letter of Recommendation) Strength(out of 5)'),
                        html.Li('Research Experience ( 0 for no experience and 1 for having an experience)'),
                        html.Li('Undergraduate CGPA is the average of grade points obtained in all the subject (out of 10)'),
                        html.Li('Chance of Admit (range from  0 to 1) --The Label')
                    ]),

                    html.P(
                        '''The size of dataset is 500 records and 9 columns and it contains
                         several parameters which are considered important during the application for Masters Programs. 
                        depending on the following factors :''',
                        className='text-content')
                    ]),

                html.Div(className='row table', children=[            
                    dash_table.DataTable(
                        id='table',
                        columns=[{"name": i, "id": i} for i in df1.columns],
                        data=df1.to_dict('records'),
                        style_cell={'textAlign': 'center'},
                        style_header={
                            'backgroundColor': '#276678',
                            'color':'#FFFF',
                            'fontWeight': 'bold'}
                    )
                ]),
             ])

        ]),
        
        # Start Dashboard Tab ******************************
        dcc.Tab(label=' DASHBOARD', className='tab-icon fa fa-bar-chart',selected_className='custom-tab--selected' , children=[ 
            
            # Start Overlay Section *********************************
            html.Section(className='row overlay-img', children=[
                html.Div(className='overlay', children=[
                    html.Div(className='banner', children=[
                        html.H2('Data Visualization'),
                        html.P("This dashboard show the relations between the features in the dataset")
                    ])
                ])
            ]),
            # End Overlay Section *************************************

            html.Div(className='container', children=[
                html.Div(className='row', children=[
                    html.Div(className='six columns', children=[
                        html.Div(className='row box', children=[
                            dcc.Graph(
                                id='bar',
                                figure= lorVSadmit_fig                          
                            )  
                        ]),          
                        html.Div(className='row box', children=[
                            dcc.Graph(
                                id='scatter1',
                                figure= gerVSadmit_fig
                            )
                            
                        ]),          
                        html.Div(className='row box', children=[
                            dcc.Graph(
                                id='scatter',
                                figure= cgpaVSadmit                                
                            )                       
                        ])
                    ]),
                    html.Div(className='six columns', children=[                             
                        html.Div(className='row box', children=[                    
                            dcc.Graph(
                                id='pie',
                                figure= pie_fig                                 
                            )
                                            
                        ]),     
                        html.Div(className='row box', children=[
                            dcc.Graph(
                                id='scatter2',
                                figure= toeflVSadmit_fig
                            )
                        ]),   
                        html.Div(className='row box', children=[
                            dcc.Graph(
                                id='line',
                                figure= rateVSadmit_fig                                 
                            )
                        ])
                    ]),
                ])
            ])

        ]), #End Dashboard Tab ******************************

        # Start ML tab
        dcc.Tab(label=' MACHINE LEARNING', className='tab-icon fa fa-lightbulb-o',selected_className='custom-tab--selected', children=[
            # Start Overlay Section *********************************
            html.Section(className='row overlay-img', children=[
                html.Div(className='overlay', children=[
                    html.Div(className='banner', children=[
                        html.H2('Machine Learning'),
                    #    html.P("This dashboard show the relations between the features in the dataset")
                    ])
                ])
            ]),
            # End Overlay Section *************************************

            html.Div(className='container', children=[
                html.H2('Model Explanation', className='block-caption'),
                html.Div(className='row', style={'margin':'15px'} , children=[ 
                    html.Div(className='twelve columns', children=[
                        html.P('''Post graduate degrees are becoming more and more a desired degree all over the world. 
                                It is an advantage for the student to have an idea a head about their probability
                                 of being admitted to a university, as a result the students can work on enhancing 
                                 the language test or the degree for their currently running courses... etc.
                                 In our project we use a regression task to predict the student admission percentage.''',
                                 className='text-content'),


                        html.H6('Steps to build our model:', className='block-caption'),
                        html.Ul(id='model-list', children=[
                            html.Li('data preprocessing(remove null values, normalization, map GRE score to the new scale)'),
                            html.Li('Apply different machine learning regression models'),
                            html.Li('Select the best model'),
                            html.Li('Save the model')
                            ]),
                        html.P('In our task we used Random Forest Regressor model from scikit-learn library', 
                                className='text-content'),

                        html.H6('Random Forest method explanation:', className='block-caption'),
                        html.P('''Random forests are an ensemble learning method for classification, regression and other tasks
                                 that work by building a multitude of decision trees at training time and generating the class 
                                 that is the class type (classification) or mean/average prediction (regression) of the 
                                 individual trees''', 
                                 className='text-content'),

                        html.Div(className='row', children=[
                                dcc.Graph(figure=importance_fig)
                            ]),

                        html.H6('Evaluation', className='block-caption'),
                        html.P('We test our model on the test set and the random forest regressor score was 85%', 
                                className='text-content'),

                        html.Div(className='row', children=[
                            dcc.Graph(figure=regression_fig)
                        ]),
                    ])   
                ]),
            ])
          
        ]),
        #End ML Tab*********************************************

        # Start Prediction Tab *********************************************
        dcc.Tab(label=' PREDICTION', className='tab-icon fa fa-line-chart',selected_className='custom-tab--selected',  children=[
            # Start Overlay Section *********************************
            html.Section(className='row overlay-img', children=[
                html.Div(className='overlay', children=[
                    html.Div(className='banner', children=[
                        html.H2('Model Prediction'),
                        html.P("Enter your information to get your expected chance of admission")
                    ])
                ])
            ]),
            # End Overlay Section *************************************

            html.Div(className='container', children=[
                html.Div(className='row', style={'margin':'15px'} , children=[
                    # Start Left Side  *****************************
                    html.Div(className='five columns border-style', children=[

                        # CGPA Slider ************************************
                        html.Div(style={'padding':'5px'}, children=[
                            html.Label('CGPA')

                        ]),

                            html.Div(style={'padding':'5px'}, children=[
                                daq.NumericInput(
                                                id = 'CGPAInput',
                                                min=5,
                                                max=10,
                                                value=7,
                                                size = 250
                                                ) 
                                    
                            ]),

                            # GRE Score Slider ************************************
                            html.Div(className='row',style={'padding':'5px'}, children=[
                                html.Label('GRE score')
                            ]),
                            html.Div(className='row', style={'padding':'5px'}, children=[
                                daq.Slider(id = 'GRESlider', min=130, max=170, value=140,
                                    handleLabel={"showCurrentValue": True, "label": "VALUE"},
                                    step=1
                                )
                            ]),

                            # TOFEL Slider ************************************
                            html.Div(className='row', style={'padding':'5px'}, children=[
                                html.Label('TOEFL iBT Score')
                            ]),
                            html.Div(className='row', style={'padding':'5px'}, children=[
                                daq.Slider(id = 'TOEFLSlider', min=61, max=120, value=90,
                                    handleLabel={"showCurrentValue": True, "label": "VALUE"},
                                    step=1
                                )
                            ]),

                            # Rating Div *************************************
                            html.Div(style={'padding':'5px'}, children=[
                                html.Label('University Rating')
                            ]),
                            html.Div(className='row', style={'padding':'5px'}, children=[
                                dcc.Dropdown(
                                    id = 'RatingDrop',
                                    options=[
                                        {'label': '1',  'value': '1'},
                                        {'label': '2',  'value': '2'},
                                        {'label': '3',  'value': '3'},
                                        {'label': '4',  'value': '4'},
                                        {'label': '5',  'value': '5'}
                                    ],
                                    value='1'
                                )
                            ]),

                            # LOR Div *****************************************
                            html.Div(className='row', style={'padding':'5px'}, children=[
                                html.Label('Letter Of Recommendation')
                            ]),
                            html.Div(className='row', style={'padding':'5px'}, children=[
                                dcc.Dropdown(
                                    id = 'LORDrop',
                                    options=[
                                        {'label': '0.5',  'value': '0.5'},
                                        {'label': '1.0',  'value': '1.0'},
                                        {'label': '1.5',  'value': '1.5'},
                                        {'label': '2.0',  'value': '2.0'},
                                        {'label': '2.5',  'value': '2.5'},
                                        {'label': '3.0',  'value': '3.0'},
                                        {'label': '3.5',  'value': '3.5'},
                                        {'label': '4.0',  'value': '4.0'},
                                        {'label': '4.5',  'value': '4.5'},
                                        {'label': '5.0',  'value': '5.0'}
                                    ],
                                    value='0.5'
                                )
                            ]),

                            # SOP DIv ***************************************
                            html.Div(className='row', style={'padding':'5px'}, children=[
                                html.Label('Statement of Purpose')
                            ]),
                            html.Div(className='row', style={'padding':'5px'}, children=[
                                dcc.Dropdown(
                                    id = 'SOPDrop',
                                    options=[
                                        {'label': '1',  'value': '1'},
                                        {'label': '2',  'value': '2'},
                                        {'label': '3',  'value': '3'},
                                        {'label': '4',  'value': '4'},
                                        {'label': '5',  'value': '5'}
                                    ],
                                    value='1'
                                )
                            ]),

                    
                            # Reaserch DIv ************************************
                            html.Div(style={'padding':'5px'}, children=[
                                html.Label('Reasearch Experience')
                            ]),
                            html.Div(className='row', style={'padding':'5px'}, children=[
                                html.Div(className='six columns', children=[
                                    dcc.RadioItems(
                                        id = 'ResearchRadio',
                                        options=[
                                            {'label': 'YES', 'value': '1'},
                                            {'label': 'NO', 'value': '0'} 
                                        ],
                                        value='0'
                                    )  
                                ])
                            ])
                        ]),
                        # End Left Side *****************************


                        # Start Right Side ***************************
                        html.Div(className='seven columns' , style={'text-align' : 'center', 'margin':'15px'}, children=[
                            
                            #Admission prediction Text
                            html.Div(children=[
                                    html.H6("Admission Probablity" , className='block-caption'),
                                    html.H5(id="prediction_result", 
                                            style={'font-weight':'bold', 'font-size':'40px', 'color':'#1687a7'}), 
                            ]),
                            
                            # Prediction bar 
                            dcc.Graph(id = 'barGraph',className='prediction-bar')

                        ])
                    ])
                ])

            ]), # ***END ML TAB****************************************

        
        ]) # ***END TABS ****************************************
    ]), 
    # End Tabs ***********************************************

    # Start Footer ***********************************************
    html.Footer(className='footer', children=[
        html.P('Copyright © 2021 Datology Group. Learning Analysis course . WS20/21')
    ])
    # End Footer *************************************************

])  # End Main Layout ***********************************************************

 # The callback function will provide one "Ouput" in the form of a string (=children)
@app.callback([Output(component_id="prediction_result",component_property="children"),
               Output(component_id="barGraph",component_property="figure")],
# The values correspnding to the three sliders are obtained by calling their id and value property
              [
               Input("GRESlider","value"),
               Input("TOEFLSlider","value"),
               Input("RatingDrop","value"),
               Input("SOPDrop","value"),
               Input("LORDrop","value"),
               Input("CGPAInput","value"),
               Input("ResearchRadio","value"),
                ])

# The input variable are set in the same order as the callback Inputs
def update_prediction(GRE, TOEFL, Rating,SOP,LOR, CGPA, Research):

    # We create a NumPy array in the form of the original features
    # ["GRE","TOEFL","Rating", "SOP","LOR", "CGPA","Research"]
  
    input_X = np.array([
                       GRE,
                       TOEFL,
                       Rating,
                       SOP,
                       LOR,
                       CGPA,
                       Research]).reshape(1,-1)
                               
    
    # Prediction is calculated based on the input_X array
    prediction = round(model.predict(input_X)[0] * 100, 2)
   
    #prepare the prediction bar chart graph
    data = go.Bar(x =[0,1,2], y = [0,prediction,0])
        
    layout = go.Layout(
            title = 'Admission Probability',
            height = 500,
            width  = 500,
            xaxis=dict(
            autorange=True,
            ticks='',
            showticklabels=False ),

            yaxis=dict(
            fixedrange=True,
            range = [ 0, 100 ],
            ticks='',
            showticklabels=True)
        )
    figure = go.Figure(data=data, layout=layout)
    # Customize aspect
    figure.update_traces(marker_color='rgb(158,202,225)', 
                        marker_line_color='rgb(8,48,107)',
                        marker_line_width=1.5, opacity=0.8)
    prediction  = '{:.2f} %'.format(prediction)   
    # And retuned to the Output of the callback function
    return prediction, figure 

if __name__ == '__main__':
    app.run_server(debug=True)