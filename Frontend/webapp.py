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

# Feature importance Visualization
df = pd.read_csv('../Dataset/admission_predict_V1.2.csv')
X = df.drop(['Chance of Admit','Serial No.'], axis=1)
y = df['Chance of Admit']

importance_frame = pd.DataFrame()
importance_frame['Features'] = X.columns
importance_frame['Importance'] = model.feature_importances_
importance_frame = importance_frame.sort_values(by=['Importance'], ascending=True)

importance_fig = px.bar(importance_frame, y='Features', x='Importance', color='Features',orientation='h')

importance_fig.update_layout(title='The impact of the various features on the chance of admission',
                            xaxis_title='Importance',
                            yaxis_title='',
                            height=500, width = 700 )
<<<<<<< HEAD
#----------------------------------------------------------------------------------
gerVSadmit_fig = px.scatter(df, x="GRE Score", 
                                y="Chance of Admit",
                                log_x=True,
                                size_max=60)
#----------------------------------------------------------------------------------
=======
#----------------------------------------------------------------------------------
gerVSadmit_fig = px.scatter(df, x="GRE Score", 
                                y="Chance of Admit",
                                log_x=True,
                                size_max=60)
#----------------------------------------------------------------------------------
>>>>>>> bbf68ecec423a8fc9e0c8ee926df2f9f40489dbc
toeflVSadmit_fig = px.scatter(df, x="TOEFL Score", 
                                y="Chance of Admit",
                                log_x=True,
                                size_max=60)
#-----------------------------------------------------------------------------------
cgpaVSadmit = px.scatter(df, x="CGPA", 
                            y="Chance of Admit",
                            log_x=True, size_max=60)
<<<<<<< HEAD
#-----------------------------------------------------------------------------------
=======
#--------------------------------------------------------------------------
>>>>>>> bbf68ecec423a8fc9e0c8ee926df2f9f40489dbc
df_count = df.groupby('University Rating', as_index = False).agg('count')
df_count ['std_count'] = df_count['LOR']
lorVSadmit_fig = px.bar(df_count, 
                        y='std_count',
                        x='University Rating', 
                        title = 'Distriution of student applications across Universities_by_rating')
#---------------------------------------------------------------------
df.sort_values(by=['University Rating'], inplace=True)
df_avg =df.groupby ('University Rating', as_index=False)['Chance of Admit'].mean()
rateVSadmit_fig=go.Figure()
rateVSadmit_fig.add_trace(go.Scatter(x=df_avg['University Rating'],
                                    y=df_avg['Chance of Admit'],
                                    mode='lines+markers'))

rateVSadmit_fig.update_layout(title='Effect of Uni Ratings on admission',
                                xaxis_title='University Rating',
                                yaxis_title='Chance of Admit')
<<<<<<< HEAD
#---------------------------------------------------------------------
=======
#----------------------------------------------------------------------
>>>>>>> bbf68ecec423a8fc9e0c8ee926df2f9f40489dbc
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

app.layout = html.Div(style={'margin':'0'}, children=[
    html.Div(className='container', children=[
    
    # Start Header ***********************************************
        html.Div(style = {'background-color':'rgb(249 249 249)','padding':'10px'}, className= 'row', children=[
            html.Div(className='two columns', children=[
                html.Img(src=app.get_asset_url('logo.png'), style={'width':'50%'}),
        ]),

            html.Div(className='eight columns',style = {'text-align':'center'}, children=[
                html.H4("Student Admission Prediction")
        ]),

            html.Div(className='two columns', style = {'text-align':'right'} , children=[
                html.A(" Source code ", href='https://github.com/LameesKadhim/SAP-project', className='fa fa-github header-links', target="_blank"),
                html.A(" Video ", href='https://github.com/LameesKadhim/SAP-project', className='fa fa-youtube-play header-links', target="_blank")
        ])
    ]),
<<<<<<< HEAD
# End Header ********************************************************

        dcc.Tabs(style={'margin':'10px 0px'},children=[

=======
    # End Header **************************************************

    dcc.Tabs(style={'margin':'10px 0px'},children=[

>>>>>>> bbf68ecec423a8fc9e0c8ee926df2f9f40489dbc
        # Start HOME  Tab*********************************************
        dcc.Tab(label=' HOME', className='custom-tab tab-icon fa fa-home',  children=[
            html.Div(className='row' , children=[

                # Start Left Side *****************************
                html.Div(className='six columns', children=[

                    # Student Admission Prediction DIv ************************************
                    html.Div(className='row', children=[
                        html.Label(className='block-caption',
                                   children=['Objective of the project']),
                        html.Label(className='text-content', children=[
                                   'Our university acceptance calculator can help you to find the probability of getting accepted into a particular university based on your profile, and it’s completely free. Enter your scores and GPA to see our predicted output. This output will give you a fair idea about your chances for a particular university.'
                                   ])
                    ]),

                ]),
                # End Left Side *****************************

                # Start Right Side ***************************
                html.Div(className='six columns', children=[

                    # LOGO
                    html.Div(className='row', children=[
                        html.Img(src=app.get_asset_url('logo.png'), style={'width': '250px'}),
                    ]),

                    # ABOUT US
                    html.Div(className='row', children=[
                        html.Label(className='block-caption', children=['What is SAP?']),
                        html.Label(className='text-content', children=[
                            'SAP(Student Admission Prediction) is the best place for the bachelor students to understand their chances of getting accepted into shortlisted universities'
                        ])
                    ]),
                    # About Dataset
                    html.Div(className='row', children=[
                        html.Label(className='block-caption',
                                   children=['About Dataset']),
                        html.Label(className='text-content', children=[
                            'This dataset was built with the purpose of helping students in shortlisting universities with their profiles. The predicted output gives them a fair idea about their chances for a particular university. We use the dataset which is available in Link below: '
                        ])
                    ]),

                ]),

                    html.Div(className='row', style={'text-align': 'right' , 'margin-down':'5px'} , children=[
                      html.A(
                           "View our dataset source link", href='https://www.kaggle.com/mohansacharya/graduate-admissions?select=Admission_Predict.csv', target="_blank")
                    ]),

                # End Right Side *********************************************
            ]),

            # Start About Us Section *****************************************
            html.Section(className='AboutUs', children=[
                    html.H3('Datology Team',style={'font-style':'bold'}),
                    
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
                            html.Img(src=app.get_asset_url('tamanna.jpg'), className='avatar'),
                            html.Div(children=[
                                html.A(href= 'https://github.com/tamanna18', className='fa fa-github social-link ', target="_blank"),
                                html.A(href= 'https://www.linkedin.com/in/tamanna-724345189/', className='fa fa-linkedin social-link', target="_blank")
                                
                            ])
                        ]),

                        html.Div(style={'float': 'left','width':'20%'}, children=[
                            html.H6("Kunal"),
                            html.Img(src=app.get_asset_url('kunal.png'), className='avatar'),
                            html.Div(children=[
                                html.A(href= 'https://github.com/kunalait', className='fa fa-github social-link ', target="_blank"),
                                html.A(href= 'https://www.linkedin.com/in/kunal-2375b515a/', className='fa fa-linkedin social-link', target="_blank")
                                
                            ])
                        ])
                    ])
            ])

        ]),
        
        # Start Dashboard Tab
        dcc.Tab(label=' DASHBOARD', className='tab-icon fa fa-bar-chart' , children=[
            html.Div(className='row', children=[            
                dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in df1.columns],
                    data=df1.to_dict('records')
                )
            ]),
            
            html.Div(className='row', children=[
                html.Div(className='six columns', children=[
                    html.Div(className='row', children=[
                        dcc.Graph(
                            id='bar',
                            figure= lorVSadmit_fig                                 
                        )  
                                            ]),          
                    html.Div(className='row', children=[
                        dcc.Graph(
                            id='scatter1',
                            figure= gerVSadmit_fig
                        )
                        
                    ]),          
                    html.Div(className='row', children=[
                        dcc.Graph(
                            id='scatter',
                            figure= cgpaVSadmit                                
                        )                       
                    ])
                ]),
                html.Div(className='six columns', children=[                             
                    html.Div(className='row', children=[                    
                        dcc.Graph(
                            id='pie',
                            figure= pie_fig                                 
                        )
                                           
                    ]),     
                    html.Div(className='row', children=[
                        dcc.Graph(
                            id='scatter2',
                            figure= toeflVSadmit_fig
                        )
                    ]),   
                    html.Div(className='row', children=[
                        dcc.Graph(
                            id='line',
                            figure= rateVSadmit_fig                                 
                        )
                    ])
                 ]),
            ])
        ]), #End Dashboard Tab ******************************

<<<<<<< HEAD
            # Start ML Tab *********************************************
            dcc.Tab(label=' Prediction', className='tab-icon fa fa-line-chart',  children=[
=======
        # Start ML tab
        dcc.Tab(label=' ML', className='tab-icon ', children=[
            html.H2('Model Explanation', style={'font-style':'bold','text-align':'center'}),
            html.Div(className='row', style={'margin':'15px'} , children=[ 
                html.Div(className='twelve columns', children=[
                    html.P('Our task is to predict the student admission probability using a regression task'),
                    html.P('Steps to build our model:'),
                    html.Ul(id='model-list', children=[
                        html.Li('data preprocessing(remove null values, normalization, map GRE score to the new scale)'),
                        html.Li('Apply different machine learning regression models'),
                        html.Li('Select the best model'),
                        html.Li('Save the model')
                        ]),
                    html.P('In our task we used Random Forest Regressor model from scikit-learn library and obtain 85% score ' )
                
               ])   
           ]),
           html.Div(className='row', children=[
               dcc.Graph(figure=importance_fig)
               ])
        ]),
        #End ML Tab
        # Start Prediction Tab *********************************************
        dcc.Tab(label=' Prediction', className='tab-icon fa fa-line-chart',  children=[
>>>>>>> bbf68ecec423a8fc9e0c8ee926df2f9f40489dbc
                
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
                                            min=1,
                                            max=10,
                                            value=5,
                                            size = 200
                                        ) 
                                
                        ]),

                        # GRE Score Slider ************************************
                        html.Div(className='row',style={'padding':'5px'}, children=[
                        #    html.Div(className='six columns', children=[
                                html.Label('GRE score')
                        #    ]),
                        ]),
                        html.Div(className='row', style={'padding':'5px'}, children=[
                        #    html.Div(className='six columns', children=[
                                daq.Slider(id = 'GRESlider', min=130, max=170, value=140,
                                    handleLabel={"showCurrentValue": True, "label": "VALUE"},
                                    step=1
                                )
                        #    ])
                        ]),

                        # TOFEL Slider ************************************
                        html.Div(className='row', style={'padding':'5px'}, children=[
                        #    html.Div(className='six columns', children=[
                                html.Label('TOEFL iBT Score')
                        #    ]),
                        ]),
                        html.Div(className='row', style={'padding':'5px'}, children=[
                        #    html.Div(className='six columns', children=[
                                daq.Slider(id = 'TOEFLSlider', min=61, max=120, value=90,
                                    handleLabel={"showCurrentValue": True, "label": "VALUE"},
                                    step=1
                                )
                        #    ])
                        ]),

                        # Rating Div *************************************
                        html.Div(style={'padding':'5px'}, children=[
                        #    html.Div(className='six columns', children=[
                                html.Label('University Rating')
                        #    ]),
                        ]),
                        html.Div(className='row', style={'padding':'5px'}, children=[
                        #    html.Div(className='six columns', children=[
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
                        #    ])
                        ]),

                        # LOR Div *****************************************
                        html.Div(className='row', style={'padding':'5px'}, children=[
                        #    html.Div(className='six columns',style={'width':'100%'}, children=[
                                html.Label('Letter Of Recommendation')
                        #    ]),
                        ]),
                        html.Div(className='row', style={'padding':'5px'}, children=[
                        #    html.Div(className='six columns', children=[
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
                        #    ])
                        ]),

                        # SOP DIv ***************************************
                        html.Div(className='row', style={'padding':'5px'}, children=[
                        #    html.Div(className='six columns', children=[
                                html.Label('Statement of Purpose')
                        #    ]),
                        ]),
                        html.Div(className='row', style={'padding':'5px'}, children=[
                        #    html.Div(className='six columns', children=[
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
                        #    ])
                        ]),

                
                        # Reaserch DIv ************************************
                        html.Div(style={'padding':'5px'}, children=[
                        #    html.Div(className='six columns', children=[
                                html.Label('Reasearch Experience')
                        #    ]),
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
<<<<<<< HEAD
                        
                        #Admission prediction Text
                        html.Div(children=[
                                html.H6("Admission Probablity"),
                                html.H5(id="prediction_result", 
                                        style={'font-weight':'bold', 'font-size':'40px'}), 
                        ]),

                        # Importance bar
                        # dcc.Graph(figure=importance_fig),
                        
                        # Prediction bar 
                        dcc.Graph(id = 'barGraph',className='prediction-bar')
=======
                        dcc.Graph(id = 'barGraph',style={'margin-left': '92px', 'margin-top': '37px','textAlign': 'center'})
>>>>>>> bbf68ecec423a8fc9e0c8ee926df2f9f40489dbc
                        
                    ])
                ])
            ]), # ***END ML TAB****************************************

        
        ]) # ***END TABS ****************************************

    ]), # End Div Container


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