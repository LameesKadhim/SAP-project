# -*- coding: utf-8 -*-
import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_table
import pandas as pd
import dash_daq as daq
import joblib
from dash.dependencies import Input, Output
import base64


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

df1 = pd.read_csv('../Dataset/admission_predict_V1.2.csv').head()

loaded_model = joblib.load('../Backend/model_V1.sav')


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# logo preprocessing
logo = 'logo.jpg'
_base64 = base64.b64encode(open(logo, 'rb').read()).decode('ascii')
# Start Header ***********************************************
app.layout = html.Div(className='container', children=[
    html.Div(className= 'row', children=[
        html.Div(className='two columns', children=[
           html.Img(src='data:image/jpg;base64,{}'.format(_base64), style = {'width':'75%'}), 
        ]),

        html.Div(className='eight columns',style = {'text-align':'center','font-style':'italic'}, children=[
            html.H4("Student Admission Prediction")
        ]),

        html.Div(className='two columns', style = {'text-align':'right'} , children=[
            html.A("Github", href='https://github.com/LameesKadhim/SAP-project', target="_blank")
        ])
    ]),
# End Header **************************************************
    dcc.Tabs(style={'margin':'15px'},children=[
        dcc.Tab(label='HOME' , children=[


        ]),
# *************************************************************
        dcc.Tab(label='DASHBOARD',  children=[
            # Read the data frame
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in df1.columns],
                data=df1.to_dict('records')
            )

        ]),
# *******************************************
        dcc.Tab(label='ML', className='custom-tab',  children=[

           # html.H1("Prediction", style={'textAlign': 'center'}),

            # Left Side
            html.Div(className='row', style={'margin':'15px'} , children=[
                html.Div(className='six columns', children=[

                    # CGPA Slider ************************************
                    html.Div(className='row', style={'padding':'5px'}, children=[
                        html.Div(className='six columns', children=[
                            html.Label('CGPA')
                        ]),
                    ]),
                    html.Div(className='row', style={'padding':'5px'}, children=[
                        html.Div(className='six columns', children=[
                            daq.Slider(id = 'CGPASlider', min=0, max=10, value=5,
                                handleLabel={"showCurrentValue": True, "label": "VALUE"},
                                step=1
                            )
                        ])
                    ]),

                    # GRE Score Slider ************************************
                    html.Div(className='row',style={'padding':'5px'}, children=[
                        html.Div(className='six columns', children=[
                            html.Label('GRE score')
                        ]),
                    ]),
                    html.Div(className='row', style={'padding':'5px'}, children=[
                        html.Div(className='six columns', children=[
                            daq.Slider(id = 'GRESlider', min=130, max=170, value=140,
                                handleLabel={"showCurrentValue": True, "label": "VALUE"},
                                step=1
                            )
                        ])
                    ]),

                    # TOFEL Slider ************************************
                    html.Div(className='row', style={'padding':'5px'}, children=[
                        html.Div(className='six columns', children=[
                            html.Label('TOEFL Score')
                        ]),
                    ]),
                     html.Div(className='row', style={'padding':'5px'}, children=[
                        html.Div(className='six columns', children=[
                            daq.Slider(id = 'TOEFLSlider', min=0, max=120, value=90,
                                handleLabel={"showCurrentValue": True, "label": "VALUE"},
                                step=1
                            )
                        ])
                    ]),

                    # Rating Div ****************************
                    html.Div(className='row', style={'padding':'5px'}, children=[
                        html.Div(className='six columns', children=[
                            html.Label('University Rating')
                        ]),
                    ]),
                    html.Div(className='row', style={'padding':'5px'}, children=[
                        html.Div(className='six columns', children=[
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
                        ])
                    ]),

                    # LOR Div *******************************
                    html.Div(className='row', style={'padding':'5px'}, children=[
                        html.Div(className='six columns',style={'width':'100%'}, children=[
                            html.Label('Letter Of Recommendation')
                        ]),
                    ]),
                    html.Div(className='row', style={'padding':'5px'}, children=[
                        html.Div(className='six columns', children=[
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
                        ])
                    ]),

                    # SOP DIv ************************************
                    html.Div(className='row', style={'padding':'5px'}, children=[
                        html.Div(className='six columns', children=[
                            html.Label('Statement of Purpose')
                        ]),
                    ]),
                    html.Div(className='row', style={'padding':'5px'}, children=[
                        html.Div(className='six columns', children=[
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
                        ])
                    ]),

            
                    # Reaserch DIv ************************************
                    html.Div(className='row', style={'padding':'5px'}, children=[
                        html.Div(className='six columns', children=[
                            html.Label('Reasearch Experience')
                        ]),
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
                    ]),


                    # Predict Button Div ***************************
                    #html.Button('Predict', id='predict-btn', n_clicks=0)      
                ]),
                # End Left Side *****************************


                # Start Right Side ***************************
                html.Div(className='six columns' , style={'text-align' : 'center', 'margin':'15px'}, children=[
                    html.H5("Admission Probablity: "),
                    html.H5(id="prediction_result"),
                    dcc.Graph(
                        figure={
                            'data': [
                                {'x': [1,2,3], 'y': [0,4,0],'type': 'bar', 'name': 'SF'}
                               
                            ]
                        }
                    )
                ])

            ])



        


        ]),
# *******************************************
    ])
])

 # The callback function will provide one "Ouput" in the form of a string (=children)
@app.callback(Output(component_id="prediction_result",component_property="children"),
# The values correspnding to the three sliders are obtained by calling their id and value property
              [
               Input("GRESlider","value"),
               Input("TOEFLSlider","value"),
               Input("RatingDrop","value"),
               Input("SOPDrop","value"),
               Input("LORDrop","value"),
               Input("CGPASlider","value"),
               Input("ResearchRadio","value"),
                ])

# The input variable are set in the same order as the callback Inputs
def update_prediction(GRE, TOEFL, Rating,SOP,LOR, CGPA, Research):

    # We create a NumPy array in the form of the original features
    # ["Pressure","Viscosity","Particles_size", "Temperature","Inlet_flow", "Rotating_Speed","pH","Color_density"]
    # Except for the X1, X2 and X3, all other non-influencing parameters are set to their mean
    input_X = np.array([
                       GRE,
                       TOEFL,
                       Rating,
                       SOP,
                       LOR,
                       CGPA,
                       Research]).reshape(1,-1)
                               
    
    # Prediction is calculated based on the input_X array
    prediction = loaded_model.predict(input_X)[0] * 100
    
    # And retuned to the Output of the callback function
    return ("{:.2f} %".format(prediction) )

if __name__ == '__main__':
    app.run_server(debug=True)