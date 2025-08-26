"""
Dataset tab component for the Student Admission Prediction application.
"""

from dash import html, dcc, dash_table
import pandas as pd


def create_dataset_tab(sample_data: pd.DataFrame = None) -> dcc.Tab:
    """
    Create the dataset tab component.
    
    Args:
        sample_data (pd.DataFrame): Sample data to display in the table
        
    Returns:
        dcc.Tab: Dataset tab component
    """
    return dcc.Tab(
        label=' DATASET',
        className='tab-icon fa fa-database',
        selected_className='custom-tab--selected',
        children=[
            # Overlay Section
            html.Section(
                className='row overlay-img',
                children=[
                    html.Div(
                        className='overlay',
                        children=[
                            html.Div(
                                className='banner',
                                children=[
                                    html.H2('Dataset Details', style={'font-size': '32pt'}),
                                    html.P("Explanation of the different features and the output of the dataset")
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # About Dataset Section
            html.Div(
                className='container',
                children=[
                    html.Label('DATASET', className='block-caption'),
                    html.Div(
                        className='row',
                        style={'text-align': 'center'},
                        children=[
                            html.P(
                                '''This dataset was built with the purpose of helping students in shortlisting 
                                universities with their profiles. The predicted output gives them a fair idea 
                                about their chances for a particular university. 
                                We use the dataset which is available in link below: ''',
                                className='text-content'
                            )
                        ]
                    ),
                    
                    html.Div(
                        className='row',
                        style={'text-align': 'left', 'margin-down': '5px'},
                        children=[
                            html.A(
                                "View our dataset source link",
                                href='https://www.kaggle.com/mohansacharya/graduate-admissions?select=Admission_Predict.csv',
                                target="_blank"
                            )
                        ]
                    ),
                    
                    html.Div(
                        className='row',
                        children=[
                            html.H6('Attributes Of Dataset', className='block-caption'),
                            html.Ul(
                                className='text-content',
                                children=[
                                    html.Li('Serial: Students serial number'),
                                    html.Li('GRE Score (130-170)'),
                                    html.Li('TOEFL Score (60-120)'),
                                    html.Li('SOP ( Statement of Purpose)'),
                                    html.Li('LOR(Letter of Recommendation) Strength(out of 5)'),
                                    html.Li('Research Experience ( 0 for no experience and 1 for having an experience)'),
                                    html.Li('Undergraduate CGPA is the average of grade points obtained in all the subject (out of 10)'),
                                    html.Li('Chance of Admit (range from  0 to 1) --The Label')
                                ]
                            ),
                            
                            html.P(
                                '''The size of dataset is 500 records and 9 columns and it contains
                                 several parameters which are considered important during the application for Masters Programs. 
                                Table below shows a sample from our dataset :''',
                                className='text-content'
                            )
                        ]
                    ),
                    
                    # Data Table
                    html.Div(
                        className='row table',
                        children=[
                            _create_data_table(sample_data) if sample_data is not None else html.P("No data available")
                        ]
                    )
                ]
            )
        ]
    )


def _create_data_table(data: pd.DataFrame) -> dash_table.DataTable:
    """
    Create a data table component.
    
    Args:
        data (pd.DataFrame): Data to display in the table
        
    Returns:
        dash_table.DataTable: Data table component
    """
    return dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in data.columns],
        data=data.to_dict('records'),
        style_cell={'textAlign': 'center'},
        style_header={
            'backgroundColor': '#276678',
            'color': '#FFFF',
            'fontWeight': 'bold'
        }
    )
