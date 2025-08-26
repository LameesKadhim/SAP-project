"""
Machine Learning tab component for the Student Admission Prediction application.
"""

from dash import html, dcc
from typing import Dict, Any


def create_ml_tab(charts: Dict[str, Any]) -> dcc.Tab:
    """
    Create the machine learning tab component.
    
    Args:
        charts (Dict[str, Any]): Dictionary containing chart figures
        
    Returns:
        dcc.Tab: Machine Learning tab component
    """
    return dcc.Tab(
        label=' MACHINE LEARNING',
        className='tab-icon fa fa-lightbulb-o',
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
                                    html.H2('Machine Learning', style={'font-size': '32pt'}),
                                    html.P("Model explanation, feature impact and model evaluation")
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Building Steps Section
            html.Div(
                className='container',
                children=[
                    html.H6('Steps to build our model:', className='block-caption'),
                    html.Ul(
                        id='model-list',
                        className='text-content',
                        children=[
                            html.Li('Data preprocessing (remove null values, normalization, map GRE score to the new scale)'),
                            html.Li('Apply different machine learning regression models'),
                            html.Li('Select the best model'),
                            html.Li('Save the model')
                        ]
                    ),
                    html.P(
                        'In our task we used Random Forest Regressor model from scikit-learn library',
                        className='text-content'
                    )
                ]
            ),
            
            # Random Forest Explanation Section
            html.Div(
                className='container',
                children=[
                    html.H6('Random Forest method explanation:', className='block-caption'),
                    html.P(
                        '''Random forests are an ensemble learning method for classification, regression and other tasks
                        that work by building a multitude of decision trees at training time and generating the class 
                        that is the class type (classification) or mean/average prediction (regression) of the 
                        individual trees''',
                        className='text-content'
                    )
                ]
            ),
            
            # Features Impact Section
            html.Div(
                className='container',
                children=[
                    html.H6('Features Impact on chance of admission:', className='block-caption'),
                    html.P(
                        '''The graph below shows the impact of various features on the chance of students' admission percentage ''',
                        className='text-content'
                    ),
                    html.Div(
                        className='row',
                        style={'margin': '0px 115px', "border": "2px #1687a7 solid"},
                        children=[
                            dcc.Graph(figure=charts.get('feature_importance'))
                        ]
                    )
                ]
            ),
            
            # Evaluation Section
            html.Div(
                className='container',
                children=[
                    html.H6('Evaluation', className='block-caption'),
                    html.P(
                        'We test our model on the test set and the random forest regressor score was 85%',
                        className='text-content'
                    ),
                    html.Div(
                        className='row',
                        style={'margin': '0px 115px', "border": "2px #1687a7 solid"},
                        children=[
                            dcc.Graph(figure=charts.get('regression_plot'))
                        ]
                    )
                ]
            )
        ]
    )
