"""
Dashboard tab component for the Student Admission Prediction application.
"""

from dash import html, dcc
from typing import Dict, Any


def create_dashboard_tab(charts: Dict[str, Any]) -> dcc.Tab:
    """
    Create the dashboard tab component.
    
    Args:
        charts (Dict[str, Any]): Dictionary containing chart figures
        
    Returns:
        dcc.Tab: Dashboard tab component
    """
    return dcc.Tab(
        label=' DASHBOARD',
        className='tab-icon fa fa-bar-chart',
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
                                    html.H2('Data Visualization', style={'font-size': '32pt'}),
                                    html.P("This dashboard show the relations between the features in the dataset")
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Charts Section
            html.Div(
                className='container',
                style={'margin-top': '10px'},
                children=[
                    html.Div(
                        className='row',
                        children=[
                            # Left Column
                            html.Div(
                                className='six columns',
                                children=[
                                    html.Div(
                                        className='row box',
                                        children=[
                                            dcc.Graph(
                                                id='bar',
                                                figure=charts.get('lor_vs_admit')
                                            )
                                        ]
                                    ),
                                    html.Div(
                                        className='row box',
                                        children=[
                                            dcc.Graph(
                                                id='scatter1',
                                                figure=charts.get('gre_vs_admit')
                                            )
                                        ]
                                    ),
                                    html.Div(
                                        className='row box',
                                        children=[
                                            dcc.Graph(
                                                id='scatter',
                                                figure=charts.get('cgpa_vs_admit')
                                            )
                                        ]
                                    )
                                ]
                            ),
                            
                            # Right Column
                            html.Div(
                                className='six columns',
                                children=[
                                    html.Div(
                                        className='row box',
                                        children=[
                                            dcc.Graph(
                                                id='pie',
                                                figure=charts.get('pie_chart')
                                            )
                                        ]
                                    ),
                                    html.Div(
                                        className='row box',
                                        children=[
                                            dcc.Graph(
                                                id='scatter2',
                                                figure=charts.get('toefl_vs_admit')
                                            )
                                        ]
                                    ),
                                    html.Div(
                                        className='row box',
                                        children=[
                                            dcc.Graph(
                                                id='line',
                                                figure=charts.get('rate_vs_admit')
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )
