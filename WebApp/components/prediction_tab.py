"""
Prediction tab component for the Student Admission Prediction application.
"""

from dash import html, dcc
import dash_daq as daq
from config.settings import (
    GRE_SCORE_RANGE, TOEFL_SCORE_RANGE, CGPA_RANGE,
    UNIVERSITY_RATING_OPTIONS, LOR_OPTIONS, SOP_OPTIONS, RESEARCH_OPTIONS
)


def create_prediction_tab() -> dcc.Tab:
    """
    Create the prediction tab component.
    
    Returns:
        dcc.Tab: Prediction tab component
    """
    return dcc.Tab(
        label=' PREDICTION',
        className='tab-icon fa fa-line-chart',
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
                                    html.H2('Model Prediction', style={'font-size': '32pt'}),
                                    html.P("Enter your information to get your expected chance of admission")
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Prediction Form Section
            html.Div(
                className='container',
                children=[
                    html.Div(
                        className='row',
                        style={'margin': '15px'},
                        children=[
                            # Left Side - Input Form
                            html.Div(
                                className='five columns border-style',
                                children=[
                                    _create_cgpa_input(),
                                    _create_gre_slider(),
                                    _create_toefl_slider(),
                                    _create_university_rating_dropdown(),
                                    _create_lor_dropdown(),
                                    _create_sop_dropdown(),
                                    _create_research_radio()
                                ]
                            ),
                            
                            # Right Side - Results
                            html.Div(
                                className='seven columns',
                                style={'text-align': 'center', 'margin-top': '35px'},
                                children=[
                                    # Admission prediction text
                                    html.Div(
                                        children=[
                                            html.H6("Admission Probability", className='prediction-text'),
                                            html.H5(id="prediction_result", className='prediction-text')
                                        ]
                                    ),
                                    
                                    # Prediction bar
                                    dcc.Graph(id='barGraph', className='prediction-bar')
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )


def _create_cgpa_input() -> html.Div:
    """Create CGPA input component."""
    return html.Div(
        style={'padding': '5px'},
        children=[
            html.Label('CGPA'),
            html.Div(
                style={'padding': '5px', 'width': '85%'},
                children=[
                    daq.NumericInput(
                        id='CGPAInput',
                        min=CGPA_RANGE['min'],
                        max=CGPA_RANGE['max'],
                        value=CGPA_RANGE['default'],
                        size=265
                    )
                ]
            )
        ]
    )


def _create_gre_slider() -> html.Div:
    """Create GRE score slider component."""
    return html.Div(
        className='row',
        style={'padding': '5px'},
        children=[
            html.Label('GRE score'),
            html.Div(
                className='row',
                style={'padding': '5px', 'width': '85%'},
                children=[
                    daq.Slider(
                        id='GRESlider',
                        min=GRE_SCORE_RANGE['min'],
                        max=GRE_SCORE_RANGE['max'],
                        value=GRE_SCORE_RANGE['default'],
                        handleLabel={"showCurrentValue": True, "label": "VALUE"},
                        step=1
                    )
                ]
            )
        ]
    )


def _create_toefl_slider() -> html.Div:
    """Create TOEFL score slider component."""
    return html.Div(
        className='row',
        style={'padding': '5px'},
        children=[
            html.Label('TOEFL iBT Score'),
            html.Div(
                className='row',
                style={'padding': '5px', 'width': '85%'},
                children=[
                    daq.Slider(
                        id='TOEFLSlider',
                        min=TOEFL_SCORE_RANGE['min'],
                        max=TOEFL_SCORE_RANGE['max'],
                        value=TOEFL_SCORE_RANGE['default'],
                        handleLabel={"showCurrentValue": True, "label": "VALUE"},
                        step=1
                    )
                ]
            )
        ]
    )


def _create_university_rating_dropdown() -> html.Div:
    """Create university rating dropdown component."""
    return html.Div(
        style={'padding': '5px'},
        children=[
            html.Label('University Rating'),
            html.Div(
                className='row',
                style={'padding': '5px', 'width': '85%'},
                children=[
                    dcc.Dropdown(
                        id='RatingDrop',
                        options=UNIVERSITY_RATING_OPTIONS,
                        value='1'
                    )
                ]
            )
        ]
    )


def _create_lor_dropdown() -> html.Div:
    """Create letter of recommendation dropdown component."""
    return html.Div(
        className='row',
        style={'padding': '5px'},
        children=[
            html.Label('Letter Of Recommendation'),
            html.Div(
                className='row',
                style={'padding': '5px', 'width': '85%'},
                children=[
                    dcc.Dropdown(
                        id='LORDrop',
                        options=LOR_OPTIONS,
                        value='0.5'
                    )
                ]
            )
        ]
    )


def _create_sop_dropdown() -> html.Div:
    """Create statement of purpose dropdown component."""
    return html.Div(
        className='row',
        style={'padding': '5px'},
        children=[
            html.Label('Statement of Purpose'),
            html.Div(
                className='row',
                style={'padding': '5px', 'width': '85%'},
                children=[
                    dcc.Dropdown(
                        id='SOPDrop',
                        options=SOP_OPTIONS,
                        value='1'
                    )
                ]
            )
        ]
    )


def _create_research_radio() -> html.Div:
    """Create research experience radio component."""
    return html.Div(
        style={'padding': '5px'},
        children=[
            html.Label('Research Experience'),
            html.Div(
                className='row',
                style={'padding': '5px'},
                children=[
                    html.Div(
                        className='six columns',
                        children=[
                            dcc.RadioItems(
                                id='ResearchRadio',
                                options=RESEARCH_OPTIONS,
                                value='0'
                            )
                        ]
                    )
                ]
            )
        ]
    )
