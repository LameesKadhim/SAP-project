"""
Header component for the Student Admission Prediction application.
"""

from dash import html
from config.settings import APP_TITLE


def create_header() -> html.Header:
    """
    Create the application header component.
    
    Returns:
        html.Header: Header component
    """
    return html.Header(
        className='header-style',
        children=[
            html.Div(
                className='container',
                children=[
                    html.Div(
                        className='ten columns',
                        children=[
                            html.H3(APP_TITLE, style={'font-family': 'fantasy'})
                        ]
                    ),
                    html.Div(
                        className='two columns',
                        style={'text-align': 'left', 'padding-top': '15px'},
                        children=[
                            html.A(
                                " Video ",
                                href='https://github.com/LameesKadhim/SAP-project',
                                className='fa fa-youtube-play header-links',
                                target="_blank"
                            ),
                            html.A(
                                " Source code ",
                                href='https://github.com/LameesKadhim/SAP-project',
                                className='fa fa-github header-links',
                                target="_blank"
                            )
                        ]
                    )
                ]
            )
        ]
    )
