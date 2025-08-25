"""
Home tab component for the Student Admission Prediction application.
"""

from dash import html, dcc


def create_home_tab() -> dcc.Tab:
    """
    Create the home tab component.
    
    Returns:
        dcc.Tab: Home tab component
    """
    return dcc.Tab(
        label=' HOME',
        className='tab-icon fa fa-home',
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
                                    html.H1('Student Admission Prediction'),
                                    html.P('SAP is the best place for the bachelor students to understand their chances of getting accepted into shortlisted universities')
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Introduction Section
            html.Div(
                className='container',
                children=[
                    html.Div(
                        className='row',
                        children=[
                            # Logo
                            html.Div(
                                className='three columns',
                                children=[
                                    html.Img(
                                        src='assets/logo.jpg',
                                        style={'width': '220px', 'padding-top': '20px'}
                                    )
                                ]
                            ),
                            # Text
                            html.Div(
                                className='nine columns',
                                children=[
                                    html.H2('Motivation', className='block-caption'),
                                    html.P(
                                        '''Post graduate degrees are becoming more and more desired degrees all over the world. 
                                        It is an advantage for the students to have an idea a head about their probability
                                        of being admitted to a university, as a result the students can work on enhancing 
                                        the language test or the degree for their currently running courses and so on.
                                        In our project we use a regression task to predict the student admission percentage.''',
                                        className='text-content',
                                        style={'margin-bottom': '20px'}
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Project Objective Section
            html.Section(
                className='section objective-overlay-img',
                children=[
                    html.Div(
                        className='objective-overlay',
                        children=[
                            html.Div(
                                className='container',
                                style={'padding-top': '5px'},
                                children=[
                                    html.H2('PROJECT OBJECTIVE', className='objective-H2'),
                                    html.Div(
                                        className='row',
                                        children=[
                                            html.Div(
                                                children=[
                                                    html.P(
                                                        '''Our university acceptance calculator can help you
                                                        to find the probability of getting accepted into a 
                                                        particular university based on your profile, and it is 
                                                        completely free. Enter your language scores and CGPA to see the
                                                        predicted output. This output will give you a fair 
                                                        idea about your chance for being admitted to a particular university.''',
                                                        className='text-content'
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
            ),
            
            # About Us Section
            html.Div(
                className='AboutUs',
                children=[
                    html.Section(
                        className='container',
                        children=[
                            html.H3('Datology Team', className='block-caption'),
                            html.Div(
                                style={'overflow': 'hidden'},
                                children=[
                                    _create_team_member("Saif Almaliki", "saif.jpg", 
                                                       "https://github.com/SaifAlmaliki",
                                                       "https://www.linkedin.com/in/saif-almaliki-5a681376/"),
                                    _create_team_member("Sepideh Hosseini", "Sepideh.jpg",
                                                       "https://github.com/Sepideh-hd",
                                                       "https://www.linkedin.com/in/sepideh-hosseini-dehkordi-16452610a/"),
                                    _create_team_member("Lamees Kadhim", "lamees.png",
                                                       "https://github.com/LameesKadhim",
                                                       "https://www.linkedin.com/in/lamees-mohammed-nazar-976587119/"),
                                    _create_team_member("Tamanna", "tamanna.jpg",
                                                       "https://github.com/tamanna18",
                                                       "https://www.linkedin.com/in/tamanna-724345189/"),
                                    _create_team_member("Kunal", "kunal.png",
                                                       "https://github.com/kunalait",
                                                       "https://www.linkedin.com/in/kunal-2375b515a/")
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )


def _create_team_member(name: str, image_file: str, github_url: str, linkedin_url: str) -> html.Div:
    """
    Create a team member component.
    
    Args:
        name (str): Team member name
        image_file (str): Image file name
        github_url (str): GitHub profile URL
        linkedin_url (str): LinkedIn profile URL
        
    Returns:
        html.Div: Team member component
    """
    return html.Div(
        style={'float': 'left', 'width': '20%'},
        children=[
            html.H6(name),
            html.Img(src=f'assets/{image_file}', className='avatar'),
            html.Div(
                children=[
                    html.A(
                        href=github_url,
                        className='fa fa-github social-link',
                        target="_blank"
                    ),
                    html.A(
                        href=linkedin_url,
                        className='fa fa-linkedin social-link',
                        target="_blank"
                    )
                ]
            )
        ]
    )
