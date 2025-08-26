"""
Main application file for the Student Admission Prediction application.
Refactored with modular design following best practices.
"""

import logging
from dash import Dash, html, dcc

# Import modules
from config.settings import EXTERNAL_STYLESHEETS, DEBUG_MODE, HOST, PORT
from data.data_loader import create_data_loader
from models.model_manager import create_model_manager
from visualizations.charts import create_chart_creator
from components.header import create_header
from components.home_tab import create_home_tab
from components.dataset_tab import create_dataset_tab
from components.dashboard_tab import create_dashboard_tab
from components.ml_tab import create_ml_tab
from components.prediction_tab import create_prediction_tab
from callbacks.prediction_callback import create_prediction_callback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure logging


def create_app() -> Dash:
    """
    Create and configure the Dash application.
    
    Returns:
        Dash: Configured Dash application
    """
    try:
        logger.info("Initializing application components")
        
        # Initialize data and model managers
        data_loader = create_data_loader()
        model_manager = create_model_manager()
        chart_creator = create_chart_creator(data_loader, model_manager)
        
        # Create Dash app
        app = Dash(__name__, external_stylesheets=EXTERNAL_STYLESHEETS)
        
        # Prepare data and charts
        df = data_loader.df
        X, y = data_loader.prepare_features_target()
        
        if df is None or X is None or y is None:
            logger.error("Failed to load data or prepare features")
            return app
        
        # Create charts
        charts = {}
        
        # Scatter plots
        scatter_plots = chart_creator.create_scatter_plots(df)
        charts.update(scatter_plots)
        
        # Distribution plots
        distribution_plots = chart_creator.create_distribution_plots(df)
        charts.update(distribution_plots)
        
        # Model-specific charts
        if model_manager.is_loaded:
            # Feature importance plot
            feature_importance = model_manager.get_feature_importance()
            if feature_importance:
                charts['feature_importance'] = chart_creator.create_feature_importance_plot(feature_importance)
            
            # Regression plot
            y_predicted = model_manager.model.predict(X)
            charts['regression_plot'] = chart_creator.create_regression_plot(y, y_predicted)
        
        # Create app layout
        app.layout = create_app_layout(data_loader, charts)
        
        # Create callbacks
        create_prediction_callback(app, model_manager, chart_creator)
        
        logger.info("Application created successfully")
        return app
        
    except Exception as e:
        logger.error(f"Error creating application: {str(e)}")
        # Return a basic app in case of error
        app = Dash(__name__)
        app.layout = html.Div([
            html.H1("Error Loading Application"),
            html.P("An error occurred while loading the application. Please check the logs.")
        ])
        return app


def create_app_layout(data_loader, charts):
    """
    Create the application layout.
    
    Args:
        data_loader: DataLoader instance
        charts: Dictionary of chart figures
        
    Returns:
        html.Div: Application layout
    """
    return html.Div([
        html.Div([
            # Header
            create_header(),
            
            # Tabs
            dcc.Tabs(
                parent_className='custom-tabs',
                className='custom-tabs-container',
                children=[
                    create_home_tab(),
                    create_dataset_tab(data_loader.get_sample_data()),
                    create_dashboard_tab(charts),
                    create_ml_tab(charts),
                    create_prediction_tab()
                ]
            )
        ]),
        
        # Footer
        html.Footer(
            className='footer',
            children=[
                html.P('Copyright © 2021 Datology Group. Learning Analysis course . WS20/21')
            ]
        )
    ])


def main():
    """Main function to run the application."""
    try:
        logger.info("Starting Student Admission Prediction application")
        
        # Create application
        app = create_app()
        
        # Run application
        logger.info(f"Starting server on {HOST}:{PORT}")
        app.run(debug=DEBUG_MODE, host=HOST, port=PORT)
        
    except Exception as e:
        logger.error(f"Error running application: {str(e)}")
        print(f"Error: {str(e)}")


if __name__ == '__main__':
    main()
