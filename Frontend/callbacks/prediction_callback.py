"""
Prediction callback module for the Student Admission Prediction application.
"""

import logging
from dash import Input, Output
from typing import Optional
from visualizations.charts import ChartCreator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_prediction_callback(app, model_manager: 'ModelManager', chart_creator: ChartCreator):
    """
    Create the prediction callback function.
    
    Args:
        app: Dash application instance
        model_manager: ModelManager instance
        chart_creator: ChartCreator instance
    """
    
    @app.callback(
        [Output(component_id="prediction_result", component_property="children"),
         Output(component_id="barGraph", component_property="figure")],
        [Input("GRESlider", "value"),
         Input("TOEFLSlider", "value"),
         Input("RatingDrop", "value"),
         Input("SOPDrop", "value"),
         Input("LORDrop", "value"),
         Input("CGPAInput", "value"),
         Input("ResearchRadio", "value")]
    )
    def update_prediction(gre, toefl, rating, sop, lor, cgpa, research):
        """
        Update prediction based on user inputs.
        
        Args:
            gre: GRE score
            toefl: TOEFL score
            rating: University rating
            sop: Statement of Purpose score
            lor: Letter of Recommendation score
            cgpa: CGPA
            research: Research experience
            
        Returns:
            tuple: (prediction_text, prediction_figure)
        """
        try:
            logger.info("Processing prediction request")
            
            # Validate inputs
            if any(param is None for param in [gre, toefl, rating, sop, lor, cgpa, research]):
                logger.warning("Some input parameters are None")
                return "Please fill all fields", {}
            
            # Convert string inputs to float
            try:
                gre = float(gre)
                toefl = float(toefl)
                rating = float(rating)
                sop = float(sop)
                lor = float(lor)
                cgpa = float(cgpa)
                research = float(research)
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting inputs to float: {str(e)}")
                return "Invalid input values", {}
            
            # Make prediction
            prediction = model_manager.predict_admission_probability(
                gre, toefl, rating, sop, lor, cgpa, research
            )
            
            if prediction is None:
                logger.error("Prediction failed")
                return "Prediction failed", {}
            
            # Create prediction bar chart
            prediction_figure = chart_creator.create_prediction_bar(prediction)
            
            if prediction_figure is None:
                logger.error("Failed to create prediction chart")
                return f"{prediction:.2f} %", {}
            
            # Format prediction text
            prediction_text = f"{prediction:.2f} %"
            
            logger.info(f"Prediction successful: {prediction_text}")
            return prediction_text, prediction_figure
            
        except Exception as e:
            logger.error(f"Unexpected error in prediction callback: {str(e)}")
            return "An error occurred", {}
