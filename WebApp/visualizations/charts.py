"""
Visualization module for creating charts and graphs for the Student Admission Prediction application.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import logging
from typing import Optional
from config.settings import CHART_HEIGHT, CHART_WIDTH, PIE_CHART_COLORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChartCreator:
    """Handles creation of various charts and visualizations."""
    
    def __init__(self, data_loader=None, model_manager=None):
        self.data_loader = data_loader
        self.model_manager = model_manager
        
    def create_regression_plot(self, y_actual: pd.Series, y_predicted: pd.Series) -> Optional[go.Figure]:
        """
        Create regression plot comparing actual vs predicted values.
        
        Args:
            y_actual (pd.Series): Actual target values
            y_predicted (pd.Series): Predicted target values
            
        Returns:
            go.Figure: Regression plot or None if error
        """
        try:
            logger.info("Creating regression plot")
            
            fig = go.Figure()
            
            # Add scatter plot of actual vs predicted
            fig.add_trace(go.Scatter(
                x=y_actual,
                y=y_predicted,
                mode='markers',
                name='actual vs. predicted'
            ))
            
            # Add regression line
            min_val = min(y_actual.min(), y_predicted.min())
            max_val = max(y_actual.max(), y_predicted.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='regression line'
            ))
            
            fig.update_layout(
                xaxis_title='Actual output',
                yaxis_title='Predicted output',
                height=CHART_HEIGHT,
                width=CHART_WIDTH
            )
            
            logger.info("Regression plot created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating regression plot: {str(e)}")
            return None
    
    def create_feature_importance_plot(self, feature_importance: dict) -> Optional[go.Figure]:
        """
        Create feature importance bar chart.
        
        Args:
            feature_importance (dict): Feature importance dictionary
            
        Returns:
            go.Figure: Feature importance plot or None if error
        """
        try:
            logger.info("Creating feature importance plot")
            
            # Create DataFrame for plotting
            importance_frame = pd.DataFrame({
                'Features': list(feature_importance.keys()),
                'Importance': list(feature_importance.values())
            })
            importance_frame = importance_frame.sort_values(by=['Importance'], ascending=True)
            
            fig = px.bar(
                importance_frame,
                y='Features',
                x='Importance',
                color='Features',
                orientation='h'
            )
            
            fig.update_layout(
                xaxis_title='Importance',
                yaxis_title='',
                height=CHART_HEIGHT,
                width=CHART_WIDTH
            )
            
            logger.info("Feature importance plot created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {str(e)}")
            return None
    
    def create_scatter_plots(self, df: pd.DataFrame) -> dict:
        """
        Create scatter plots for various features vs admission chance.
        
        Args:
            df (pd.DataFrame): Dataset
            
        Returns:
            dict: Dictionary containing scatter plots
        """
        plots = {}
        
        try:
            logger.info("Creating scatter plots")
            
            # GRE vs Admission
            plots['gre_vs_admit'] = px.scatter(
                df, x="GRE Score", y="Chance of Admit",
                log_x=True, size_max=60,
                title='GRE vs. Chance of admission'
            )
            
            # TOEFL vs Admission
            plots['toefl_vs_admit'] = px.scatter(
                df, x="TOEFL Score", y="Chance of Admit",
                log_x=True, size_max=60,
                title='TOEFL vs. Chance of admission'
            )
            
            # CGPA vs Admission
            plots['cgpa_vs_admit'] = px.scatter(
                df, x="CGPA", y="Chance of Admit",
                log_x=True, size_max=60,
                title='CGPA vs. Chance of admission'
            )
            
            logger.info("Scatter plots created successfully")
            return plots
            
        except Exception as e:
            logger.error(f"Error creating scatter plots: {str(e)}")
            return {}
    
    def create_distribution_plots(self, df: pd.DataFrame) -> dict:
        """
        Create distribution and statistical plots.
        
        Args:
            df (pd.DataFrame): Dataset
            
        Returns:
            dict: Dictionary containing distribution plots
        """
        plots = {}
        
        try:
            logger.info("Creating distribution plots")
            
            # Student distribution across universities
            df_count = df.groupby('University Rating', as_index=False).agg('count')
            df_count['std_count'] = df_count['LOR']
            
            plots['lor_vs_admit'] = px.bar(
                df_count,
                y='std_count',
                x='University Rating',
                title='Student Distribution across Universities'
            )
            
            # University rating effect on admission chance
            df_sorted = df.sort_values(by=['University Rating'])
            df_avg = df_sorted.groupby('University Rating', as_index=False)['Chance of Admit'].mean()
            
            plots['rate_vs_admit'] = go.Figure()
            plots['rate_vs_admit'].add_trace(go.Scatter(
                x=df_avg['University Rating'],
                y=df_avg['Chance of Admit'],
                mode='lines+markers'
            ))
            plots['rate_vs_admit'].update_layout(
                title='Effect of Uni Ratings on avg. admission chance',
                xaxis_title='University Rating',
                yaxis_title='Avg. Chance of Admit'
            )
            
            # Pie chart for university distribution
            total = df_count['std_count'].sum()
            df_count['percentage'] = df_count['std_count'] / total
            
            plots['pie_chart'] = px.pie(
                df_count,
                values='percentage',
                names='University Rating',
                title="Percentage of students across universities"
            )
            
            plots['pie_chart'].update_traces(
                hoverinfo='label+percent',
                textfont_size=15,
                textinfo='label+percent',
                marker=dict(
                    colors=PIE_CHART_COLORS,
                    line=dict(color='#FFFFFF', width=2)
                )
            )
            
            logger.info("Distribution plots created successfully")
            return plots
            
        except Exception as e:
            logger.error(f"Error creating distribution plots: {str(e)}")
            return {}
    
    def create_prediction_bar(self, prediction: float) -> Optional[go.Figure]:
        """
        Create prediction bar chart for the prediction tab.
        
        Args:
            prediction (float): Prediction value (0-100)
            
        Returns:
            go.Figure: Prediction bar chart or None if error
        """
        try:
            logger.info("Creating prediction bar chart")
            
            data = go.Bar(x=[0, 1, 2], y=[0, prediction, 0])
            
            layout = go.Layout(
                title='Admission Probability Graph',
                title_x=0.5,
                height=500,
                width=500,
                xaxis=dict(
                    autorange=True,
                    ticks='',
                    showticklabels=False
                ),
                yaxis=dict(
                    fixedrange=True,
                    range=[0, 100],
                    ticks='',
                    showticklabels=True
                )
            )
            
            fig = go.Figure(data=data, layout=layout)
            fig.update_traces(
                marker_color='rgb(158,202,225)',
                marker_line_color='rgb(8,48,107)',
                marker_line_width=1.5,
                opacity=0.8
            )
            
            logger.info("Prediction bar chart created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating prediction bar chart: {str(e)}")
            return None


def create_chart_creator(data_loader=None, model_manager=None) -> ChartCreator:
    """
    Factory function to create a ChartCreator instance.
    
    Args:
        data_loader: DataLoader instance
        model_manager: ModelManager instance
        
    Returns:
        ChartCreator: Initialized chart creator
    """
    return ChartCreator(data_loader, model_manager)
