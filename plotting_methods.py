import plotly.express as px
import pandas as pd

class PlottingMethods:
    """
    A modular class for generating granular Plotly charts wrapped in HTML.
    """
    
    @staticmethod
    def bar_chart(df: pd.DataFrame, x_col: str, y_col: str = None, title: str = "Bar Chart") -> str:
        """Generates an HTML-wrapped bar chart."""
        if df.empty or x_col not in df.columns:
            return "<p>Error: Invalid data or column for Bar Chart.</p>"
            
        if y_col:
            fig = px.bar(df, x=x_col, y=y_col, title=title)
        else:
            counts = df[x_col].value_counts().reset_index()
            counts.columns = [x_col, 'count']
            fig = px.bar(counts, x=x_col, y='count', title=title)
            
        return fig.to_html(full_html=False)

    @staticmethod
    def pie_chart(df: pd.DataFrame, names_col: str, title: str = "Pie Chart") -> str:
        """Generates an HTML-wrapped pie chart."""
        if df.empty or names_col not in df.columns:
            return "<p>Error: Invalid data or column for Pie Chart.</p>"
            
        counts = df[names_col].value_counts().reset_index()
        counts.columns = [names_col, 'count']
        fig = px.pie(counts, names=names_col, values='count', title=title)
        return fig.to_html(full_html=False)

    @staticmethod
    def histogram(df: pd.DataFrame, x_col: str, title: str = "Histogram") -> str:
        """Generates an HTML-wrapped histogram."""
        if df.empty or x_col not in df.columns:
             return "<p>Error: Invalid data or column for Histogram.</p>"
             
        fig = px.histogram(df, x=x_col, title=title)
        return fig.to_html(full_html=False)