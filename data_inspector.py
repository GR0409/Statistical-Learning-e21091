import pandas as pd
import numpy as np
import scipy.stats as ss
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OrdinalEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DataInspector:
    """
    End-to-end tool for CSV data ingestion, cleaning, feature engineering, 
    and statistical visualization for local environments.
    """
    def __init__(self):
        self.df = None
        self.original_df = None

    # ==========================================
    # 1. Data Ingestion & Sanitization
    # ==========================================
    def upload_data(self, filepath=None):
        """Handles local file ingestion via Command Prompt."""
        if filepath is None:
            print("Please provide a path to a CSV file.")
            filepath = input("File path (or press Enter to use default Titanic dataset): ")
        
        # Fallback to web dataset if user just hits Enter
        if not filepath.strip():
            print("\nNo file provided. Falling back to web Titanic dataset...")
            filepath = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

        try:
            # Garbage String Handling at ingestion
            garbage_strings = ['?', 'n/a', 'N/A', 'NULL', 'null', ' ', '']
            self.df = pd.read_csv(filepath, na_values=garbage_strings)
            self.original_df = self.df.copy()
            print(f"\nSuccessfully loaded data!")
            
            self._auto_type_correction()
        except Exception as e:
            print(f"Error loading file: {e}")

    def _auto_type_correction(self):
        """Forces conversion to numeric if it doesn't result in an entirely null column."""
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                temp_converted = pd.to_numeric(self.df[col], errors='coerce')
                if not temp_converted.isna().all() or self.df[col].isna().all():
                    self.df[col] = temp_converted
        print("Auto-type correction applied.")

    # ==========================================
    # 2. Structural Analysis & Cleaning
    # ==========================================
    def data_summary(self):
        """Displays row/column counts, a 20-row preview, and type breakdowns."""
        if self.df is None: return "No data loaded."
        
        print("\n" + "-" * 50)
        print(f"Dataset Shape: {self.df.shape[0]} Rows, {self.df.shape[1]} Columns")
        print("-" * 50)
        
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        print(f"Numerical Columns ({len(num_cols)}): {num_cols}")
        print(f"Categorical Columns ({len(cat_cols)}): {cat_cols}")
        print("-" * 50)
        print("First 20 Rows Preview:")
        # Changed 'display' to 'print' for local cmd execution
        print(self.df.head(20).to_string()) 

    def handle_missing_values(self, strategy='mean', constant_value=0):
        """Imputes missing values using mean, median, mode, or a constant."""
        if self.df is None: return
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean':
            self.df[num_cols] = self.df[num_cols].fillna(self.df[num_cols].mean())
        elif strategy == 'median':
            self.df[num_cols] = self.df[num_cols].fillna(self.df[num_cols].median())
        elif strategy == 'mode':
            for col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        elif strategy == 'constant':
            self.df = self.df.fillna(constant_value)
            
        print(f"Missing values handled using '{strategy}' strategy.")

    def remove_duplicates(self):
        """Prunes exact row matches."""
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        print(f"Removed {initial_rows - len(self.df)} duplicate rows.")

    def handle_outliers(self, columns, action='flag'):
        """IQR-based outlier detection to flag or delete rows."""
        if self.df is None: return
        if isinstance(columns, str): columns = [columns]
            
        outlier_mask = pd.Series([False] * len(self.df), index=self.df.index)
        
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                col_outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                outlier_mask = outlier_mask | col_outliers

        if action == 'flag':
            self.df['is_outlier'] = outlier_mask
            print(f"Flagged {outlier_mask.sum()} outlier rows across {columns}.")
        elif action == 'delete':
            self.df = self.df[~outlier_mask]
            print(f"Deleted {outlier_mask.sum()} outlier rows based on {columns}.")

    def delete_rows(self, rows_str):
        """Accepts a comma-separated string of indices to drop."""
        indices = [int(x.strip()) for x in rows_str.split(',') if x.strip().isdigit()]
        valid_indices = [i for i in indices if i in self.df.index]
        self.df = self.df.drop(index=valid_indices)
        print(f"Deleted {len(valid_indices)} rows.")

    def delete_columns(self, cols_str):
        """Accepts a comma-separated string of column names to drop."""
        cols = [x.strip() for x in cols_str.split(',')]
        valid_cols = [c for c in cols if c in self.df.columns]
        self.df = self.df.drop(columns=valid_cols)
        print(f"Deleted columns: {valid_cols}")

    # ==========================================
    # 3. Feature Engineering (Normalization)
    # ==========================================
    def extract_normalized_numeric_data(self, strategy='standard'):
        """Supports minmax, standard (Z-score), and robust scaling."""
        num_df = self.df.select_dtypes(include=[np.number]).copy()
        if num_df.empty: return num_df
        
        if strategy == 'minmax':
            scaler = MinMaxScaler()
        elif strategy == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
            
        scaled_data = scaler.fit_transform(num_df)
        return pd.DataFrame(scaled_data, columns=num_df.columns, index=num_df.index)

    def extract_normalized_categorical_data(self, strategy='onehot'):
        """Supports onehot, ordinal, and uniform (scaled 0-1) encoding."""
        cat_df = self.df.select_dtypes(exclude=[np.number]).copy()
        if cat_df.empty: return cat_df
        
        if strategy == 'onehot':
            return pd.get_dummies(cat_df, drop_first=True)
        elif strategy == 'ordinal':
            encoder = OrdinalEncoder()
            encoded = encoder.fit_transform(cat_df)
            return pd.DataFrame(encoded, columns=cat_df.columns, index=cat_df.index)
        elif strategy == 'uniform':
            encoder = OrdinalEncoder()
            scaler = MinMaxScaler()
            encoded = scaler.fit_transform(encoder.fit_transform(cat_df))
            return pd.DataFrame(encoded, columns=cat_df.columns, index=cat_df.index)

    def get_unified_dataframe(self, num_strategy='standard', cat_strategy='onehot'):
        """Merges scaled numerical and encoded categorical data."""
        num_norm = self.extract_normalized_numeric_data(strategy=num_strategy)
        cat_norm = self.extract_normalized_categorical_data(strategy=cat_strategy)
        return pd.concat([num_norm, cat_norm], axis=1)

    # ==========================================
    # 4. Advanced Interactive Visualization
    # ==========================================
    def univariate_subplots(self, column):
        """Generates a 3-panel subplot: Box, Scatter, and Histogram for a numeric column."""
        if column not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[column]):
            print("Requires a valid numeric column.")
            return
            
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Box Plot", "Index vs Value Scatter", "Histogram"))
        
        fig.add_trace(go.Box(x=self.df[column], name="Box"), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df[column], mode='markers', name="Scatter"), row=1, col=2)
        fig.add_trace(go.Histogram(x=self.df[column], name="Histogram"), row=1, col=3)
        
        fig.update_layout(title_text=f"Univariate Analysis: {column}", height=400)
        fig.show() # Opens in your local web browser

    def plot_relationship(self, col1, col2):
        """Smart relationship detector that picks the correct chart type."""
        is_col1_num = pd.api.types.is_numeric_dtype(self.df[col1])
        is_col2_num = pd.api.types.is_numeric_dtype(self.df[col2])
        
        if is_col1_num and is_col2_num:
            fig = px.scatter(self.df, x=col1, y=col2, trendline="ols", title=f"Scatter: {col1} vs {col2}")
        elif not is_col1_num and not is_col2_num:
            counts = self.df.groupby([col1, col2]).size().reset_index(name='count')
            fig = px.bar(counts, x=col1, y='count', color=col2, barmode='group', title=f"Grouped Bar: {col1} vs {col2}")
        else:
            cat_col, num_col = (col1, col2) if not is_col1_num else (col2, col1)
            fig = px.box(self.df, x=cat_col, y=num_col, points="all", title=f"Box Plot: {num_col} by {cat_col}")
            
        fig.show() # Opens in your local web browser

    def categorical_frequency(self, column):
        """Bar charts displaying both raw counts and percentage labels."""
        if pd.api.types.is_numeric_dtype(self.df[column]):
            print("Requires a categorical column.")
            return
            
        counts = self.df[column].value_counts().reset_index()
        counts.columns = [column, 'Count']
        counts['Percentage'] = (counts['Count'] / counts['Count'].sum() * 100).round(2).astype(str) + '%'
        
        fig = px.bar(counts, x=column, y='Count', text='Percentage', 
                     title=f"Categorical Frequency: {column}")
        fig.update_traces(textposition='outside')
        fig.show() # Opens in your local web browser

    # ==========================================
    # 5. Deep Statistical Insights
    # ==========================================
    def _cramers_v(self, x, y):
        """Calculates Cramér's V statistic for categorical-categorical association."""
        confusion_matrix = pd.crosstab(x, y)
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

    def _eta(self, cat, num):
        """Calculates Eta (via ANOVA) for categorical-continuous association."""
        valid_mask = ~cat.isna() & ~num.isna()
        cat, num = cat[valid_mask], num[valid_mask]
        
        grand_mean = num.mean()
        sst = ((num - grand_mean)**2).sum()
        ssb = sum(len(num[cat == val]) * ((num[cat == val].mean() - grand_mean)**2) for val in cat.unique())
        
        return np.sqrt(ssb / sst) if sst != 0 else 0

    def plot_all_associations_heatmap(self):
        """Visualizes relationships across ALL data types in a unified heatmap."""
        cols = self.df.columns
        matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
        
        for col1 in cols:
            for col2 in cols:
                is_num1 = pd.api.types.is_numeric_dtype(self.df[col1])
                is_num2 = pd.api.types.is_numeric_dtype(self.df[col2])
                
                if col1 == col2:
                    matrix.loc[col1, col2] = 1.0
                elif is_num1 and is_num2:
                    matrix.loc[col1, col2] = self.df[col1].corr(self.df[col2], method='pearson')
                elif not is_num1 and not is_num2:
                    matrix.loc[col1, col2] = self._cramers_v(self.df[col1], self.df[col2])
                else:
                    cat_col = col1 if not is_num1 else col2
                    num_col = col2 if not is_num1 else col1
                    matrix.loc[col1, col2] = self._eta(self.df[cat_col], self.df[num_col])
                    
        fig = px.imshow(matrix, text_auto=".2f", aspect="auto", 
                        color_continuous_scale='RdBu_r', 
                        title="Unified Correlation/Association Heatmap")
        fig.show() # Opens in your local web browser