# Import the classes from the other files in the same directory
from data_inspector import DataInspector
from plotting_methods import PlottingMethods

def main():
    # 1. Initialize the Inspector and Plotting Tools
    inspector = DataInspector()
    plotter = PlottingMethods()

    # 2. Upload Data (Will prompt you in the command line)
    print("--- STAGE 1: INGESTION ---")
    inspector.upload_data() 
    
    # Check if data loaded successfully before proceeding
    if inspector.df is None:
        print("Data failed to load. Exiting.")
        return

    # 3. Structural Analysis
    print("\n--- STAGE 2: SUMMARY & SANITIZATION ---")
    inspector.data_summary()

    # 4. Impute and Clean
    inspector.handle_missing_values(strategy='median')
    inspector.remove_duplicates()

    # Targeted deletions
    inspector.delete_columns("PassengerId, Name, Ticket, Cabin")

    # Outlier management
    inspector.handle_outliers(columns='Fare', action='flag')

    # 5. Normalization Extraction
    print("\n--- STAGE 3: FEATURE ENGINEERING ---")
    unified_df = inspector.get_unified_dataframe(num_strategy='robust', cat_strategy='onehot')
    print("Unified Encoded Data Preview:")
    # Changed 'display' to 'print' for local cmd execution
    print(unified_df.head().to_string())

    # 6. Advanced Visualization
    print("\n--- STAGE 4: ADVANCED VISUALIZATION ---")
    print("Generating charts... Look at your web browser tabs!")
    
    # Univariate Subplots
    inspector.univariate_subplots('Age')

    # Smart Relationships (Num-Num, Cat-Num, Cat-Cat)
    inspector.plot_relationship('Age', 'Fare')       # Num-Num
    inspector.plot_relationship('Survived', 'Age')   # Cat-Num
    inspector.plot_relationship('Survived', 'Sex')   # Cat-Cat

    # Frequency Plot
    inspector.categorical_frequency('Embarked')

    # 7. Deep Statistical Insights
    print("\n--- STAGE 5: DEEP STATISTICAL INSIGHTS ---")
    inspector.plot_all_associations_heatmap()

    # 8. Test Modular Plotting Methods (HTML generation)
    print("\n--- STAGE 6: HTML PLOTTING EXPORTS ---")
    html_bar = plotter.bar_chart(inspector.df, 'Pclass', title="Passenger Class Distribution")
    print(f"Generated HTML string length: {len(html_bar)} characters.")
    print("Process complete!")

if __name__ == "__main__":
    main()