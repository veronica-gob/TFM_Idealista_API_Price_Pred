import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List, Optional
import warnings

class EdaMethods:

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def plot_distribution(self, column: str) -> None:
        """
        Plots a histogram with a kernel density estimate (KDE) overlay for a specified numerical column,
        as well as a QQ-Plot to assess normality.

        Parameters:
        ----------
        column : str
            The name of the column in the DataFrame for which to plot the distribution.

        Returns:
        -------
        None
            This function does not return any values. It directly displays the histogram and QQ-Plot.
        """
        
        # Create histogram with KDE overlay
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data[column], bins=30, kde=True)
        
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()
        
        # Create QQ-Plot
        plt.figure(figsize=(10, 6))
        stats.probplot(self.data[column], dist="norm", plot=plt)
        plt.title(f'QQ-Plot of {column}')
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Sample Quantiles')
        plt.grid()
        plt.show()

    def plot_counts(self, column: str) -> None:
        """
        Plots a bar chart showing the count of properties in each district, 
        with a maximum of 30 items displayed.

        Parameters:
        ----------
        column : str
            The name of the column in the DataFrame representing district names.

        Returns:
        -------
        None
            This function does not return any values. It directly displays the bar plot.
        """
        # Count occurrences of each district
        counts = self.data[column].value_counts()

        # Limit to the top 10 districts
        counts = counts.head(10)

        # Create a bar plot for district counts
        plt.figure(figsize=(10, 6))
        sns.barplot(x=counts.index, y=counts.values, palette='Set2')

        plt.title('Number of Properties by District')
        plt.xlabel(column)
        plt.ylabel('Number of Properties')
        plt.xticks(rotation=45)
        plt.show()

    def plot_boxplot_for_categories(self, categorical_column: str, numerical_column: str) -> None:
        """
        Plots a box plot for the numerical column distribution for each category in the categorical column,
        limiting to the top 10 categories by count and ordering them by the higher values of the numerical column.

        Parameters:
        ----------
        categorical_column : str
            The name of the column in the DataFrame representing categorical values (e.g., districts, types).
        
        numerical_column : str
            The name of the column representing the numerical values (e.g., price).
        
        Returns:
        -------
        None
            This function does not return any values. It directly displays the box plot.
        """
        
        # Ensure that the columns exist in the DataFrame
        if categorical_column not in self.data.columns or numerical_column not in self.data.columns:
            print(f"Error: '{categorical_column}' or '{numerical_column}' column not found in the DataFrame.")
            return

        # Drop rows with NaN values in the categorical or numerical column
        data_clean = self.data.dropna(subset=[categorical_column, numerical_column])

        if data_clean.empty:
            print(f"Error: No valid data found for '{categorical_column}' or '{numerical_column}'.")
            return
        
        # Get the top 10 categories by count
        top_categories = data_clean[categorical_column].value_counts().head(10).index

        # Filter the data to include only the top 10 categories
        filtered_data = data_clean[data_clean[categorical_column].isin(top_categories)]

        # Order the categories by the mean (or other aggregation) of the numerical column
        category_order = filtered_data.groupby(categorical_column)[numerical_column].mean().sort_values(ascending=False).index

        # Create a boxplot for each category in the filtered data
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=categorical_column, y=numerical_column, data=filtered_data, 
                    order=category_order, palette='Set2')

        # Set plot titles and labels
        plt.title(f'{numerical_column} Distribution by {categorical_column}')
        plt.xlabel(categorical_column)
        plt.ylabel(numerical_column)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Display the plot
        plt.show()


    def plot_scatter_matrix(self, columns: List[str], title: Optional[str] = 'Scatter Plots for All Numerical Variables') -> None:
        """
        Plots a scatter matrix (pair plot) for specified numerical columns in a DataFrame.
        
        Parameters:
        ----------
        columns : List[str]
            A list of column names to include in the scatter matrix.
        title : Optional[str], default 'Scatter Plots for All Numerical Variables'
            The title for the scatter matrix plot.

        Returns:
        -------
        None
            This function does not return any values. It directly displays the scatter matrix plot.
        """
        
        # Create scatter matrix for specified columns
        sns.pairplot(self.data[columns])
        
        # Set the title with adjusted position
        plt.suptitle(title, y=1.02)
        plt.show()

    def plot_correlation_heatmap(self, columns: List[str], figsize: Optional[tuple] = (12, 8)) -> None:
        """
        Plots a heatmap of the correlation matrix for specified numerical columns in a DataFrame.

        Parameters:
        ----------
        columns : List[str]
            A list of column names to include in the correlation heatmap.
        figsize : Optional[tuple], default (12, 8)
            Size of the figure for the heatmap.

        Returns:
        -------
        None
            This function does not return any values. It directly displays the heatmap.
        """
        
        # Calculate correlation matrix for specified columns
        corr_matrix = self.data[columns].corr().round(1)
        
        # Plot heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

    def plot_vif(self, columns: List[str]) -> None:
        """
        Calculates and prints the Variance Inflation Factor (VIF) for specified numerical columns in a DataFrame.

        Parameters:
        ----------
        columns : List[str]
            A list of column names to include in the VIF calculation.

        Returns:
        -------
        None
            This function does not return any values. It directly prints the VIF values.
        """
        
        # Create a DataFrame to store VIF results
        vif_data = pd.DataFrame()
        vif_data["Variable"] = columns
        
        # Calculate VIF for the specified columns
        X = self.data[columns]
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        
        # Print the VIF values
        print("Variance Inflation Factor (VIF):")
        print(vif_data)

    def plot_boolean_boxplots(self, price_column: str) -> None:
        """
        Plots box plots for all boolean columns in a DataFrame to visualize
        their effect on the specified price column. The plots display side-by-side 
        price distributions for each boolean column, using softer colors for True and False.

        Parameters:
        ----------
        price_column : str
            The name of the column in the DataFrame representing price values.

        Returns:
        -------
        None
            This function does not return any values. It directly displays the plots.
        """
        
        # Suppress warnings
        warnings.filterwarnings("ignore")
        
        # Identify boolean columns
        bool_columns = self.data.select_dtypes(include=['bool', 'boolean']).columns.tolist()

        # Set up the plotting area (2 plots per row)
        num_plots = len(bool_columns)
        num_cols = 2  # Number of columns for the grid
        num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate number of rows needed

        plt.figure(figsize=(10, 4 * num_rows))  # Adjust figure size as needed

        # Use Set2 palette for softer colors
        palette = sns.color_palette('Set2', n_colors=2)  # Two colors for True and False

        for i, col in enumerate(bool_columns):
            # Create a box plot for each boolean variable with softer colors
            plt.subplot(num_rows, num_cols, i + 1)
            sns.boxplot(x=col, y=price_column, data=self.data, palette=palette)
            plt.title(f'Price Distribution by {col.capitalize()}')
            plt.xlabel(col.capitalize())
            plt.ylabel('Price')

        # Adjust layout and show the plots
        plt.tight_layout()
        plt.show()

    def plot_boxplot(self, column: str) -> None:
        """
        Plots a box plot for a specified numerical column.

        Parameters:
        ----------
        column : str
            The name of the numerical column to visualize with a box plot.

        Returns:
        -------
        None
            This function does not return any values. It directly displays the box plot.
        """
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=self.data[column], palette='Set2')
        plt.title(f'Box Plot of {column}')
        plt.ylabel(column.capitalize())
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
        
    def plot_scatter(self, x: str, y: str, hue: Optional[str] = None, figsize: tuple = (10, 6)) -> None:
        """
        Plots a scatter plot for two specified columns, with an optional hue.

        Parameters:
        ----------
        x : str
            The name of the column for the x-axis.
        y : str
            The name of the column for the y-axis.
        hue : Optional[str], default None
            The name of the column to use for color coding the points (optional).
        figsize : tuple, default (10, 6)
            The size of the figure.

        Returns:
        -------
        None
            This function does not return any values. It directly displays the scatter plot.
        """
        plt.figure(figsize=figsize)
        sns.scatterplot(x=x, y=y, hue=hue, data=self.data, palette='Set2')
        plt.title(f'Scatter Plot of {y} vs {x}')
        plt.xlabel(x.capitalize())
        plt.ylabel(y.capitalize())
        plt.grid(True)
        plt.show()
