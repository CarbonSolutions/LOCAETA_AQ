import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from pandas.api.types import CategoricalDtype
from great_tables import GT
import logging

# Suppress all warnings in jupyter notebook
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class report_processor: 
    """
    A class to generate report results
    """

    def __init__(self, cfg):
        """
        Initialize the processor with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary with file paths and parameters
        """
        self.cfg = cfg

    def plot_health_benefit(self, combined_df, y_label, output_dir): 
        # Convert DataFrame from wide to long format
        df_long = combined_df.melt(id_vars=["Endpoint", "Race"], var_name="Run", value_name="Value")

        # Remove rows with NaN values
        df_long = df_long.dropna(subset=["Value"])

        # Skip zero values if desired
        df_long = df_long[df_long["Value"] != 0]

        # Set Seaborn style
        sns.set(style="whitegrid")

        # Determine the number of unique "Endpoint" categories with data
        endpoints_with_data = df_long["Endpoint"].unique()
        num_endpoints = len(endpoints_with_data)

        # Adjust figure size dynamically
        fig_width = min(25, num_endpoints * 8)
        fig_height = 10

        # Create figure and axes manually for better control
        fig = plt.figure(figsize=(fig_width, fig_height))

        # Create a grid of subplots
        if num_endpoints <= 3:
            ncols = num_endpoints
        else:
            ncols = 3
            
        nrows = (num_endpoints + ncols - 1) // ncols
        gridspec = fig.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.4, wspace=0.3)

        # Create each subplot individually
        for i, endpoint in enumerate(endpoints_with_data):
            row, col = i // ncols, i % ncols
            ax = fig.add_subplot(gridspec[row, col])
            
            # Filter data for this endpoint
            endpoint_data = df_long[df_long["Endpoint"] == endpoint]
            
            # Skip if no data
            if len(endpoint_data) == 0:
                continue
                
            # Create the bar plot
            ax = sns.barplot(data=endpoint_data, x="Race", y="Value", hue="Run", ax=ax)

            # Set title and labels
            ax.set_title(f"Endpoint: {endpoint}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Race-Ethnicity", fontsize=15)
            ax.set_ylabel(y_label, fontsize=15)

            # Regular rotation for other endpoints
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
            
            # Remove legend from individual subplots (we'll add a common one later)
            ax.get_legend().remove()

            # Add vertical gridlines between categories
            # Get the actual x tick positions
            tick_positions = ax.get_xticks()
            for j in range(len(tick_positions)):
                if j > 0:  # Skip the first one to avoid a line at the left edge
                    # Place gridline between tick positions
                    midpoint = (tick_positions[j-1] + tick_positions[j]) / 2
                    ax.axvline(x=midpoint, color='grey', linestyle='--', linewidth=0.5, alpha=0.6)
            
            # Remove legend from individual subplots (we'll add a common one later)
            if hasattr(ax, 'get_legend') and ax.get_legend() is not None:
                ax.get_legend().remove()

        # Add a single legend for the entire figure
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, title="Run", loc="upper center", 
                bbox_to_anchor=(0.5, 0.05), ncol=min(5, len(labels)), frameon=True)

        if '\n' in y_label:
            filename_label = y_label.replace('\n', '')
        else:
            filename_label = y_label

        # Improve overall layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Make room for the legend
        plt.savefig(os.path.join(output_dir, f'{filename_label}.png'), dpi=300, bbox_inches='tight')
        logger.info(f"Health bar plots are saved successfully: {os.path.join(output_dir, f'{filename_label}.png')}")

    def plot_only_mortality(self, combined_df, y_label, output_dir): 

        # Filter the dataframe to get only rows containing "Mortality All Cause" in the Endpoint
        mortality_df = combined_df[combined_df['Endpoint'].str.contains('Mortality All Cause') & combined_df['Race'].str.contains('ALL') ].copy()

        mortality_df.drop(columns='Race', inplace=True)
        mortality_df.set_index('Endpoint', inplace=True)

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Create the bar plot
        ax = mortality_df.plot(kind='bar')

        # Customize the plot (optional)
        plt.title('')
        plt.xlabel('')
        plt.ylabel(y_label)
        plt.xticks(fontsize=12, rotation=0) # Rotate x-axis labels for better readability
        # Add grid lines for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add vertical gridlines between categories
        # Get the actual x tick positions
        tick_positions = ax.get_xticks()
        for j in range(len(tick_positions)):
            if j > 0:  # Skip the first one to avoid a line at the left edge
                # Place gridline between tick positions
                midpoint = (tick_positions[j-1] + tick_positions[j]) / 2
                ax.axvline(x=midpoint, color='grey', linestyle='--', linewidth=0.5, alpha=0.6)

        # Add a single legend for the entire figure
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, labels, title="Run", loc="upper center", 
                bbox_to_anchor=(0.5, -0.1), ncol=min(2, len(labels)), frameon=True)
            
        # Add value labels on top of bars
        for container in ax.containers:
            if 'Monetized' in y_label:
                ax.bar_label(container, fmt='{:,.0f}')
            else:
                ax.bar_label(container, fmt='{:,.0f}')

        if '\n' in y_label:
            filename_label = y_label.replace('\n', '')
        else:
            filename_label = y_label

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{filename_label} only mortality.png'), dpi=300, bbox_inches='tight')
        # Show the plot
        logger.info(f"Mortality only plots are saved successfully: {os.path.join(output_dir, f'{filename_label} only mortality.png')}")


    def plot_area_weighted_average_all_runs(self, df, run_list, output_dir, output_file):
        # Define your desired order of species
        species_order = ["NH3", "SOA", "NOx","SOx","PNH4", "PNO3", "PSO4", "PrimPM25", "TotalPM25"]  # Modify as needed

        # Convert DataFrame from wide to long format
        df_long = df.melt(id_vars=["Species"], var_name="Run", value_name="Area-Weighted Average")

        # Convert "Species" to a categorical type with the defined order
        species_cat = CategoricalDtype(categories=species_order, ordered=True)
        df_long["Species"] = df_long["Species"].astype(species_cat)

        # Set run order based on run_list
        run_cat = CategoricalDtype(categories=run_list, ordered=True)
        df_long["Run"] = df_long["Run"].astype(run_cat)

        # Sort the DataFrame to ensure correct plotting order
        df_long = df_long.sort_values(["Species","Run"])

        # Set Seaborn style
        sns.set_style("whitegrid")

        # Create figure and bar plot
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(data=df_long, x="Species", y="Area-Weighted Average", hue="Run", palette="tab10")

        # Customize the plot
        plt.title("Area-Weighted Averages by Species Across Runs", fontsize=25, fontweight="bold", pad=20)
        plt.xlabel("Species", fontsize=20)
        plt.ylabel("Area-Weighted Average [ug/m³]", fontsize=18)
        plt.xticks(rotation=45, fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(title="Run", title_fontsize=14, fontsize=16, frameon=True)
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        # Add subtle vertical gridlines between categories
        for i in range(len(ax.get_xticks())):
            ax.axvline(x=i - 0.5, color='grey', linestyle='--', linewidth=0.5, alpha=0.3)

        # Show plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, output_file), dpi=300, bbox_inches='tight')
        logger.info(f"Combined concentration plots are saved successfully: {os.path.join(output_dir, output_file)}")

    def format_if_not_zero(self, val, format_string):
        """Formats the value if it's not zero, otherwise returns the original value."""
        return format_string.format(val) if val != 0 else val

    def save_area_weighted_avg_for_all_runs (self, df, output_dir, output_file):
        format_string = "{:.4E}"
        # Apply the formatting
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].apply(lambda x: self.format_if_not_zero(x, format_string))

        df.to_csv(os.path.join(output_dir, output_file), index=False)
        logger.info(f"Combined mean concentrations are saved in {os.path.join(output_dir, output_file)}")
        logger.info(GT(df))


    def read_and_prepare_benmap_csv(self, runname, output_root, benmap_target_file, benmap_output):
        """
        Read, filter, and prepare a BenMAP CSV file for merging.

        Returns
        -------
        df_mean : pd.DataFrame
            DataFrame containing Mean values (filtered, renamed).
        df_normalized : pd.DataFrame or None
            DataFrame containing normalized Mean_per_Pop values if applicable.
        """
        output_path = os.path.join(output_root, runname)
        file_path = os.path.join(output_path, benmap_target_file)

        df = pd.read_csv(file_path)
        df = df[df['Mean'] != 0]  # Filter out zero Mean
        df = df[~df['Race'].str.contains("WHITE", case=False, na=False)]  # Remove White

        if benmap_output == 'incidence':
            df.rename(
                columns={'Mean': runname, 'Mean_per_Pop': f'{runname}_(per_million)'},
                inplace=True
            )
            df_mean = df[['Endpoint', 'Race', runname]]
            df_normalized = df[['Endpoint', 'Race', f'{runname}_(per_million)']]
            return df_mean, df_normalized
        else:
            df.rename(columns={'Mean': runname}, inplace=True)
            df_mean = df[['Endpoint', 'Race', runname]]
            return df_mean, None


    def merge_benmap_dfs(self, existing_df, new_df):
        """Merge two BenMAP DataFrames on Endpoint and Race."""
        if existing_df is None:
            return new_df
        return pd.merge(existing_df, new_df, on=['Endpoint', 'Race'], how='outer')


    def save_and_plot_results(self, combined_df, combined_df_normalized, benmap_output, out_dir):
        """Save combined results and generate summary plots."""
        if benmap_output == 'incidence':
            self.plot_health_benefit(combined_df, 'Annual Health Benefits', out_dir)
            self.plot_health_benefit(combined_df_normalized, "Normalized Health Benefits (per million)", out_dir)
            self.plot_only_mortality(combined_df, 'Annual Health Benefits', out_dir)

            combined_df.to_csv(os.path.join(out_dir, "Health_Benefit.csv"), index=False)
            logger.info(f"Health benefits are saved in {os.path.join(out_dir, 'Health_Benefit.csv')}")

            combined_df_normalized.to_csv(os.path.join(out_dir, "Normalized_Health_Benefit.csv"), index=False)
            logger.info(f"Normalized health benefits are saved in {os.path.join(out_dir, 'Normalized_Health_Benefit.csv')}")

        elif benmap_output == 'valuation':
            self.plot_health_benefit(combined_df, 'Monetized Annual Health Benefits (million $)', out_dir)
            self.plot_only_mortality(combined_df, 'Monetized Annual Health Benefits (million $)', out_dir)
            combined_df.to_csv(os.path.join(out_dir, "Monetized_Health_Benefit.csv"), index=False)
            logger.info(f"Valuation of health benefits are saved in {os.path.join(out_dir, 'Monetized_Health_Benefit.csv')}")

