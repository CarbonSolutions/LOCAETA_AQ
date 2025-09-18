import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import numpy as np

class DataCenterEmissionProcessor:
    """
    A class to handle Data Center emission data processing and NEI integration.
    """
    
    def __init__(self, config):
        """
        Initialize the processor with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary with file paths and parameters
        """
        self.config = config
        self.NEI_cols = ['NOx','SOx', 'NH3', 'VOC', 'PM2_5']

    def load_nei(self, nei_file):
        nei = gpd.read_file(nei_file)
        nei.reset_index(drop=True, inplace=True)

        col_dict = {poll: f"{poll}_nei" for poll in self.NEI_cols}
        nei.rename(columns=col_dict, inplace=True)
        return nei

    def reformat_datacenter(self, df):
        pollutant_cols = [col for col in df.columns if "_tons_final" in col]
        base_cols = [col for col in df.columns if "_tons_base" in col]
        cols_needed = pollutant_cols + base_cols + ["eis", "cambium_gea", "DOE/EIA ORIS plant or facility code"]

        df = df[cols_needed].dropna(subset=["eis"])
        df = df.astype({"eis": "int64", "DOE/EIA ORIS plant or facility code": "int64"})
        df.rename(columns={"eis": "EIS_ID", "DOE/EIA ORIS plant or facility code": "oris_ID"}, inplace=True)

        df_grouped = df.groupby("EIS_ID").agg(
            {**{col: "sum" for col in pollutant_cols + base_cols}, "cambium_gea": "first"}
        ).reset_index()
        return df_grouped


    def find_minimal_unique_identifier_columns(self, df, max_combination_size=30):
        cols = df.columns.tolist()
        for r in range(1, min(len(cols), max_combination_size) + 1):
            for combo in combinations(cols, r):
                if not df.duplicated(subset=combo).any():
                    return list(combo)
        return None


    def mapping_datacenter_to_nei(self, nei_with_dc, nei_all_pt, unique_cols, is_base):
        if is_base:
            pollutant_map = {
                "NOx": "NOx_tons_base",
                "PM2_5": "PM2.5_tons_base",
                "VOC": "VOC_tons_base",
                "NH3": "NH3_tons_base",
                "SOx": "SO2_tons_base",
            }
        else:
            pollutant_map = {
                "NOx": "NOx_tons_final",
                "PM2_5": "PM2.5_tons_final",
                "VOC": "VOC_tons_final",
                "NH3": "NH3_tons_final",
                "SOx": "SO2_tons_final",
            }

        nei_with_dc["was_mapped"] = True

        for nei_col, dc_col in pollutant_map.items():
            total_by_eis = nei_with_dc.groupby("EIS_ID")[f"{nei_col}_nei"].transform("sum")
            nei_with_dc[f"{nei_col}_split"] = nei_with_dc[f"{nei_col}_nei"] / total_by_eis.replace(0, pd.NA)

            mask_zero_total = (total_by_eis == 0) & nei_with_dc[dc_col].notna() & (nei_with_dc[dc_col] != 0)
            for eid in nei_with_dc.loc[mask_zero_total, "EIS_ID"].unique():
                match_rows = nei_with_dc["EIS_ID"] == eid
                n_rows = match_rows.sum()
                nei_with_dc.loc[match_rows, f"{nei_col}_split"] = 1.0 / n_rows

            nei_with_dc[nei_col] = nei_with_dc[f"{nei_col}_split"] * nei_with_dc[dc_col]

        
        nei_all_pt_final = nei_all_pt.merge(
            nei_with_dc[unique_cols + ["was_mapped", "cambium_gea"] + list(pollutant_map.keys())],
            on=unique_cols,
            how="left",
        )


        for k in pollutant_map:
            nei_all_pt_final[k] = pd.to_numeric(nei_all_pt_final[k], errors="coerce")
            nei_all_pt_final[k] = nei_all_pt_final[k].fillna(nei_all_pt_final[f"{k}_nei"]).infer_objects(copy=False)

        return nei_all_pt_final

    def plot_diagnostics(self, gdf, scenario, output_dir):
        """Plot before/after emission sums for quick validation."""

        plot_totals = (
            gdf.groupby("case")[self.NEI_cols].sum().reset_index()
            .melt(id_vars="case", var_name="pollutant", value_name="tons")
        )

        # --- NEI emissions (no case distinction, just one total per pollutant) ---
        nei_cols = [f"{p}_nei" for p in self.NEI_cols]
        gdf_final = gdf[gdf["case"] == "final"]
        nei_totals = gdf_final[nei_cols].sum().reset_index()
        nei_totals.columns = ["pollutant", "tons"]
        nei_totals["pollutant"] = nei_totals["pollutant"].str.replace("_nei", "", regex=False)
        nei_totals["case"] = "NEI"

        # --- Combine ---
        plot_totals = pd.concat([plot_totals, nei_totals], ignore_index=True)

        plt.figure(figsize=(8,5))
        ax = sns.barplot(
            data=plot_totals,
            x="pollutant", y="tons", hue="case"
        )
        plt.title(f"Datacenter {scenario} emissions: Base vs Final")
        plt.ylabel("Emissions (tons)")

        # Add values on top of bars
        for p in ax.patches:
            height = p.get_height()
            if abs(height) > 0: 
                ax.annotate(
                    f"{height:,.0f}",  # format value
                    (p.get_x() + p.get_width() / 2., height/2),
                    ha='center', va='bottom',
                    fontsize=10,
                    rotation=90
                )

        plot_path = os.path.join(output_dir, f"{scenario}_emission_diagnostic.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved diagnostic plot {plot_path}")



    def plot_diff_emis(self, outputdir, gdf_emis, emis_region):
        """Plot difference in DataCenter emissions (Base vs Sens) and save figure."""

        base_pollutants = [f'{poll}_base' for poll in self.NEI_cols]
        sens_pollutants = [f'{poll}_sens' for poll in self.NEI_cols]

        grouped_combined = pd.DataFrame()
        multi_region_indices = []

        # Process each pollutant
        for i, (base_col, sens_col) in enumerate(zip(base_pollutants, sens_pollutants)):
            grouped_sum = gdf_emis.groupby('cambium_ge_sens')[[base_col, sens_col]].sum().reset_index()

            if len(grouped_sum) > 1:
                multi_region_indices.append(i)
            else:
                if grouped_combined.empty:
                    grouped_combined['cambium_ge_sens'] = grouped_sum['cambium_ge_sens']
                grouped_combined[base_col] = grouped_sum[base_col].values
                grouped_combined[sens_col] = grouped_sum[sens_col].values

        # Multi-region plots
        if multi_region_indices:
            fig, axes = plt.subplots(nrows=len(multi_region_indices), ncols=1, figsize=(10, 6*len(multi_region_indices)))
            if len(multi_region_indices) == 1:
                axes = [axes]  # ensure iterable

            for idx, ax in zip(multi_region_indices, axes):
                base_col = base_pollutants[idx]
                sens_col = sens_pollutants[idx]
                grouped_sum = gdf_emis.groupby('cambium_ge_sens')[[base_col, sens_col]].sum().reset_index()
                grouped_sum.sort_values(by=base_col, ascending=False, inplace=True)

                x = np.arange(len(grouped_sum))
                width = 0.35
                bars1 = ax.bar(x - width/2, grouped_sum[base_col], width, label=base_col)
                bars2 = ax.bar(x + width/2, grouped_sum[sens_col], width, label=sens_col)

                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.annotate(f'{height:,.0f}',
                                    xy=(bar.get_x() + bar.get_width()/2, height/2),
                                    xytext=(0, 3),
                                    textcoords='offset points',
                                    ha='center', va='bottom', fontsize=9, rotation=90)

                ax.set_xlabel('Cambium_gea region')
                ax.set_ylabel('Total Emissions [tons/yr]')
                col = base_col.replace('_base', '')
                ax.set_title(f'Total {col} Emissions by Cambium Regions')
                ax.set_xticks(x)
                ax.set_xticklabels(grouped_sum['cambium_ge_sens'], rotation=45, ha='right')
                ax.legend()

        # Single region summary plot
        if not grouped_combined.empty:
            base_vals = [grouped_combined[f'{p}_base'].iloc[0] for p in self.NEI_cols]
            sens_vals = [grouped_combined[f'{p}_sens'].iloc[0] for p in self.NEI_cols]

            x = np.arange(len(self.NEI_cols))
            width = 0.35
            fig, ax = plt.subplots(figsize=(10, 6))
            bars1 = ax.bar(x - width/2, base_vals, width, label='Base')
            bars2 = ax.bar(x + width/2, sens_vals, width, label='Sens')

            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:,.0f}',
                                xy=(bar.get_x() + bar.get_width()/2, height),
                                xytext=(0, 3),
                                textcoords='offset points',
                                ha='center', va='bottom', fontsize=9)

            ax.set_xlabel('Pollutant')
            ax.set_ylabel('Total Emissions [tons/yr]')
            ax.set_title(f'Total Emissions by Pollutant (Base vs Sens) at {emis_region}')
            ax.set_xticks(x)
            ax.set_xticklabels(self.NEI_cols, rotation=45, ha='right')
            ax.legend()

        plt.tight_layout()
        fig_path = os.path.join(outputdir, f'Total_Difference_{emis_region}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {fig_path}")

    def run_datacenter_emissions_plots(self, config):
        """Run DataCenter emissions plotting for all configured regions."""

        for scenario in config["target_scenarios"]:

            scenario = str(scenario)
            # Example placeholder logic (replace with your mapping steps)
            print(f"Processing scenario: {scenario}")


            base_emis_dir = os.path.join(config['output']['output_dir'], f"{scenario}_base")
            base_emis_file = os.path.join(base_emis_dir, f"{scenario}_base.shp")
            gdf_base_emis = gpd.read_file(base_emis_file)
            gdf_base_emis.reset_index(drop=True, inplace=True)

            sens_emis_dir = os.path.join(config['output']['output_dir'], f"{scenario}")
            sens_file = os.path.join(sens_emis_dir, f"{scenario}.shp")
            gdf_sens_emis = gpd.read_file(sens_file)
            gdf_sens_emis.reset_index(drop=True, inplace=True)

            gdf_base_emis['EIS_ID'] = gdf_base_emis['EIS_ID'].astype(int)
            gdf_sens_emis['EIS_ID'] = gdf_sens_emis['EIS_ID'].astype(int)

            # Print rows where EIS_ID is null in gdf_base_emis
            null_rows_base = gdf_base_emis[gdf_base_emis['EIS_ID'].isnull()]
            print("Null EIS_ID in gdf_base_emis:")
            print(null_rows_base)

            # Print rows where EIS_ID is null in gdf_compare
            null_rows_compare = gdf_sens_emis[gdf_sens_emis['EIS_ID'].isnull()]
            print("Null EIS_ID in gdf_sens_emis:")
            print(null_rows_compare)

            print(gdf_base_emis['EIS_ID'].dtype)
            print(gdf_sens_emis['EIS_ID'].dtype)

            key_cols = ["EIS_ID", "SCC", "rel_point_", "source_fil", "was_mapped"]
            compare_cols = gdf_sens_emis.columns.difference(key_cols + ["geometry"])

            merged = gdf_base_emis[key_cols + list(compare_cols)].merge(
                gdf_sens_emis[key_cols + list(compare_cols)],
                on=key_cols,
                suffixes=("_base", "_sens")
            )

            diff_mask = (merged[[f"{col}_base" for col in compare_cols]].values !=
                        merged[[f"{col}_sens" for col in compare_cols]].values).any(axis=1)

            diff_combined = merged[diff_mask].merge(
                gdf_base_emis[key_cols + ["geometry"]],
                on=key_cols,
                how="left"
            )

            diff_combined = gpd.GeoDataFrame(diff_combined, geometry="geometry")

            scenario_output_dir = os.path.join(config['output']['plots_dir'],f"{scenario}")
            os.makedirs(scenario_output_dir, exist_ok=True)

            self.plot_diff_emis(scenario_output_dir, diff_combined, scenario)
            
            if config['subregional_scenarios']: 
                for emis_region in config['subregional_scenarios']:

                    emis_region = str(emis_region)
                    sens_emis_dir = os.path.join(config['output']['output_dir'], f"{scenario}_{emis_region}")
                    sens_file = os.path.join(sens_emis_dir , f"{scenario}_{emis_region}.shp")
                    gdf_sens_emis = gpd.read_file(sens_file)
                    gdf_sens_emis.reset_index(drop=True, inplace=True)

                    key_cols = ["EIS_ID", "SCC", "rel_point_", "source_fil", "was_mapped"]
                    compare_cols = gdf_sens_emis.columns.difference(key_cols + ["geometry"])

                    merged = gdf_base_emis[key_cols + list(compare_cols)].merge(
                        gdf_sens_emis[key_cols + list(compare_cols)],
                        on=key_cols,
                        suffixes=("_base", "_sens")
                    )

                    diff_mask = (merged[[f"{col}_base" for col in compare_cols]].values !=
                                merged[[f"{col}_sens" for col in compare_cols]].values).any(axis=1)

                    diff_combined = merged[diff_mask].merge(
                        gdf_base_emis[key_cols + ["geometry"]],
                        on=key_cols,
                        how="left"
                    )

                    diff_combined = gpd.GeoDataFrame(diff_combined, geometry="geometry")

                    scenario_output_dir = os.path.join(config['output']['plots_dir'],f"{scenario}_{emis_region}")
                    os.makedirs(scenario_output_dir, exist_ok=True)

                    self.plot_diff_emis(scenario_output_dir, diff_combined, f"{scenario}_{emis_region}")

    def process_datacenter(self, config):

        os.makedirs(config['output']['output_dir'], exist_ok=True)
        os.makedirs(config['output']['plots_dir'], exist_ok=True)

        # Load NEI base data
        nei_all_pt = self.load_nei(config['combined_nei_file'])

        for scenario in config["target_scenarios"]:
            scenario = str(scenario)
            # Example placeholder logic (replace with your mapping steps)
            print(f"Processing scenario: {scenario}")

            dc_file = os.path.join(config['input']['datacenter_csv_dir'], f"300MW_national_{scenario}.csv")
            egrid = pd.read_csv(dc_file)
            egrid = self.reformat_datacenter(egrid)

            nei_with_dc = nei_all_pt[nei_all_pt["EIS_ID"].isin(egrid["EIS_ID"])].copy()
            nei_with_dc.drop(columns=["height", "diam", "temp", "velocity"], errors="ignore", inplace=True)
            unique_cols = self.find_minimal_unique_identifier_columns(nei_with_dc) or ["EIS_ID"]

            nei_with_dc = nei_with_dc.merge(egrid, on="EIS_ID", how="left")

            case_dfs = []  # store mapped dfs for plotting
            for is_base in [False, True]:
                nei_final = self.mapping_datacenter_to_nei(nei_with_dc, nei_all_pt, unique_cols, is_base)

                mapped_df = nei_final[nei_final["was_mapped"] == True].copy()
                unmapped_df = nei_final[nei_final["was_mapped"] != True].copy()

                # suffix handling
                if is_base:
                    suffix = "_base"
                else:
                    suffix = ""   # no suffix for final case

                final_output_dir = os.path.join(config['output']['output_dir'], f"{scenario}{suffix}")
                os.makedirs(final_output_dir, exist_ok=True)

                if not mapped_df.empty:
                    mapped_file = os.path.join(final_output_dir, f"{scenario}{suffix}.shp")
                    mapped_df.to_file(mapped_file, driver="ESRI Shapefile")
                    print(f"Saved {mapped_file}")

                if not unmapped_df.empty:
                    rest_file = os.path.join(final_output_dir, f"{scenario}{suffix}_rest_NEI.shp")
                    unmapped_df.to_file(rest_file, driver="ESRI Shapefile")
                    print(f"Saved {rest_file}")

                mapped_df["case"] = "base" if is_base else "final"
                case_dfs.append(mapped_df)

                # Combine into one DataFrame
                plot_df = pd.concat(case_dfs, ignore_index=True)

            # diagnostic plot for each scenario
            if config['output']["plots_diagnostics"]:
                final_plots_dir = os.path.join(config['output']['plots_dir'], f"{scenario}")
                os.makedirs(final_plots_dir, exist_ok=True)
                self.plot_diagnostics(plot_df, scenario, final_plots_dir)

    def process_subregional_emis(self, config):

        # if subregional_scenarios are not empty, process the subregional_scenarios
        if config['subregional_scenarios']: 

            for scenario in config["target_scenarios"]:

                scenario = str(scenario)
                
                emission_summary = []

                # Example placeholder logic (replace with your mapping steps)
                print(f"Processing scenario: {scenario}")

                # Read base emissions

                base_emis_dir = os.path.join(config['output']['output_dir'], f"{scenario}_base")
                base_emis =os.path.join(base_emis_dir, f"{scenario}_base.shp")
                gdf_base = gpd.read_file(base_emis)
                gdf_base.reset_index(drop=True, inplace=True)

                # Read final emissions
                final_emis_dir = os.path.join(config['output']['output_dir'], f"{scenario}")
                final_emis =os.path.join(final_emis_dir, f"{scenario}.shp")
                gdf_final = gpd.read_file(final_emis)
                gdf_final.reset_index(drop=True, inplace=True)

                print("debug base", base_emis)
                print(gdf_base.head())

                print("debug final", final_emis)
                print(gdf_final.head())



                for subset_region in config['subregional_scenarios']:

                    subset_region = str(subset_region)

                    print("subset is happening for ", subset_region)

                    print("subset_region =", subset_region, type(subset_region))
                    print(gdf_final['cambium_ge'].unique())

                    base_subset = gdf_base[gdf_base['cambium_ge'] != subset_region]
                    final_subset = gdf_final[gdf_final['cambium_ge'] == subset_region]

                    # Compare two sums of emissions
                    base_region = gdf_base[gdf_base['cambium_ge'] == subset_region]
                    summary_entry = {
                        'Region': subset_region,
                    }

                    for pol in self.NEI_cols:
                        summary_entry[f'{pol}_tons_base'] = base_region[pol].sum()
                        summary_entry[f'{pol}_tons_final'] = final_subset[pol].sum()

                    emission_summary.append(summary_entry)
                            
                    combined_gdf = pd.concat([base_subset, final_subset], ignore_index=True)

                    print("# of rows must be same: ", gdf_base.shape, gdf_final.shape, combined_gdf.shape)

                    if combined_gdf.shape[0] == base_subset.shape[0] + final_subset.shape[0]: 
                        print (f"GOOD : # of row by {subset_region} is {final_subset.shape[0]}")
                    else:
                        print (f"BAD : {subset_region} doesn't result in same total rows {base_subset.shape[0]} {final_subset.shape[0]}  {combined_gdf.shape[0]}")

                    final_output_dir = os.path.join(config['output']['output_dir'], f"{scenario}_{subset_region}")
                    os.makedirs(final_output_dir, exist_ok=True)
                    filename = os.path.join(final_output_dir, f"{scenario}_{subset_region}.shp")
                    combined_gdf.to_file(filename, driver='ESRI Shapefile')
                    print(f"Saved {subset_region} emissions to {filename}")

                # Convert to DataFrame
                summary_df = pd.DataFrame(emission_summary)

                # Save to CSV
                final_output_dir = os.path.join(config['output']['output_dir'], f"{scenario}")
                output_csv = os.path.join(final_output_dir, f'{scenario}_emission_summary_by_region.csv')
                summary_df.to_csv(output_csv, index=False)



