import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import io

import plotly.graph_objects as go
from plotly.subplots import make_subplots

today = date.today()
import time


def show_scenario():

    #===============================================================================================================================
    # Set Up
    #===============================================================================================================================
    months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'] 
    # planning_years = [2023, 2024, 2025, 2026]
    params_file_loc = 'data/'

    spend_prefix = "M_P_"  
    inc_prefix = "TIncT_P_" 
    cpt_prefix = "CPT_P_"
    mcpt_prefix = "MCPT_P_" 
    minc_prefix = "nMTIncT_P_"

    # Model versions
    # ********************************************************************
    model_versions = pd.read_csv(params_file_loc + 'model_versions.csv')

    # Media label mapping
    # ********************************************************************
    media_mapping = pd.read_csv(params_file_loc + 'media_label.csv')
    media_mapping = media_mapping.set_index('media_code').to_dict()['media_label']

    # Time structure
    # ********************************************************************
    time_ref  = pd.read_csv('data/DT_snowflake.csv') 
    time_ref = time_ref[[
        'FIS_WK_END_DT', 'FIS_YR_NB', 'FIS_MO_NB', 'FIS_QTR_NB', 'FIS_WK_NB', 
    ]].drop_duplicates() 
    time_ref = time_ref[time_ref['FIS_YR_NB'] >= 2023]
    time_ref['FIS_WK_END_DT'] = pd.to_datetime(time_ref['FIS_WK_END_DT']).dt.date
    time_ref = time_ref.sort_values(by=['FIS_WK_END_DT']) 


    counting_months = time_ref.groupby(['FIS_YR_NB', 'FIS_MO_NB']).size().reset_index(name='weeks_count') 
    counting_months['lag_weeks'] = counting_months.groupby(['FIS_YR_NB'])['weeks_count'].cumsum().shift(fill_value=0) 
    counting_months.loc[counting_months.FIS_MO_NB == 1, 'lag_weeks'] = 0

    time_ref = time_ref.merge(counting_months, how = 'left', on = ['FIS_YR_NB', 'FIS_MO_NB']).drop_duplicates() 

    counting_months2 = time_ref[['FIS_YR_NB', 'FIS_MO_NB']].drop_duplicates()
    counting_months2 = counting_months2.reset_index(drop=True)
    counting_months2['lag_months'] = counting_months2.index   

    time_ref = time_ref.merge(counting_months2, how = 'left', on = ['FIS_YR_NB', 'FIS_MO_NB']).drop_duplicates() 
    df_time = time_ref.reset_index()




    #===============================================================================================================================
    # Scenario Functions
    #===============================================================================================================================
    def compute_reward_X(X, spend_data, planning_year, threshold):
        """
        Compute the rewards over a 104-week period for a given media type and calculate
        marginal increments and costs, with an updated multiplier calculation based on a threshold.

        Args:
            X (str): The media type (e.g., "X1").
            spend_data (pd.DataFrame): DataFrame containing spending data with columns:
                - 'FIS_MO_NB': Fiscal month number.
                - Media spending columns (e.g., 'X1', 'X2', 'X3').
            planning_year (int): The fiscal year of the current spending plan.
            threshold (float): A value between 0 and 100. Months with spending percentage
                greater than threshold/100 will be counted towards the multiplier.

        Returns:
            tuple: A tuple containing:
                - reward_df (pd.DataFrame): DataFrame of shape (104, 13) with the reward curves.
                - minc_X (float): Total marginal rewards for media X over 12 months.
                - mc_X (float): Total marginal costs for media X over 12 months.
        """
        nonlocal df_curve, df_params, df_time

        # ================================================================================================================
        # Step 1: Extract the spending array and compute the multiplier
        # ================================================================================================================
        spend_array = spend_data[X].values  # Spending for media X over 12 months
        # Compute total spending S_total
        S_total = spend_array.sum()

        # Handle the case when total spending is zero to avoid division by zero
        if S_total == 0:
            spend_percentages = np.zeros_like(spend_array)
        else:
            # Compute the percentage of total spend per month
            spend_percentages = spend_array / S_total

        # Compute the multiplier based on the threshold
        multiplier = np.sum(spend_percentages > (threshold / 100))

        # If multiplier is zero, set it to 1 to avoid division by zero later
        if multiplier == 0:
            multiplier = 1

        # ================================================================================================================
        # Step 2: Compute the timing stuff
        # ================================================================================================================
        # Prepare a list to hold the 104-length arrays for each month
        monthly_arrays = []

        # Get fiscal years and months from spend_data
        fiscal_months = spend_data['FIS_MO_NB'].values

        # Filter df_time for the relevant fiscal year
        df_time_filtered = df_time[df_time['FIS_YR_NB'] == planning_year]

        # Group by fiscal month and count the number of weeks in each month
        weeks_in_month = df_time_filtered.groupby('FIS_MO_NB')['FIS_WK_NB'].nunique()
        weeks_in_month = weeks_in_month.reindex(range(1, 13), fill_value=0)

        # Compute cumulative weeks to determine the starting week for each month
        cumulative_weeks = weeks_in_month.cumsum()
        start_weeks = [0] + cumulative_weeks.values.tolist()[:-1]

        # Initialize total marginal increments and costs
        minc_X = 0.0
        mc_X = 0.0

        # ================================================================================================================
        # Loop over each month to compute the rewards
        # ================================================================================================================
        for idx, S_mo in enumerate(spend_array):
            fiscal_month = fiscal_months[idx]

            if S_mo == 0:
                # If the spending is zero, append an array of zeros
                monthly_array = np.zeros(104)
                monthly_arrays.append(monthly_array)
                continue

            # Step 2: Calculate annualized spending and find the closest match in df_params
            S_yr = S_mo * multiplier
            M_P_X_col = spend_prefix + X
            TIncT_P_X_col = inc_prefix + X
            nMTIncT_P_X_col = minc_prefix + X  # Marginal rewards column
            Pct_Delta_col = 'Pct_Delta'   

            # Find the index of the closest spending value in df_params
            idx_closest = (np.abs(df_params[M_P_X_col] - S_yr)).idxmin()

            # Retrieve values from df_params
            reward_yr = df_params.at[idx_closest, TIncT_P_X_col]
            reward_mo = reward_yr / multiplier

            # Calculate minc_mo and mc_mo for marginal cost per reward
            minc_mo = df_params.at[idx_closest, nMTIncT_P_X_col]
            mc_mo = df_params.at[idx_closest, Pct_Delta_col] * S_yr


            # Accumulate total marginal increments and costs
            minc_X += minc_mo
            mc_X += mc_mo

            # Step 3: Generate the reward curve for the month
            timing_curve = df_curve[X].values  # Timing curve of length 52
            monthly_reward_curve = reward_mo * timing_curve


            # Step 4: Create a 104-length array with appropriate leading zeros
            start_week = start_weeks[fiscal_month - 1]
            leading_zeros = int(start_week)
            trailing_zeros = 104 - leading_zeros - 52
            if trailing_zeros < 0:
                # If there is an overlap beyond 104 weeks, truncate the array
                monthly_array = np.concatenate([
                    np.zeros(leading_zeros),
                    monthly_reward_curve[:104 - leading_zeros]
                ])
            else:
                monthly_array = np.concatenate([
                    np.zeros(leading_zeros),
                    monthly_reward_curve,
                    np.zeros(trailing_zeros)
                ])
            monthly_arrays.append(monthly_array)
            

        # Aggregate the monthly arrays into a final reward curve
        monthly_arrays = [arr[:104] for arr in monthly_arrays]  # Ensure each array is at most 104 elements
        # Pad any shorter arrays to exactly 104 elements
        monthly_arrays = [np.pad(arr, (0, 104 - len(arr)), 'constant') if len(arr) < 104 else arr for arr in monthly_arrays]
        reward_X = np.sum(monthly_arrays, axis=0)

        # Construct the output DataFrame
        columns = [f'Month_{i+1}' for i in range(12)] + ['aggregated']
        data = np.column_stack(monthly_arrays + [reward_X])
        reward_df = pd.DataFrame(data, columns=columns)

        return reward_df, minc_X, mc_X
    


    def compute_plan_reward(spend_data, planning_year, lead_years, lag_years, threshold):
        """
        Compute the reward curves for all medias over an extended time frame, broken down by month and aggregated.

        Args:
            spend_data (pd.DataFrame): DataFrame containing spending data with columns:
                - 'FIS_MO_NB': Fiscal month number.
                - Media spending columns (e.g., 'X1', 'X2', 'X3').
            lead_years (int): Number of years to add as leading zeros.
            lag_years (int): Number of years to add as trailing zeros.2
            planning_year (int): The fiscal year of the current spending plan.
            threshold (float): Threshold value for reward calculations.

        Returns:
            list: A list of 13 numpy arrays:
                - First 12 arrays represent the monthly rewards summed across all media
                - Last array represents the aggregated rewards summed across all media
                Each array includes leading and trailing zeros.
        """
        # Ensure global access to necessary dataframes
        # global df_curve, df_params, df_time

        # Step 1: Get the list of media columns from spend_data
        medias = spend_data.columns.tolist()[1:]

        # Initialize lists to store the monthly and aggregated reward arrays
        monthly_rewards = [[] for _ in range(12)]  # One list for each month
        aggregated_rewards = []  # List for aggregated rewards

        # Loop over each media to compute its reward curves
        for media in medias:
            # Extract spending data for the media
            media_spend_data = spend_data[['FIS_MO_NB', media]]

            # Call compute_reward_X for the current media
            reward_df = compute_reward_X(media, media_spend_data, planning_year, threshold)[0]

            # Extract monthly columns and aggregated column
            for month in range(12):
                month_col = f'Month_{month+1}'
                monthly_rewards[month].append(reward_df[month_col].values)
            
            # Extract and store aggregated rewards
            aggregated_rewards.append(reward_df['aggregated'].values)

        # Step 2: Sum the reward arrays across all media for each month and aggregated
        summed_monthly_rewards = [np.sum(month_arrays, axis=0) for month_arrays in monthly_rewards]
        summed_aggregated_rewards = np.sum(aggregated_rewards, axis=0)

        # Step 3: Add leading and trailing zeros to each array
        leading_zeros = np.zeros(lead_years * 52)
        trailing_zeros = np.zeros(lag_years * 52)

        # Create final list of extended arrays (12 monthly + 1 aggregated)
        extended_reward_curves = []
        
        # Process monthly arrays
        for monthly_reward in summed_monthly_rewards:
            extended_monthly = np.concatenate([leading_zeros, monthly_reward, trailing_zeros])
            extended_reward_curves.append(extended_monthly)
        
        # Process aggregated array
        extended_aggregated = np.concatenate([leading_zeros, summed_aggregated_rewards, trailing_zeros])
        extended_reward_curves.append(extended_aggregated)

        return extended_reward_curves




    def plan_forecast_craft(spend_data, planning_year, lead_years, lag_years, cutoff):
        """
        Generate monthly and quarterly forecast tables based on spending data and reward calculations.

        Args:
            spend_data (pd.DataFrame): DataFrame containing spending data with columns:
                - 'FIS_MO_NB': Fiscal month number.
                - Media spending columns (e.g., 'NTV', 'ING', 'STR').
            planning_year (int): The fiscal year of the current spending plan.
            lead_years (int): Number of years to add as leading zeros.
            lag_years (int): Number of years to add as trailing zeros.
            cutoff (float): Threshold value for reward calculations.

        Returns:
            tuple: A tuple containing:
                - craft_mo (pd.DataFrame): DataFrame with monthly forecast data.
                - craft_qtr (pd.DataFrame): DataFrame with quarterly forecast data.
        """
        df_time_scenario = df_time[df_time['FIS_YR_NB'].between(planning_year - lead_years, planning_year + lag_years)]
        weekly_table = df_time_scenario[['FIS_WK_END_DT', 'FIS_YR_NB', 'FIS_QTR_NB', 'FIS_MO_NB']]
        results = compute_plan_reward(spend_data, planning_year, lead_years, lag_years, cutoff)

        names = []
        for i in range(len(results)-1):
            serl = list(results[i])
            if len(serl) < weekly_table.shape[0]:
                serl = serl + [0] * (weekly_table.shape[0] - len(serl))
            if len(serl) > weekly_table.shape[0]:
                serl = serl[:weekly_table.shape[0]]
            col_name = str(planning_year) + ' ' + months[i]
            names.append(col_name)
            weekly_table[col_name] = serl

        # Monthly Table
        # ================================================================================================================
        monthly_table = weekly_table.groupby(['FIS_YR_NB', 'FIS_MO_NB'])[names].sum().reset_index()
        rewards = monthly_table.iloc[:, 2:].values.T
        monthly_table.FIS_MO_NB.replace(dict(zip(range(1, 13), months)), inplace=True) 
        monthly_table['timeline'] = monthly_table.FIS_YR_NB.astype(str) + " " + monthly_table.FIS_MO_NB.astype(str) 
        
        shard1 = pd.DataFrame({'Spending Month': names, "Spend": spend_data.iloc[:, 1:].sum(axis = 1).values})
        shard2 = pd.DataFrame(rewards)
        shard2.columns = monthly_table['timeline'].values
        craft_mo = pd.concat([shard1, shard2], axis=1) 


        # Monthly Table
        # ================================================================================================================
        quarter_table = weekly_table.groupby(['FIS_YR_NB', 'FIS_QTR_NB'])[names].sum().reset_index()
        rewards = quarter_table.iloc[:, 2:].values.T 
        rewards = rewards.reshape(4, 3, -1) # Turning monthly tracking into quarterly tracking
        rewards = rewards.sum(axis = 1)
        quarter_table.FIS_QTR_NB = quarter_table.FIS_QTR_NB.astype(str)
        quarter_table['timeline'] = quarter_table.FIS_YR_NB.astype(str) + " Q" + quarter_table.FIS_QTR_NB.astype(str)

        names = [str(planning_year) + " Q" + str(x) for x in range(1, 5)]
        shard1 = pd.DataFrame({'Spending Quarter': names, 
                            "Spend": spend_data.iloc[:, 1:].values.sum(axis = 1).reshape(4, 3).sum(axis = 1)})
        shard2 = pd.DataFrame(rewards)
        shard2.columns = quarter_table['timeline'].values
        craft_qtr = pd.concat([shard1, shard2], axis=1) 

        return craft_mo, craft_qtr
    

    def forecast_table_summarizer(table):
        shard1 = ['Total Attendance', ""]
        shard2 = list(table.iloc[:, 2:].sum(axis = 0).values)

        table.loc[-1] = np.array(shard1 + shard2)
        table = table.reset_index(drop = True)

        shard1 = table.iloc[:-1]
        shard2 = table.iloc[[-1]]

        table2 = pd.concat([shard2, shard1], axis = 0).reset_index(drop = True)
        return table2




    def build_plan_summary(spend_data, planning_year, threshold, unit_revenue):
        """
        Build a summary report of the spending plan with total spend, total reward,
        cost per reward, and marginal cost per reward for each media and aggregated.

        Args:
            spend_data (pd.DataFrame): DataFrame containing spending data with columns:
                - 'FIS_MO_NB': Fiscal month number.
                - Media spending columns (e.g., 'YOT', 'FAB').
            planning_year (int): The fiscal year of the current spending plan.

        Returns:
            pd.DataFrame: A DataFrame 'plan_summary' with columns:
                - 'Total Spend'
                - 'Total Reward'
                - 'Cost per Reward'
                - 'Marginal Cost per Reward'
            and rows:
                - 'Total' (aggregated over all medias)
                - One row per media (e.g., 'YOT', 'FAB')
        """
        # Ensure nonlocal access to necessary dataframes
        nonlocal df_curve, df_params, df_time

        # Initialize dictionaries to store summary data
        total_spend_dict = {}
        total_reward_dict = {}
        minc_X_dict = {}
        mc_X_dict = {}

        # List of media columns (exclude 'FIS_MO_NB')
        medias = spend_data.columns.tolist()[1:]

        # Loop over each media to compute summary metrics
        for media in medias:
            # Extract spending data for the media
            media_spend_data = spend_data[['FIS_MO_NB', media]]


            # Total spend for media: sum of monthly spend
            total_spend = media_spend_data[media].sum()
            total_spend_dict[media] = total_spend


            # Call compute_reward_X for the current media
            reward_df, minc_X, mc_X = compute_reward_X(media, media_spend_data, planning_year, threshold)

            # Total reward for media: sum of the 'aggregated' column
            total_reward = reward_df['aggregated'].sum()
            total_reward_dict[media] = total_reward

            # Store minc_X and mc_X
            minc_X_dict[media] = minc_X
            mc_X_dict[media] = mc_X


        # Calculate aggregated totals
        total_spend_agg = sum(total_spend_dict.values())
        total_reward_agg = sum(total_reward_dict.values())
        minc_X_agg = sum(minc_X_dict.values())
        mc_X_agg = sum(mc_X_dict.values())


        # Build the summary DataFrame
        plan_summary = pd.DataFrame(columns=['Total Spend', 'Total Attendance', 'Cost per Attendance', 'Marginal Cost per Attendance'])

        # Add aggregated totals to the summary
        plan_summary.loc['Total'] = {
            'Total Spend': total_spend_agg,
            'Total Attendance': total_reward_agg,
            'Cost per Attendance': total_spend_agg / total_reward_agg if total_reward_agg != 0 else np.nan,
            'Marginal Cost per Attendance': mc_X_agg / minc_X_agg if minc_X_agg != 0 else np.nan
        }

        # Add individual media data to the summary
        for media in medias:
            total_spend = total_spend_dict[media]
            total_reward = total_reward_dict[media]
            minc_X = minc_X_dict[media]
            mc_X = mc_X_dict[media]

            plan_summary.loc[media] = {
                'Total Spend': total_spend,
                'Total Attendance': total_reward,
                'Cost per Attendance': total_spend / total_reward if total_reward != 0 else np.nan,
                'Marginal Cost per Attendance': mc_X / minc_X if minc_X != 0 else np.nan
            }

        # Reorder rows: aggregated totals first, then individual medias
        plan_summary = plan_summary.reset_index().rename(columns={'index': 'Media'})
        plan_summary = plan_summary[['Media', 'Total Spend', 'Total Attendance', 'Cost per Attendance', 'Marginal Cost per Attendance']]

        # Add ROAS and MROAS
        plan_summary['ROAS'] = unit_revenue * plan_summary['Total Attendance'] / plan_summary['Total Spend']
        plan_summary['MROAS'] = 1 + ((unit_revenue - plan_summary['Marginal Cost per Attendance']) / plan_summary['Marginal Cost per Attendance'])


        # Round up columns
        plan_summary['Total Spend'] = plan_summary['Total Spend'].astype(int)
        plan_summary['Total Attendance'] = plan_summary['Total Attendance'].astype(int)
        plan_summary['Cost per Attendance'] = plan_summary['Cost per Attendance'].round(1)
        plan_summary['Marginal Cost per Attendance'] = plan_summary['Marginal Cost per Attendance'].round(1)
        plan_summary['ROAS'] = plan_summary['ROAS'].round(1)
        plan_summary['MROAS'] = plan_summary['MROAS'].round(1)


        return plan_summary
    


    def scenario_plots(scenarios, metrics, channels, colors, title, ylabel1, ylabel2, currency_symbol):
        # **********************************************************************************
        # Read numbers
        # **********************************************************************************
        board1 = scenarios[0]
        board2 = scenarios[1]

        metric1 = metrics[0]
        metric2 = metrics[1]

        metric1_s1 = board1[metric1].values
        metric1_s2 = board2[metric1].values
        max_metric1 =  max(max(metric1_s1), max(metric1_s2))

        metric2_s1 = board1[metric2].values
        metric2_s2 = board2[metric2].values
        max_metric2 = max(max(metric2_s1), max(metric2_s2))

        metric1_compare = [(s2 - s1) / s1 * 100 for s1, s2 in zip(metric1_s1, metric1_s2)]
        metric2_compare = [(s2 - s1) / s1 * 100 for s1, s2 in zip(metric2_s1, metric2_s2)]

        color1_s1 = colors[0]
        color1_s2 = colors[1]
        color2_s1 = colors[2]
        color2_s2 = colors[3]

        default_textpos = 0.5 * max_metric1

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # **********************************************************************************
        # The bar plots
        # **********************************************************************************
        # Scenario 1
        # ....................................................
        fig.add_trace(
            go.Bar(
                x= channels,
                y= metric1_s1,
                name= "1",
                marker_color= color1_s1
            ),
            secondary_y=False,
        )

        # Scenario 2
        # ....................................................
        fig.add_trace(
            go.Bar(
                x= channels,
                y= metric1_s2,
                name= "2",
                marker_color= color1_s2
            ),
            secondary_y=False,
        )




        # **********************************************************************************
        # Line plots
        # **********************************************************************************
        # Scenario 1
        # ....................................................
        fig.add_trace(
            go.Scatter(
                x=channels,
                y= metric2_s1,
                name="MROAS 1",
                line=dict(
                    color= color2_s1, 
                    width=3, dash='dash',
                    shape='spline',  # This creates a smooth curve
                    smoothing=1.3    # Adjust smoothing intensity (0.5-1.5 range works well)
                    ),
                marker=dict(
                    size=10,         # Larger marker size
                    color=color2_s1,
                    line=dict(
                        width=1,
                        color='white'
                    )
                )

            ),
            secondary_y=True,
        )

        # Scenario 2
        # ....................................................
        fig.add_trace(
            go.Scatter(
                x=channels,
                y= metric2_s2,
                name="MROAS 2",
                line=dict(color= color2_s2, 
                        width=3, dash='dash',
                        shape='spline',  # This creates a smooth curve
                        smoothing=1.3    # Adjust smoothing intensity (0.5-1.5 range works well)
                    ),
                marker=dict(
                    size=10,         # Larger marker size
                    color=color2_s2,
                    line=dict(
                        width=1,
                        color='white'
                    )
                )
            ),
            secondary_y=True,
        )




        # **********************************************************************************
        # Costmetics
        # **********************************************************************************

        # Annotations for metric 1 change
        for i, channel in enumerate(channels):
            change = metric1_compare[i]
            
            # Determine color based on change
            color = "green" if change >= 0 else "red"
            
            # Format text with plus/minus sign and percentage
            if change >= 0:
                text = f"+{change:.1f}%"  # Add plus sign for positive changes
            else:
                text = f"{change:.1f}%"   # Negative sign is automatically included
            
            # Improved positioning logic
            current_value = max(metric1_s1[i], metric1_s2[i])
            
            # If the value is very small (less than 5% of the maximum), use a fixed position
            if current_value < 0.03 * max_metric1:
                ypos = 0.12 * max_metric1  # Position at 15% of max height for very small values
            # If it's the maximum value, add a bit more space
            elif current_value >= 0.95 * max_metric1:
                ypos = 1.05 * max_metric1  # Position at 110% of max for the largest values
            # For medium values, position proportionally
            else:
                ypos = current_value + (0.125 * max_metric1)  # Position above the bar with consistent spacing
            
            # Add the annotation without arrows
            fig.add_annotation(
                x=channel,
                y=ypos, 
                text=text,
                showarrow=False,  # No arrow
                font=dict(
                    color=color, 
                    size=14,      # Slightly larger font for better visibility
                    weight='bold' # Make it bold for emphasis
                ),
                align='center',
                bgcolor='rgba(255,255,255,0.7)',  # Semi-transparent white background
                bordercolor=color,
                borderwidth=1,
                borderpad=3
            )




        fig.update_layout(
            # Wider plot for spacing
            width=1300,
            height=700,
            # Extra large left margin
            margin=dict(t=80, r=50, b=100, l=150),
            # Title styling
            title=dict(
                text= title,
                font=dict(
                    size=28,
                    color= color1_s2,
                    weight='bold'
                ),
                x=0.35
            ),
            # Other layout
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom", 
                y=-0.15,
                xanchor="center",
                x=0.5
            ),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        # Set x-axis properties
        fig.update_xaxes(
            title_text="",
            showgrid=False,
            showline=True,
            linewidth=2,
            linecolor='lightgray'
        )

        # Set y-axes properties
        fig.update_yaxes(
            title_text= ylabel1,
            title_font=dict(size=16),
            range=[0, 1.2 * max(max(metric1_s2), max(metric1_s1))],
            showgrid=True,
            gridcolor='lightgray',
            secondary_y=False,
            tickformat=','
        )

        fig.update_yaxes(
            title_text= ylabel2,
            title_font=dict(size=16),
            range=[0, 1.2 * max(max(metric2_s2), max(metric2_s1))],
            showgrid=False,
            secondary_y=True,
            tickprefix= currency_symbol,
            ticksuffix='.0'
        )

        return fig

























    #===============================================================================================================================
    # Page content begins now 
    #===============================================================================================================================
    st.write("")
    st.write("")

    # Reset session keys 
    # ********************************************************************
    if 'scenario_page_loaded' not in st.session_state:
        # Clear password related states on first load of the page
        for key in ['scenario_region_validated', 'scenario_region', 'scenario_region_code']:
            st.session_state.pop(key, None)
        st.session_state['scenario_page_loaded'] = True


    # Refresh other functionalities, requiring password again
    # ********************************************************************
    st.session_state['refresh_minimizer'] = "Yes"
    st.session_state['refresh_maximizer'] = "Yes"

    # Initialize session state variables
    # ********************************************************************
    if 'scenario_computed' not in st.session_state:
        st.session_state['scenario_computed'] = False

    # Initialize password validation state and add a new state for tracking active tab
    if 'scenario_region_validated' not in st.session_state:
        st.session_state['scenario_region_validated'] = "Not Validated"
    if 'scenario_region' not in st.session_state:
        st.session_state['scenario_region'] = ""
    if 'refresh_scenario' not in st.session_state:
        st.session_state['refresh_scenario'] = "No"


    whitespace = 15
    list_tabs = "Input Tab", "Output Tab"
    tab1, tab2 = st.tabs([s.center(whitespace,"\u2001") for s in list_tabs])

    #------------------------------------------------------------------------------------------------------------
    # Input Tab
    #------------------------------------------------------------------------------------------------------------
    # Input Tab
    with tab1:
        # Check if tab has changed and reset validation if needed
        current_tab = "No"
        if st.session_state['refresh_scenario'] != current_tab:
            st.session_state['scenario_region_validated'] = "Not Validated"
            st.session_state['scenario_region'] = ""
            st.session_state['refresh_scenario'] = current_tab

        # Get region code from user
        region_code_scenario = st.text_input("Please enter the region password", key="scenario_password_input")
        st.session_state['scenario_region_code'] = region_code_scenario
        
        # Validate password
        if region_code_scenario:
            # Assuming model_versions is a DataFrame with 'password' and 'scenario_region' columns
            check_scenario = model_versions.loc[model_versions.password == region_code_scenario, 'region'].values
            
            if len(check_scenario) == 1:
                st.session_state['scenario_region_validated'] = "Validated"
                st.session_state['scenario_region'] = check_scenario[0]
            else:
                st.session_state['scenario_region_validated'] = "Not Validated"


        
        # Display messages
        if st.session_state['scenario_region_validated'] == "Not Validated" and region_code_scenario:
            st.error("Please enter the correct region password to proceed")

        elif st.session_state['scenario_region_validated'] == "Validated":


            # Displaying region info, load region input files
            # ********************************************************************
            region = st.session_state['scenario_region']
            file_params = params_file_loc + region + "/input_300pct.csv"
            file_curve = params_file_loc + region + "/input_mediaTiming.csv"
            file_base = params_file_loc + region + "/input_base.csv"

            mmm_year = model_versions.loc[model_versions.region == region, 'update'].values[0]
            adjust_ratio = model_versions.loc[model_versions.region == region, 'adjust'].values[0]
            price = model_versions.loc[model_versions.region == region, 'price'].values[0]
            currency = model_versions.loc[model_versions.region == region, 'currency'].values[0]

            message = f"** {region} scenarios will be based on model results for fiscal year {mmm_year}"
            st.markdown(
                f"<p style='font-size: 6px; color: #4e98ff; font-style: italic;'>{message}</p>",
                unsafe_allow_html=True
            )

            # Set up dataframes 
            # ********************************************************************
            df_base = pd.read_csv(file_base)
            df_base.columns = ['FIS_MO_NB'] + list(media_mapping.keys())
            df_base['FIS_MO_NB'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

            df_curve  = pd.read_csv(file_curve)
            media_list = df_curve.columns.tolist()

            df_params  = pd.read_csv(file_params)
            df_params.columns = [x.replace("TlncT", 'TIncT') for x in df_params.columns]
            names = list(df_params.columns)
            names2 = [s.replace("FABING", "ING") for s in names]
            names2 = [s.replace("DIS_BAN", "BAN") for s in names2]
            names2 = [s.replace("DIS_AFF", "AFF") for s in names2]
            df_params.columns = names2



            # User choosing spending plans
            # ******************************************************************** 
            if 'results_df' not in st.session_state:
                st.session_state['results_df'] = pd.DataFrame() 

            uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

            if uploaded_files:
                #.......................................................................................................
                # User uploading the files
                #.......................................................................................................
                dfs = []
                file_names = []
                for file in uploaded_files:
                    df = pd.read_csv(file)
                    dfs.append(df)
                    file_names.append(file.name) 

                #.......................................................................................................
                # User choosing file for scenario 1 & scenario 2 
                #.......................................................................................................
                st.write("")
                col1, col2 = st.columns(2)

                with col1:
                    scnr1_name = st.selectbox("Scenario 1", file_names)
                    scnr1_index = file_names.index(scnr1_name)
                    scnr1 = dfs[scnr1_index] 
                    st.session_state['scnr1_file'] = file_names[scnr1_index]
                    st.session_state['scnr1_table'] = scnr1  

                    scnr1_summary = []
                    for x in scnr1.columns[1:]:
                        scnr1_summary.append([x, int(np.round(scnr1[x].sum(), 0))]) 
                    scnr1_summary = pd.DataFrame(scnr1_summary, columns=['Media', 'Spending']) 
                    scnr1_summary = pd.DataFrame(scnr1_summary.values.T, columns = scnr1.columns[1:]).fillna(0)
                    scnr1_summary = scnr1_summary.iloc[1:, :]
                    scnr1_summary.rename(columns = media_mapping, inplace=True)
                    scnr1_summary = scnr1_summary.T.reset_index()
                    scnr1_summary.columns = ['Media', 'Annual Spending - Scenario 1']

                    scnr1_revised = scnr1.copy()
                    scnr1_revised.columns = ['FIS_MO_NB'] + list(media_mapping.keys())  
                    scnr1_revised['FIS_MO_NB'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

                with col2:
                    scnr2_name = st.selectbox("Scenario 2", file_names)
                    scnr2_index = file_names.index(scnr2_name)
                    scnr2 = dfs[scnr2_index] 
                    st.session_state['scnr2_file'] = file_names[scnr2_index]
                    st.session_state['scnr2_table'] = scnr2  

                    scnr2_summary = []
                    for x in scnr2.columns[1:]:
                        scnr2_summary.append([x, int(np.round(scnr2[x].sum(), 0))])
                    scnr2_summary = pd.DataFrame(scnr2_summary, columns=['Media', 'Spending'])
                    scnr2_summary = pd.DataFrame(scnr2_summary.values.T, columns = scnr2.columns[1:]).fillna(0)
                    scnr2_summary = scnr2_summary.iloc[1:, :]
                    scnr2_summary.rename(columns = media_mapping, inplace=True)
                    scnr2_summary = scnr2_summary.T.reset_index()
                    scnr2_summary.columns = ['Media', 'Annual Spending - Scenario 2']

                    scnr2_revised = scnr2.copy()
                    scnr2_revised.columns = ['FIS_MO_NB'] + list(media_mapping.keys())
                    scnr2_revised['FIS_MO_NB'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

                    scenarios_summary = scnr1_summary.merge(scnr2_summary, on = 'Media', how = 'left') 
                    scenarios_summary = scenarios_summary.T
                    names = scenarios_summary.iloc[0, :].values
                    scenarios_summary.columns = names
                    scenarios_summary = scenarios_summary.iloc[1:, :]

                container = st.container()
                with container:
                    numRows = scenarios_summary.shape[0]
                    st.dataframe(scenarios_summary, height = (numRows + 1) * 35 + 3)


                inv_map = {v: k for k, v in media_mapping.items()}
                error_medias_scnr1 = [] 
                medias_UB_scnr1 = []
                for x in scenarios_summary.columns:
                    x_code = inv_map[x]
                    spending = scenarios_summary[x].values[0] 
                    spending_UB = df_params.loc[df_params['PCT_Change'] == 300, 'M_P_' + x_code].values[0]
                    if spending > spending_UB:
                        error_medias_scnr1.append(x) 
                        medias_UB_scnr1.append(spending_UB)

                if len(error_medias_scnr1) > 0:
                    st.error("The following medias in Scenario 1 exeeded upper bound for annnual spending, please adjust the spending plans before running the analysis -- ")
                    for i in np.arange(len(error_medias_scnr1)):
                        x = error_medias_scnr1[i]
                        x_ub = np.round(medias_UB_scnr1[i], 0)
                        x_ub = format(x_ub, ",")
                        st.error(x + " exceeded annual upper bound of $" + str(x_ub)) 



                error_medias_scnr2 = [] 
                medias_UB_scnr2 = []
                for x in scenarios_summary.columns:
                    x_code = inv_map[x]
                    spending = scenarios_summary[x].values[1] 
                    spending_UB = df_params.loc[df_params['PCT_Change'] == 300, 'M_P_' + x_code].values[0]
                    if spending > spending_UB:
                        error_medias_scnr2.append(x) 
                        medias_UB_scnr2.append(spending_UB)

                if len(error_medias_scnr2) > 0:
                    st.error("The following media in Scenario 2 exeeded upper bound for annnual spending, please adjust the spending plans before running the analysis -- ")
                    for i in np.arange(len(error_medias_scnr2)):
                        x = error_medias_scnr2[i]
                        x_ub = np.round(medias_UB_scnr2[i], 0)
                        x_ub = format(x_ub, ",")
                        st.error(x + " exceeded annual upper bound of $" + x_ub)



                st.write("")
                st.write("")
                st.write("")

                st.divider()
                if st.button("Run Analysis", help = 'Click on the button to run scenario analysis, this should take 10 - 15 seconds'):
                    if len(error_medias_scnr1) > 0 or len(error_medias_scnr2) > 0:
                        st.error("Please address the spending plan errors above before running the analysis")
                    else:
                        with st.spinner("I'm working on it ..."):

                            # Summary Table
                            # ******************************************************************** 
                            results1 = build_plan_summary(scnr1_revised, 2024, adjust_ratio, price)
                            # results1.iloc[:, 1:] = results1.iloc[:, 1:].round(1)
                            st.session_state['results1'] = results1

                            results2 = build_plan_summary(scnr2_revised, 2024, adjust_ratio, price)
                            # results2.iloc[:, 1:] = results2.iloc[:, 1:].round(1)
                            st.session_state['results2'] = results2 


                            # Timing Results - Monthly
                            # ********************************************************************
                            mo_base = plan_forecast_craft(df_base, mmm_year, 0, 2, adjust_ratio)[0]
                            
                            mo_scnr1 = plan_forecast_craft(scnr1_revised, mmm_year + 1, 1, 1, adjust_ratio)[0]
                            forecast_craft_mo_s1 = pd.concat([mo_base, mo_scnr1], axis = 0)
                            forecast_craft_mo_s1.iloc[:, 1:] = forecast_craft_mo_s1.iloc[:, 1:].round(1)
                            forecast_craft_mo_s1 = forecast_table_summarizer(forecast_craft_mo_s1)


                            mo_scnr2 = plan_forecast_craft(scnr2_revised, mmm_year + 1, 1, 1, adjust_ratio)[0]
                            forecast_craft_mo_s2 = pd.concat([mo_base, mo_scnr1], axis = 0)
                            forecast_craft_mo_s2.iloc[:, 1:] = forecast_craft_mo_s2.iloc[:, 1:].round(1)
                            forecast_craft_mo_s2 = forecast_table_summarizer(forecast_craft_mo_s2)

                            st.session_state['forecast_crafts_mo'] = [forecast_craft_mo_s1, forecast_craft_mo_s2]

                            # Timing Results - Quarterly
                            # ********************************************************************
                            qtr_base = plan_forecast_craft(df_base, mmm_year, 0, 2, adjust_ratio)[1]
                            
                            qtr_scnr1 = plan_forecast_craft(scnr1_revised, mmm_year + 1, 1, 1, adjust_ratio)[1]
                            forecast_craft_qtr_s1 = pd.concat([qtr_base, qtr_scnr1], axis = 0)
                            forecast_craft_qtr_s1.iloc[:, 1:] = forecast_craft_qtr_s1.iloc[:, 1:].round(1)
                            forecast_craft_qtr_s1 = forecast_table_summarizer(forecast_craft_qtr_s1)


                            qtr_scnr2 = plan_forecast_craft(scnr2_revised, mmm_year + 1, 1, 1, adjust_ratio)[1]
                            forecast_craft_qtr_s2 = pd.concat([qtr_base, qtr_scnr1], axis = 0)
                            forecast_craft_qtr_s2.iloc[:, 1:] = forecast_craft_qtr_s2.iloc[:, 1:].round(1)
                            forecast_craft_qtr_s2 = forecast_table_summarizer(forecast_craft_qtr_s2)

                            st.session_state['forecast_crafts_qtr'] = [forecast_craft_qtr_s1, forecast_craft_qtr_s2]

                        st.session_state["scenario_computed"] = True
                        st.success("Success! Please check the results in the following tabs 👉")



    #------------------------------------------------------------------------------------------------------------
    # Output Tab
    #-----------------------------------------------------------------------------------------------------------
    with tab2:
        scenario_status = st.session_state['scenario_computed']
        
        if scenario_status == False:
            st.write("Please upload scenario files and run the analysis in the first tab")


        else:
            # ********************************************************************************************************************
            # Preparing Result Tables
            # ********************************************************************************************************************
            results1 = st.session_state['results1']
            results2 = st.session_state['results2']
            forecast_crafts_mo = st.session_state['forecast_crafts_mo']
            forecast_crafts_qtr = st.session_state['forecast_crafts_qtr']





            st.write("")
            st.write("")
            viewing = st.selectbox("Select result format to view", ['Media Summary', 'Increment Forecast'])


            if viewing == 'Media Summary':


                # Summary Tables
                # *****************************************************************************************************
                col1, col2 = st.columns(2)
            
                with col1:
                    st.write("")
                    st.markdown("## Scenario 1")
                    container = st.container()
                    table = results1
                    table['Media'] = table['Media'].replace(media_mapping).fillna(table['Media'])
                    with container:
                        numRows = table.shape[0]
                        st.dataframe(table, height = (numRows + 1) * 35 + 3, hide_index=True)


                with col2:
                    st.write("")
                    st.markdown("## Scenario 2")
                    container = st.container()
                    table = results2
                    table['Media'] = table['Media'].replace(media_mapping).fillna(table['Media'])
                    with container:
                        numRows = table.shape[0]
                        st.dataframe(table, height = (numRows + 1) * 35 + 3, hide_index=True)


                # Plots
                # *****************************************************************************************************

                fig = scenario_plots(
                    scenarios = [results1.iloc[1:, :], results2.iloc[1:, :]],

                    metrics = ['Total Spend', 'MROAS'],

                    channels = list(media_mapping.values()),

                    colors = ['rgb(174, 139, 113)', 
                            'rgb(140, 63, 12)',
                            'rgb(174, 139, 113)',
                            'rgb(140, 63, 12)'
                            
                            ],

                    title = "Media budget & MROAS variation per touchpoint", 

                    ylabel1 = "", ylabel2= "", currency_symbol = currency 
                )
                st.plotly_chart(fig)


                fig = scenario_plots(
                    scenarios = [results1.iloc[1:, :], results2.iloc[1:, :]],

                    metrics = ['Total Attendance', 'Cost per Attendance'],

                    channels = list(media_mapping.values()),

                    colors = ['rgb(188, 214, 150)', 
                            'rgb(36, 84, 40)',
                            'rgb(188, 214, 150)', 
                            'rgb(36, 84, 40)'
                            ],

                    title = "Incremental attendance & CPA evolution", 

                    ylabel1 = "", ylabel2= "", currency_symbol = currency 
                )
                st.plotly_chart(fig)




            if viewing == 'Increment Forecast':
                col1, col2 = st.columns(2)
                with col1:
                    scenario = st.radio("", ['Scenario 1', 'Scenario 2'])
                with col2:
                    timeframe  = st.radio("", ['Monthly', 'Quarterly'])


                if (scenario == 'Scenario 1') & (timeframe == 'Monthly'):
                    table = forecast_crafts_mo[0]
                    table.iloc[:, 2:] = table.iloc[:, 2:].astype(float).astype(int)
                    container = st.container()
                    with container:
                        numRows = table.shape[0]
                        st.dataframe(table, height = (numRows + 1) * 35 + 3, hide_index=True)

                if (scenario == 'Scenario 1') & (timeframe == 'Quarterly'):
                    table = forecast_crafts_qtr[0]
                    table.iloc[:, 2:] = table.iloc[:, 2:].astype(float).astype(int)
                    container = st.container()
                    with container:
                        numRows = table.shape[0]
                        st.dataframe(table, height = (numRows + 1) * 35 + 3, hide_index=True)

                if (scenario == 'Scenario 2') & (timeframe == 'Monthly'):
                    table = forecast_crafts_mo[1]
                    table.iloc[:, 2:] = table.iloc[:, 2:].astype(float).astype(int)
                    container = st.container()
                    with container:
                        numRows = table.shape[0]
                        st.dataframe(table, height = (numRows + 1) * 35 + 3, hide_index=True)

                if (scenario == 'Scenario 2') & (timeframe == 'Quarterly'):
                    table = forecast_crafts_qtr[1]
                    table.iloc[:, 2:] = table.iloc[:, 2:].astype(float).astype(int)
                    container = st.container()
                    with container:
                        numRows = table.shape[0]
                        st.dataframe(table, height = (numRows + 1) * 35 + 3, hide_index=True)



        # st.session_state["scenario_computed"] = False