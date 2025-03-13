import streamlit as st
import pandas as pd
import numpy as np
import io

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import time
from datetime import date
today = date.today()


def show_maximization():

    #===============================================================================================================================
    # Set Up
    #===============================================================================================================================
    months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'] 
    months_full = [
        "October",
        "November",
        "December",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        ]
    planning_years = [2023, 2024, 2025, 2026]
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
    media_mapping_file = pd.read_csv(params_file_loc + 'media_label.csv')

    # Time structure
    # ********************************************************************
    time_ref  = pd.read_csv('data/DT_snowflake.csv') 
    time_ref = time_ref[[
        'FIS_WK_END_DT', 'FIS_YR_NB', 'FIS_MO_NB', 'FIS_QTR_NB', 'FIS_WK_NB', 
    ]].drop_duplicates() 
    time_ref = time_ref[time_ref['FIS_YR_NB'].isin(planning_years)]
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
        # Ensure nonlocal access to necessary dataframes
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
        multiplier = np.sum(spend_percentages >= (threshold / 100))

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
        global df_curve, df_params, df_time

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
        weekly_table = df_time[['FIS_WK_END_DT', 'FIS_YR_NB', 'FIS_QTR_NB', 'FIS_MO_NB']]
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
        board1['Media'] = channels
        board2 = scenarios[1]
        board2['Media'] = channels

        metric1 = metrics[0]
        metric2 = metrics[1]

        # Drop medias with no spending
        # ******************************************************************************
        drops = []
        set1 = set(board1.loc[board1[metric1] == 0, 'Media'].values) 
        set2 = set(board2.loc[board2[metric1] == 0, 'Media'].values)
        drops.append(list(set1.intersection(set2)))
        board1 = board1[~board1['Media'].isin(drops[0])]
        board2 = board2[~board2['Media'].isin(drops[0])] 
        channels = [x for x in channels if x not in drops[0]]
        print(channels)

        # Gather metrics 
        # ******************************************************************************
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













    # ================================================================================================================================
    # Optimizer Function
    # ================================================================================================================================ 
    def budget_maximizer(
            spend_plan,   # dataframe of month and media spend
            planning_months, # fiscal month number
            planning_year, # fiscal year,
            base_year, # spend dataframe of MMM modeling year
            df_bounds, # dataframe of media bounds 
            cutoff # media percentage cutoff for determining the multiplier 
            ):
        
        # Utility function to convert an spend array into spend dataframe
        # ================================================================================================================
        def revise_spend(spend_plan, newSpend_array):
            spend_plan_revised = spend_plan.copy()
            spend_plan_revised.iloc[:, 1:] = spend_plan.iloc[:, 1:]
            for i, month in enumerate(planning_months):
                spend_plan_revised.loc[spend_plan_revised['FIS_MO_NB'] == month, spend_plan.columns[1:]] = newSpend_array[i*len(spend_plan.columns[1:]):(i+1)*len(spend_plan.columns[1:])]
            return spend_plan_revised
        
        # Utility function to compute the reward for a given spend plan in planning months
        # ================================================================================================================
        def current_reward_calculator(plan, planning_year, planning_months, planning_weeks, cutoff):
            plan_rewards = compute_plan_reward(plan, planning_year, 1, 0, cutoff)
            current_reward = 0 
            for x in planning_months:
                current_reward += plan_rewards[x-1][planning_weeks].sum() 
            return current_reward
        
        # Step 0: Filter df_time and get planning_weeks
        # ================================================================================================================
        df_time_filtered = df_time[df_time['FIS_YR_NB'].isin([planning_year, planning_year - 1])].reset_index(drop=True)
        planning_weeks = df_time_filtered[
            (df_time_filtered['FIS_YR_NB'] == planning_year) & 
            (df_time_filtered['FIS_MO_NB'].isin(planning_months))
        ].index.tolist()


        # Step 1: Compute reward_debit
        # ================================================================================================================
        reward_credit_year0 = compute_plan_reward(base_year, planning_year - 1, 0, 1, cutoff)[-1][planning_weeks].sum()
        reward_credit_year1 = 0
        if min(planning_months) > 1:
            plan_rewards = compute_plan_reward(spend_plan, planning_year, 1, 0, cutoff) 
            for i in np.arange(min(planning_months) - 1):
                print(i)
                reward_credit_year1 += plan_rewards[i][planning_weeks].sum() 

        reward_credit = reward_credit_year0 + reward_credit_year1

        
        # Step 2: Set spendings in the spending month to 0
        # ================================================================================================================
        spend_plan_wiped = spend_plan.copy()
        spend_plan_wiped.loc[spend_plan_wiped['FIS_MO_NB'].isin(planning_months), spend_plan_wiped.columns[1:]] = 0



        
        # Step 3: For each {media, month}, compute the potential gain percentgae (due to media timing)
        # ================================================================================================================
        result_data = []
        for month in planning_months:
            # Compute available weeks
            available_weeks = df_time_filtered[
                (df_time_filtered['FIS_YR_NB'] == planning_year) & 
                (df_time_filtered['FIS_MO_NB'].between(month, planning_months[-1]))
            ].shape[0]
            # Compute cumulative indices for each media
            cumulative_indices = df_curve.iloc[:available_weeks, :].sum()
            result_data.append(cumulative_indices.tolist())
        cumu_timing_df = pd.DataFrame(result_data, index=planning_months)
        cumu_timing_df.columns = media_list


        # Step 4: Flatten media timing index, media names, and media spend
        # ================================================================================================================
        array_cumu_timing = cumu_timing_df.values.flatten()
        array_media_entries = np.tile(media_list, len(planning_months))
        spend0 = []
        for month in planning_months:
            spend0.extend(spend_plan[spend_plan['FIS_MO_NB'] == month].iloc[:, 1:].values.flatten())
        spend0 = np.array(spend0)

        # Step 5: Form the lower bound array and make the array of multipliers at {month, media} level
        # ================================================================================================================
        array_lb = []
        for spend, media in zip(spend0, array_media_entries):
            lb_pct = df_bounds.loc[df_bounds['Media'] == media, 'LB'].values[0]
            ub_pct = df_bounds.loc[df_bounds['Media'] == media, 'UB'].values[0]
            
            array_lb.append(spend * lb_pct)
        array_lb = np.array(array_lb)

        spend_lb = spend_plan_wiped.copy()
        for i, month in enumerate(planning_months):
            spend_lb.loc[spend_lb['FIS_MO_NB'] == month, spend_plan.columns[1:]] = array_lb[i*len(spend_plan.columns[1:]):(i+1)*len(spend_plan.columns[1:])]


        multipliers = [] 
        for x in media_list:
            x_sum = spend_lb[x].values.sum()

            multipliers.append(len([x for x in spend_lb[x].values if x/x_sum >= (cutoff / 100)]))

        multipliers = pd.DataFrame({
            'Media': media_list,
            'LB': multipliers
        })
        multipliers = np.tile(multipliers.LB.values, len(planning_months))


        # Step 6: Create the reward panel
        # ================================================================================================================
        reward_panels = []
        for i, (media_entry, cumu_timing) in enumerate(zip(array_media_entries, array_cumu_timing)):
            multiplier = multipliers[i]

            # Define variable names
            spend_varname = spend_prefix + media_entry
            inc_varname = inc_prefix + media_entry
            cpt_varname = cpt_prefix + media_entry

            # Create reward_panel for this entry
            reward_panel = df_params[['PCT_Change', spend_varname, inc_varname]].copy()

            # Adjust spend and inc columns
            reward_panel[spend_varname] /= multiplier
            reward_panel[inc_varname] = (reward_panel[inc_varname] / multiplier) * cumu_timing

            # Create CPT column
            reward_panel[cpt_varname] = reward_panel[spend_varname] / reward_panel[inc_varname]

            # Rename columns
            reward_panel.columns = ['pct', f'S_{i}', f'R_{i}', f'CPT_{i}']
            
            reward_panels.append(reward_panel)
            
        # Combine all reward panels and rename to df_params_monthly
        df_params_monthly = pd.concat(reward_panels, axis=1)

        # Keep only one 'pct' column
        df_params_monthly = df_params_monthly.loc[:, ~df_params_monthly.columns.duplicated()]
        df_params_monthly = df_params_monthly.fillna(0)


        # Step 7 : form the upperbound array
        # ================================================================================================================
        array_ub = []
        array_300 = []
        for i, media in enumerate(array_media_entries):
            max_spend = df_params_monthly.loc[df_params_monthly['pct'] == 300, f'S_{i}'].values[0]
            array_300.append(max_spend)
        array_300 = np.array(array_300)
        for spend, media in zip(spend0, array_media_entries):
            ub_pct = df_bounds.loc[df_bounds['Media'] == media, 'UB'].values[0]
            array_ub.append(min(spend * ub_pct, array_300[len(array_ub)]))
        array_ub = np.array(array_ub)


        # Step 8: Initialize the iteration trackers
        # ================================================================================================================
        current_budget = np.sum(array_lb)
        remaining_budget = np.sum(spend0) - np.sum(array_lb)
        print(f"Benchmark Budget: {np.sum(spend0)}")
        print(f"Current Budget: {current_budget}") 
        print(f"Remaining Budget: {remaining_budget}")

        # Start with the scenario where each media is at their lowest boundary
        spend1 = array_lb.copy()

        # Create allocation_rank dataframe
        allocation_rank = pd.DataFrame({
            'entry': range(len(spend1)),
            'current_CPT': np.zeros(len(spend1)),
            'updates' : np.zeros(len(spend1))
        })

        for i, (spend, media) in enumerate(zip(spend1, array_media_entries)):
            # Find the closest spend value in df_params_monthly
            closest_row = df_params_monthly[f'S_{i}'].sub(spend).abs().idxmin()
            # Get the corresponding reward and CPT values
            cpt = df_params_monthly.loc[closest_row, f'CPT_{i}']
            # Update current_reward and allocation_rank
            allocation_rank.loc[i, 'current_CPT'] = cpt


        # Sort allocation_rank
        allocation_rank = allocation_rank.sort_values('current_CPT').reset_index(drop=True)
        entries_at_upper_bound = list(allocation_rank.loc[allocation_rank.current_CPT == 0, 'entry'].values)
        allocation_rank = allocation_rank[allocation_rank['current_CPT'] > 0]
        allocation_rank.current_CPT = allocation_rank.current_CPT.round(0)


        # Function for one iterative step
        # ================================================================================================================
        def update_entry(entry, target_cpt, spend1, allocation_rank, remaining_budget):
            i = allocation_rank.loc[entry, 'entry']
            # Skip if this entry has already hit upper bound
            if i in entries_at_upper_bound:
                print(f"Skipping Entry {i} ({array_media_entries[i]}) as it has already reached upper bound")
                return spend1, remaining_budget  # Return both values


            print("Allocating to entry", i, "Media", array_media_entries[i])
            current_spend = spend1[i]
            old_cpt = allocation_rank.loc[entry, 'current_CPT']
            
            # Find the row with target CPT
            target_row = df_params_monthly[f'CPT_{i}'].sub(target_cpt).abs().idxmin()
            next_row = np.minimum(target_row + 1, df_params_monthly.shape[0] - 1)
            
            new_spend = df_params_monthly.loc[next_row, f'S_{i}']

            # Check if new_spend exceeds the upper bound
            if new_spend >= array_ub[i]:
                print(f"Cannot allocate to ${new_spend}, adjusting to upper bound ${array_ub[i]}") 
                new_spend = array_ub[i]
                closest_row = df_params_monthly[f'S_{i}'].sub(new_spend).abs().idxmin()
                new_cpt = df_params_monthly.loc[closest_row, f'CPT_{i}']
                entries_at_upper_bound.append(i)
                print(f"Entry {i} ({array_media_entries[i]}) has reached its upper bound and will not be allocated any more budget.")
            else:
                new_cpt = df_params_monthly.loc[next_row, f'CPT_{i}']

            spend_increase = new_spend - current_spend
            
            if spend_increase > remaining_budget:
                # Insufficient remaining budget
                new_spend = current_spend + remaining_budget
                closest_row = df_params_monthly[f'S_{i}'].sub(new_spend).abs().idxmin()
                new_reward = df_params_monthly.loc[closest_row, f'R_{i}']
                new_cpt = df_params_monthly.loc[closest_row, f'CPT_{i}']
                allocated_budget = remaining_budget
                remaining_budget = 0
            else:
                allocated_budget = spend_increase
                remaining_budget -= spend_increase

            spend1[i] = new_spend
            if spend1[i] == array_ub[i]:
                entries_at_upper_bound.append(i)
                print(f"Entry {i} ({array_media_entries[i]}) has reached its upper bound and will not be allocated any more budget.")

            allocation_rank.loc[entry, 'current_CPT'] = new_cpt
            allocation_rank.loc[entry, 'updates'] += 1

            # Find the media with the target CPT
            target_entry = allocation_rank[allocation_rank['current_CPT'] == target_cpt].iloc[0].name
            target_media_index = allocation_rank.loc[target_entry, 'entry']
            target_media = array_media_entries[target_media_index]

            print(f"Media: {array_media_entries[i]} (Entry {i})" )
            print(f"Target CPT: {target_cpt:.4f} (from {target_media})")
            print(f"Spend: {current_spend:.2f} >>> {new_spend:.2f}  (allocated {spend_increase:.2f})")
            print(f"CPT: {old_cpt:.2f} >>> {new_cpt:.2f}")
            print(f"Remaining budget: {remaining_budget:.2f}")
            print("--------------------")

            return spend1, allocation_rank, remaining_budget
        

        # Main optimization loop
        # ================================================================================================================
        iteration = 1
        while remaining_budget > 0:
            print(f"\nIteration {iteration}:")
            allocation_rank['current_CPT'] = allocation_rank['current_CPT'].round(0)
            allocation_rank = allocation_rank.sort_values('current_CPT')

            # Instead of removing entries at upper bound, create a filtered view for allocation decisions
            remaining_entries = allocation_rank[~allocation_rank.entry.isin(entries_at_upper_bound)]
            lowest_cpt = remaining_entries['current_CPT'].min() if not remaining_entries.empty else 0
            lowest_cpt_entries = remaining_entries[remaining_entries['current_CPT'] == lowest_cpt]

            # Deciding running case:
            special_case = 0
            if len(remaining_entries) == 0:
                special_case = 1
            if remaining_entries['current_CPT'].nunique() == 1 and len(remaining_entries) > 1:
                special_case = 2
            if len(remaining_entries) == 1:
                special_case = 3
            if len(lowest_cpt_entries) > 1 and remaining_entries['current_CPT'].nunique() > 1:
                special_case = 4

            # Special Case 1: All entries have reached their upper bounds
            # .............................................................................................................
            if special_case == 1:
                print("Running Special Case 1")
                print("All entries have reached their upper bounds. Ending allocation process.")
                break

            # Special Case 2: All entries have the same CPT
            # .............................................................................................................
            if special_case == 2:
                print("Running Special Case 2")
                print("All remaining entries have the same CPT. Allocating fixed amount to entry with most room.")

                # Calculate remaining capacity for each entry
                remaining_capacity = []
                for idx in remaining_entries.index:
                    i = allocation_rank.loc[idx, 'entry']
                    capacity = array_ub[i] - spend1[i]
                    remaining_capacity.append({
                        'entry_idx': idx,
                        'entry': i,
                        'capacity': capacity
                    })

                # Sort by remaining capacity (descending)
                remaining_capacity = sorted(remaining_capacity, key=lambda x: x['capacity'], reverse=True)

                # Select entry with most capacity
                selected = remaining_capacity[0]
                entry = selected['entry_idx']
                i = selected['entry'] 
                print(f"Selected {array_media_entries[i]} (Entry {i}) with remaining capacity of ${selected['capacity']:.2f}")

                # Cap allocation at 3000 or distance to upper bound
                distance_to_upper = array_ub[i] - spend1[i]
                max_allocation = min(3000, distance_to_upper, remaining_budget)
                print(f"Allocating {max_allocation:.2f} to {array_media_entries[i]} (Entry {i})")

                # Update spend and lookup new reward/CPT
                current_spend = spend1[i]
                spend1[i] += max_allocation
                remaining_budget -= max_allocation

                # Find closest row in df_params_monthly for the new spend level
                closest_row = df_params_monthly[f'S_{i}'].sub(spend1[i]).abs().idxmin()
                new_cpt = df_params_monthly.loc[closest_row, f'CPT_{i}']

                # Update allocation_rank
                allocation_rank.loc[entry, 'current_CPT'] = new_cpt
                allocation_rank.loc[entry, 'updates'] += 1

                if spend1[i] >= array_ub[i]:
                    entries_at_upper_bound.append(i)
                    print(f"Entry {i} ({array_media_entries[i]}) has reached its upper bound and will not be allocated any more budget.")

            # Special case 3: Single entry remaining
            # ...........................................................................................................
            elif special_case == 3:
                print("Running special case 3")
                entry = remaining_entries.index[0]
                i = remaining_entries.loc[entry, 'entry']
                current_spend = spend1[i]
                distance_to_upper = array_ub[i] - current_spend
                
                while remaining_budget > 0 and spend1[i] < array_ub[i]:
                    quarter_budget = min(remaining_budget / 4, distance_to_upper / 4)
                    if quarter_budget < 250:  # Break if allocation becomes too small
                        quarter_budget = remaining_budget
                        
                    spend1[i] += quarter_budget
                    remaining_budget -= quarter_budget
                    distance_to_upper = array_ub[i] - spend1[i]
                    
                    if spend1[i] >= array_ub[i]:
                        entries_at_upper_bound.append(i)
                        break

                # Update CPT after all allocations
                closest_row = df_params_monthly[f'S_{i}'].sub(spend1[i]).abs().idxmin()
                new_cpt = df_params_monthly.loc[closest_row, f'CPT_{i}']
                allocation_rank.loc[entry, 'current_CPT'] = new_cpt
                allocation_rank.loc[entry, 'updates'] += 1

            # Special case 4: Multiple entries with the same lowest CPT
            # ...........................................................................................................
            # Check for multiple entries with the same lowest CPT
            if special_case == 4:
                print("Running Special Case 4")
                next_cpt = allocation_rank[allocation_rank['current_CPT'] > lowest_cpt]['current_CPT'].min()
                print(f"Multiple entries with lowest CPT {lowest_cpt:.4f}. Updating to next CPT {next_cpt:.4f}")
                
                # Calculate required spend for each entry
                spend_requirements = []
                for entry in lowest_cpt_entries.index:
                    i = allocation_rank.loc[entry, 'entry']
                    current_spend = spend1[i]
                    
                    # Find required spend for target CPT
                    target_row = df_params_monthly[f'CPT_{i}'].sub(next_cpt).abs().idxmin()
                    next_row = np.minimum(target_row + 1, df_params_monthly.shape[0] - 1)
                    required_spend = df_params_monthly.loc[next_row, f'S_{i}'] - current_spend
                    
                    spend_requirements.append({
                        'entry': entry,
                        'required_spend': required_spend
                    })

                # Sort by required spend
                spend_requirements = sorted(spend_requirements, key=lambda x: x['required_spend'])
                print(spend_requirements)
                
                # Process entries in order of required spend
                for req in spend_requirements:
                    results = update_entry(req['entry'], next_cpt, spend1, allocation_rank, remaining_budget)
                    spend1 = results[0]
                    allocation_rank = results[1]
                    remaining_budget = results[2]
                    if remaining_budget <= 0: 
                        print("Stopping allocation as goal has been reached or budget depleted.")
                        break


            # Normal Case: Update the entry with lowest CPT
            # ...........................................................................................................
            if special_case == 0:
                # Normal case: update the entry with the lowest CPT
                print("Running Normal Case")
                entry = remaining_entries.index[0]
                # Look up next CPT from full allocation_rank, not just remaining_entries
                entry_cpt = allocation_rank.loc[entry, 'current_CPT']
                next_cpt = allocation_rank[allocation_rank['current_CPT'] > entry_cpt]['current_CPT'].min()
                results = update_entry(entry, next_cpt, spend1, allocation_rank, remaining_budget)
                spend1 = results[0]
                allocation_rank = results[1]
                remaining_budget = results[2]

            iteration += 1
            print("")
            print("")
            print("")

        # ================================================================================================================
        # Wrapping up the results
        # ================================================================================================================

        # Create the final spend plan
        final_spend_plan = revise_spend(spend_plan, spend1)

        # Compute total reward
        earned_reward_original = current_reward_calculator(spend_plan, planning_year, planning_months, planning_weeks, cutoff)
        earned_reward_optimized = current_reward_calculator(final_spend_plan, planning_year, planning_months, planning_weeks, cutoff)

        original_rewards = np.round(earned_reward_original + reward_credit).astype(int)
        optimized_rewards = np.round(earned_reward_optimized + reward_credit).astype(int)
        rewards = [original_rewards, optimized_rewards]

        return final_spend_plan, rewards
    

    def optimization_summary(plan0, plan1, planning_year, rewards, cutoff, unit_revenue):

        # Table 1 - Aggregate Summary
        # ****************************************************************************************************************
        contents = ['Total Spend', 'Total Reward', 'Total Reward in Planning Period', 'Cost per Reward', 'Marginal Cost per Reward']
        summary0 = build_plan_summary(plan0, planning_year, cutoff, unit_revenue).drop(columns = ['ROAS', 'MROAS'])
        summary1 = build_plan_summary(plan1, planning_year, cutoff, unit_revenue).drop(columns = ['ROAS', 'MROAS'])

        total_original = summary0.iloc[0, 1:].values.astype(int).tolist()
        planning_original = rewards[0]
        collect_original = total_original[:2] + [planning_original] + total_original[2:]

        total_optimized = summary1.iloc[0, 1:].values.astype(int).tolist()
        planning_optimized = rewards[1]
        collect_optimized = total_optimized[:2] + [planning_optimized] + total_optimized[2:]

        table1 = pd.DataFrame({
            'Contents': contents,
            'Original': collect_original,
            'Optimized': collect_optimized
        })

        table1['Change'] = table1['Optimized'] - table1['Original']
        table1['Change (%)'] = np.round((table1.Optimized / table1.Original - 1) * 100, 1)


        # Table 2 - Media Level Summary
        # ****************************************************************************************************************
        values = []
        for x in media_list:
            collect0 = np.round(summary0[summary0.Media == x].iloc[:, 1:].values, 1)[0]
            spend0 = collect0[0]
            reward0_total = collect0[1]
            reward0_plan = np.round(compute_reward_X(x, plan0, planning_year, cutoff)[0]['aggregated'].values[:52].sum(),1) 
            cpa0 = collect0[2]
            mcpa0 = collect0[3]
            values.append([x, 'original', spend0, reward0_total, reward0_plan, cpa0, mcpa0])

            collect1 = np.round(summary1[summary1.Media == x].iloc[:, 1:].values, 1)[0]
            spend1 = collect1[0]
            reward1_total = collect1[1]
            reward1_plan = np.round(compute_reward_X(x, plan1, planning_year, cutoff)[0]['aggregated'].values[:52].sum(),1)
            cpa1 = collect1[2]
            mcpa1 = collect1[3]
            values.append([x, 'optimized', spend1, reward1_total, reward1_plan, cpa1, mcpa1])

        table2 = pd.DataFrame(values, columns=['Media', 'Version', 'Total Spend', 'Total Reward', 'Total Reward in Planning Period', 'Cost per Reward', 'Marginal Cost per Reward'])
        table2.iloc[:, 2:] = np.round(table2.iloc[:, 2:], 1)
        
        craft = pd.DataFrame(table2['Media'])
        for x in table2.columns[2:]:
            df = table2[['Media', 'Version', x]]
            hierarchical_df = df.set_index(['Media', 'Version']).unstack(level=1)
            hierarchical_df.columns = [f'{col[0]} ({col[1]})' for col in hierarchical_df.columns]
            hierarchical_df = hierarchical_df.reset_index()
            hierarchical_df[x + ' Change(%)'] = 100 * (hierarchical_df[x + ' (optimized)'] / hierarchical_df[x + ' (original)'] - 1)
            hierarchical_df[x + ' Change(%)'] = np.round(hierarchical_df[x + ' Change(%)'], 1)
            craft = craft.merge(hierarchical_df, how = 'left', on = "Media").drop_duplicates()
        table2 = craft

        return [plan1, table1, table2]
    


 


















    #===============================================================================================================================
    # Page content begins now 
    #===============================================================================================================================
    st.write("")
    st.write("")

    # Reset session keys 
    # ********************************************************************
    if 'maximizer_page_loaded' not in st.session_state:
        for key in ['maximizer_region_validated', 'maximizer_region', 'maximizer_region_code']:
            st.session_state.pop(key, None)
        st.session_state['maximizer_page_loaded'] = True
    
    # Refresh other functionalities
    # ********************************************************************
    st.session_state['refresh_scenario'] = "Yes"
    st.session_state['refresh_minimizer'] = "Yes"
    st.session_state["scenario_computed"] = False


    # Initialize session state variables
    # ********************************************************************
    if 'maximizer_done' not in st.session_state:
        st.session_state['maximizer_done'] = False

    # Initialize password validation state and add a new state for tracking active tab
    if 'maximizer_region_validated' not in st.session_state:
        st.session_state['maximizer_region_validated'] = "Not Validated"
    if 'maximizer_region' not in st.session_state:
        st.session_state['maximizer_region'] = ""
    if 'refresh_maximizer' not in st.session_state:
        st.session_state['refresh_maximizer'] = "No"


    # User inputs
    # ********************************************************************
    whitespace = 15
    list_tabs = "Input Tab", "Output Tab"
    tab1, tab2 = st.tabs([s.center(whitespace,"\u2001") for s in list_tabs])

    #------------------------------------------------------------------------------------------------------------
    # Input Tab
    #------------------------------------------------------------------------------------------------------------
    with tab1:
        # Check if tab has changed and reset validation if needed
        current_tab = "No"
        if st.session_state['refresh_maximizer'] != current_tab:
            st.session_state['maximizer_region_validated'] = "Not Validated"
            st.session_state['maximizer_region'] = ""
            st.session_state['refresh_maximizer'] = current_tab

        # Get region code from user
        region_code_maximizer = st.text_input("Please enter the region password", key="maximizer_password_input")
        st.session_state['maximizer_region_code'] = region_code_maximizer
        
        # Validate password
        if region_code_maximizer:
            # Assuming model_versions is a DataFrame with 'password' and 'maximizer_region' columns
            check_maximizer = model_versions.loc[model_versions.password == region_code_maximizer, 'region'].values
            
            if len(check_maximizer) == 1:
                st.session_state['maximizer_region_validated'] = "Validated"
                st.session_state['maximizer_region'] = check_maximizer[0]
            else:
                st.session_state['maximizer_region_validated'] = "Not Validated"


        # Display messages
        if st.session_state['maximizer_region_validated'] == "Not Validated" and region_code_maximizer:
            st.error("Please enter the correct region password to proceed")

        elif st.session_state['maximizer_region_validated'] == "Validated":
            # User input 1: select region >>> prepare parameters
            # ******************************************************************** 
            region = st.session_state['maximizer_region']
            file_params = params_file_loc + region + "/input_300pct.csv"
            file_curve = params_file_loc + region + "/input_mediaTiming.csv"
            file_base = params_file_loc + region + "/input_base.csv"

            media_mapping = media_mapping_file[media_mapping_file.region == region].set_index('media_code').to_dict()['media_label']
            media_mapping_inverse = {value: key for key, value in media_mapping.items()}
            mmm_year = model_versions.loc[model_versions.region == region, 'update'].values[0]
            adjust_ratio = model_versions.loc[model_versions.region == region, 'adjust'].values[0]
            price = model_versions.loc[model_versions.region == region, 'price'].values[0]
            currency = model_versions.loc[model_versions.region == region, 'currency'].values[0]

            message = f"** {region} scenarios will be based on model results for fiscal year {mmm_year}"
            st.markdown(
                f"<p style='font-size: 6px; color: #4e98ff; font-style: italic;'>{message}</p>",
                unsafe_allow_html=True
            )

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


            col1, col2 = st.columns(2)
            # User input 2: Choosing planning month range
            # ********************************************************************
            def split_months_indices(months, start_month, end_month):
                # Find the indices of the start and end months
                start_index = months.index(start_month)
                end_index = months.index(end_month)
                
                # Handle the case where the start month comes after the end month in the list
                if start_index <= end_index:
                    # Get the indices of the months between the start and end months (inclusive)
                    indices_between = list(range(start_index, end_index + 1))
                    # Get the indices of the rest of the months
                    indices_rest = list(range(start_index)) + list(range(end_index + 1, len(months)))
                else:
                    # Get the indices of the months between the start and end months (inclusive), considering the wrap-around
                    indices_between = list(range(start_index, len(months))) + list(range(end_index + 1))
                    # Get the indices of the rest of the months
                    indices_rest = list(range(end_index + 1, start_index))
                
                # Convert 0-based indices to 1-based indices
                indices_between = [index + 1 for index in indices_between]
                indices_rest = [index + 1 for index in indices_rest]
                
                return indices_between, indices_rest
            

            with col1:
                start_month, end_month = st.select_slider(
                    "Planning Period",
                    options= months_full,
                    value=("October", "September"),
                )
                planning_months, non_planning_months = split_months_indices(months_full, start_month, end_month)

            with col2:
                st.write("")



            # User input 3: choosing baseline spending, using MMM year as default
            # ******************************************************************************************** 
            st.write("")  
            st.write("Please enter the initial spending plan")
            spend_blueprint = df_base.T.iloc[1:,:].reset_index()
            spend_blueprint.columns = [['Media'] + months_full]
            for x in spend_blueprint.columns[1:]:
                spend_blueprint[x] = spend_blueprint[x].astype(float).round(0)
            spend_blueprint['Media'] = spend_blueprint['Media'].replace(media_mapping)


            container = st.container()
            with container:
                num_rows = spend_blueprint.shape[0]
                spend_plan = st.data_editor(
                    spend_blueprint,
                    height = (num_rows + 1) * 35 + 3, 
                    disabled = ['Media'],     
                    hide_index = True                 
                ) 

            spend_plan['Media'] = spend_plan['Media'].replace(media_mapping_inverse)
            spend_plan = spend_plan.T
            spend_plan.columns = spend_plan.iloc[0]
            spend_plan = spend_plan.iloc[1:].reset_index()
            spend_plan.rename(columns= {'index' : 'FIS_MO_NB'}, inplace = True)
            spend_plan['FIS_MO_NB'] = spend_plan['FIS_MO_NB'].apply(lambda x: months_full.index(x) + 1) 

            
            # User input 4: Choosing media bounds
            # ********************************************************************************************
            st.write("") 
            st.write("Please choose adjusted bounds for each media channel as percentage.")
            df_bounds = pd.DataFrame({
                'Media' : media_list,
                'Lower Bound (%)' : 80,
                'Upper Bound (%)' : 120
            })
            df_bounds['Media'] = df_bounds['Media'].replace(media_mapping)
            df_bounds_user = df_bounds.T.iloc[1:, :]
            df_bounds_user.columns = df_bounds.Media.values
            df_bounds_user

            container = st.container()
            with container:
                num_rows = df_bounds_user.shape[0]
                df_bounds_user = st.data_editor(
                    df_bounds_user,
                    height = (num_rows + 1) * 35 + 3,                
                ) 

            df_bounds_coded = df_bounds_user.rename(columns = media_mapping_inverse)
            df_bounds_coded = df_bounds_coded.T.reset_index()
            df_bounds_coded.columns = ['Media', 'LB', 'UB']
            for x in df_bounds_coded.columns[1:]:
                df_bounds_coded[x] = df_bounds_coded[x].astype(float) / 100

            # Error Catching
            # *********************************************************************************************
            # Skip for now



            # Runnning the optimizer
            # *********************************************************************************************
            if st.button("Let's begin!"):
                with st.spinner("I'm working on it ..."):
                    crafts = budget_maximizer(spend_plan, 
                                            planning_months, 
                                            planning_years[1], 
                                            df_base,
                                            df_bounds_coded, 
                                            adjust_ratio)
                    result_package = optimization_summary(
                        spend_plan,
                        crafts[0],
                        2024,
                        crafts[1],
                        adjust_ratio,
                        price
                    )

                    result_package.append(build_plan_summary(spend_plan, 2024, adjust_ratio, price))
                    result_package.append(build_plan_summary(result_package[0], 2024, adjust_ratio, price))

                    """
                    Summary of result package items: 
                    [0] - Optimized Spend
                    [1] - Aggregate Summary
                    [2] - Summary by Media
                    [3] - Summary for original spend plan
                    [4] - Summary for optimized spend plan
                    """

                    st.success("Optimization performed successfully! Please check the results in the output tab 👉")
                    st.session_state['maximizer_results'] = result_package
                    st.session_state["maximizer_done"] = True



    with tab2:
        scenario_status = st.session_state['maximizer_done']
        if scenario_status:
            result_package = st.session_state['maximizer_results']
            st.write("")
            st.write("")
            viewing = st.selectbox("Select result format to view",['Optimized Spend','Aggregate Summary','Summary by Media'])

            if viewing == "Optimized Spend":
                optimized_spend = result_package[0]
                for x in optimized_spend.columns[1:]:
                    optimized_spend[x] = optimized_spend[x].astype(float).round(1)
                nrows = optimized_spend.shape[0]
                optimized_spend_showing = optimized_spend.rename(columns = media_mapping)

                container = st.container()
                with container:
                    st.dataframe(optimized_spend_showing, 
                                height = (nrows + 1) * 35 + 3, 
                                hide_index=True)
                    

                # Plots
                # *****************************************************************************************************
                summary0 = result_package[3]
                summary1 = result_package[4]
                fig = scenario_plots(
                    scenarios = [summary0.iloc[1:, :], summary1.iloc[1:, :]],

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
                    scenarios = [summary0.iloc[1:, :], summary1.iloc[1:, :]],

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



                    
            if viewing == 'Aggregate Summary':
                summary = result_package[1]

                # Some post-processing
                # ********************************************************************************************
                summary = summary[summary.Contents != 'Total Reward'] # Hide total (12-month) reward

                nrows = summary.shape[0]
                
                container = st.container()
                with container:
                    st.dataframe(summary, 
                                height = (nrows + 1) * 35 + 3, 
                                hide_index=True)


            if viewing == 'Summary by Media':
                summary = result_package[2]
                nrows = summary.shape[0]
                
                container = st.container()
                with container:
                    st.dataframe(summary, 
                                height = (nrows + 1) * 35 + 3, 
                                hide_index=True)



        else:
            st.write("Please complete the steps in the Input Tab and run optimization")