import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import io

today = date.today()


def show_scenario():

    #===============================================================================================================================
    # Set Up
    #===============================================================================================================================
    months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'] 
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
    media_mapping = pd.read_csv(params_file_loc + 'media_label.csv')
    media_mapping = media_mapping.set_index('media_code').to_dict()['media_label']

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




    def build_plan_summary(spend_data, planning_year, threshold):
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
        plan_summary = pd.DataFrame(columns=['Total Spend', 'Total Reward', 'Cost per Reward', 'Marginal Cost per Reward'])

        # Add aggregated totals to the summary
        plan_summary.loc['Total'] = {
            'Total Spend': total_spend_agg,
            'Total Reward': total_reward_agg,
            'Cost per Reward': total_spend_agg / total_reward_agg if total_reward_agg != 0 else np.nan,
            'Marginal Cost per Reward': mc_X_agg / minc_X_agg if minc_X_agg != 0 else np.nan
        }

        # Add individual media data to the summary
        for media in medias:
            total_spend = total_spend_dict[media]
            total_reward = total_reward_dict[media]
            minc_X = minc_X_dict[media]
            mc_X = mc_X_dict[media]

            plan_summary.loc[media] = {
                'Total Spend': total_spend,
                'Total Reward': total_reward,
                'Cost per Reward': total_spend / total_reward if total_reward != 0 else np.nan,
                'Marginal Cost per Reward': mc_X / minc_X if minc_X != 0 else np.nan
            }

        # Reorder rows: aggregated totals first, then individual medias
        plan_summary = plan_summary.reset_index().rename(columns={'index': 'Media'})
        plan_summary = plan_summary[['Media', 'Total Spend', 'Total Reward', 'Cost per Reward', 'Marginal Cost per Reward']]


        return plan_summary
    




























    #===============================================================================================================================
    # Page content begins now 
    #===============================================================================================================================
    st.write("")
    st.write("")

    # Initialize session state variables
    # ********************************************************************
    if 'scenario_computed' not in st.session_state:
                st.session_state['scenario_computed'] = False


    # User inputs
    # ********************************************************************
    whitespace = 15
    list_tabs = "Input Tab", "Output Tab"
    tab1, tab2 = st.tabs([s.center(whitespace,"\u2001") for s in list_tabs])

    #------------------------------------------------------------------------------------------------------------
    # Input Tab
    #------------------------------------------------------------------------------------------------------------
    with tab1:
        # User input 1: select region >>> prepare parameters
        # ******************************************************************** 
        region = st.selectbox("Select Region", ["UK", 'Italy'], index = 0)
        file_params = params_file_loc + region + "/input_300pct.csv"
        file_curve = params_file_loc + region + "/input_mediaTiming.csv"
        file_base = params_file_loc + region + "/input_base.csv"

        mmm_year = model_versions.loc[model_versions.region == region, 'update'].values[0]
        adjust_ratio = model_versions.loc[model_versions.region == region, 'adjust'].values[0]
        message = f"** {region} results will be based on media mix model (MMM) on fiscal year {mmm_year}"
        st.markdown(
            f"<p style='font-size: 6px; color: #a65407; font-style: italic;'>{message}</p>",
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


        # User input 2: scenario spend plans
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
                        results1 = build_plan_summary(scnr1_revised, 2024, adjust_ratio)
                        results1.iloc[:, 1:] = results1.iloc[:, 1:].round(1)
                        st.session_state['results1'] = results1

                        results2 = build_plan_summary(scnr2_revised, 2024, adjust_ratio)
                        results2.iloc[:, 1:] = results2.iloc[:, 1:].round(1)
                        st.session_state['results2'] = results2 


                        # Timing Results - Monthly
                        # ********************************************************************
                        
                        mo_base = plan_forecast_craft(df_base, 2023, 0, 2, adjust_ratio)[0]
                        mo_scnr1 = plan_forecast_craft(scnr1_revised, 2024, 1, 1, adjust_ratio)[0]
                        mo_scnr2 = plan_forecast_craft(scnr2_revised, 2025, 2, 0, adjust_ratio)[0]
                        forecast_craft_mo = pd.concat([mo_base, mo_scnr1, mo_scnr2], axis = 0)
                        forecast_craft_mo.iloc[:, 1:] = forecast_craft_mo.iloc[:, 1:].round(1)
                        st.session_state['forecast_craft_mo'] = forecast_craft_mo

                        # Timing Results - Quarterly
                        # ********************************************************************
                        qtr_base = plan_forecast_craft(df_base, 2023, 0, 2, adjust_ratio)[1]
                        qtr_scnr1 = plan_forecast_craft(scnr1_revised, 2024, 1, 1, adjust_ratio)[1]
                        qtr_scnr2 = plan_forecast_craft(scnr2_revised, 2025, 2, 0, adjust_ratio)[1]
                        forecast_craft_qtr = pd.concat([qtr_base, qtr_scnr1, qtr_scnr2], axis = 0)
                        forecast_craft_qtr.iloc[:, 1:] = forecast_craft_qtr.iloc[:, 1:].round(1)
                        st.session_state['forecast_craft_qtr'] = forecast_craft_qtr

                    st.session_state["scenario_computed"] = True
                    st.success("Success! Please check the results in the following tabs 👉")









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
            forecast_craft_mo = st.session_state['forecast_craft_mo']
            forecast_craft_qtr = st.session_state['forecast_craft_qtr']





            st.write("")
            st.write("")
            viewing = st.selectbox("Select result format to view", ['Media Summary', 'Increment Forecast'])


            if viewing == 'Media Summary':

                col1, col2 = st.columns(2)
            
                with col1:
                    st.write("")
                    st.markdown("## Scenario 1")
                    container = st.container()
                    with container:
                        numRows = results1.shape[0]
                        st.dataframe(results1, height = (numRows + 1) * 35 + 3)


                with col2:
                    st.write("")
                    st.markdown("## Scenario 2")
                    container = st.container()
                    with container:
                        numRows = results2.shape[0]
                        st.dataframe(results2, height = (numRows + 1) * 35 + 3)





            if viewing == 'Increment Forecast':
                timeframe  = st.radio("", ['Monthly', 'Quarterly'])

                if timeframe == 'Monthly':
                    container = st.container()
                    with container:
                        numRows = forecast_craft_mo.shape[0]
                        st.dataframe(forecast_craft_mo, height = (numRows + 1) * 35 + 3, hide_index=True)
                    
                if timeframe == 'Quarterly':
                    container = st.container()
                    with container:
                        numRows = forecast_craft_qtr.shape[0]
                        st.dataframe(forecast_craft_qtr, height = (numRows + 1) * 35 + 3, hide_index=True)