import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import io

today = date.today()

def show_sandbox():

    #===============================================================================================================================
    # Set Up
    #===============================================================================================================================
    months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'] 
    planning_years = [2023, 2024, 2025, 2026]
    media_mapping = {
        'NTV':'National TV', 
        'LTV': 'Local TV',
        'CTV': 'Connected TV',
        'TTV' : 'TTV',
        'STR' : 'Streaming',
        'YOT' : 'YouTube',
        'CIN' : 'Cinema',
        'DIS' : 'Display',
        'SEM' : 'Paid Search',
        'FAB' : 'Facebook',
        'SNP' : 'Snapchat',
        'TWT' : 'Twitter',
        'PIN' : 'Pinterest',
        'TIK' : 'TikTok',
        'DRD' : 'Digital Radio',
        'LRD' : 'Local Radio',
        'OOH' : 'Out of Home'
    }


    # Time structure
    # ********************************************************************
    time_ref  = pd.read_csv('DT_snowflake.csv') 
    time_ref = time_ref[[
        'FIS_WK_END_DT', 'FIS_YR_NB', 'FIS_MO_NB', 'FIS_WK_NB', 
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
    time_ref = time_ref.reset_index()



    # Media Timing & 300% Table
    # ********************************************************************
    params_mediaTiming  = pd.read_csv('input_mediaTiming.csv')
    medias_mediaTiming = params_mediaTiming.columns[1:]

    params_300pct  = pd.read_csv('input_300pct.csv')


    # Base year mongthly spending
    # ********************************************************************
    baseyear = pd.read_csv('scenario2.csv') 





    #===============================================================================================================================
    # Scenario Functions
    #===============================================================================================================================


    def scnr_inc_X(
            spending_data,  # Spending file 
            x,   # Media name
            year_index  # [0, 1, 2, 3] -- 0 is the base year, i.e. FY23   
            ): 
        # Output Items to fillin
        #-------------------------------------------------------------------------------------------------------------
        planned_spendings = [] # Planned spending for each month, length = 12
        monthly_table_M = []   # Each month's ncrement from the current month spending, without considering media timing, length = 12 
        dummy_burnCurve_M = [] # List of burn curves for media X from jsut focal month spending, length = 12
        MAI_MA_months = []  # Margianl attendance given current month's spending, length = 12
        MC_months = [] # Marginal cost given current months's spending, length = 12
        
        # Use the base year spendings as the starting point of the annual spending, to locate percentage on the 300 table
        # ----------------------------------------------------------------------------------------------------------------
        spending_tracker = list(baseyear[x])
        spending_300pct = list(params_300pct['M_P_' + x].values) 

        # Fill in the monthly data --  monthly_table_M,  MC_months, MAI_MA_months 
        #-------------------------------------------------------------------------------------------------------------------------------------------
        for t in np.arange(1, 13):
            
            # month spending >>> year spending >>> year spending on 300% table >>> percentage location
            # ***************************************************************************************
            S = spending_data.loc[spending_data['fm'] == t, x].values[0] 
            planned_spendings.append(S) 

            spending_tracker = spending_tracker[1:] + [S]
            S_yr = sum(spending_tracker)

            curveLoc_300 = np.abs(np.array(spending_300pct) - S_yr).argmin() 
            curveLoc_300 = params_300pct.iloc[curveLoc_300, :]['PCT_Change'] # Corresponding spending percentage in the 300% table, i.e. location on the media satuation curve 
            curve_ref = params_300pct.loc[params_300pct['PCT_Change'] == curveLoc_300, :]  # Grab all the metrics corresponding to the location on the media satuation curve

            # [Annual Increment, Annual Spending on 300% table] >>> Annual CPT >>> Monthly Increment
            # ***************************************************************************************
            simu_S_yr = curve_ref['M_P_' + x].values[0]
            simu_Inc_yr = curve_ref['TIncT_P_' + x].values[0]
            CPT_M_yr = simu_S_yr / simu_Inc_yr 
            Increment = S / CPT_M_yr  
            row = [x, t, S, Increment]
            monthly_table_M.append(row)
            
            # [Annual Percentage Delta, Annual Spending] >>> Annual MC 
            # ***************************************************************************************
            simu_pctDelta_yr = curve_ref['Pct_Delta'].values[0]
            simu_MC_yr = simu_pctDelta_yr * simu_S_yr 
            MC_months.append(simu_MC_yr) 
            
            # [Annual Percentage Delta, Annual Spending] >>> Annual MC 
            # ***************************************************************************************
            simu_nMIncT_yr = curve_ref['nMTIncT_P_' + x].values[0]
            MAI_MA_months.append(simu_nMIncT_yr) 
        
    
        # Up to now, we have increment from each month's spending   
        monthly_table_M = pd.DataFrame(monthly_table_M, columns=['media', 'fm', 'spend', 'increment'])  


        # Apply media timing to the monthly increments, i.e. creating the burn curve
        #------------------------------------------------------------------------------------------------------------------------------------------- 
        inc_curve = params_mediaTiming[x].values

        # Burn curve from the start
        # **************************************************************************************** 
        for t in monthly_table_M.fm.values:
            timeline = [0] * time_ref.shape[0]
            start = time_ref.loc[(time_ref.FIS_MO_NB == t) & (time_ref.FIS_YR_NB == planning_years[year_index]), 'lag_weeks'].values[0] 
            timeline[start: start + len(inc_curve)] = inc_curve

            inc = monthly_table_M.loc[monthly_table_M.fm == t, 'increment'].values[0]
            spread = inc * np.array(timeline)
            focal_spread = list(spread)[start: start + len(inc_curve)]
            dummy_burnCurve_M.append(focal_spread)

        return ([
            dummy_burnCurve_M,
            MAI_MA_months, 
            MC_months,
            planned_spendings,
            monthly_table_M
        ]) 
    


    def scnr_results_package(scenario_name, year_index, data):  
        fill = [] 

        scenario = data.reset_index()
        scenario.rename(columns= {'index': 'fm'}, inplace=True)
        scenario.fm = scenario.fm + 1 
        for media in medias_mediaTiming:
            row = [
                scenario_name, 
                year_index, 
                media, 
                scnr_inc_X(scenario, media,  year_index)[3],
                scnr_inc_X(scenario, media,  year_index)[0], 
                scnr_inc_X(scenario, media,  year_index)[1],
                scnr_inc_X(scenario, media,  year_index)[2]
            ]
            fill.append(row)    
                
        results = pd.DataFrame(fill, columns=['scenario', 'year', 'media', 
                                            'spending', 'list_burnCurve', 'list_MAI_MA', 'list_MC'])     
        return results




    def scnr_burncurve_M_monthly(lookup, media, year_index):
        burnCurve_M = lookup.loc[(lookup.media == media) & (lookup.year == year_index), 'list_burnCurve'].values[0]
        curves_M = []

        for i in np.arange(1, 13):
            inc_1month = burnCurve_M[i - 1]  # burn curve for media M from spending in month i, weekly level 

            # Convert the weekly level curve to monthly level curve
            # ****************************************************************************************
            time_stamp = time_ref.loc[(time_ref.FIS_YR_NB == planning_years[year_index]) &
                                    (time_ref.FIS_MO_NB == i), 'index'].values[0] 
            shard = time_ref.iloc[time_stamp: time_stamp + len(inc_1month), :]  
            shard['inc'] = inc_1month
            inc_1month = shard.groupby(['FIS_YR_NB', 'FIS_MO_NB'])['inc'].sum().reset_index()['inc'].values 

            # Append to the media collector curves_M, 
            # It's a list of monthly burn curve from 12 months spending for media M
            # Note that the curve is not considering which year is it yet, i.e. they all starts from index 0
            # ****************************************************************************************
            curves_M.append(inc_1month) 

        return curves_M

    


    def scnr_burnCurve_yearT(lookup, year_index, spending_data):

        # 1) Collecting monthly burn curves for all medias 
        # 2) Aggreaget to get monthly burn curve from all media's spending in each 12 months
        # ***************************************************************************************
        burnCurve_medias = []
        for media in medias_mediaTiming:
            burnCurve_medias.append(pd.DataFrame(scnr_burncurve_M_monthly(lookup, media, year_index)).values) 

        shape = burnCurve_medias[0].shape
        burnCurve_year = np.zeros(shape)
        for i in burnCurve_medias:
            burnCurve_year += i 

        # 1) Putting the single-month burn cruves to a larger time scale, i.e. the scale of the whole planning period
        # ***********************************************************************************************************
        burnCurve_final = []
        for i in np.arange(1, 13):
            timeline = np.zeros(12 * len(planning_years)) 
            fill = burnCurve_year[i - 1]
            start = time_ref[ (time_ref.FIS_YR_NB == planning_years[year_index]) & (time_ref.FIS_MO_NB == i)]['lag_months'].values[0] 
            timeline[start: start + len(fill)] = fill 
            burnCurve_final.append(timeline) 
        burnCurve_final = pd.DataFrame(burnCurve_final)
        burnCurve_final.columns = months * 4 
        

        # Wrapping up and formating
        # *********************************************************************************************************** 
        craft = burnCurve_final.copy() 
        craft['Inc_Attn'] = craft.iloc[:, :].sum(axis = 1)
        craft['Cost'] = spending_data.iloc[:, 1:].sum(axis = 1) 
        craft['CPA'] = craft['Cost'] / craft['Inc_Attn'] 
        craft['Month'] = months 
        craft['Fiscal Year'] = planning_years[year_index]

        header = ['Fiscal Year', 'Month', 'Cost', 'Inc_Attn', 'CPA']
        shard1 = craft.iloc[:, -len(header):]
        shard1 = shard1[header]
        shard2 = craft.iloc[:, 0: -len(header)]
        shard2.columns = pd.MultiIndex.from_product([planning_years, months])
        craft_monthly = pd.concat([shard1, shard2], axis = 1)


        # Creating quarterly table
        # *********************************************************************************************************** 
        shards = []
        for i in range(4):
            shard = burnCurve_final.iloc[:, 12*i : 12*(i+1)].T
            shard['fiscal_quarter'] = [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3 
            shard = shard.groupby('fiscal_quarter').sum().T
            shard['spending_quarter'] = [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3 
            shard = shard[['spending_quarter', 1, 2, 3, 4]]
            shard = pd.DataFrame(shard.values, columns= ['spending_quarter', 'Q1', 'Q2', 'Q3', 'Q4'])
            shard = shard.groupby('spending_quarter').sum()
            shard = pd.DataFrame(shard).reset_index().drop(columns = ['spending_quarter'])
            shards.append(shard)
        
        quarterly_inc = pd.concat(shards, axis = 1) 
        years = []
        for yr in planning_years:
            piece = [str(yr)] * 4
            years.append(piece)
        years = np.array(years).flatten() 
        quarters = quarterly_inc.columns
        timeline_cols= [years[i] + ' ' + quarters[i] for i in range(16)]  
        quarterly_inc.columns = timeline_cols
        quarterly_inc['Total Attendance'] = quarterly_inc.sum(axis = 1) 

        spends = spending_data.copy()
        spends['total_spend'] = spends.iloc[:, 1:].sum(axis = 1)
        spends['fiscal_quarter'] = [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3 
        spends = spends.groupby('fiscal_quarter')['total_spend'].sum()
        spends = pd.DataFrame(spends).reset_index()


        craft_quarterly = quarterly_inc.copy() 
        craft_quarterly['Total Cost'] = spends['total_spend'].values
        craft_quarterly['CPA'] = craft_quarterly['Total Cost'] / craft_quarterly['Total Attendance']  
        craft_quarterly['Fiscal Year'] = planning_years[year_index]
        craft_quarterly['Fiscal Quarter'] = [1, 2, 3, 4]
        craft_quarterly = craft_quarterly[['Fiscal Year', 'Fiscal Quarter', 'Total Cost', 'Total Attendance', 'CPA'] + timeline_cols] 
        for i in craft_quarterly.columns[1:]:
            craft_quarterly[i] = np.round(craft_quarterly[i], 0).astype(int)


        return [craft_monthly, craft_quarterly]
    



    def scnr_summary_table(lookup, year_index):
        names = ['media', 'cost', 'inc', 'CPA', 'MCPA']
        table = []
        MA_months = []
        MC_months = [] 
        for x in medias_mediaTiming:
            pool = lookup.loc[(lookup.media == x) & (lookup.year == year_index), :]
            
            cost = sum(pool.spending.values[0])

            inc = pool.list_burnCurve.values[0]
            inc = pd.DataFrame(inc)
            inc['total'] = inc.sum(axis = 1)
            inc = inc['total'].sum()

            CPA = cost / inc

            MA_sum = sum(pool.list_MAI_MA.values[0])
            MA_months.append(MA_sum)
            MC_sum = sum(pool.list_MC.values[0])
            MC_months.append(MC_sum)
            MCPA = MC_sum / MA_sum 

            row = [x, cost, inc, CPA, MCPA]
            table.append(row)

        table = pd.DataFrame(table, columns= names).fillna(np.nan)

        # Computing Aggregated Results
        # *************************************************************************
        agg_cost = int(np.round(table.cost.sum(), 0))
        agg_inc = int(np.round(table.inc.sum(), 0))
        agg_CPA = int(np.round(agg_cost / agg_inc, 0)) 
        agg_MCPA = int(np.round(sum(MC_months) / sum(MA_months), 0))


        # Putting together & some formatting 
        # *************************************************************************
        final_craft = table.copy().fillna(0)
        for x in names[1:]:
            final_craft[x] = final_craft[x].astype(float).round(0).astype(int)

        final_craft.loc[final_craft.CPA == 0, 'CPA'] = np.nan
        final_craft.loc[final_craft.MCPA == 0, 'MCPA'] = np.nan
        final_craft.columns = ['Media', 'Annual Cost', 'Annual Attendance', 'CPA', 'MCPA']
        final_craft['Media'].replace(media_mapping, inplace=True)
        
        return [final_craft, 
                [agg_cost, agg_inc, agg_CPA, agg_MCPA]]
    




    #===============================================================================================================================
    # Page content begins now 
    #===============================================================================================================================
    # st.header("Scenario Planning") 
    # st.write("")
    st.write("")
    st.write("")
    

    whitespace = 15
    list_tabs = "Input Tab", "Output Tab"
    tab1, tab2 = st.tabs([s.center(whitespace,"\u2001") for s in list_tabs])

    
    
    with tab1:
        st.write("")
        st.header("Select spending plans here") 
        st.write("")
        st.write("")
        
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
                spending_UB = params_300pct.loc[params_300pct['PCT_Change'] == 3, 'M_P_' + x_code].values[0]
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
                spending_UB = params_300pct.loc[params_300pct['PCT_Change'] == 3, 'M_P_' + x_code].values[0]
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

                        # Results Package
                        # ********************************************************************
                        year0 = scnr_results_package('scenario0', 0, baseyear)
                        year1 = scnr_results_package('scenario1', 1, scnr1)
                        year2 = scnr_results_package('scenario2', 2, scnr2)
                        results_df = pd.concat([year0, year1, year2], axis=0)
                        st.session_state['results_df'] = results_df

                        # Summary Table
                        # ********************************************************************
                        results1 = scnr_summary_table(results_df, 1)
                        st.session_state['results1'] = results1
                        results2 = scnr_summary_table(results_df, 2)
                        st.session_state['results2'] = results2


                        # Timing Results - Monthly
                        # ********************************************************************
                        curves0 = scnr_burnCurve_yearT(results_df, 0, baseyear)
                        curves1 = scnr_burnCurve_yearT(results_df, 1, scnr1)
                        curves2 = scnr_burnCurve_yearT(results_df, 2, scnr2) 


                        craft = pd.concat([curves0[0], curves1[0], curves2[0]], axis = 0)

                        for c in craft.columns[2:]:
                            craft[c] = craft[c].round(0).astype(int)

                        summation= craft.iloc[:, 5:].sum(axis = 0).values
                        craft.loc[-1] = ['Total Increment', np.nan, np.nan, np.nan, np.nan] + list(summation)
                        craft.reset_index(drop=True, inplace=True)

                        new_columns = list(craft.columns[0 : 5])
                        for shard in craft.columns[5:]:
                            new_columns.append(str(shard[0]) + '-' + shard[1])
                        craft.columns = new_columns
                        shard1 = craft[craft['Fiscal Year'] == 'Total Increment']
                        shard2 = craft[craft['Fiscal Year'] != 'Total Increment']
                        craft_monthly = pd.concat([shard1, shard2], axis = 0)
                        st.session_state['craft_monthly'] = craft_monthly

                        # Timing Results - Quarterly
                        # ********************************************************************
                        craft = pd.concat([curves0[1], curves1[1], curves2[1]], axis = 0)
                        for c in craft.columns[2:]:
                            craft[c] = craft[c].round(0).astype(int)

                        summation= craft.iloc[:, 5:].sum(axis = 0).values
                        craft.loc[-1] = ['Total Increment', np.nan, np.nan, np.nan, np.nan] + list(summation)
                        craft.reset_index(drop=True, inplace=True)

                        shard1 = craft[craft['Fiscal Year'] == 'Total Increment']
                        shard2 = craft[craft['Fiscal Year'] != 'Total Increment']
                        craft_quarterly = pd.concat([shard1, shard2], axis = 0)
                        st.session_state['craft_quarterly'] = craft_quarterly

                    st.success("Success! Please check the results in the following tabs 👉")







    with tab2:
        results_df = st.session_state['results_df']
        
        if results_df.shape[0] == 0:
            st.write("Please upload scenario files and run the analysis in the first tab")


        else:
            # ********************************************************************************************************************
            # Preparing Result Tables
            # ********************************************************************************************************************
            results1 = st.session_state['results1']
            scnr1_file = st.session_state['scnr1_file']
            scnr1_table = st.session_state['scnr1_table']
            media_summary1 = results1[0]
            agg_summary1 = pd.DataFrame(
                        [[
                            'Aggregate', 
                            results1[1][0], 
                            results1[1][1],
                            results1[1][2],
                            results1[1][3]
                        ]], 
                        columns = ['File Name', 'Aggregate Cost', 'Aggregate Attendance', 'CPA', 'MCPA']
                    )


            results2 = st.session_state['results2']
            scnr2_file = st.session_state['scnr2_file']
            scnr2_table = st.session_state['scnr2_table']
            media_summary2 = results2[0]
            agg_summary2 = pd.DataFrame(
                        [[
                            'Aggregate', 
                            results2[1][0], 
                            results2[1][1],
                            results2[1][2],
                            results2[1][3]
                        ]], 
                        columns = ['File Name', 'Aggregate Cost', 'Aggregate Attendance', 'CPA', 'MCPA']
                    )
            
            
            craft_monthly = st.session_state['craft_monthly']

            craft_quarterly = st.session_state['craft_quarterly']


            summary_scnr1 = agg_summary1.copy()
            summary_scnr1 .columns = media_summary1.columns
            summary_scnr1  = pd.concat([summary_scnr1 , media_summary1], axis = 0)


            summary_scnr2  = agg_summary2.copy()
            summary_scnr2.columns = media_summary2.columns
            summary_scnr2 = pd.concat([summary_scnr2, media_summary2], axis = 0)


            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writter:
                scnr1_table.to_excel(writter, sheet_name = "Scenario 1 Spending", index = False)
                scnr2_table.to_excel(writter, sheet_name = "Scenario 2 Spending", index = False)
                summary_scnr1.to_excel(writter, sheet_name = 'Scenario 1 Summary', index = False) 
                summary_scnr2.to_excel(writter, sheet_name = 'Scenario 2 Summary', index = False)
                craft_monthly.to_excel(writter, sheet_name = 'Monthly Increment', index = False)
                craft_quarterly.to_excel(writter, sheet_name = 'Quarterly Increment', index = False)


            st.write("")
            st.download_button(
                label = 'Download Results Package in Excel 📦', 
                data = buffer, 
                file_name =  "scenario_" + today.strftime("%b_%d_%Y") +  ".xlsx",
                mime = 'application/vnd.ms-excel')




            st.write("")
            st.write("")
            viewing = st.selectbox("Select result format to view", ['Media Summary', 'Increment Forecast'])


            if viewing == 'Media Summary':

                col1, col2 = st.columns(2)
            
                with col1:
                    st.write("")
                    st.markdown("## Scenario 1")
                    
                    st.write("Aggregate Summary")
                    st.dataframe(
                        agg_summary1, hide_index=True
                    )

                    st.write("Detail by Media")
                    container = st.container()
                    with container:
                        numRows = media_summary1.shape[0]
                        st.dataframe(media_summary1, height = (numRows + 1) * 35 + 3, hide_index=True)


                with col2:
                    st.write("")
                    st.markdown("## Scenario 2")

                    st.write("Aggregate Summary")
                    st.dataframe(
                        agg_summary2, hide_index=True
                    )

                    st.write("Detail by Media")
                    container = st.container()
                    with container:
                        numRows = media_summary2.shape[0]
                        st.dataframe(media_summary2, height = (numRows + 1) * 35 + 3, hide_index=True)

            if viewing == 'Increment Forecast':
                timing = st.radio("", ['Monthly', 'Quarterly'])

                if timing == 'Monthly':
                    container = st.container()
                    with container:
                        numRows = craft_monthly.shape[0]
                        st.dataframe(craft_monthly, height = (numRows + 1) * 35 + 3, hide_index=True)
                    
                if timing == 'Quarterly':
                    container = st.container()
                    with container:
                        numRows = craft_quarterly.shape[0]
                        st.dataframe(craft_quarterly, height = (numRows + 1) * 35 + 3, hide_index=True)

            



    # with tab3:
    #     results_df = st.session_state['results_df']

    #     if results_df.shape[0] == 0:
    #         st.write("Please upload scenario files and run the analysis in the first tab")


    #     else:
    #         craft_monthly = st.session_state['craft_monthly']
    #         craft_quarterly = st.session_state['craft_quarterly']

    #         timing = st.radio("", ['Monthly', 'Quarterly'])

    #         if timing == 'Monthly':
    #             container = st.container()
    #             with container:
    #                 numRows = craft_monthly.shape[0]
    #                 st.dataframe(craft_monthly, height = (numRows + 1) * 35 + 3, hide_index=True)
                    
                    

    #         if timing == 'Quarterly':
    #             container = st.container()
    #             with container:
    #                 numRows = craft_quarterly.shape[0]
    #                 st.dataframe(craft_quarterly, height = (numRows + 1) * 35 + 3, hide_index=True)


    