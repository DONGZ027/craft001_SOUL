import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import date

today = date.today()


def show_maximization():

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
    

    # Variable Names
    # ***************************************************************************
    spend_prefix = 'M_P_' 
    inc_prefix = 'TIncT_P_'
    mcpt_prefix = 'nMCPT_P_' 
    cpt_prefix = 'CPT_P_'

    # Loading Inputs
    # ***************************************************************************
    params_300pct  = pd.read_csv('input_300pct.csv')
    baseyear = pd.read_csv('scenario2.csv')
    


    # Processing 
    # ***************************************************************************
    media_list = list(baseyear.columns[1:])

    lookup_1 = params_300pct[params_300pct.PCT_Change == 0.001]
    lookup_100 = params_300pct[params_300pct.PCT_Change == 1]
    lookup_300 = params_300pct[params_300pct.PCT_Change == 3]

    fill = []
    st.write()
    for i in np.arange(len(media_list)):
        x = media_list[i]
        x_code = list(media_mapping.keys())[i]
        fill.append([x_code, lookup_100[spend_prefix + x_code].values[0]])
    lookup_base = pd.DataFrame(fill, columns = ['media', 'spend0'])



    #===============================================================================================================================
    # Scenario Functions
    #===============================================================================================================================
    def opt_Array2Inc(X, name_X):  # X = monthly spending of a year for a media channel; name_X = media channel name
        if sum(X) == 0:
            return 0
        else:
            spending_300pct = list(params_300pct[spend_prefix + name_X].values) 
            spending_tracker = list(baseyear[name_X].values)

            Increments = []
            for i in np.arange(0, 12):
                S = X[i] 
                spending_tracker = spending_tracker[1:] + [S] 
                S_yr = sum(spending_tracker)
                curveLoc_300 = np.abs(np.array(spending_300pct) - S_yr).argmin() 
                curveLoc_300 = params_300pct.iloc[curveLoc_300, :]['PCT_Change']
                curve_ref = params_300pct.loc[params_300pct['PCT_Change'] == curveLoc_300, :]

                simu_S_yr = curve_ref[spend_prefix + name_X].values[0]
                simu_Inc_yr = curve_ref['TIncT_P_' + name_X].values[0]
                CPT_M_yr = simu_S_yr / simu_Inc_yr 
                Increment = S / CPT_M_yr 
                Increments.append(Increment) 

            annual_increments = sum(Increments)
            return annual_increments
        

    def opt_DF2Inc(df0):
        total = []
        for x in df0.columns:
            total.append(opt_Array2Inc(df0[x].values, x)) 
        return sum(total) 




    #===============================================================================================================================
    # Optimizer Functions
    #===============================================================================================================================
    
    # ############################################################# #
    #       Dynamic Allocation Based on MCPT / CPT                  #                   
    # ############################################################# #
    def opt_dynamicSat(data, bounds, metrics):
        # I. Set up 
        # **********************************************************************************************
        for x in data.columns:
            if data[x].sum() == 0:
                data.drop(columns = x, inplace = True)

        serving_table = []
        finished_tables = [] 
        warnings = [] 

        spend_metric = metrics[0]
        inc_metric = metrics[1]
        sat_metric = metrics[2]

        initial_values = pd.DataFrame(data.sum(axis = 0)).reset_index() # Convert from monthly to annual spend
        initial_values.columns = ['media', 'spend']

        bread = []
        for x in data.columns:
            LB_pct = bounds.loc[bounds.media == x, 'lb'].values[0]
            bread.append(list(data[x].values * LB_pct))
        bread = pd.DataFrame(bread).T
        bread.columns = data.columns 

        cake = data.sum().sum() - bread.sum().sum()

        print("Cake %:",  cake / data.sum().sum())
        
        print("Initial Cake:", cake)

        for x in data.columns:
            # print(x)
            spend0 = lookup_base[lookup_base.media == x].spend0.values[0]
            spend1 = initial_values[initial_values.media == x].spend.values[0] 
            spend2 = bread[x].values.sum()

            pct1 = np.round(spend1 / spend0, 3)
            pct2 = np.round(spend2 / spend0, 3)
            # print(pct1, pct2)
            # print('...................')

            sat1 = params_300pct[params_300pct.PCT_Change == pct1][sat_metric + x].values[0]
            sat2 = np.round(params_300pct[params_300pct.PCT_Change == pct2][sat_metric + x].values[0], 0) 

            inc1 = params_300pct[params_300pct.PCT_Change == pct1][inc_metric + x].values[0] 
            inc2 = params_300pct[params_300pct.PCT_Change == pct2][inc_metric + x].values[0]

            serving_table.append([x, spend0, spend1, spend2, pct1, pct2, sat1, sat2, inc1, inc2])

        serving_table = pd.DataFrame(serving_table, 
                                     columns = ['media', 'spend0', 'spend1', 'spend2', 
                                                'pct1', 'pct2', 
                                                'sat1', 'sat2', 
                                                'inc1', 'inc2']).sort_values(by = 'sat2', ascending = True).reset_index(drop = True) 
        serving_table = serving_table.reset_index()
        serving_table.rename(columns = {'index' : 'order0'}, inplace = True)

        
        

        # II. While loop to attribute remaining spending to medias, following low sat to high sat order
        # ***********************************************************************************************
        print('')
        print('')
        print('Optimization Begins')
        print('.........................................')
        while cake > 0:
            print("cake before:", cake)
            leaving_guest = 'Nobody yet'
            min_sat = serving_table.sat2.min() 
            priority_group = list(serving_table[serving_table.sat2 == min_sat].media) 
            

            # Decide which scenario is current iteration
            # -------------------------------------------
            running_block = 1 
            if len(serving_table.sat2.unique()) == 1:
                running_block = 2
            if (serving_table.shape[0] == 1 and cake > 0) | (cake < 1):
                running_block = 3


            # 1) Normal case, when media candidates have different saturation
            # ----------------------------------------------------------------
            if running_block == 1:
                guest1 = priority_group[0]
                UB_PCT = bounds.loc[bounds.media == guest1, 'ub'].values[0]
                guest1_UB = serving_table[serving_table.media == guest1].spend1.values[0] * UB_PCT
                guest1_base = serving_table[serving_table.media == guest1].spend0.values[0]
                sat_values = params_300pct[sat_metric + guest1].values 
                sat_current = serving_table.loc[serving_table.media == guest1, 'sat2'].values[0]
                sat_loc0 = np.abs(np.array(sat_values) - sat_current).argmin() 

                guest2 = [x for x in serving_table.media.values if x not in priority_group][0]
                sat_goal = serving_table[serving_table.media == guest2].sat2.values[0] 
                print(guest1, ">>>",  guest2)
                sat_loc1 = np.abs(np.array(sat_values) - sat_goal).argmin() 
                if sat_loc1 == sat_loc0:
                    sat_loc1 = sat_loc0 + 5


                guest1_cake = params_300pct.iloc[sat_loc1, :][spend_metric + guest1] - serving_table[serving_table.media == guest1].spend2.values[0]
                guest1_meal = serving_table.loc[serving_table.media == guest1, 'spend2'].values[0] + guest1_cake 

                if guest1_meal > guest1_UB: 
                    leaving_guest = guest1
                    guest1_cake = guest1_UB - serving_table.loc[serving_table.media == guest1, 'spend2'].values[0]
                    guest1_meal = serving_table.loc[serving_table.media == guest1, 'spend2'].values[0] + guest1_cake

                if cake < guest1_cake:
                    guest1_cake = cake
                    guest1_meal = serving_table.loc[serving_table.media == guest1, 'spend2'].values[0] + guest1_cake
                
                # Updating the serving table
                # ---------------------------
                print("guest 1" , guest1, "gets:", guest1_cake)
                serving_table.loc[serving_table.media == guest1, 'spend2'] =  guest1_meal
                serving_table.loc[serving_table.media == guest1, 'pct2'] = np.round(serving_table.loc[serving_table.media == guest1, 'spend2'].values[0] / guest1_base , 3)
                serving_table.loc[serving_table.media == guest1, 'sat2'] = params_300pct.iloc[sat_loc1, :][sat_metric + guest1]
                serving_table['sat2'] = np.round(serving_table['sat2'], 0)
                serving_table.loc[serving_table.media == guest1, 'inc2'] = params_300pct.loc[params_300pct.PCT_Change == serving_table.loc[serving_table.media == guest1, 'pct2'].values[0], inc_metric + guest1].values[0]
                serving_table = serving_table.sort_values(by = ['sat2'], ascending = True).reset_index(drop = True) 


                # Updating the cake
                # -----------------
                cake = cake - guest1_cake
                
                # Updating guest list on serving table and finished tables
                # --------------------------------------------------------
                if leaving_guest != 'Nobody yet':
                    finished_tables.append(serving_table.loc[serving_table.media == leaving_guest, :])
                    serving_table = serving_table[serving_table.media != leaving_guest]



            # 2) When all medias have the same saturation
            # ----------------------------------------------------------------
            if running_block == 2:
                print('Running block 2')
                serving_table = serving_table.sort_values(by = 'order0', ascending = True).reset_index(drop = True) 

                guest1 = serving_table.media.values[0]
                UB_PCT = bounds.loc[bounds.media == guest1, 'ub'].values[0]
                guest1_UB = serving_table[serving_table.media == guest1].spend1.values[0] * UB_PCT
                guest1_base = serving_table[serving_table.media == guest1].spend0.values[0]

                guest1_cake = cake * 0.2
                guest1_meal = serving_table.loc[serving_table.media == guest1, 'spend2'].values[0] + guest1_cake 

                if guest1_meal > guest1_UB:
                    leaving_guest = guest1 
                    guest1_cake = guest1_UB - serving_table.loc[serving_table.media == guest1, 'spend2'].values[0]
                    guest1_meal = serving_table.loc[serving_table.media == guest1, 'spend2'].values[0] + guest1_cake 

                # Updating the serving table
                # ---------------------------
                print("guest 1" , guest1, "gets:", guest1_cake)
                serving_table.loc[serving_table.media == guest1, 'spend2'] =  guest1_meal
                serving_table.loc[serving_table.media == guest1, 'pct2'] = np.round(serving_table.loc[serving_table.media == guest1, 'spend2'].values[0] / guest1_base , 3)
                serving_table.loc[serving_table.media == guest1, 'sat2'] = params_300pct.iloc[sat_loc1, :][sat_metric + guest1]
                serving_table['sat2'] = np.round(serving_table['sat2'], 0)
                serving_table.loc[serving_table.media == guest1, 'inc2'] = params_300pct.loc[params_300pct.PCT_Change == serving_table.loc[serving_table.media == guest1, 'pct2'].values[0], inc_metric + guest1].values[0]
                serving_table = serving_table.sort_values(by = ['sat2'], ascending = True).reset_index(drop = True) 

                # Updating the cake
                # -----------------
                cake = cake - guest1_cake
                
                # Updating guest list on serving table and finished tables
                # --------------------------------------------------------
                if leaving_guest != 'Nobody yet':
                    finished_tables.append(serving_table.loc[serving_table.media == leaving_guest, :])
                    serving_table = serving_table[serving_table.media != leaving_guest]



            # 3) When remaining cake is tiny, or only 1 media candidate left
            # ----------------------------------------------------------------
            if running_block == 3:
                print('Running block 3')

                guest1 = serving_table.media.values[0]
                UB_PCT = bounds.loc[bounds.media == guest1, 'ub'].values[0]
                guest1_UB = serving_table[serving_table.media == guest1].spend1.values[0] * UB_PCT
                guest1_base = serving_table[serving_table.media == guest1].spend0.values[0] 

                guest1_cake = cake
                guest1_meal = serving_table.loc[serving_table.media == guest1, 'spend2'].values[0] + guest1_cake


                # Updating the serving table
                # ---------------------------
                print("guest 1" , guest1, "gets:", guest1_cake)
                serving_table.loc[serving_table.media == guest1, 'spend2'] =  guest1_meal
                serving_table.loc[serving_table.media == guest1, 'pct2'] = np.round(serving_table.loc[serving_table.media == guest1, 'spend2'].values[0] / guest1_base , 3)
                serving_table.loc[serving_table.media == guest1, 'sat2'] = params_300pct.iloc[sat_loc1, :][sat_metric + guest1]
                serving_table['sat2'] = np.round(serving_table['sat2'], 0)
                serving_table.loc[serving_table.media == guest1, 'inc2'] = params_300pct.loc[params_300pct.PCT_Change == serving_table.loc[serving_table.media == guest1, 'pct2'].values[0], inc_metric + guest1].values[0]
                serving_table = serving_table.sort_values(by = ['sat2'], ascending = True).reset_index(drop = True) 


                # Updating the cake
                # -----------------
                cake = cake - guest1_cake
                


        # III. Putting media candidates and maximized medias together
        # ***********************************************************************************************
        final_bill = serving_table.copy()
        if len(finished_tables) > 0:
            for x in finished_tables:
                final_bill = pd.concat([final_bill, x], axis = 0) 
        final_bill['r'] = final_bill.spend2 / final_bill.spend1 


        # III - 2. Computing an aggregate
        # ***********************************************************************************************
        total = ['Annual Total']
        for col in final_bill.columns[1:]:
            total.append(final_bill[col].sum())
        final_bill.loc[-1] = total 
        shard1 = final_bill.loc[final_bill.media == 'Annual Total', :] 
        shard2 = final_bill.loc[final_bill.media != 'Annual Total', :]
        final_bill = pd.concat([shard2, shard1], axis = 0)
        # final_bill.loc[final_bill.Media == 'Annual Total', 'Change (%)'] = 0



        # IV. Adjusting spending to monthly level
        # ***********************************************************************************************
        optimized = data.copy()
        for x in optimized.columns:
            ratio = final_bill.loc[final_bill.media == x, 'r'].values[0]
            optimized[x] = optimized[x] * ratio


        inc1 = opt_DF2Inc(data)
        inc2 = opt_DF2Inc(optimized)
        delta = (inc2 - inc1) / inc1

        names = optimized.columns
        optimized['Month'] = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
        optimized = optimized[['Month'] + list(names)]


        # V. Exporting Results 
        # ***********************************************************************************************
        return([
            optimized,
            final_bill,
            [inc1, inc2, delta]
        ]) 



    #===============================================================================================================================
    # Page content begins now 
    #===============================================================================================================================
    st.write("")
    st.write("")



    whitespace = 15
    list_tabs = "Input Tab", "Output Tab"
    tab1, tab2 = st.tabs([s.center(whitespace,"\u2001") for s in list_tabs])
    

    with tab1:
        st.write("")
        st.header("Provide initial spending plan file and set bounds for optimization") 
        st.write("")
        st.write("")
        if 'results_maximization' not in st.session_state:
            st.session_state['results_maximization'] = [] 
        # st.session_state['results_maximization'] = [] 

        step1_done = False
        step1_error = False

        step2_done = False

        #.......................................................................................................
        # Step 1.  User choose initial spending plan 
        #.......................................................................................................
        st.write("")
        st.divider()
        st.write("Step 1: Upload initial spending plan")
        uploaded_files = st.file_uploader("Upload CSV scenario files", type="csv", accept_multiple_files=True)
        if uploaded_files:
            dfs = []
            file_names = []
            for file in uploaded_files:
                df = pd.read_csv(file)
                dfs.append(df)
                file_names.append(file.name)  
            input_file = st.selectbox("Choose one of the input files as initial scenario", file_names)
            file_index = file_names.index(input_file)
            input_table = dfs[file_index].copy() 
            input_table = input_table.drop(columns = ['Month'])
            st.session_state['input_table'] = input_table

            annual_summary = []
            for x in input_table.columns:
                annual_summary.append([x, int(np.round(input_table[x].sum(), 0))]) 
            annual_summary = pd.DataFrame(annual_summary, columns=['Media', 'Spending']) 
            annual_summary = pd.DataFrame(annual_summary.values.T, columns = input_table.columns).fillna(0)
            annual_summary = annual_summary.iloc[1:, :]
            
            names = list(annual_summary.columns)
            annual_summary[' '] = ['Annual Spending']
            annual_summary = annual_summary[[' '] + names]
            st.dataframe(annual_summary, hide_index = True)

            step1_done = True

            # Error Catching
            # ***************************************************************************
            error_medias_scnr1 = [] 
            medias_UB_scnr1 = []
            for i in np.arange(len(annual_summary.columns[1:])):
                x = list(media_mapping.values())[i]
                x_code = list(media_mapping.keys())[i]
                spending = annual_summary[x].values[0] 
                spending_UB = params_300pct.loc[params_300pct['PCT_Change'] == 3, spend_prefix + x_code].values[0]
                if spending > spending_UB:
                    error_medias_scnr1.append(x) 
                    medias_UB_scnr1.append(spending_UB)


            if len(error_medias_scnr1) > 0:
                step1_error = True
                st.error("The following medias in current scenario exeeded upper bound for annnual spending, please adjust the spending plans before running the analysis -- ")
                for i in np.arange(len(error_medias_scnr1)):
                    x = error_medias_scnr1[i]
                    x_ub = np.round(medias_UB_scnr1[i], 0)
                    x_ub = int(x_ub)
                    x_ub = format(x_ub, ",")
                    st.error(x + " exceeded annual upper bound of $" + str(x_ub)) 



            
            
     


 



        #.......................................................................................................
        # Step 2.  User adjust media adjustment bounds 
        #.......................................................................................................
        st.write("")
        st.divider()
        st.write("Step 2: Choose media adjustment range for optimization")
        media_bounds = []
        if step1_done == False:
            st.write("")
        
        elif step1_error == True:
            st.error("Please adjust the spending plans before proceeding to Step")
        else:
            for i in np.arange(len(annual_summary.columns[1:])):
                x = list(media_mapping.values())[i] 
                x_code = list(media_mapping.keys())[i]
                spend_now = annual_summary[x].values[0] 
                spend_UB = params_300pct.loc[params_300pct['PCT_Change'] == 3, spend_prefix + x_code].values[0] 
                max_adjust = spend_UB / spend_now
                # max_adjust = np.round(max_adjust, 2) - 0.1
                adjust0 = 0.8
                adjust1 = 1.2
                min_adjust = 0 
                
                media_bounds.append([x, min_adjust, adjust0, adjust1, max_adjust]) 
            
            media_bounds = pd.DataFrame(media_bounds, 
                                        columns = ['Media', 
                                                'Lower Bound %', 'Adjust Minimum %  🖋️', 
                                                'Adjust Maximum % 🖋️', 'Upper Bound %'])
            # media_bounds['media_code'] = list(media_mapping.keys())
            
            container = st.container()
            with container:
                num_rows = media_bounds.shape[0]
                st.data_editor(
                    media_bounds,
                    height = (num_rows + 1) * 35 + 3, 
                    column_config={
                        "Adjust Minimum %": st.column_config.NumberColumn(
                        help = "Please choose a number between Minimum and Maximum",
                        # format = "%d 🖋️",
                        ),

                        "Adjust Maximum %": st.column_config.NumberColumn(
                        help = "Please choose a number between Minimum and Maximum",
                        # format = "%d 🖋️",
                        )
                    },
                    disabled = ['Media', 'Lower Bound %', 'Upper Bound %'],      
                    hide_index = True                 
                ) 

            

            # Tables for optimization function
            # **************************************************************************
            step2_done = True
            st.divider()

    
        #.......................................................................................................
        # Step 3.  Optimization Button
        #.......................................................................................................
        if step2_done == False:
            st.write("")

        else:
            backstage_bounds = media_bounds.copy().drop(columns = ['Media', 'Lower Bound %', 'Upper Bound %'])
            backstage_bounds['media'] = list(media_mapping.keys()) 
            backstage_bounds.columns = ['lb', 'ub', 'media']
            
            backstage_spend = input_table.copy()
            backstage_spend.columns = list(media_mapping.keys()) 

            # st.write(backstage_bounds)
            
            zero_spend_medias = []
            for x in backstage_spend.columns:
                if backstage_spend[x].sum() == 0:
                    zero_spend_medias.append(x)

            if st.button("Let's begin the optimization!"):
                with st.spinner("I'm working on it ..."):
                    model_spends = []
                    model_reports = []
                    model_incs = []

                    model1 = opt_dynamicSat(backstage_spend, backstage_bounds, [spend_prefix, inc_prefix, mcpt_prefix])
                    model1_spend = model1[0]
                    model1_report = model1[1]
                    model1_inc = model1[2][1]
                    model_spends.append(model1_spend)
                    model_reports.append(model1_report)
                    model_incs.append(model1_inc)

                    model2 = opt_dynamicSat(backstage_spend, backstage_bounds, [spend_prefix, inc_prefix, cpt_prefix])
                    model2_spend = model2[0]
                    model2_report = model2[1]
                    model2_inc = model2[2][1]
                    model_spends.append(model2_spend)
                    model_reports.append(model2_report)
                    model_incs.append(model2_inc)
                    
                    winner_index = model_incs.index(max(model_incs)) 
                    proposal_optimized = model_spends[winner_index]

                    check = model_reports[winner_index]
                
                    summary_table = []
                    for x in proposal_optimized.columns[1:]:
                        spend1 = backstage_spend[x].values.sum() 
                        spend2 = proposal_optimized[x].values.sum() 
                        change =  np.round(100 *(spend2 - spend1) / spend1, 2)

                        inc1 = opt_Array2Inc(backstage_spend[x].values, x)
                        inc2 = opt_Array2Inc(proposal_optimized[x].values, x) 

                        summary_table.append([x, spend1, spend2, change, inc1, inc2]) 
                    summary_table = pd.DataFrame(summary_table, 
                                                columns = ['media', 'spend1', 'spend2', 'change(%)', 'inc1', 'inc2'])
                
                    for x in summary_table.columns[1:]:
                        summary_table[x] = np.round(summary_table[x], 1)


                    if len(zero_spend_medias) > 0:
                        for x in zero_spend_medias:
                            proposal_optimized[x] = 0
                            summary_table.loc[-1] = [x, 0, 0, 0, 0, 0]


                
                    st.session_state['results_maximization'] = [proposal_optimized, summary_table, check]


                    st.success("Optimization completed successfully! Please check the results in the next tab 👉")


    with tab2:
        results_maximization = st.session_state['results_maximization'] 

        if len(results_maximization) == 0:
            st.write("No results to display yet, please input initial spending plan in the first tab and run the optimizer.")


        else:
            # ====================================================================================================================
            # Preparing results table for display
            # ====================================================================================================================

            # Spend Plans
            # ******************************************************************
            results_spend0 = st.session_state['input_table'] 
            names = list(results_spend0.columns)
            results_spend0['Month'] = months
            results_spend0 = results_spend0[['Month'] + names]
            
            spend_plan = results_maximization[0]
            results_spend1 = spend_plan.rename(columns = media_mapping)

            summary0 = ['Annual Total']
            summary1 = ['Annual Total']
            
            for x in media_mapping.values():
                summary0.append(results_spend0[x].sum())
                summary1.append(results_spend1[x].sum())

            results_spend0.loc[-1] = summary0
            shard1 = results_spend0.loc[results_spend0.Month == 'Annual Total', :]
            shard2 = results_spend0.loc[results_spend0.Month != 'Annual Total', :]
            results_spend0 = pd.concat([shard1, shard2], axis = 0)
            results_spend0 = results_spend0.reset_index(drop = True) 

            results_spend1.loc[-1] = summary1
            shard1 = results_spend1.loc[results_spend1.Month == 'Annual Total', :]
            shard2 = results_spend1.loc[results_spend1.Month != 'Annual Total', :]
            results_spend1 = pd.concat([shard1, shard2], axis = 0)
            results_spend1 = results_spend1.reset_index(drop = True) 

            # Summary Table
            # ******************************************************************
            summary_table = results_maximization[1]
            
            summary_table = summary_table.reset_index(drop = True) 

            results_summary = summary_table.copy()
            results_summary.media.replace(media_mapping, inplace = True)
            results_summary = results_summary.rename(columns = {'media': 'Media', 
                                                                'spend1': 'Spend Before', 
                                                                'spend2': 'Spend After', 
                                                                'change(%)': 'Change (%)', 
                                                                'inc1': 'Incremental Attendance Before', 
                                                                'inc2': 'Incremental Attendance After'})
            for x in ['Spend Before', 'Spend After', 'Incremental Attendance Before', 'Incremental Attendance After']:
                results_summary[x] = results_summary[x].apply(lambda x: "{:,.0f}".format(x))
            
            # # total = ['Annual Total']
            # # for col in results_summary.columns[1:]:
            # #     total.append(results_summary[col].sum())
            # # results_summary.loc[-1] = total 
            # # shard1 = results_summary.loc[results_summary.Media == 'Annual Total', :] 
            # # shard2 = results_summary.loc[results_summary.Media != 'Annual Total', :]
            # # results_summary = pd.concat([shard2, shard1], axis = 0)
            # # results_summary.loc[results_summary.Media == 'Annual Total', 'Change (%)'] = 0

            nrows_summary = results_summary.shape[0]
            results_summary = results_summary.style.applymap(lambda v: 'color: red' if v < 0 else 'color: green', subset=['Change (%)'])


            

            # Download Button for Results
            # ****************************************************************** 
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writter:
                results_spend0.to_excel(writter, sheet_name = "Initial Spend Scnario", index = False)
                results_spend1.to_excel(writter, sheet_name = "Optimized Spend Scenario", index = False) 
                results_summary.to_excel(writter, sheet_name = "Summary Table", index = False) 

            st.write("")
            st.download_button(
                label = 'Download Results Package in Excel 📦', 
                data = buffer, 
                file_name =  "Attendance_Maximization_" + today.strftime("%b_%d_%Y") +  ".xlsx",
                mime = 'application/vnd.ms-excel')
            


            st.write("")
            st.write("")
            viewing = st.selectbox("Select result format to view", ['Optimized Spending Scenario', 'Optimization Summary'])


            if viewing  == "Optimized Spending Scenario":
                st.write("")
                st.write("")
                
                st.dataframe(results_spend1, height = 500)


            if viewing == "Optimization Summary":
                st.write("")
                st.write("")
                
                container = st.container()
                with container:
                    st.dataframe(results_summary, height = (nrows_summary + 1) * 35 + 3, hide_index=True)
            



            # col1, col2 = st.columns(2)
            # with col1: 
            #     viewing = st.selectbox("Select a table to view", ['Optimized Spending Plan', 'Summary Table'])
            # with col2:
            #     st.write("")


            # if viewing == 'Optimized Spending Plan':

            #     st.write("")
            #     spend_plan = results_maximization[0]
            #     numRows = spend_plan.shape[0]

            #     container = st.container()
            #     with container:
            #         st.dataframe(spend_plan, height = (numRows + 1) * 35 + 3)
                
            #     st.write("") 


                # summary = []
                # for x in spend_plan.columns:
                #     summary.append([x, int(np.round(spend_plan[x].sum(), 0))]) 
                # summary = pd.DataFrame(summary, columns=['Media', 'Spending']) 
                # summary = pd.DataFrame(summary.values.T, columns = backstage_spend.columns).fillna(0)
                # # summary = summary.iloc[1:, :]
                # st.dataframe(summary) 



            # if viewing == 'Summary Table':

                # st.write("")

                # summary_table = results_maximization[1]
                # summary_table = summary_table.reset_index(drop = True) 
                # numRows = summary_table.shape[0]


                # summary_table = summary_table.style.applymap(lambda v: 'color: red' if v < 0 else 'color: green', subset=['change(%)'])
                # summary_table = summary_table.format(precision = 1)




                # container = st.container()
                # with container:
                #     st.dataframe(summary_table, height = (numRows + 1) * 35 + 3)



        # st.write("")
        # col1, col2 = st.columns(2)
        # with col1: 
            
        #     bounds = st.slider(
        #         "Select the bounds for the optimization: ",
        #         50, 250, (80, 120))
            
        # with col2:
        #     st.write("")


        # bounds = [x / 100 for x in bounds]
        
        # numRows = blueprint.shape[0]
        # container = st.container()
        # with container:
        #     user_panel = st.data_editor(
        #         blueprint,

        #         disabled = ['Month'],
        #         hide_index = True, 
        #         height = (numRows + 1) * 35 + 3
        #         )
        

        # summary = []
        # for x in user_panel.columns[1:]:
        #     summary.append([x, int(np.round(user_panel[x].sum(), 0))]) 
        # summary = pd.DataFrame(summary, columns=['Media', 'Spending']) 
        # summary = pd.DataFrame(summary.values.T, columns = user_panel.columns[1:]).fillna(0)
        # summary = summary.iloc[1:, :]
        # st.dataframe(summary) 


        # proposal = user_panel[user_panel.columns[1:]] 

        # zero_spend_medias = []
        # for x in proposal.columns:
        #     if proposal[x].sum() == 0:
        #         zero_spend_medias.append(x)


        # if st.button("Let's begin the optimization!"):
            
        #     # Run optimizers, compare results, select winner
        #     with st.spinner("I'm working on it ..."):

        #         model_spends = []
        #         model_reports = []
        #         model_incs = []


        #         model1 = opt_dynamicSat(proposal, bounds, [spend_prefix, inc_prefix, mcpt_prefix])
        #         model1_spend = model1[0]
        #         model1_report = model1[1]
        #         model1_inc = model1[2][1]
        #         model_spends.append(model1_spend)
        #         model_reports.append(model1_report)
        #         model_incs.append(model1_inc)

        #         model2 = opt_dynamicSat(proposal, bounds, [spend_prefix, inc_prefix, cpt_prefix])
        #         model2_spend = model2[0]
        #         model2_report = model2[1]
        #         model2_inc = model2[2][1]
        #         model_spends.append(model2_spend)
        #         model_reports.append(model2_report)
        #         model_incs.append(model2_inc)


        #         winner_index = model_incs.index(max(model_incs)) 


        #         # results1 = opt_dynamicSat(proposal, bounds, [spend_prefix, inc_prefix, mcpt_prefix])
        #         # # results1_optimizedInc = results1[2][1]
        #         # results = opt_dynamicSat(proposal, bounds, [spend_prefix, inc_prefix, cpt_prefix])
        #         # # results2_optimizedInc = results2[2][1]

        #         # # results = results1
        #         # # if results2_optimizedInc > results1_optimizedInc:
        #         # #     results = results2 
            
        #         proposal_optimized = model_spends[winner_index]

        #         check = model_reports[winner_index]
                
        #         summary_table = []
        #         for x in proposal_optimized.columns[1:]:
        #             spend1 = proposal[x].values.sum() 
        #             spend2 = proposal_optimized[x].values.sum() 
        #             change =  np.round(100 *(spend2 - spend1) / spend1, 2)

        #             inc1 = opt_Array2Inc(proposal[x].values, x)
        #             inc2 = opt_Array2Inc(proposal_optimized[x].values, x) 

        #             summary_table.append([x, spend1, spend2, change, inc1, inc2]) 
        #         summary_table = pd.DataFrame(summary_table, 
        #                                     columns = ['media', 'spend1', 'spend2', 'change(%)', 'inc1', 'inc2'])
                
        #         for x in summary_table.columns[1:]:
        #             summary_table[x] = np.round(summary_table[x], 1)


        #         if len(zero_spend_medias) > 0:
        #             for x in zero_spend_medias:
        #                 proposal_optimized[x] = 0
        #                 summary_table.loc[-1] = [x, 0, 0, 0, 0, 0]


                
        #         st.session_state['results_maximization'] = [proposal_optimized, summary_table, check]



                

        # #         st.success("Optimization completed successfully! Please check the results in the next tab 👉")




            
            
                
                
        

        




    