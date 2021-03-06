import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

### QC FUNCTIONS ### 

def load_randox(filename, mappings):
    '''
    Imports measurement uncertainty excel file from Randox Acusera, then
    maps these to the Assay and Measurands via the Mappings file
    
    Drops any statistics for QCs that are below the limit of quantitation
    (if defined in the mappings file)
    '''
    randoxMU = pd.read_excel(filename)
    randoxMU = randoxMU.rename(columns={'Assay':'Randox'})
    randoxMU = randoxMU.drop(columns=['Intra','Inter','ExpandedUOM'])
    randoxMU = randoxMU.sort_values(by=['Randox','Mean'])
    
    ## Calculate the percentage CV
    randoxMU['% CV'] = round(randoxMU['UOM']/randoxMU['Mean'] * 100,2)
    
    ## Amino acid QCs are all stored in Randox under the same name.
    ## For CSF, select all AAA QCs, create a copy and add a suffix to distinguish these
    csf_aa = randoxMU[randoxMU['Instrument'] == 'AAA+'].copy()
    csf_aa['Randox'] = csf_aa['Randox'].astype(str) + '(CSF)'
    
    ## For urine/plasma, identify urine QC (except phosphoethanolamine) by the lot number starting with "AAU"
    randoxMU['Randox'][randoxMU['Lot Name'].str[:3] == 'AAU'] = randoxMU['Randox'].astype(str) + ' (Urine)' 
    
    ## Add the CSF amino acid QCs to the rest of the QC data
    ## (needs to be after previous step to avoid adding "urine" suffix)
    randoxMU = pd.concat([randoxMU,csf_aa],ignore_index=True)

    ## Read the mappings csv file and join this with the QC data to show the Assay and Measurand name, and import LoQ
    mappings = pd.read_csv(mappings)
    qc_data = mappings.merge(right=randoxMU,on='Randox',how='inner')
    
    ## Drop any QC values that are outside the reportable interval
    qc_data = qc_data.query('Mean >= reportable_range_lower')
    qc_data = qc_data.query('Mean <= reportable_range_upper')
    
    ## Drop duplicates
    qc_data = qc_data.drop_duplicates()
    
    return qc_data

def qc_lot_summary(qc_data, assay, count_thresh=1):
    '''
    Returns a dataframe containing lot number statistics for the assay QC data
    Excludes any statistics with fewer than count_thresh values
    '''
    ## Define the order in which columns should appear
    column_order = ['Count','Mean','UOM','% CV']
    
    ## Filter the main qc data file for only the assay we are interested in, and drop unnecessary columns
    assay_qc_data = qc_data[qc_data['Assay'] == assay].drop(columns=['Assay','Randox'])
    
    ## Drop any QC values where there are fewer than count_thresh datapoints
    assay_qc_data = assay_qc_data[assay_qc_data['Count'] >= count_thresh]
    
    ## Pivot the data and reorder levels
    assay_qc_pivot = assay_qc_data.pivot(index='Measurand', values=column_order, columns=['Lot Name','Instrument'])
    assay_qc_pivot = assay_qc_pivot.swaplevel(0,2, axis=1).sort_index(axis=1)

    ## Reorder columns
    assay_qc_pivot = assay_qc_pivot.reindex(column_order, level=2, axis=1)
    
    ## Merge with mappings to show all measurands (so that they appear even if no QC data) and order by sort order
    mappings = pd.read_csv("data\\raw_data\\mappings.csv")
    all_measurands = mappings[mappings['Assay'] == assay]
    all_measurands = all_measurands.reset_index()[['Measurand','Order']]

    assay_qc_pivot = assay_qc_pivot.merge(right=all_measurands,on='Measurand',how='outer')
    assay_qc_pivot = assay_qc_pivot.sort_values(by='Order', ignore_index=True)
    assay_qc_pivot = assay_qc_pivot.drop(columns='Order')
    
    ## some jiggery pokery to get the multindex back
    assay_qc_pivot = assay_qc_pivot.set_index('Measurand')
    assay_qc_pivot.columns = pd.MultiIndex.from_tuples(assay_qc_pivot.columns)
    
    ## Fill blanks
    assay_qc_pivot = assay_qc_pivot.fillna('')
    
    ## Round the data to 2 decimal places
    assay_qc_data = assay_qc_data.round(2)
    
    return assay_qc_pivot

def conditional_mean(series):
    '''
    Calculate mean only if maximum value is not more than double the minimum value
    '''
    if (series.max() / series.min()) < 2:
        return series.mean()
    else:
        return NaN

def qc_aggregated(qc_data, assay, count_thresh=1):
    '''
    Returns total number of QC datapoints for each analyte and average of each lot number measurement uncertainty and %CV
    Excludes any statistics with fewer than count_thresh values
    '''
    mappings = pd.read_csv("data\\raw_data\\mappings.csv")
    all_measurands = mappings[mappings['Assay'] == assay]
    all_measurands = all_measurands.reset_index()[['Measurand','Order','IsRatio','ratio_numerator','ratio_denominator']]

    ## Filter qc data for assay values only
    filtered = qc_data[qc_data['Assay'] == assay].drop(columns=['Assay','Randox'])
    
    ## Ignore lots with low numbers of data points
    filtered = filtered[filtered['Count'] >= count_thresh]
    
    ## Calculate aggregate functions
    aggregated = filtered.groupby('Measurand').agg({'Count':'sum',
                                              'UOM':'mean',
                                              '% CV':'mean'})
        
    ## Merge with mappings to show all measurands (so that they appear even if no QC data) and order by sort order
    aggregated = aggregated.merge(right=all_measurands,on='Measurand',how='outer')
    aggregated = aggregated.sort_values(by='Order', ignore_index=True)
    aggregated = aggregated.set_index('Measurand')

    ## Calculate simple ratios
    # need to rename column so it doesn't have a space for df. naming
    aggregated = aggregated.rename(columns={'% CV':'value'})
    aggregated['value'] = aggregated.apply(
    lambda x: np.sqrt(float(aggregated.loc[x.ratio_numerator]['value'])**2 + float(aggregated.loc[x.ratio_denominator]['value'])**2)
    if x.IsRatio else x.value
    ,axis=1
    )
    
    # rename the column again
    aggregated = aggregated.rename(columns={'value':'% CV'})

    ## Calculate expanded uncertainty
    aggregated['Expanded uncertainty'] = aggregated['% CV'] * 2
    
    ## Drop columns
    aggregated = aggregated.drop(columns=['Order','IsRatio','ratio_numerator','ratio_denominator'])
    
    ## Round values
    aggregated = aggregated.round(2)
    
    ## Fill blanks
    aggregated = aggregated.fillna('')
    
    return aggregated

def qc_lot_summary_with_means(qc_data, assay, count_thresh=1):
    '''
    Returns a dataframe containing both lot number statistics and aggregated statistics
    Excludes any statistics with fewer than count_thresh values
    '''
    assay_qc_pivot = qc_lot_summary(qc_data, assay, count_thresh)
    aggregated = qc_aggregated(qc_data, assay, count_thresh)
    
    assay_qc_pivot[('All instrument','All lots','Count')] = aggregated['Count']
    assay_qc_pivot[('All instrument','All lots','UOM')] = aggregated['UOM']
    assay_qc_pivot[('All instrument','All lots','% CV')] = aggregated['% CV']
    assay_qc_pivot[('All instrument','All lots','Expanded uncertainty')] = aggregated['Expanded uncertainty']
    
    ## Round values
    assay_qc_pivot = assay_qc_pivot.round(2)
    
    ## Fill blanks
    assay_qc_pivot = assay_qc_pivot.fillna('')
    
    return assay_qc_pivot

def assay_qc_data_export(df, count_thresh):
    '''
    Createsa a qc summary table for each assay (excluding any statistics
    with fewer than count_thresh value per lot number / instrument.
    
    Exports each table to a .csv file in the processed data folder
    '''
    for assay in df['Assay'].unique():
        try:
            filepath = os.path.abspath('') + '\\data\\processed\\qc_summary_tables\\' + assay + '.csv'
            table = qc_lot_summary_with_means(df, assay, count_thresh)
            table.to_csv(filepath)
            print(f'{assay} succesfully exported')
        except:
            print(f'!!! Error in exporting data for {assay}')
            
### EQA FUNCTIONS ###

def combined_uncertainty(std,target_value_uncert):
    return np.sqrt(std**2 + target_value_uncert**2)
    
def load_eqa(folder):
    '''
    Load EQA detail from folder
    '''
    df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(folder, "*.csv"))),join='outer',ignore_index=True)
    df['Standard Uncertainty'] = df['Standard Uncertainty'].fillna(0)
    df = df.rename(columns={'Analyte':'EQA analyte name'})
    return df
            
def standard_uncertainty_data(filename):
    '''
    Extract just the standard uncertainty from an analyte information UKNEQAS scrape
    '''
    
    df = pd.read_csv(filename)
    standard_uncertainty = df[df['Name'] == 'Standard Uncertainty']
    
    #Assumes that there are three samples per distribution for all schemes (might be dangerous)
    standard_uncertainty['Distribution'][0::3] = standard_uncertainty['Distribution'][0::3].astype(str) + 'A'
    standard_uncertainty['Distribution'][1::3] = standard_uncertainty['Distribution'][1::3].astype(str) + 'B'
    standard_uncertainty['Distribution'][2::3] = standard_uncertainty['Distribution'][2::3].astype(str) + 'C'
    
    standard_uncertainty = standard_uncertainty.rename(columns={'Value':'Standard Uncertainty','Analyte name':'Analyte','Distribution':'Specimen'})
    standard_uncertainty = standard_uncertainty.drop(columns='Name')
    return standard_uncertainty

def eqa_calculations(df):
    '''
    Calculate % bias and % uncertainty in target value for all numeric results and targets
    '''
    # Remove rows with non-numeric result or target values
    # https://stackoverflow.com/questions/21771133/finding-non-numeric-rows-in-dataframe-in-pandas/44178063
    data_columns = ['Result','Targ']
    num_df = (df.drop(data_columns, axis=1)
         .join(df[data_columns].apply(pd.to_numeric, errors='coerce')))

    num_df = num_df[num_df[data_columns].notnull().all(axis=1)]
    
    ##Calculate percentage bias
    num_df['% Bias'] = round(100*(num_df['Result'] - num_df['Targ'])/num_df['Targ'],1)
    
    ## Calculate percentage uncertainty in the target value
    num_df['% uncertainty in target value'] = round(100*num_df['Standard Uncertainty']/num_df['Targ'],1)
    
    return num_df

def eqa_scheme_bias_plot(df, scheme):
    '''
    Plot %bias against target value for all analytes in a single scheme
    '''
    df = df[df['Scheme name'] == scheme]    
    
    g = sns.FacetGrid(data=df, col='EQA analyte name', col_wrap=5, sharex=False)
    g = g.map(plt.scatter, "Targ", "% Bias", edgecolor="w")
    g.fig.suptitle(f'EQA results: {scheme}', ha='left', x=0, weight='bold')
    g.fig.subplots_adjust(top=0.8)
    g.set_xlabels('Target value')
    g.savefig(fname = f'data\\processed\\eqa_bias_plot\\{scheme}.png', dpi=200)
    
def eqa_assay_bias_plot(df, assay):
    '''
    Plot %bias against target value for all analytes in a single scheme
    '''
    # Select EQA data for assay
    df = df[(df['Assay'] == assay)]
    
    # Drop any analytes that do not have an EQA scheme
    df = df.dropna(subset=['EQA analyte name'])
    
    # Plot the data (if there is any)
    if len(df) > 0:
        g = sns.FacetGrid(data=df, col='Measurand', col_wrap=5, sharex=False)
        g = g.map(plt.scatter, "Targ", "% Bias", edgecolor="w")
        g.fig.suptitle(f'EQA results: {assay}', ha='left', x=0, weight='bold')
        g.fig.subplots_adjust(top=0.8)
        g.set_xlabels('Target value')
        g.savefig(fname = f'data\\processed\\eqa_bias_plot\\{assay}.png', dpi=200)
    
    # Otherwise tell us that there is no data
    else:
        print(f'No EQA data for {assay}')

def eqa_bias_multi_plot(df):
    '''
    Plot % bias against target value for all analytes in all assays in a dataframe df
    '''
    for assay in df['Assay'].unique():
        eqa_assay_bias_plot(df, assay)

def eqa_summary_statistics(df,assay):
    '''
    Display a table showing the mean bias, standard deviation of the bias, average % uncertainty in the target value,
    combined uncertainty of the mean % bias and expanded uncertainty of % bias (using a coverage factor of 2)
    for the specified assay
    '''
    eqa_summary = df[df['Assay'] == assay]
    eqa_summary = eqa_summary.groupby('Measurand').agg({'Specimen':'count','Targ':['min','max'],'% Bias':['mean','std'],'% uncertainty in target value':'mean'})
        
    ## Combine standard deviation of bias with average uncertainty in the target value
    eqa_summary['Combined uncertainty of % bias'] = eqa_summary.apply(lambda x: combined_uncertainty(x[('% Bias','std')],x[('% uncertainty in target value', 'mean')]), axis = 1)

    ## Merge with all measurands and sort
    mappings = pd.read_csv("data\\raw_data\\mappings.csv")
    all_measurands = mappings[mappings['Assay'] == assay]
    all_measurands = all_measurands.reset_index()[['Measurand','Order','IsRatio','ratio_numerator','ratio_denominator']]
    
    eqa_summary = eqa_summary.merge(right=all_measurands,on='Measurand',how='outer')
    eqa_summary = eqa_summary.sort_values(by='Order', ignore_index=True)

    ## Reset index
    eqa_summary = eqa_summary.set_index('Measurand')
    
    ## Calculate ratios
    # rename columns so that can use dot naming
    eqa_summary = eqa_summary.rename(columns={('% Bias', 'mean'):'bias',('Combined uncertainty of % bias','' ):'biasstd'})
    
    # Calculate bias of ratio
    eqa_summary['bias'] = eqa_summary.apply(
    lambda x: ((float(100 + eqa_summary.loc[x.ratio_numerator]['bias']) / float(100 + eqa_summary.loc[x.ratio_denominator]['bias']))-1)*100
    if x.IsRatio else x.bias,
    axis=1
    ) 
    
    # calculate combined uncertainty of bias for ratios
    eqa_summary['biasstd'] = eqa_summary.apply(
    lambda x: np.sqrt(float(eqa_summary.loc[x.ratio_numerator]['biasstd'])**2 + float(eqa_summary.loc[x.ratio_denominator]['biasstd'])**2)
    if x.IsRatio else x.biasstd
    ,axis=1
    )
    
    # rename the column back again
    eqa_summary = eqa_summary.rename(columns={'bias':('% Bias', 'mean'),'biasstd':('Combined uncertainty of % bias','' )})

    # Calculate expanded uncertainty of % bias using a coverage factor of 2
    eqa_summary['Expanded uncertainty of % bias'] = eqa_summary[('Combined uncertainty of % bias','')].apply(lambda x: x*2)
    
    ## Drop columns
    eqa_summary = eqa_summary.drop(columns=['Order','IsRatio','ratio_numerator','ratio_denominator'])        
    
    ## Round values
    eqa_summary = eqa_summary.round(2)
    
    ## Fill blanks
    eqa_summary = eqa_summary.fillna('')
    
    return eqa_summary

def assay_eqa_data_export(df):
    '''
    Createsa a eqa summary table for each assay.
    
    Exports each table to a .csv file in the processed data folder
    '''
    for assay in df['Assay'].unique():
        try:
            filepath = os.path.abspath('') + '\\data\\processed\\eqa_summary_tables\\' + assay + '.csv'
            table =  eqa_summary_statistics(df,assay)
            table.to_csv(filepath)
            print(f'{assay} succesfully exported')
        except:
            print(f'!!! Error in exporting data for {assay}')

#### PERFORMANCE TARGETS FUNCTIONS ###

def total_error(cv,bias):
    return round(abs(bias) + 1.65*cv,1)

def performance(value, optimal, desirable, minimal):
    if abs(value) <= optimal:
        return "Optimal"
    elif abs(value) <= desirable:
        return "Desirable"
    elif abs(value) <= minimal:
        return "Minimal"
    elif abs(value) > minimal:
        return "Not met"
    else:
        return "Not evaluated"
        
def load_performance_targets(filepath):
    '''
    Load performance targets data from Excel file
    '''
    performance_targets = pd.read_excel(filepath,skiprows=2
                                       ,names=['Assay','Measurand','Biol CVi','Biol CVg'
                                               ,'Anal CV Optimal','Anal CV Desirable','Anal CV Minimal'
                                              ,'Bias Optimal','Bias Desirable','Bias Minimal'
                                              ,'TE Optimal','TE Desirable','TE Minimal','Source of performance targets'])
    performance_targets = performance_targets.drop(columns=['Biol CVi','Biol CVg','Source of performance targets'])
    performance_targets = performance_targets.round(1)
    return performance_targets

def performance_table(qc_data, eqa_data, assay, count_thresh, targets):
    '''
    Create table summarising imprecision, bias and total error performance against
    performance targets
    '''
    ### Load imprecision data
    imprec = qc_aggregated(qc_data,assay,count_thresh)
    
    # Select only the % CV column and measurand name
    imprec = imprec.reset_index()[['Measurand','% CV']]
    
    ### Load bias data
    bias = eqa_summary_statistics(eqa_data,assay)
    
    # Select only the % Bias column and measurand name
    bias.columns = bias.columns.get_level_values(0)
    bias = bias.reset_index()[['Measurand',('% Bias', 'mean')]]
    bias = bias.rename(columns={('% Bias', 'mean'):'% Bias'})
    
    ### Merge imprecision and bias data
    df = imprec.merge(right=bias,on='Measurand',how='outer')
    df[['% CV','% Bias']] = df[['% CV','% Bias']].apply(pd.to_numeric, errors='ignore')
    
    ### Calculate total error
    df['Total error'] = df.apply(lambda x: total_error(x['% CV'],x['% Bias']),axis=1)
    
    ### Merge with performance targets
    targets = targets[targets['Assay'] == assay]
    df = df.merge(right=targets,on='Measurand',how='outer')
    #df = df.fillna('')
    
    ### Calculate performance against performance targets  
    df['CV performance'] = df.apply(lambda x: performance(x['% CV'],x['Anal CV Optimal'],x['Anal CV Desirable'],x['Anal CV Minimal']), axis = 1)
    df['Bias performance'] = df.apply(lambda x: performance(x['% Bias'],x['Bias Optimal'],x['Bias Desirable'],x['Bias Minimal']), axis = 1)
    df['TAE performance'] = df.apply(lambda x: performance(x['Total error'],x['TE Optimal'],x['TE Desirable'],x['TE Minimal']), axis = 1)
    
    ## Select only required columns
    df = df[['Measurand','% CV','CV performance','% Bias','Bias performance','Total error','TAE performance']]
    df = df.set_index('Measurand')
    df[['% CV','% Bias','Total error']] = df[['% CV','% Bias','Total error']].fillna('')
    
    return df

def assay_performance_data_export(qc_data, eqa_data, count_thresh, targets, combined=True):
    '''
    Creates a performance summary table for each assay, either as a separate .csv or combined.
    
    Exports each table to a separate .csv file in the processed data folder
    '''
    if combined:
        filepath = os.path.abspath('') + '\\data\\processed\\performance_summary_tables\\combined.csv'
        df = pd.DataFrame()
        
        for assay in qc_data['Assay'].unique():
            try:
                table =  performance_table(qc_data, eqa_data, assay, count_thresh, targets)
                table['Assay'] = assay
                df = pd.concat([df,table],ignore_index=False)
                print(f'Performance table for {assay} succesfully exported') 
            except:
                print(f'!!! Error for assay: {assay}')                
        
        df.to_csv(filepath,index=True)
        print('Performance table succesfully exported')    
        
    else:
        for assay in qc_data['Assay'].unique():
            try:
                filepath = os.path.abspath('') + '\\data\\processed\\performance_summary_tables\\' + assay + '.csv'
                table =  performance_table(qc_data, eqa_data, assay, count_thresh, targets)
                table.to_csv(filepath)
                print(f'{assay} succesfully exported')
            except:
                print(f'!!! Error in exporting data for {assay}')