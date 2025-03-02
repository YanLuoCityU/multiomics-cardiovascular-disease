import os
import time
import logging
import pandas as pd
import numpy as np

# diag_min_dates: extract the minimum date of the diagnoses
def diag_min_dates(diag_df, date_df, diag_list, rename):
    '''
    
    Note: The corresponding ICD-10 diagnosis codes can be found in data-field Field 41270 and the two fields can be linked using the array structure.
    
    diag_df is the dataframe with only diagnoses information;
    date_df is the dataframe with the corresponding dates information for the diagnoses;
    diag_list is the list of interested diagnoses.
    
    E.g.,:
        If diag_df is as follows:
            diag_df = {
                'eid': [1000120, 1000260, 1000329, 1000449, 1000515],
                '41270-0.0': ['C679', 'E780', 'D171', 'E119', 'A047'],
                '41270-0.1': ['D221', 'I200', 'D649', 'E780', 'A419'],
                '41270-0.2': ['D231', 'I251', 'E149', 'F171', 'C442'],
                '41270-0.3': ['D313', 'M1399', 'E611', 'H409', 'C444'],
                '41270-0.4': ['E669', 'M6593', 'E669', 'I10', 'C445'],
                '41270-0.5': ['E780', 'R001', 'E780', 'I251', 'C830'],
                '41270-0.6': ['E785', 'S5250', 'G258', 'I48', 'C911'],
                '41270-0.7': ['F171', 'T810', 'H251', 'I489', 'C97'],
                '41270-0.8': ['G560', 'W019', 'H268', 'I501', 'D044'],
                ...
            }
            
            cad_list = ['I21', 'I210', 'I211', 'I212', 'I213', 'I214', 'I219', 'I21X', 
                        'I22', 'I220', 'I221', 'I228', 'I229', 
                        'I23', 'I230', 'I231', 'I232', 'I233', 'I234', 'I235', 'I236', 'I238', 
                        'I24', 'I240', 'I241', 'I248', 'I249', 
                        'I25', 'I250', 'I251', 'I252', 'I253', 'I254', 'I255', 'I256', 'I258', 'I259', 'Z951', 'Z955']
        
        The result_dict will be: {1000260: [2], 1000449: [5]}
        
        If date_df is as follows:
            date_df = {
                'eid': [1000120, 1000260, 1000329, 1000449, 1000515],
                '41280-0.0': ['2022-04-15', '2015-05-26', '2011-01-11', None, None],
                '41280-0.1': ['2022-09-14', '2015-05-26', '2014-03-20', None, None],
                '41280-0.2': ['2022-09-14', '2015-05-26', '2014-03-20', None, None],
                '41280-0.3': ['2022-04-15', None, '2013-06-25', None, None],
                '41280-0.4': ['2022-04-15', None, '1999-02-23', None, None],
                '41280-0.5': ['2022-04-15', None, '1999-02-23', '2013-06-25', None],
                '41280-0.6': ['2010-06-14', None, '2014-03-20', None, None],
                '41280-0.7': ['2010-06-14', None, '1997-06-09', None, None],
                '41280-0.8': ['2010-06-14', None, '1999-02-23', None, None],
                ...
            }
        
        The min_dates will be: [(1000260, Timestamp('2015-05-26 00:00:00')), (1000449, Timestamp('2013-06-25 00:00:00'))]
    '''
    # Get the list of 'eid' that contain any diagnosis in diag_list
    mask = diag_df.iloc[:, 1:].isin(diag_list).any(axis=1)
    sub_diag_df = diag_df[mask]

    # Get the column indices that match the diagnoses in diag_list
    diag_positions = sub_diag_df.iloc[:, 1:].apply(lambda row: np.where(row.isin(diag_list))[0] + 1, axis=1)

    # Extract the corresponding dates from the date DataFrame
    def get_min_date(eid, positions):
        row = date_df[date_df['eid'] == eid].iloc[0, positions]
        dates = pd.to_datetime(row.dropna().values)
        return dates.min() if not dates.empty else pd.NaT

    # Calculate the minimum date using apply and a lambda function
    min_dates = sub_diag_df.apply(lambda row: get_min_date(row['eid'], diag_positions.loc[row.name]), axis=1)

    # Create a DataFrame and rename the date column
    min_dates_df = pd.DataFrame({
        'eid': sub_diag_df['eid'],
        rename: min_dates
    })

    return min_dates_df

# death_min_dates: extract the date of cardiovascular death
def death_min_dates(death_df, diag_list, rename):
    '''
    diag_df is the dataframe with diagnoses information;
    diag_list is the list of interested diagnoses.
    '''
    condition_columns = death_df.columns[3:]
    mask = death_df[condition_columns].apply(lambda x: x.astype(str).str.startswith(tuple(diag_list)), axis=1)
    min_dates_df = death_df[mask.any(axis=1)][['eid', '40000-0.0']]
    min_dates_df['40000-0.0'] = pd.to_datetime(min_dates_df['40000-0.0'], format='%Y-%m-%d')
    min_dates_df.rename(columns = {'40000-0.0': rename}, inplace=True)
    
    return min_dates_df

# get_date_interval: calculate the interval between two dates
'''
def get_date_interval(start_date_var, end_date_var, df, unit):
    start_date = pd.to_datetime(df[start_date_var], format='%Y-%m-%d')
    end_date = pd.to_datetime(df[end_date_var], format='%Y-%m-%d')
    nb_of_dates = start_date.shape[0]
    days = [(end_date[i] - start_date[i]).days for i in range(nb_of_dates)]
    if unit == 'day':
        return pd.DataFrame(days)
    elif unit == 'year':
        years = [ele/365 for ele in days]
        return pd.DataFrame(years)
'''

def get_date_interval(start_date_var, end_date_var, df, unit):
    start_date = pd.to_datetime(df[start_date_var], format='%Y-%m-%d')
    end_date = pd.to_datetime(df[end_date_var], format='%Y-%m-%d')
    days = (end_date - start_date).dt.days
    if unit == 'year':
        days /= 365
    return pd.DataFrame(days)

# define_event_time: define the event time for the interested outcomes
def define_event_time(df, outcome):
    df = df.copy()
    df[f'bl2{outcome}_yrs_raw'] = get_date_interval('bl_date', f'{outcome}_date', df, 'year')
    df[f'bl2{outcome}_yrs'] = df[[f'bl2{outcome}_yrs_raw', 'bl2end_yrs', 'bl2death_yrs']].min(axis=1)
    df[f'{outcome}'] = 1
    df[f'{outcome}'].loc[df[f'bl2{outcome}_yrs_raw'].isnull()] = 0
    mapping = {
        'cad': 'Coronary artery disease',
        'stroke': 'Stroke',
        'hf': 'Heart failure',
        'af': 'Atrial fibrillation',
        'va': 'Ventricular arrhythmias',
        'pad': 'Peripheral artery diseases',
        'aaa': 'Abdominal aortic aneurysm',
        'vt': 'Vein thrombosis',
        'ar': 'Arrhythmias',
        'mace': 'Major adverse cardiovascular event',
        'cvd': 'Cardiovascular diseases',
        'cved': 'Cardiovascular endpoints with cardiovascular death'
    }
    logger.info(f"{mapping[outcome]}: {df[outcome].value_counts()}")
    
    return df

# Define the path
data_dir = '/home/ukb/data/phenotype_data/'
resources_dir = '/home/ukb/data/resources/'
output_dir = '/your path/multiomics-cardiovascular-disease/data/processed/outcomes/'
log_dir = '/your path/multiomics-cardiovascular-disease/saved/log/DataPreparation/'

# Set up logger
log_filename = os.path.join(log_dir, 'OutcomeGenerator.log')
logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

start_time = time.time()

# Read data
logger.info('Reading data.')
health_outcomes_df = pd.read_csv(data_dir + 'health_outcomes.csv', low_memory=False)
outcomes_info_df = pd.read_csv(output_dir + 'OutcomesBasicInfo.csv', low_memory=False)


# List of the interested diagnoses
logger.info('Loading list of the interested diagnoses.')

## Coronary artery disease
cad_icd10_list = ['I21', 'I210', 'I211', 'I212', 'I213', 'I214', 'I219', 'I21X', 
                  'I22', 'I220', 'I221', 'I228', 'I229', 
                  'I23', 'I230', 'I231', 'I232', 'I233', 'I234', 'I235', 'I236', 'I238', 
                  'I24', 'I240', 'I241', 'I248', 'I249', 
                  'I25', 'I250', 'I251', 'I252', 'I253', 'I254', 'I255', 'I256', 'I258', 'I259']
cad_icd9_list = ['410', '4109', '412', '4129', '414', '4140', '4141', '4148', '4149']
cad_opcs_list = ['K40', 'K401', 'K402', 'K403', 'K404', 'K408', 'K409', 
                 'K41', 'K411', 'K412', 'K413', 'K414', 'K418', 'K419', 
                 'K42', 'K421', 'K422', 'K423', 'K424', 'K428', 'K429', 
                 'K43', 'K431', 'K432', 'K433', 'K434', 'K438', 'K439',
                 'K44', 'K441', 'K442', 'K448', 'K449',
                 'K45', 'K451', 'K452', 'K453', 'K454', 'K455', 'K456', 'K458', 'K459',
                 'K46', 'K461', 'K462', 'K463', 'K464', 'K465', 'K468', 'K469',
                 'K49', 'K491', 'K492', 'K493', 'K494', 'K498', 'K499',
                 'K50', 'K501', 'K502', 'K503', 'K504', 'K508', 'K509',
                 'K75', 'K751', 'K752', 'K753', 'K754', 'K758', 'K759']

## Stroke
stroke_icd10_list = ['I60', 'I600', 'I601', 'I602', 'I603', 'I604', 'I605', 'I606', 'I607', 'I608', 'I609',
                     'I61', 'I610', 'I611', 'I612', 'I613', 'I614', 'I615', 'I616', 'I618', 'I619',
                     'I62', 'I620', 'I621', 'I629', 
                     'I63', 'I630', 'I631', 'I632', 'I633', 'I634', 'I635', 'I636', 'I638', 'I639', 'I64']
stroke_icd9_list = ['430', '4309', '431', '4319', '434', '4340', '4341', '4349', '436', '4369']

## Heart Failure
hf_icd10_list = ['I110', 'I130', 'I132', 'I255', 
                 'I420', 'I421', 'I422', 'I425', 'I426', 'I427', 'I428', 'I429',
                 'I50', 'I500', 'I501', 'I509']
hf_icd9_list = ['4254', '428', '4280', '4281', '4289']

## Atrial fibrillation
af_icd10_list = ['I48', 'I480', 'I481', 'I482', 'I483', 'I484', 'I489']
af_icd9_list = ['4273']
af_opcs_list = ['K571', 'K621', 'K622', 'K623', 'K624', 'X501', 'X502']

## Ventricular Arrhythmias
va_icd10_list = ['I46', 'I460', 'I461', 'I469', 'I470', 'I472', 'I490']
va_opcs_list = ['K571', 'K641', 'X503', 'X504', 'X508', 'X509']

## Peripheral artery disease
pad_icd10_list = ['I700', 'I7000', 'I7001', 'I702', 'I7020', 'I7021', 'I708', 'I7080', 'I709', 'I7090', 'I738', 'I739']
pad_icd9_list = ['4400', '4402', '4438', '4439']
pad_opcs_list = ['L216', 'L513', 'L516', 'L518', 'L521', 'L522', 'L541', 'L544', 'L548', 
                 'L591', 'L592', 'L593', 'L594', 'L595', 'L596', 'L597', 'L598', 
                 'L601', 'L602', 'L631', 'L635', 'L639', 'L667']

## Abdominal aortic aneurysm
aaa_icd10_list = ['I713', 'I714']
aaa_icd9_list = ['4413', '4414']
aaa_opcs_list = ['L18', 'L181', 'L182', 'L183', 'L184', 'L185', 'L186', 'L188', 'L189',
                 'L19', 'L191', 'L192', 'L193', 'L194', 'L195', 'L196', 'L198', 'L199',
                 'L254', 'L27', 'L271', 'L272', 'L273', 'L274', 'L275', 'L276', 'L278', 'L279',
                 'L28', 'L281', 'L282', 'L283', 'L284', 'L285', 'L286', 'L288', 'L289', 'L464']

## Aortic aneurysm (to be included)
aa_icd10_list = ['I71', 'I710', 'I711', 'I712', 'I713', 'I714', 'I715', 'I716', 'I717', 'I718', 'I719']
aa_icd9_list = ['441', '4410', '4411', '4412', '4413', '4414', '4415', '4416', '4417']
aa_opcs_list = ['L18', 'L181', 'L182', 'L183', 'L184', 'L185', 'L186', 'L188', 'L189',
                 'L19', 'L191', 'L192', 'L193', 'L194', 'L195', 'L196', 'L198', 'L199',
                 'L254', 'L27', 'L271', 'L272', 'L273', 'L274', 'L275', 'L276', 'L278', 'L279',
                 'L28', 'L281', 'L282', 'L283', 'L284', 'L285', 'L286', 'L288', 'L289', 'L464']

## Venous thromboembolism
vt_icd10_list = ['I26', 'I260', 'I269', 'I80', 'I800', 'I801', 'I802', 'I803', 'I808', 'I809', 'I81', 'I820', 'I822', 'I823', 'I828', 'I829']
vt_icd9_list = ['4151', '451', '4510', '4511', '4512', '4518', '4519', '4532', '4538']
vt_opcs_list = ['L791', 'L902']

## Cardiovascular death
cvd_death_list = ['I' + str(i) for i in range(100)]

######################################################################################################
########################### ICD-9/ICD-10 + OPCS  #####################################################
######################################################################################################
logger.info('Extracting the minimum date of the diagnoses.')

# Diagnoses - ICD10
diag_df = health_outcomes_df[['eid'] + ['41270-0.' + str(i) for i in range(259)]]
date_df = health_outcomes_df[['eid'] + ['41280-0.' + str(i) for i in range(259)]]

cad_icd10_df = diag_min_dates(diag_df, date_df, cad_icd10_list, 'cad_icd10_date')
stroke_icd10_df = diag_min_dates(diag_df, date_df, stroke_icd10_list, 'stroke_icd10_date')
hf_icd10_df = diag_min_dates(diag_df, date_df, hf_icd10_list, 'hf_icd10_date')
af_icd10_df = diag_min_dates(diag_df, date_df, af_icd10_list, 'af_icd10_date')
va_icd10_df = diag_min_dates(diag_df, date_df, va_icd10_list, 'va_icd10_date')
pad_icd10_df = diag_min_dates(diag_df, date_df, pad_icd10_list, 'pad_icd10_date')
aaa_icd10_df = diag_min_dates(diag_df, date_df, aaa_icd10_list, 'aaa_icd10_date')
vt_icd10_df = diag_min_dates(diag_df, date_df, vt_icd10_list, 'vt_icd10_date')

# Diagnosis - ICD9
diag_df = health_outcomes_df[['eid'] + ['41271-0.' + str(i) for i in range(47)]]
date_df = health_outcomes_df[['eid'] + ['41281-0.' + str(i) for i in range(47)]]

cad_icd9_df = diag_min_dates(diag_df, date_df, cad_icd9_list, 'cad_icd9_date')
stroke_icd9_df = diag_min_dates(diag_df, date_df, stroke_icd9_list, 'stroke_icd9_date')
hf_icd9_df = diag_min_dates(diag_df, date_df, hf_icd9_list, 'hf_icd9_date')
af_icd9_df = diag_min_dates(diag_df, date_df, af_icd9_list, 'af_icd9_date')
pad_icd9_df = diag_min_dates(diag_df, date_df, pad_icd9_list, 'pad_icd9_date')
aaa_icd9_df = diag_min_dates(diag_df, date_df, aaa_icd9_list, 'aaa_icd9_date')
vt_icd9_df = diag_min_dates(diag_df, date_df, vt_icd9_list, 'vt_icd9_date')

# Operative procedures - OPCS
diag_df = health_outcomes_df[['eid'] + ['41272-0.' + str(i) for i in range(126)]]
date_df = health_outcomes_df[['eid'] + ['41282-0.' + str(i) for i in range(126)]]

cad_opcs_df = diag_min_dates(diag_df, date_df, cad_opcs_list, 'cad_opcs_date')
af_opcs_df = diag_min_dates(diag_df, date_df, af_opcs_list, 'af_opcs_date')
va_opcs_df = diag_min_dates(diag_df, date_df, va_opcs_list, 'va_opcs_date')
pad_opcs_df = diag_min_dates(diag_df, date_df, pad_opcs_list, 'pad_opcs_date')
aaa_opcs_df = diag_min_dates(diag_df, date_df, aaa_opcs_list, 'aaa_opcs_date')
vt_opcs_df = diag_min_dates(diag_df, date_df, vt_opcs_list, 'vt_opcs_date')

# Underlying (primary) cause of death: ICD10 + Contributory (secondary) cause of death: ICD10
death_df = health_outcomes_df[['eid', '40000-0.0', '40001-0.0'] + ['40002-0.'+str(i) for i in range(15)]]

cad_death_df = death_min_dates(death_df, cad_icd10_list, 'cad_death_date')
stroke_death_df = death_min_dates(death_df, stroke_icd10_list, 'stroke_death_date')
hf_death_df = death_min_dates(death_df, hf_icd10_list, 'hf_death_date')
af_death_df = death_min_dates(death_df, af_icd10_list, 'af_death_date')
va_death_df = death_min_dates(death_df, va_icd10_list, 'va_death_date')
pad_death_df = death_min_dates(death_df, pad_icd10_list, 'pad_death_date')
aaa_death_df = death_min_dates(death_df, aaa_icd10_list, 'aaa_death_date')
vt_death_df = death_min_dates(death_df, vt_icd10_list, 'vt_death_date')
cvd_death_df = death_min_dates(death_df, cvd_death_list, 'cvd_death_date')



######################################################################################################
########################### Pool  ####################################################################
######################################################################################################
logger.info('Pooling the results.')

# Coronary artery disease
cad_df = pd.merge(outcomes_info_df, cad_icd10_df, how = 'left', on =['eid'])
cad_df = pd.merge(cad_df, cad_icd9_df, how = 'left', on =['eid'])
cad_df = pd.merge(cad_df, cad_opcs_df, how = 'left', on =['eid'])
cad_df = pd.merge(cad_df, cad_death_df, how = 'left', on =['eid'])
cad_df['cad_date'] = cad_df[['cad_icd10_date', 'cad_icd9_date', 'cad_opcs_date', 'cad_death_date']].min(axis=1)
cad_df = define_event_time(cad_df, 'cad')

# Stroke
stroke_df = pd.merge(outcomes_info_df, stroke_icd10_df, how = 'left', on =['eid'])
stroke_df = pd.merge(stroke_df, stroke_icd9_df, how = 'left', on =['eid'])
stroke_df = pd.merge(stroke_df, stroke_death_df, how = 'left', on =['eid'])
stroke_df['stroke_date'] = stroke_df[['stroke_icd10_date', 'stroke_icd9_date', 'stroke_death_date']].min(axis=1)
stroke_df = define_event_time(stroke_df, 'stroke')

# Heart failure
hf_df = pd.merge(outcomes_info_df, hf_icd10_df, how = 'left', on =['eid'])
hf_df = pd.merge(hf_df, hf_icd9_df, how = 'left', on =['eid'])
hf_df = pd.merge(hf_df, hf_death_df, how = 'left', on =['eid'])
hf_df['hf_date'] = hf_df[['hf_icd10_date', 'hf_icd9_date', 'hf_death_date']].min(axis=1)
hf_df = define_event_time(hf_df, 'hf')

# Atrial fibrillation
af_df = pd.merge(outcomes_info_df, af_icd10_df, how = 'left', on =['eid'])
af_df = pd.merge(af_df, af_icd9_df, how = 'left', on =['eid'])
af_df = pd.merge(af_df, af_opcs_df, how = 'left', on =['eid'])
af_df = pd.merge(af_df, af_death_df, how = 'left', on =['eid'])
af_df['af_date'] = af_df[['af_icd10_date', 'af_icd9_date', 'af_opcs_date', 'af_death_date']].min(axis=1)
af_df = define_event_time(af_df, 'af')

# Ventricular arrhythmias
va_df = pd.merge(outcomes_info_df, va_icd10_df, how = 'left', on =['eid'])
va_df = pd.merge(va_df, va_opcs_df, how = 'left', on =['eid'])
va_df = pd.merge(va_df, va_death_df, how = 'left', on =['eid'])
va_df['va_date'] = va_df[['va_icd10_date', 'va_opcs_date', 'va_death_date']].min(axis=1)
va_df = define_event_time(va_df, 'va')

# Peripheral artery disease
pad_df = pd.merge(outcomes_info_df, pad_icd10_df, how = 'left', on =['eid'])
pad_df = pd.merge(pad_df, pad_icd9_df, how = 'left', on =['eid'])
pad_df = pd.merge(pad_df, pad_opcs_df, how = 'left', on =['eid'])
pad_df = pd.merge(pad_df, pad_death_df, how = 'left', on =['eid'])
pad_df['pad_date'] = pad_df[['pad_icd10_date', 'pad_icd9_date', 'pad_opcs_date', 'pad_death_date']].min(axis=1)
pad_df = define_event_time(pad_df, 'pad')

# Abdominal aortic aneurysm
aaa_df = pd.merge(outcomes_info_df, aaa_icd10_df, how = 'left', on =['eid'])
aaa_df = pd.merge(aaa_df, aaa_icd9_df, how = 'left', on =['eid'])
aaa_df = pd.merge(aaa_df, aaa_opcs_df, how = 'left', on =['eid'])
aaa_df = pd.merge(aaa_df, aaa_death_df, how = 'left', on =['eid'])
aaa_df['aaa_date'] = aaa_df[['aaa_icd10_date', 'aaa_icd9_date', 'aaa_opcs_date', 'aaa_death_date']].min(axis=1)
aaa_df = define_event_time(aaa_df, 'aaa')

# Venous thromboembolism
vt_df = pd.merge(outcomes_info_df, vt_icd10_df, how = 'left', on =['eid'])
vt_df = pd.merge(vt_df, vt_icd9_df, how = 'left', on =['eid'])
vt_df = pd.merge(vt_df, vt_opcs_df, how = 'left', on =['eid'])
vt_df = pd.merge(vt_df, vt_death_df, how = 'left', on =['eid'])
vt_df['vt_date'] = vt_df[['vt_icd10_date', 'vt_icd9_date', 'vt_opcs_date', 'vt_death_date']].min(axis=1)
vt_df = define_event_time(vt_df, 'vt')

# Cardiovascular death
cvd_death_df = pd.merge(outcomes_info_df, cvd_death_df, how = 'left', on =['eid'])
cvd_death_df['cvd_death'] = 1
cvd_death_df['cvd_death'].loc[cvd_death_df['cvd_death_date'].isnull()] = 0
cvd_death_df['bl2cvd_death_yrs_raw'] = get_date_interval('bl_date', 'cvd_death_date', cvd_death_df, 'year')
cvd_death_df['bl2cvd_death_yrs'] = cvd_death_df[['bl2cvd_death_yrs_raw', 'bl2end_yrs', 'bl2death_yrs']].min(axis=1)
logger.info(f"Cardiovascular death: {cvd_death_df['cvd_death'].value_counts()}")

# Arrhythmias
ar_df = pd.merge(outcomes_info_df, af_df[['eid', 'af', 'af_date']], how = 'left', on =['eid'])
ar_df = pd.merge(ar_df, va_df[['eid', 'va', 'va_date']], how = 'left', on =['eid'])
ar_df['ar_date'] = ar_df[['af_date', 'va_date']].min(axis=1)
ar_df = define_event_time(ar_df, 'ar')

# MACE
mace_df = pd.merge(outcomes_info_df, cad_df[['eid', 'cad', 'cad_date']], how = 'left', on =['eid'])
mace_df = pd.merge(mace_df, stroke_df[['eid', 'stroke', 'stroke_date']], how = 'left', on =['eid'])
mace_df['mace_date'] = mace_df[['cad_date', 'stroke_date']].min(axis=1)
mace_df = define_event_time(mace_df, 'mace')

# Cardiovascular diseases
cvd_df = pd.merge(outcomes_info_df, cad_df[['eid', 'cad', 'cad_date']], how = 'left', on =['eid'])
cvd_df = pd.merge(cvd_df, stroke_df[['eid', 'stroke', 'stroke_date']], how = 'left', on =['eid'])
cvd_df = pd.merge(cvd_df, hf_df[['eid', 'hf', 'hf_date']], how = 'left', on =['eid'])
cvd_df = pd.merge(cvd_df, af_df[['eid', 'af', 'af_date']], how = 'left', on =['eid'])
cvd_df = pd.merge(cvd_df, va_df[['eid', 'va', 'va_date']], how = 'left', on =['eid'])
cvd_df = pd.merge(cvd_df, pad_df[['eid', 'pad', 'pad_date']], how = 'left', on =['eid'])
cvd_df = pd.merge(cvd_df, aaa_df[['eid', 'aaa', 'aaa_date']], how = 'left', on =['eid'])
cvd_df = pd.merge(cvd_df, vt_df[['eid', 'vt', 'vt_date']], how = 'left', on =['eid'])
cvd_df['cvd_date'] = cvd_df[['cad_date', 'stroke_date', 'hf_date', 'af_date', 'va_date', 'pad_date', 'aaa_date', 'vt_date']].min(axis=1)
cvd_df = define_event_time(cvd_df, 'cvd')

# Cardiovascular endpoints with cardiovascular death
cved_df = pd.merge(outcomes_info_df, cad_df[['eid', 'cad', 'cad_date']], how = 'left', on =['eid'])
cved_df = pd.merge(cved_df, stroke_df[['eid', 'stroke', 'stroke_date']], how = 'left', on =['eid'])
cved_df = pd.merge(cved_df, hf_df[['eid', 'hf', 'hf_date']], how = 'left', on =['eid'])
cved_df = pd.merge(cved_df, af_df[['eid', 'af', 'af_date']], how = 'left', on =['eid'])
cved_df = pd.merge(cved_df, va_df[['eid', 'va', 'va_date']], how = 'left', on =['eid'])
cved_df = pd.merge(cved_df, pad_df[['eid', 'pad', 'pad_date']], how = 'left', on =['eid'])
cved_df = pd.merge(cved_df, aaa_df[['eid', 'aaa', 'aaa_date']], how = 'left', on =['eid'])
cved_df = pd.merge(cved_df, vt_df[['eid', 'vt', 'vt_date']], how = 'left', on =['eid'])
cved_df = pd.merge(cved_df, cvd_death_df[['eid', 'cvd_death', 'cvd_death_date']], how = 'left', on =['eid'])
cved_df['cved_date'] = cved_df[['cad_date', 'stroke_date', 'hf_date', 'af_date', 'va_date', 'pad_date', 'aaa_date', 'vt_date', 'cvd_death_date']].min(axis=1)
cved_df = define_event_time(cved_df, 'cved')

# Pool all outcomes
outcomes_df = outcomes_info_df.copy()
outcomes_df = pd.merge(outcomes_df, cad_df[['eid', 'cad', 'cad_date', 'bl2cad_yrs']], how = 'left', on =['eid'])
outcomes_df = pd.merge(outcomes_df, stroke_df[['eid', 'stroke', 'stroke_date', 'bl2stroke_yrs']], how = 'left', on =['eid'])
outcomes_df = pd.merge(outcomes_df, hf_df[['eid', 'hf', 'hf_date', 'bl2hf_yrs']], how = 'left', on =['eid'])
outcomes_df = pd.merge(outcomes_df, af_df[['eid', 'af', 'af_date', 'bl2af_yrs']], how = 'left', on =['eid'])
outcomes_df = pd.merge(outcomes_df, va_df[['eid', 'va', 'va_date', 'bl2va_yrs']], how = 'left', on =['eid'])
outcomes_df = pd.merge(outcomes_df, pad_df[['eid', 'pad', 'pad_date', 'bl2pad_yrs']], how = 'left', on =['eid'])
outcomes_df = pd.merge(outcomes_df, aaa_df[['eid', 'aaa', 'aaa_date', 'bl2aaa_yrs']], how = 'left', on =['eid'])
outcomes_df = pd.merge(outcomes_df, vt_df[['eid', 'vt', 'vt_date', 'bl2vt_yrs']], how = 'left', on =['eid'])
outcomes_df = pd.merge(outcomes_df, cvd_death_df[['eid', 'cvd_death', 'cvd_death_date', 'bl2cvd_death_yrs']], how = 'left', on =['eid'])
outcomes_df = pd.merge(outcomes_df, ar_df[['eid', 'ar', 'ar_date', 'bl2ar_yrs']], how = 'left', on =['eid'])
outcomes_df = pd.merge(outcomes_df, mace_df[['eid', 'mace', 'mace_date', 'bl2mace_yrs']], how = 'left', on =['eid'])
outcomes_df = pd.merge(outcomes_df, cvd_df[['eid', 'cvd', 'cvd_date', 'bl2cvd_yrs']], how = 'left', on =['eid'])
outcomes_df = pd.merge(outcomes_df, cved_df[['eid', 'cved', 'cved_date', 'bl2cved_yrs']], how = 'left', on =['eid'])

# Save the outcomes
outcomes_df.to_csv(output_dir + 'Outcomes.csv', index=False)

# Record end time
end_time = time.time()
total_time = end_time - start_time
logger.info(f'Total time: {total_time:.3f} seconds')


'''
Dictionary for diagnoses and operations:

- Data-Coding 87: Description: ICD9 - WHO International Classification of Diseases
- Data-Coding 19: Description: ICD10 - WHO International Classification of Diseases
- Data-Coding 240: Description: OPCS4 codes used to specify medical procedures and operations in Health Episode Statistics records
'''