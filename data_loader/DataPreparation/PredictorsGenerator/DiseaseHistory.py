import os
import time
import logging
import gc
import pandas as pd
import numpy as np

# self_diag: extract the self-reported diagnoses
def self_diag(diag_df, diag_list, field, outcome):
    '''
    diag_df is the dataframe with only diagnoses information;
    diag_list is the list of interested diagnoses.
    '''
    eid_list = diag_df[diag_df.iloc[:, 1:].apply(lambda x: x.isin(diag_list).any(), axis=1)]['eid'].tolist()
    diag_df[f'{outcome}_{field}'] = diag_df['eid'].apply(lambda x: 1 if x in eid_list else 0)
    diag_df = diag_df[['eid', f'{outcome}_{field}']]
    
    return diag_df

# diag_min_dates: extract the minimum date of the diagnoses
def diag_min_dates(diag_df, date_df, diag_list, rename):
    '''
    diag_df is the dataframe with only diagnoses information;
    date_df is the dataframe with the corresponding dates information for the diagnoses;
    diag_list is the list of interested diagnoses.
    '''
    eid_list = diag_df[diag_df.iloc[:, 1:].apply(lambda x: x.isin(diag_list).any(), axis=1)]['eid'].tolist()
    sub_diag_df = diag_df[diag_df['eid'].isin(eid_list)]
    
    result_dict = {}
    for idx, row in sub_diag_df.iterrows():
        eid = row['eid']
        codes = row[1:].values 
        positions = [i for i, code in enumerate(codes) if code in diag_list]
        if positions:
            result_dict[eid] = positions
            
    min_dates = []
    for eid, positions in result_dict.items():
        row = date_df[date_df['eid'] == eid]
        dates = row.iloc[0, [pos + 1 for pos in positions]].dropna().tolist()
        dates = pd.to_datetime(dates)
        if not dates.empty:
            min_date = dates.min()
            min_dates.append((eid, min_date))
    min_dates_df = pd.DataFrame(min_dates, columns=['eid', 'min_date'])
    min_dates_df.rename(columns = {'min_date': rename}, inplace=True)
    
    return min_dates_df

# get_date_interval: calculate the interval between two dates
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
    
# Define the path
data_dir = '/home/ukb/data/phenotype_data/'
resources_dir = '/home/ukb/data/resources/'
output_dir = '/your path/multiomics-cardiovascular-disease/data/processed/covariates/'
log_dir = '/your path/multiomics-cardiovascular-disease/saved/log/DataPreparation/'

# Set up logger
log_filename = os.path.join(log_dir, 'DiseaseHistory.log')
logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

start_time = time.time()

# Read data
logger.info('Reading data...')
population_char_df = pd.read_csv(data_dir + 'population_char.csv', low_memory=False)
health_outcomes_df = pd.read_csv(data_dir + 'health_outcomes.csv', low_memory=False)
medication_df = pd.read_csv(data_dir + 'medication.csv', low_memory=False)
pc_df = population_char_df[['eid', '53-0.0']] # Date of attending assessment centre
pc_df.rename(columns = {'53-0.0': 'bl_date'}, inplace=True)

del population_char_df
gc.collect()

# List of the interested diagnoses
logger.info('Loading list of the interested diagnoses...')
## Diabetes
diab_20002_list = [1220, 1221, 1222, 1223]
diab_icd10_list = ['E10', 'E100', 'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E107', 'E108', 'E109', 
                   'E11', 'E110', 'E111', 'E112', 'E113', 'E114', 'E115', 'E116', 'E117', 'E118', 'E119', 
                   'E12', 'E120', 'E121', 'E122', 'E123', 'E124', 'E125', 'E126', 'E127', 'E128', 'E129', 
                   'E13', 'E130', 'E131', 'E132', 'E133', 'E134', 'E135', 'E136', 'E137', 'E138', 'E139',
                   'E14', 'E140', 'E141', 'E142', 'E143', 'E144', 'E145', 'E146', 'E147', 'E148', 'E149']
diab_icd9_list = ['250', '2500', '25000', '25001', '25009', '2501', '25010', '25011', '25019', 
                  '2502', '25020', '25021', '25029', '2503', '2504', '2505', '2506', '2507', '2509', '25090', '25091', '25099']

## Hypertension
hypt_20002_list = [1065, 1072]
hypt_icd10_list = ['I10', 'I11', 'I110', 'I119', 'I12', 'I120', 'I129', 'I13', 'I130', 'I131', 'I132', 'I139', 'I15', 'I150', 'I151', 'I152', 'I158', 'I159']
hypt_icd9_list = ['401', '4010', '4011', '4019', '402', '4020', '4021', '4029', '403', '4030', '4031', '4039', '404', '4040', '4041', '4049', '405', '4050', '4051', '4059']

## Coronary artery disease
cad_20002_list = [1066, 1075]
cad_20004_list = [1070, 1095, 1523]
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
stroke_20002_list = [1081, 1086, 1491, 1583]
stroke_icd10_list = ['I60', 'I600', 'I601', 'I602', 'I603', 'I604', 'I605', 'I606', 'I607', 'I608', 'I609',
                     'I61', 'I610', 'I611', 'I612', 'I613', 'I614', 'I615', 'I616', 'I618', 'I619',
                     'I62', 'I620', 'I621', 'I629', 
                     'I63', 'I630', 'I631', 'I632', 'I633', 'I634', 'I635', 'I636', 'I638', 'I639', 'I64']
stroke_icd9_list = ['430', '4309', '431', '4319', '434', '4340', '4341', '4349', '436', '4369']

## Heart Failure
hf_20002_list = [1076, 1079]
hf_icd10_list = ['I110', 'I130', 'I132', 'I255', 
                 'I420', 'I421', 'I422', 'I425', 'I426', 'I427', 'I428', 'I429',
                 'I50', 'I500', 'I501', 'I509']
hf_icd9_list = ['4254', '428', '4280', '4281', '4289']

## Atrial fibrillation
af_20002_list = [1471, 1483]
af_20004_list = [1524]
af_icd10_list = ['I48', 'I480', 'I481', 'I482', 'I483', 'I484', 'I489']
af_icd9_list = ['4273']
af_opcs_list = ['K571', 'K621', 'K622', 'K623', 'K624', 'X501', 'X502']

## Ventricular Arrhythmias
va_icd10_list = ['I46', 'I460', 'I461', 'I469', 'I470', 'I472', 'I490']
va_opcs_list = ['K571', 'K641', 'X503', 'X504', 'X508', 'X509']

## Peripheral artery disease
pad_20002_list = [1067, 1087, 1088]
pad_20004_list = [1102, 1108]
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

## Venous thromboembolism
vt_20002_list = [1068, 1093, 1094]
vt_icd10_list = ['I26', 'I260', 'I269', 'I80', 'I800', 'I801', 'I802', 'I803', 'I808', 'I809', 'I81', 'I820', 'I822', 'I823', 'I828', 'I829']
vt_icd9_list = ['4151', '451', '4510', '4511', '4512', '4518', '4519', '4532', '4538']
vt_opcs_list = ['L791', 'L902']


######################################################################################################
########################### Self-reported conditions and operations  #################################
######################################################################################################
logger.info('Defining disease history using self-reported conditions and operations...')
# Non-cancer illness code, self-reported
diag_df = health_outcomes_df[['eid'] + ['20002-0.' + str(i) for i in range(34)]]

diab_20002_df = self_diag(diag_df, diab_20002_list, 20002, 'diab')
hypt_20002_df = self_diag(diag_df, hypt_20002_list, 20002, 'hypt')
cad_20002_df = self_diag(diag_df, cad_20002_list, 20002, 'cad')
hf_20002_df = self_diag(diag_df, hf_20002_list, 20002, 'hf')
af_20002_df = self_diag(diag_df, af_20002_list, 20002, 'af')
stroke_20002_df = self_diag(diag_df, stroke_20002_list, 20002, 'stroke')
pad_20002_df = self_diag(diag_df, pad_20002_list, 20002, 'pad')
vt_20002_df = self_diag(diag_df, vt_20002_list, 20002, 'vt')
del diag_df
gc.collect()

# Operation code, self-reported
diag_df = health_outcomes_df[['eid'] + ['20004-0.' + str(i) for i in range(32)]]

cad_20004_df = self_diag(diag_df, cad_20004_list, 20004, 'cad')
af_20004_df = self_diag(diag_df, af_20004_list, 20004, 'af')
pad_20004_df = self_diag(diag_df, pad_20004_list, 20004, 'pad')
del diag_df
gc.collect()

# Others
diab_reported_df = pd.merge(diab_20002_df, health_outcomes_df[['eid', '2443-0.0', '2986-0.0']], how = 'left', on =['eid'])
diab_reported_df = pd.merge(diab_reported_df, medication_df[['eid', '6153-0.0', '6177-0.0']], how = 'left', on =['eid'])
diab_reported_df['diab_self_report'] = ((diab_reported_df['diab_20002'] == 1) |
                                        (diab_reported_df['2443-0.0'] == 1) |
                                        (diab_reported_df['2986-0.0'] == 1) |
                                        (diab_reported_df['6153-0.0'] == 3) |
                                        (diab_reported_df['6177-0.0'] == 3)).astype(int)

hypt_reported_df = pd.merge(hypt_20002_df, health_outcomes_df[['eid', '6150-0.0']], how = 'left', on =['eid'])
hypt_reported_df['hypt_self_report'] = ((hypt_reported_df['hypt_20002'] == 1) | (hypt_reported_df['6150-0.0'] == 4)).astype(int)

cad_reported_df = pd.merge(cad_20002_df, cad_20004_df, how = 'left', on =['eid'])
cad_reported_df = pd.merge(cad_reported_df, health_outcomes_df[['eid', '6150-0.0']], how = 'left', on =['eid'])
cad_reported_df['cad_self_report'] = ((cad_reported_df['cad_20002'] == 1) |
                                      (cad_reported_df['cad_20004'] == 1) |
                                      (cad_reported_df['6150-0.0'] == 1)).astype(int)

stroke_reported_df = pd.merge(stroke_20002_df, health_outcomes_df[['eid', '6150-0.0']], how = 'left', on =['eid'])
stroke_reported_df['stroke_self_report'] = ((stroke_reported_df['stroke_20002'] == 1) | (stroke_reported_df['6150-0.0'] == 3)).astype(int)

hf_reported_df = hf_20002_df.copy()
hf_reported_df.rename(columns = {'hf_20002': 'hf_self_report'}, inplace=True)

af_reported_df = pd.merge(af_20002_df, af_20004_df, how = 'left', on =['eid'])
af_reported_df['af_self_report'] = ((af_reported_df['af_20002'] == 1) | (af_reported_df['af_20004'] == 1)).astype(int)

pad_reported_df = pd.merge(pad_20002_df, pad_20004_df, how = 'left', on =['eid'])
pad_reported_df['pad_self_report'] = ((pad_reported_df['pad_20002'] == 1) | (pad_reported_df['pad_20004'] == 1)).astype(int)

vt_reported_df = pd.merge(vt_20002_df, health_outcomes_df[['eid', '6152-0.0']], how = 'left', on =['eid'])
vt_reported_df['vt_self_report'] = ((vt_reported_df['vt_20002'] == 1) |
                                    (vt_reported_df['6152-0.0'] == 5) |
                                    (vt_reported_df['6152-0.0'] == 7)).astype(int)

del diab_20002_df, hypt_20002_df, cad_20002_df, hf_20002_df, af_20002_df, stroke_20002_df, pad_20002_df, vt_20002_df
del cad_20004_df, af_20004_df, pad_20004_df
gc.collect()

######################################################################################################
########################### ICD-9/ICD-10 + OPCS  #####################################################
######################################################################################################
logger.info('Defining disease history using ICD and OPCS codes...')
# Diagnoses - ICD10
diag_df = health_outcomes_df[['eid'] + ['41270-0.' + str(i) for i in range(259)]]
date_df = health_outcomes_df[['eid'] + ['41280-0.' + str(i) for i in range(259)]]

diab_icd10_df = diag_min_dates(diag_df, date_df, diab_icd10_list, 'diab_icd10_date')
hypt_icd10_df = diag_min_dates(diag_df, date_df, hypt_icd10_list, 'hypt_icd10_date')
cad_icd10_df = diag_min_dates(diag_df, date_df, cad_icd10_list, 'cad_icd10_date')
stroke_icd10_df = diag_min_dates(diag_df, date_df, stroke_icd10_list, 'stroke_icd10_date')
hf_icd10_df = diag_min_dates(diag_df, date_df, hf_icd10_list, 'hf_icd10_date')
af_icd10_df = diag_min_dates(diag_df, date_df, af_icd10_list, 'af_icd10_date')
va_icd10_df = diag_min_dates(diag_df, date_df, va_icd10_list, 'va_icd10_date')
pad_icd10_df = diag_min_dates(diag_df, date_df, pad_icd10_list, 'pad_icd10_date')
aaa_icd10_df = diag_min_dates(diag_df, date_df, aaa_icd10_list, 'aaa_icd10_date')
vt_icd10_df = diag_min_dates(diag_df, date_df, vt_icd10_list, 'vt_icd10_date')
del diag_df, date_df
gc.collect()

# Diagnosis - ICD9
diag_df = health_outcomes_df[['eid'] + ['41271-0.' + str(i) for i in range(47)]]
date_df = health_outcomes_df[['eid'] + ['41281-0.' + str(i) for i in range(47)]]

diab_icd9_df = diag_min_dates(diag_df, date_df, diab_icd9_list, 'diab_icd9_date')
hypt_icd9_df = diag_min_dates(diag_df, date_df, hypt_icd9_list, 'hypt_icd9_date')
cad_icd9_df = diag_min_dates(diag_df, date_df, cad_icd9_list, 'cad_icd9_date')
stroke_icd9_df = diag_min_dates(diag_df, date_df, stroke_icd9_list, 'stroke_icd9_date')
hf_icd9_df = diag_min_dates(diag_df, date_df, hf_icd9_list, 'hf_icd9_date')
af_icd9_df = diag_min_dates(diag_df, date_df, af_icd9_list, 'af_icd9_date')
pad_icd9_df = diag_min_dates(diag_df, date_df, pad_icd9_list, 'pad_icd9_date')
aaa_icd9_df = diag_min_dates(diag_df, date_df, aaa_icd9_list, 'aaa_icd9_date')
vt_icd9_df = diag_min_dates(diag_df, date_df, vt_icd9_list, 'vt_icd9_date')
del diag_df, date_df
gc.collect()

# Operative procedures - OPCS
diag_df = health_outcomes_df[['eid'] + ['41272-0.' + str(i) for i in range(126)]]
date_df = health_outcomes_df[['eid'] + ['41282-0.' + str(i) for i in range(126)]]

cad_opcs_df = diag_min_dates(diag_df, date_df, cad_opcs_list, 'cad_opcs_date')
af_opcs_df = diag_min_dates(diag_df, date_df, af_opcs_list, 'af_opcs_date')
va_opcs_df = diag_min_dates(diag_df, date_df, va_opcs_list, 'va_opcs_date')
pad_opcs_df = diag_min_dates(diag_df, date_df, pad_opcs_list, 'pad_opcs_date')
aaa_opcs_df = diag_min_dates(diag_df, date_df, aaa_opcs_list, 'aaa_opcs_date')
vt_opcs_df = diag_min_dates(diag_df, date_df, vt_opcs_list, 'vt_opcs_date')
del diag_df, date_df
gc.collect()

######################################################################################################
########################### Pool  ####################################################################
######################################################################################################
logger.info('Pooling the results...')
# Diabetes
diab_df = pd.merge(pc_df, diab_icd10_df, how = 'left', on =['eid'])
diab_df = pd.merge(diab_df, diab_icd9_df, how = 'left', on =['eid'])
diab_df = pd.merge(diab_df, diab_reported_df, how = 'left', on =['eid'])
diab_df['diab_date'] = diab_df[['diab_icd10_date', 'diab_icd9_date']].min(axis=1)
diab_df['bl2diab_yrs'] = get_date_interval('bl_date', 'diab_date', diab_df, 'year')
diab_df['diab_hist'] = ((diab_df['diab_self_report'] == 1) | (diab_df['bl2diab_yrs'] < 0)).astype(int)
del diab_icd10_df, diab_icd9_df, diab_reported_df
gc.collect()

# Hypertension
hypt_df = pd.merge(pc_df, hypt_icd10_df, how = 'left', on =['eid'])
hypt_df = pd.merge(hypt_df, hypt_icd9_df, how = 'left', on =['eid'])
hypt_df = pd.merge(hypt_df, hypt_reported_df, how = 'left', on =['eid'])
hypt_df['hypt_date'] = hypt_df[['hypt_icd10_date', 'hypt_icd9_date']].min(axis=1)
hypt_df['bl2hypt_yrs'] = get_date_interval('bl_date', 'hypt_date', hypt_df, 'year')
hypt_df['hypt_hist'] = ((hypt_df['hypt_self_report'] == 1) | (hypt_df['bl2hypt_yrs'] < 0)).astype(int)
del hypt_icd10_df, hypt_icd9_df, hypt_reported_df
gc.collect()

# Coronary artery disease
cad_df = pd.merge(pc_df, cad_icd10_df, how = 'left', on =['eid'])
cad_df = pd.merge(cad_df, cad_icd9_df, how = 'left', on =['eid'])
cad_df = pd.merge(cad_df, cad_opcs_df, how = 'left', on =['eid'])
cad_df = pd.merge(cad_df, cad_reported_df, how = 'left', on =['eid'])
cad_df['cad_date'] = cad_df[['cad_icd10_date', 'cad_icd9_date', 'cad_opcs_date']].min(axis=1)
cad_df['bl2cad_yrs'] = get_date_interval('bl_date', 'cad_date', cad_df, 'year')
cad_df['cad_hist'] = ((cad_df['cad_self_report'] == 1) | (cad_df['bl2cad_yrs'] < 0)).astype(int)
del cad_icd10_df, cad_icd9_df, cad_opcs_df, cad_reported_df
gc.collect()

# Stroke
stroke_df = pd.merge(pc_df, stroke_icd10_df, how = 'left', on =['eid'])
stroke_df = pd.merge(stroke_df, stroke_icd9_df, how = 'left', on =['eid'])
stroke_df = pd.merge(stroke_df, stroke_reported_df, how = 'left', on =['eid'])
stroke_df['stroke_date'] = stroke_df[['stroke_icd10_date', 'stroke_icd9_date']].min(axis=1)
stroke_df['bl2stroke_yrs'] = get_date_interval('bl_date', 'stroke_date', stroke_df, 'year')
stroke_df['stroke_hist'] = ((stroke_df['stroke_self_report'] == 1) | (stroke_df['bl2stroke_yrs'] < 0)).astype(int)
del stroke_icd10_df, stroke_icd9_df, stroke_reported_df
gc.collect()

# Heart failure
hf_df = pd.merge(pc_df, hf_icd10_df, how = 'left', on =['eid'])
hf_df = pd.merge(hf_df, hf_icd9_df, how = 'left', on =['eid'])
hf_df = pd.merge(hf_df, hf_reported_df, how = 'left', on =['eid'])
hf_df['hf_date'] = hf_df[['hf_icd10_date', 'hf_icd9_date']].min(axis=1)
hf_df['bl2hf_yrs'] = get_date_interval('bl_date', 'hf_date', hf_df, 'year')
hf_df['hf_hist'] = ((hf_df['hf_self_report'] == 1) | (hf_df['bl2hf_yrs'] < 0)).astype(int)
del hf_icd10_df, hf_icd9_df, hf_reported_df
gc.collect()

# Atrial fibrillation
af_df = pd.merge(pc_df, af_icd10_df, how = 'left', on =['eid'])
af_df = pd.merge(af_df, af_icd9_df, how = 'left', on =['eid'])
af_df = pd.merge(af_df, af_opcs_df, how = 'left', on =['eid'])
af_df = pd.merge(af_df, af_reported_df, how = 'left', on =['eid'])
af_df['af_date'] = af_df[['af_icd10_date', 'af_icd9_date', 'af_opcs_date']].min(axis=1)
af_df['bl2af_yrs'] = get_date_interval('bl_date', 'af_date', af_df, 'year')
af_df['af_hist'] = ((af_df['af_self_report'] == 1) | (af_df['bl2af_yrs'] < 0)).astype(int)
del af_icd10_df, af_icd9_df, af_opcs_df, af_reported_df
gc.collect()

# Ventricular arrhythmias
va_df = pd.merge(pc_df, va_icd10_df, how = 'left', on =['eid'])
va_df = pd.merge(va_df, va_opcs_df, how = 'left', on =['eid'])
va_df['va_date'] = va_df[['va_icd10_date', 'va_opcs_date']].min(axis=1)
va_df['bl2va_yrs'] = get_date_interval('bl_date', 'va_date', va_df, 'year')
va_df['va_hist'] = (va_df['bl2va_yrs'] < 0).astype(int)
del va_icd10_df, va_opcs_df
gc.collect()

# Peripheral artery disease
pad_df = pd.merge(pc_df, pad_icd10_df, how = 'left', on =['eid'])
pad_df = pd.merge(pad_df, pad_icd9_df, how = 'left', on =['eid'])
pad_df = pd.merge(pad_df, pad_opcs_df, how = 'left', on =['eid'])
pad_df = pd.merge(pad_df, pad_reported_df, how = 'left', on =['eid'])
pad_df['pad_date'] = pad_df[['pad_icd10_date', 'pad_icd9_date', 'pad_opcs_date']].min(axis=1)
pad_df['bl2pad_yrs'] = get_date_interval('bl_date', 'pad_date', pad_df, 'year')
pad_df['pad_hist'] = ((pad_df['pad_self_report'] == 1) | (pad_df['bl2pad_yrs'] < 0)).astype(int)
del pad_icd10_df, pad_icd9_df, pad_opcs_df, pad_reported_df
gc.collect()

# Abdominal aortic aneurysm
aaa_df = pd.merge(pc_df, aaa_icd10_df, how = 'left', on =['eid'])
aaa_df = pd.merge(aaa_df, aaa_icd9_df, how = 'left', on =['eid'])
aaa_df = pd.merge(aaa_df, aaa_opcs_df, how = 'left', on =['eid'])
aaa_df['aaa_date'] = aaa_df[['aaa_icd10_date', 'aaa_icd9_date', 'aaa_opcs_date']].min(axis=1)
aaa_df['bl2aaa_yrs'] = get_date_interval('bl_date', 'aaa_date', aaa_df, 'year')
aaa_df['aaa_hist'] = (aaa_df['bl2aaa_yrs'] < 0).astype(int)
del aaa_icd10_df, aaa_icd9_df, aaa_opcs_df
gc.collect()

# Venous thromboembolism
vt_df = pd.merge(pc_df, vt_icd10_df, how = 'left', on =['eid'])
vt_df = pd.merge(vt_df, vt_icd9_df, how = 'left', on =['eid'])
vt_df = pd.merge(vt_df, vt_opcs_df, how = 'left', on =['eid'])
vt_df = pd.merge(vt_df, vt_reported_df, how = 'left', on =['eid'])
vt_df['vt_date'] = vt_df[['vt_icd10_date', 'vt_icd9_date', 'vt_opcs_date']].min(axis=1)
vt_df['bl2vt_yrs'] = get_date_interval('bl_date', 'vt_date', vt_df, 'year')
vt_df['vt_hist'] = ((vt_df['vt_self_report'] == 1) | (vt_df['bl2vt_yrs'] < 0)).astype(int)
del vt_icd10_df, vt_icd9_df, vt_opcs_df, vt_reported_df
gc.collect()

# Pool all outcomes
feature_df = pd.merge(pc_df['eid'], diab_df[['eid', 'diab_hist']], how = 'left', on =['eid'])
feature_df = pd.merge(feature_df, hypt_df[['eid', 'hypt_hist']], how = 'left', on =['eid'])
feature_df = pd.merge(feature_df, cad_df[['eid', 'cad_hist']], how = 'left', on =['eid'])
feature_df = pd.merge(feature_df, stroke_df[['eid', 'stroke_hist']], how = 'left', on =['eid'])
feature_df = pd.merge(feature_df, hf_df[['eid', 'hf_hist']], how = 'left', on =['eid'])
feature_df = pd.merge(feature_df, af_df[['eid', 'af_hist']], how = 'left', on =['eid'])
feature_df = pd.merge(feature_df, va_df[['eid', 'va_hist']], how = 'left', on =['eid'])
feature_df = pd.merge(feature_df, pad_df[['eid', 'pad_hist']], how = 'left', on =['eid'])
feature_df = pd.merge(feature_df, aaa_df[['eid', 'aaa_hist']], how = 'left', on =['eid'])
feature_df = pd.merge(feature_df, vt_df[['eid', 'vt_hist']], how = 'left', on =['eid'])
feature_df['cvd_hist'] = ((feature_df['cad_hist'] == 1) |
                          (feature_df['stroke_hist'] == 1) |
                          (feature_df['hf_hist'] == 1) |
                          (feature_df['af_hist'] == 1) |
                          (feature_df['va_hist'] == 1) |
                          (feature_df['pad_hist'] == 1) | 
                          (feature_df['aaa_hist'] == 1) |
                          (feature_df['vt_hist'] == 1)).astype(int)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
logger.info(feature_df.describe())
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

# Save the data
feature_df.to_csv(output_dir + 'DiseaseHistory.csv', index=False)

# Record end time
end_time = time.time()
total_time = end_time - start_time
logger.info(f'Total time: {total_time:.3f} seconds')