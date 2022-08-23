from genericpath import exists
import pandas as pd
import os
import shutil
dir_all = '/home/PJLAB/niujingqi/Nifty_data'
i = 0
dataset_names = os.listdir(dir_all)
for dataset_name in dataset_names:
    dir = os.path.join(dir_all, dataset_name)
    dataset_name = 'TCIA-Covid-19' #'ACRIN-NSCLC-FDG-PET
    info_csv_file_dir = os.path.join(dir,'save-nifty-meta-info.csv' )
    try:
        info_df = pd.read_csv(info_csv_file_dir)

    except:
        continue
    if not os.path.exists(os.path.join(dir, 'CT')):
        os.mkdir(os.path.join(dir, 'CT'))
    CT_dir = os.path.join(dir, 'CT')
    if not os.path.exists(os.path.join(dir, 'MR')):
        os.mkdir(os.path.join(dir, 'MR'))
    MR_dir = os.path.join(dir, 'MR')
    for i, row in info_df.iterrows():
        if row['Modality (0008|0060)'] == 'MR':
            file_name = row['NIfty Save Name']
            if file_name.split('-')[0] =='hk':
                file_name_new = dataset_name + '-' + file_name.split('-')[1]
            else:
                file_name_new = file_name
            try:
                shutil.move(os.path.join(dir, file_name), os.path.join(MR_dir, file_name_new))
            except:
                print(file_name)
                i = i+1
        if row['Modality (0008|0060)'] == 'CT':
            file_name = row['NIfty Save Name']
            if file_name.split('-')[0] =='hk':
                file_name_new = dataset_name + '-' + file_name.split('-')[1]
            else:
                file_name_new = file_name
            try:
                shutil.move(os.path.join(dir, file_name), os.path.join(CT_dir, file_name_new))
            except:
                print(file_name)
                i = i+1
    print('the error number is', i)