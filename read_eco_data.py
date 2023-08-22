import pandas as pd
from datetime import datetime

def conv_to_datetime(series):
    for lst in series:
        for t_e in lst:
            t_e[0] = pd.to_datetime(t_e[0], unit='ms')

def remove_esg_after_2023(df): 
    filtered = df.drop(df[df['esg'].apply(lambda x: x[0][0] > pd.to_datetime('01-01-2023'))].index)
    return filtered

def get_score(new_df):
    diff_array = []

    esg = new_df['esg']
    esg_ind = new_df['esg_industry']

    for i in range(len(new_df['esg'])): 
        s = 0; 
        ind_avg = 0; 

        for j in range(len(esg[i])):
            t_esg_com = esg[i][j]
            t_esg_ind = esg_ind[i][j]

            t = t_esg_com[0]

            if(t > pd.to_datetime('01-01-2023')):
                break
            
            ind_avg += t_esg_ind[1]; 
            s += (t_esg_com[1] - t_esg_ind[1])

        s /= ind_avg

        diff_array.append(s); 
        
    new_df['eco_score'] = diff_array

def preprocess(filepath): 
    df = pd.read_json('./data.json')

    conv_to_datetime(df['esg'])
    conv_to_datetime(df['esg_industry'])

    new_df = df.drop(df[df['esg'].apply(lambda x: len(x) == 0)].index)
    new_df = remove_esg_after_2023(new_df)
    new_df = new_df.reset_index()
    new_df.drop(columns=['index'], inplace=True)

    get_score(new_df) 

    new_df = new_df[new_df['eco_score'] > 0]
    new_df = new_df.reset_index()
    new_df.drop(['index', 'esg', 'esg_industry', 'industry', 'name'], axis = 1, inplace = True)

    return new_df