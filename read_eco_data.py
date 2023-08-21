import pandas as pd
from datetime import datetime

def conv_to_datetime(series):
    for lst in series:
        for t_e in lst:
            t_e[0] = pd.to_datetime(t_e[0], unit='ms')

def preprocess(filepath): 
    df = pd.read_json('./data.json')

    conv_to_datetime(df['esg'])
    conv_to_datetime(df['esg_industry'])

    new_df = df.drop(df[df['esg'].apply(lambda x: len(x) == 0)].index)

    new_df = new_df.reset_index()

    diff_array = []

    esg = new_df.esg
    esg_ind = new_df.esg_industry

    for i in range(len(esg)): 
        assert(len(esg[i]) == len(esg_ind[i])) 

        s = 0; 

        for j in range(len(esg[i])):
            t_esg_com = esg[i][j]
            t_esg_ind = esg_ind[i][j]

            t = t_esg_com[0]

            if(t > pd.to_datetime('01-01-2023')):
                break

            s += (t_esg_com[1] - t_esg_ind[1])/t_esg_ind[1]; 
        
        s /= len(esg[i]); 

        diff_array.append(s); 
        
    new_df['eco_score'] = diff_array 

    new_df.drop(['esg', 'esg_industry', 'industry', 'name'], axis = 1, inplace = True)

    return new_df