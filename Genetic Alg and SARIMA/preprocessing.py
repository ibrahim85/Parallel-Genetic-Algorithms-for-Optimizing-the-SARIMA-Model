"""
Usage:

1. As a module

    from preprocessing import preprocess
    preprocess('input_file.txt')

2. As a command line tool

Specify the name of the preprocessed file

    python3 preprocessing.py input_file.txt preprocessed.csv
    
or derive the name from the input file
    
    python3 preprocessing.py input_file.txt
    (This generates input_file_preprocessed.csv)

"""

import pandas as pd

def preprocess(filename):
    """Preprocess NCDC weather data"""
    
    fields = ['STN', 'WBAN', 'YEARMODA', 'TEMP', 'TEMP_count', 'DEWP', 'DEWP_count', 'SLP', 'SLP_count', 'STP', 'STP_count', 'VISIB', 'VISIB_count', 'WDSP', 'WDSP_count', 'MXSPD', 'GUST', 'MAX', 'MIN', 'PRCP', 'SNDP', 'FRSHTT']
    
    df = pd.read_csv(filename, 
                  sep=r'\s+', 
                  names=fields, 
                  header=0, 
                  parse_dates=['YEARMODA'], 
                  na_values={'TEMP':[9999.9], 
                             'DEWP':[9999.9], 
                             'SLP':[9999.9], 
                             'STP':[9999.9], 
                             'VISIB':[999.9], 
                             'WDSP':[999.9], 
                             'MXSPD':[999.9], 
                             'GUST':[999.9], 
                             'MAX':['9999.9'], # doesn't matter whether float or str
                             'MIN':['9999.9'], 
                             'PRCP':['99.99'],
                             'SNDP':['999.9']}
                 )
    
    flagged = df.copy()

    def strip_flag(x):
        if type(x) is float:
            return x
        elif type(x) is str:
            return float(x[:-1]) if '*' in x else float(x)
    def extract_flag(x):
        if type(x) is float:
            return False
        elif type(x) is str:
            return True if '*' in x else False
    
    flagged['MAX'] = df['MAX'].map(strip_flag)
    flagged['MAX_flag'] = df['MAX'].map(extract_flag)
    flagged['MIN'] = df['MIN'].map(strip_flag)
    flagged['MIN_flag'] = df['MIN'].map(extract_flag)
    
    flagged['PRCP'] = df['PRCP'].map(lambda x: float(x[:-1]) if type(x) is str else x)
    PRCP_flag = df['PRCP'].map(lambda x: x[-1] if type(x) is str else x)
    PRCP_dummies = pd.get_dummies(PRCP_flag).add_prefix('PRCP_')
    preprocessed = flagged.join(PRCP_dummies)
    
    return preprocessed

if __name__ == '__main__':
    
    from sys import argv
    from os.path import basename

    if len(argv) == 3:
        preprocessed = preprocess(argv[1])
        preprocessed.to_csv(argv[2])
    elif len(argv) == 2:
        preprocessed = preprocess(argv[1])
        filename = basename(argv[1])
        preprocessed.to_csv(filename.split('.')[0] + '_preprocessed.csv', index=False)
    else:
        raise Exception('Not correct number of arguments')