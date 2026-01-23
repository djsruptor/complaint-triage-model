import pandas as pd
import numpy as np

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    df.rename(columns={'Consumer complaint narrative': 'Complaint text'}, inplace=True)

    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('?', '')
    
    df['high_priority'] = np.where(
        (df['Timely_response'] == 'No') | 
        (df['Consumer_disputed'] == 'Yes'), 
        1, 0
    )
    
    df.drop(columns=['Product', 'Sub-product', 'Issue', 'Sub-issue', 'Company_public_response', 'Company', 'State', 'ZIP_code', 'Tags', 'Consumer_consent_provided', 
                    'Submitted_via', 'Date_sent_to_company', 'Date_received', 'Company_response_to_consumer', 'Timely_response', 'Consumer_disputed'],
                    inplace=True)

    return df