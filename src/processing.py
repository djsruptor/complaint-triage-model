import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def preprocess_data(file_path='data/raw_data.csv'):
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
                    'Submitted_via', 'Date_sent_to_company', 'Date_received', 'Company_response_to_consumer', 'Timely_response', 'Consumer_disputed', 'Complaint_ID'],
                    inplace=True)

    return df

def split_data(
        df, 
        target='high_priority', 
        text_col = 'Complaint_text', 
        test_size=0.2, val_size=0.25, 
        random_state=3
):
    df_full_train, df_test = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df[target], 
        random_state=random_state
    )

    df_train, df_val = train_test_split(
        df_full_train, 
        test_size=val_size, 
        stratify=df_full_train[target], 
        random_state=random_state
    )

    y_train = df_train[target].astype(int)
    y_val   = df_val[target].astype(int)
    y_test  = df_test[target].astype(int)

    X_train = df_train[text_col].astype(str)
    X_val   = df_val[text_col].astype(str)
    X_test  = df_test[text_col].astype(str)

    return X_train, X_val, X_test, y_train, y_val, y_test