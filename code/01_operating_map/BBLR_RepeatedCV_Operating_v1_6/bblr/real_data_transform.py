import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

def prepare_and_split_data(path, cat_features, num_features, target, test_size, split_random_state=42, 
                           drop_features=None, subset_size=None, subset_random_state=42, 
                           number_of_noise_cols_added=0, noise_random_state=42):
    # Upload dataset, take its subset and drop unnecessary features
    df = pd.read_csv(path)
    if subset_size is not None:
        df = df.sample(n=subset_size, random_state=subset_random_state).reset_index(drop=True)
    if drop_features is not None:
        df = df.drop(columns=drop_features)

    # choose only the specified features and target
    df = df[cat_features + num_features + [target]]
    
    # One-Hot Encoding and Standard Scaling 
    X = df.drop(columns=[target])
    y = df[target]
    
    # Adding noise columns if specified
    if number_of_noise_cols_added > 0:
        np.random.seed(noise_random_state)
        noise_data = np.random.normal(size=(X.shape[0], number_of_noise_cols_added))
        new_cols = [f'noise_{i+1}' for i in range(number_of_noise_cols_added)]
        noise_df = pd.DataFrame(noise_data, columns=new_cols)
        X = pd.concat([X, noise_df], axis=1)
        num_features += new_cols  # Adding noise columns to numerical features list for scaling
    
    # Splitting the data into training and test sets
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=split_random_state
    )

    # Preprocessing: One-Hot Encoding for categorical features and Standard Scaling for numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])

    # Scaling the data
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    return X_train, X_test, y_train, y_test