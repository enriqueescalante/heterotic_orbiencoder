# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Enrique_Escalante-Notario
# Instituto de Fisica, UNAM
# email: <enriquescalante@gmail.com>
# Distributed under terms of the GPLv3 license.
# data.py
# --------------------------------------------------------


from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

# Adapted function for data importation of principal dataset
class CustomDataset(Dataset):

    def __init__(self, file_name):
        ohe = OneHotEncoder(dtype=np.int8)
        file_out = pd.read_csv(file_name)

        data_tensor = pd.DataFrame(ohe.fit_transform(file_out).toarray())
        
        x = data_tensor.iloc[:,:].values
        y = file_out.iloc[:, -1].values

        self.X_train = torch.tensor(x,dtype=torch.float)
        self.y_train = torch.tensor(y)

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]
    
    
# Adapted function for data importation of alternative dataset
class CustomDatasetExtra(Dataset):
    def __init__(self, file_principal_name, file_alternative_name):
        ohe = OneHotEncoder(dtype=np.int8)
        file_principal = pd.read_csv(file_principal_name)
        file_alternati = pd.read_csv(file_alternative_name)
        trans = ohe.fit(file_principal)
        data_tensor = pd.DataFrame(trans.transform(file_alternati).toarray())
        
        x = data_tensor.iloc[:,:].values
        y = file_out.iloc[:, -1].values

        self.X_train = torch.tensor(x,dtype=torch.float)
        self.y_train = torch.tensor(y)

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]
    
    
# Adapted function to calculate one hot encoding lengths each features of dataset
def lenght_ohe(PATH):
    datos = pd.read_csv(PATH)
    lengths = []
    for column in datos.columns:
        lengths.append(datos[column].nunique())
    return sum(lengths)

# Adapted function to calculete  # same function

def lenghts_features(name_dataset):
    datos = pd.read_csv(name_dataset)
    lengths = []
    for column in datos.columns:
        lengths.append(datos[column].nunique())
    return lengths