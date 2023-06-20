# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Enrique_Escalante-Notario
# Instituto de Fisica, UNAM
# email: <enriquescalante@gmail.com>
# Distributed under terms of the GPLv3 license.
# loss.py
# --------------------------------------------------------

# Adapted loss function, which takes chuncks of ohe vector 

def CustomLossFunction(data, output, criterion, lenghts_data):
    ini = 0
    train_loss = 0
    for ind in lenghts_data:
        curr_data = data[:][:,ini:ini+ind]
        curr_target = output[:][:,ini:ini+ind]
        train_loss += criterion(curr_target,curr_data)
        ini = ind + ini
    return train_loss