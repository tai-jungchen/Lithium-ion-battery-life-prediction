"""
Author: Alex (Tai-Jung) Chen

This python script loads the battery data from .mat to .pkl. After loading, the .pkl file will be used by the
downstream file, data_preprocessing.py.

Reference code: https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation
"""
import h5py
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pickle


def data_loader(batch_date):
    matFilename = f'Data/{batch_date}_batchdata_updated_struct_errorcorrect.mat'
    if batch_date == "2017-05-12":
        batch_label = "1"
        batch_key = "b1c"
    elif batch_date == "2017-06-30":
        batch_label = "2"
        batch_key = "b2c"
    elif batch_date == "2018-04-12":
        batch_label = "3"
        batch_key = "b3c"
    else:
        raise Exception("Invalid batch date.")

    f = h5py.File(matFilename)
    batch = f['batch']

    num_cells = batch['summary'].shape[0]
    bat_dict = {}
    for i in range(num_cells):
        cl = f[batch['cycle_life'][i, 0]][:]
        policy = f[batch['policy_readable'][i, 0]][:].tobytes()[::2].decode()
        summary_IR = np.hstack(f[batch['summary'][i, 0]]['IR'][0, :].tolist())
        summary_QC = np.hstack(f[batch['summary'][i, 0]]['QCharge'][0, :].tolist())
        summary_QD = np.hstack(f[batch['summary'][i, 0]]['QDischarge'][0, :].tolist())
        summary_TA = np.hstack(f[batch['summary'][i, 0]]['Tavg'][0, :].tolist())
        summary_TM = np.hstack(f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())
        summary_TX = np.hstack(f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())
        summary_CT = np.hstack(f[batch['summary'][i, 0]]['chargetime'][0, :].tolist())
        summary_CY = np.hstack(f[batch['summary'][i, 0]]['cycle'][0, :].tolist())
        summary = {'IR': summary_IR, 'QC': summary_QC, 'QD': summary_QD, 'Tavg':
            summary_TA, 'Tmin': summary_TM, 'Tmax': summary_TX, 'chargetime': summary_CT,
                   'cycle': summary_CY}
        cycles = f[batch['cycles'][i, 0]]
        cycle_dict = {}
        for j in range(cycles['I'].shape[0]):
            I = np.hstack((f[cycles['I'][j, 0]][:]))
            Qc = np.hstack((f[cycles['Qc'][j, 0]][:]))
            Qd = np.hstack((f[cycles['Qd'][j, 0]][:]))
            Qdlin = np.hstack((f[cycles['Qdlin'][j, 0]][:]))
            T = np.hstack((f[cycles['T'][j, 0]][:]))
            Tdlin = np.hstack((f[cycles['Tdlin'][j, 0]][:]))
            V = np.hstack((f[cycles['V'][j, 0]][:]))
            dQdV = np.hstack((f[cycles['discharge_dQdV'][j, 0]][:]))
            t = np.hstack((f[cycles['t'][j, 0]][:]))
            cd = {'I': I, 'Qc': Qc, 'Qd': Qd, 'Qdlin': Qdlin, 'T': T, 'Tdlin': Tdlin, 'V': V, 'dQdV': dQdV, 't': t}
            cycle_dict[str(j)] = cd

        cell_dict = {'cycle_life': cl, 'charge_policy': policy, 'summary': summary, 'cycles': cycle_dict}
        key = batch_key + str(i)
        bat_dict[key] = cell_dict

    # check code
    # plt.plot(bat_dict[f'{batch_key}43']['summary']['cycle'], bat_dict[f'{batch_key}43']['summary']['QD'])
    # plt.show()
    # plt.plot(bat_dict['b1c43']['cycles']['10']['Qd'], bat_dict['b1c43']['cycles']['10']['V'])

    output_name = f"batch{batch_label}.pkl"
    with open(output_name, 'wb') as fp:
        pickle.dump(bat_dict, fp)


if __name__ == '__main__':
    # data_loader("2017-05-12")
    # data_loader("2017-06-30")
    # data_loader("2018-04-12")
    data_loader("2025-02-01")