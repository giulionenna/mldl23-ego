from train_classifier_multimodal import main_train as multimodal_train
from train_classifier_midFusion import main_train as midFusion_train
import numpy as np
import pandas as pd
import torch
def main():
    np.random.seed(13696641)
    torch.manual_seed(13696641)
    ablation_list = [{'temporal_type': 'Baseline', 'ablation': {'gsd': False, 'gtd': False, 'grd': False, 'domainA': False}},#0
                     {'temporal_type': 'TRN', 'ablation': {'gsd': False, 'gtd': False, 'grd': False, 'domainA': False}},#1
                     {'temporal_type': 'Baseline', 'ablation': {'gsd': True, 'gtd': False, 'grd': False, 'domainA': False}},#2
                     {'temporal_type': 'TRN', 'ablation': {'gsd': True, 'gtd': False, 'grd': False, 'domainA': False}},#3
                     {'temporal_type': 'Baseline', 'ablation': {'gsd': False, 'gtd': True, 'grd': False, 'domainA': False}},#4
                     {'temporal_type': 'TRN', 'ablation': {'gsd': False, 'gtd': True, 'grd': False, 'domainA': False}},#5
                     {'temporal_type': 'TRN', 'ablation': {'gsd': False, 'gtd': False, 'grd': True, 'domainA': False}},#6
                     {'temporal_type': 'TRN', 'ablation': {'gsd': True, 'gtd': True, 'grd': True, 'domainA': False}},#7
                     {'temporal_type': 'TRN', 'ablation': {'gsd': True, 'gtd': True, 'grd': True, 'domainA': True}}]#8  
    
    col = ['abl',
             'D1-D2', 
             'D1-D3',
             'D2-D1',
             'D2-D3',
             'D3-D1',
             'D3-D2']

    #vec = [4,5]
    vec = [4,5]
    for i in vec:
        final_table = pd.DataFrame(columns=col)
        ablation_entry = ablation_list[i]
        temporal_type = ablation_entry['temporal_type']
        ablation = ablation_entry['ablation']
        weights = {'gamma': 0.3, 'l_s': 0.75, 'l_r': 0.5, 'l_t': 0.75}
        domains = ['D1', 'D2','D3']
        score = {}
        best_acc = {}
        config={'config': [temporal_type, ablation, weights]}
        for i in domains:
            for j in domains:
                if (i != j):
                    shift = [i,j];
                    _, s = multimodal_train(temporal_type, ablation, weights, shift)
                    score[i+'-'+j] = s['last']
                    best_acc[i+'-'+j] = s['best']

        new_row = {'abl': str(ablation_entry), 
                   'type':"top1",
                    'D1-D2': score['D1-D2'],
                    'D1-D3': score['D1-D3'],
                    'D2-D1': score['D2-D1'],
                    'D2-D3': score['D2-D3'],
                    'D3-D1': score['D3-D1'],
                    'D3-D2': score['D3-D2']}
        final_table = final_table.append(new_row, ignore_index=True)       
        new_row = {'abl': str(ablation_entry), 
                   'type':"best",
                    'D1-D2': best_acc['D1-D2'],
                    'D1-D3': best_acc['D1-D3'],
                    'D2-D1': best_acc['D2-D1'],
                    'D2-D3': best_acc['D2-D3'],
                    'D3-D1': best_acc['D3-D1'],
                    'D3-D2': best_acc['D3-D2']}
        final_table = final_table.append(new_row, ignore_index=True)
        run_name = "ta3n_noBN"    
        table_name =  "table_results/"+run_name+"_"+temporal_type+'_gsd_'+ str(ablation_entry['ablation']['gsd'])+ \
                                    '_gtd_'+str(ablation_entry['ablation']['gtd'])+'_grd_'\
                                     +str(ablation_entry['ablation']['grd'])+'domainA'+str(ablation_entry['ablation']['domainA']) \
                                     +".csv"
        final_table.to_csv(table_name)

if __name__ == '__main__':
    main()