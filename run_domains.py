from train_classifier_multimodal import main_train as multimodal_train
from train_classifier_midFusion import main_train as midFusion_train
import numpy as np
import pandas as pd
import torch
def main():
    np.random.seed(13696641)
    torch.manual_seed(13696641)
    ablation_list = [{'temporal_type': 'Baseline', 'ablation': {'gsd': False, 'gtd': False, 'grd': False, 'domainA': False}},
                     {'temporal_type': 'TRN', 'ablation': {'gsd': False, 'gtd': False, 'grd': False, 'domainA': False}},
                     {'temporal_type': 'Baseline', 'ablation': {'gsd': True, 'gtd': False, 'grd': False, 'domainA': False}},
                     {'temporal_type': 'TRN', 'ablation': {'gsd': True, 'gtd': False, 'grd': False, 'domainA': False}},
                     {'temporal_type': 'Baseline', 'ablation': {'gsd': False, 'gtd': True, 'grd': False, 'domainA': False}},
                     {'temporal_type': 'TRN', 'ablation': {'gsd': False, 'gtd': True, 'grd': False, 'domainA': False}},
                     {'temporal_type': 'TRN', 'ablation': {'gsd': False, 'gtd': False, 'grd': True, 'domainA': False}},
                     {'temporal_type': 'TRN', 'ablation': {'gsd': True, 'gtd': True, 'grd': True, 'domainA': False}},
                     {'temporal_type': 'TRN', 'ablation': {'gsd': True, 'gtd': True, 'grd': True, 'domainA': True}}]  
    
    col = ['abl',
             'D1-D2', 
             'D1-D3',
             'D2-D1',
             'D2-D3',
             'D3-D1',
             'D3-D2']

    final_table = pd.DataFrame(columns=col)
    
    for ablation_entry in ablation_list:   
        temporal_type = ablation_entry['temporal_type']
        ablation = ablation_entry['ablation']
        weights = {'gamma': 0.3, 'l_s': 0.5, 'l_r': 1, 'l_t': 0.5}
        domains = ['D1', 'D2','D3']
        score = {}
        config={'config': [temporal_type, ablation, weights]}
        for i in domains:
            for j in domains:
                if (i != j):
                    shift = [i,j];
                    _, s = multimodal_train(temporal_type, ablation, weights, shift)
                    score[i+'-'+j] = s['last']
        
        new_row = {'abl': str(ablation_entry), 
                    'D1-D2': score['D1-D2'],
                    'D1-D3': score['D1-D3'],
                    'D2-D1': score['D2-D1'],
                    'D2-D3': score['D2-D3'],
                    'D3-D1': score['D3-D1'],
                    'D3-D2': score['D3-D2']}
        final_table = final_table.append(new_row, ignore_index=True)        

        final_dict = {'score': score, 'config': config}

        with open('results.txt', 'w') as f:
            print(final_dict, file=f)
        

    print(weights)

    print(score)
    final_table.to_csv('final_table.csv')

if __name__ == '__main__':
    main()