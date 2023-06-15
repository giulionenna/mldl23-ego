from train_classifier_multimodal import main_train as multimodal_train
from train_classifier_midFusion import main_train as midFusion_train
import numpy as np
import torch
def main():
    np.random.seed(13696641)
    torch.manual_seed(13696641)
    
    
    temporal_type = 'Baseline'
    ablation = {'gsd': False, 'gtd': False, 'grd': False, 'domainA': False}
    weights = {'gamma': 0, 'l_s': 1, 'l_r': 1, 'l_t': 1}
    domains = ['D1', 'D2','D3']
    score = {}
    config={'config': [temporal_type, ablation, weights]}
    for i in domains:
        for j in domains:
            if (i != j):
                shift = [i,j];
                _, s = midFusion_train(temporal_type, ablation, weights, shift)
                score[i+'-'+j] = s

    final_dict = {'score': score, 'config': config}
    with open('results.txt', 'w') as f:
        print(final_dict, file=f)


    print(weights)

    print(score)

if __name__ == '__main__':
    main()