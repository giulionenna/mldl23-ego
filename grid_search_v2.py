from train_classifier_multimodal import main_train
import torch
import numpy as np

def main():
    
    temporal_type = 'TRN'
    ablation = {'gsd': True, 'gtd': True, 'grd': True, 'domainA': False}
    weights = {'gamma': 0, 'l_s': 0, 'l_r': 0, 'l_t': 0}
    shifts = ['D1', 'D2', 'D3']
    shift_vec = [['D1', 'D2'], 
                 ['D1', 'D3'],
                 ['D2', 'D1'],
                 ['D2', 'D3'],
                 ['D3', 'D1'],
                 ['D3', 'D2']]
    shift_vec = [#['D1', 'D2'], 
                 ['D1', 'D3'],
                 ['D2', 'D1'],
                 #['D2', 'D3'],
                 ['D3', 'D1'],
                 #['D3', 'D2']
                 ]

    best_acc = 0

    f = open('grid_results.txt', 'a')
    f.write("\n\n\n #################################### \n")
    f.write(" \t\t NEW GRID SEARCH RUN \n")
    f.write(" #################################### \n\n")

    l_grid = [0.5, 0.75, 1]
    for loss in ['l_s', 'l_r', 'l_t']:
        best_acc = 0 
        for weight in l_grid:
            f.write('##### Testing '+ loss+ ' = '+str(weight)+ '\n')
            acc = 0
            weights[loss] = weight
            for s in shift_vec:
                l, score = main_train(temporal_type, ablation, weights, s)
                log = s[0]+'-'+s[1]+ '\t best acc: ' + str(score['best'])+ '\t last acc: ' + str(score['last']) + '\n'
                f.write(log)
                acc += score['best']
            acc = acc/len(shift_vec)
            f.write('##### Avg acc for '+loss+' = '+str(weight)+' is: '+str(acc)+ '\n')
            if acc>=best_acc:
                best_acc = acc
                best_weight = weight
        weights[loss] = best_weight
        f.write('##### Best '+loss+': '+ str(best_weight)+'\n')


    #Gamma Search
    if(False):
        gamma_grid = [0, 0.1, 0.3, 1]
        f.write('##### Grid Search on parameter GAMMA \n')
        for gamma in gamma_grid:
            acc = 0
            weights['gamma'] = gamma
            f.write('##### Testing GAMMA = '+str(gamma)+ '\n')
            for s in shift_vec:
                loss, score = main_train(temporal_type, ablation, weights, s)
                log = s[0]+'-'+s[1]+ '\t best acc: ' + str(score['best'])+ '\t last acc: ' + str(score['last']) + '\n'
                f.write(log)
                acc += score['best']
            acc = acc/len(shift_vec)
            f.write('##### Avg acc for gamma  '+ str(gamma)+'is: '+str(acc)+' \n')
            if acc>=best_acc:
                best_acc = acc
                best_gamma = gamma
        weights['gamma'] = best_gamma
        f.write('##### Best gamma: '+ str(best_gamma)+ '\n')
    

    print(weights)

    print(score)
    f.close()

if __name__ == '__main__':
    main()