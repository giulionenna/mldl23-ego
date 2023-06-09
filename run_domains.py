from train_classifier_multimodal import main_train as multimodal_train
from train_classifier_midFusion import main_train as midFusion_train

def main():
    temporal_type = 'Baseline'
    ablation = {'gsd': False, 'gtd': False, 'grd': False, 'domainA': False}
    weights = {'gamma': 0, 'l_s': 1, 'l_r': 1, 'l_t': 1}
    domains = ['D1', 'D2','D3']

    for i in domains:
        for j in domains:
            if (i != j):
                shift = [i,j];
                _, score = midFusion_train(temporal_type, ablation, weights, shift)


    print(weights)

    print(score)

if __name__ == '__main__':
    main()