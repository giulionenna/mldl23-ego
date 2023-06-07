from train_classifier_multimodal import main_train

def main():
    temporal_type = 'TRN'
    ablation = {'gsd': True, 'gtd': True, 'grd': True, 'domainA': False}
    weights = {'gamma': 0, 'l_s': 1, 'l_r': 1, 'l_t': 1}
    shift = ['D1', 'D2']

    best_score = 0

    gamma_grid = [0, 0.05, 0.1, 0.3]
    for gamma in gamma_grid:
        weights['gamma'] = gamma
        loss, score = main_train(temporal_type, ablation, weights, shift)
        if score>=best_score:
            best_score = score
            best_gamma = gamma
    weights['gamma'] = best_gamma

    l_grid = [0, 0.5, 0.75, 1]
    for loss in ['l_s', 'l_r', 'l_t']:
        best_score = 0
        for weight in l_grid:
            weights[loss] = weight
            l, score = main_train(temporal_type, ablation, weights, shift)
            if score>=best_score:
                best_score = score
                best_weight = weight
        weights[loss] = best_weight

    print(weights)

    print(score)

if __name__ == '__main__':
    main()