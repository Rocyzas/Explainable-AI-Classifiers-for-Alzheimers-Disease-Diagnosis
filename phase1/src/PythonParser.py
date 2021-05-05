import argparse

def parserTrainARGV(argv):
    # Should test argv

    parser = argparse.ArgumentParser(description='Train the models, specifying multiple argv parameters')

    parser.add_argument('-clf', '--classifier', required = True, type=checkClassifiers,
        help="Which algorithm to train classifier (allowed <SVM> - support vector machines, <DT> - Gradient boosted decision trees, <LR> - logistic regression)")
    parser.add_argument('-cl', '--classification', required = True, type=checkClassification,
        help="Specify binary classification type: <HC_AD> - healty case(0) vs alzheimers disease(1), "\
            "<MCI_AD> - mild cognitive impairment(0) vs alzheimers disease(1),"\
            " <HC_MCI> - heatly case(0) vs mild cognitive impairment(1)")
    # parser.add_argument('-b', '--BSCV', required = True, type=str2bool,
    #     help="1 for Bayes Search CV - cross validation method for automatically selecting hyperparameters giving highest accuracy;"\
    #         " 0 for quicker, but less precise, already manually set hyperparameters")
    parser.add_argument('-s', '--save', required = True, type=str2bool,
        help="1 to save the models and feature importance by weights from that classifier (all which were specified in the same command line);"\
            " 0 not to save models")

    args = parser.parse_args()

    return args

def parserExplainARGV(argv):
    # Should test argv
    parser = argparse.ArgumentParser(description='Explain data rows, specifying multiple argv parameters')
    parser.add_argument('-clf', '--classifier', required = True, type=checkClassifiers,
        help="Which model to explain instances (allowed <SVM> - support vector machines, <DT> - Gradient boosted decision trees, <LR> - logistic regression)")
    parser.add_argument('-cl', '--classification', required = True, type=checkClassification,
        help="Specify binary classification type: <HC_AD> - healty case(0) vs alzheimers disease(1), "\
            "<MCI_AD> - mild cognitive impairment(0) vs alzheimers disease(1),"\
            " <HC_MCI> - heatly case(0) vs mild cognitive impairment(1)")

    parser.add_argument('-d', '--data', required = True,
        help="Path to the file of data rows which needs to be explained")
    # parser.add_argument('-m', '--models', required = True,
    #     help="Path to the model directory")

    args = parser.parse_args()

    return args

def checkClassifiers(value):
    if value.lower() in ['svm', 'svc']:
        return 'SVM'
    if value.lower() in ['lr']:
        return 'LR'
    if value.lower() in ['dt', 'gdt', 'gbdt']:
        return 'DT'
    if value.lower() in ['all', 'a', '*']:
        return 'ALL'
    else:
        raise argparse.ArgumentTypeError("Classifier should be either: 'DT', 'LR', 'SVM', 'ALL'")

def checkClassification(value):
    if value.lower() in ['hc_ad', 'ad_hc','hcad', 'adhc']:
        return 'HC_AD'
    if value.lower() in ['mci_ad', 'ad_mci','mciad', 'admci']:
        return 'MCI_AD'
    if value.lower() in ['hc_mci', 'mci_hc','mcihc', 'hcmci']:
        return 'HC_MCI'
    if value.lower() in ['all', 'a']:
        return 'ALL'
    if value.lower() in ['multi', 'm']:
        return 'MULTI'
    else:
        raise argparse.ArgumentTypeError("Required format for binary classification: 'HC_AD' or 'MCI_AD' or 'HC_MCI' or 'ALL'")

def str2bool(value):
    if isinstance(value, bool):
       return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')
