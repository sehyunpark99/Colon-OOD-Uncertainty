"""
Script to evaluate a single model. 
"""
import os
import json
import math
import torch
import time
import argparse
import numpy as np
import torch.backends.cudnn as cudnn

# Import network models
from net.resnet import resnet50
import colon_data

# Import metrics to compute
from metrics.classification_metrics import (
    test_classification_net,
    test_classification_net_logits,
    test_classification_net_ensemble
)
from metrics.calibration_metrics import expected_calibration_error
from metrics.uncertainty_confidence import entropy, logsumexp
from metrics.ood_metrics import get_roc_auc, get_roc_auc_logits, get_roc_auc_ensemble

# Import GMM utils
from utils.gmm_utils import get_embeddings, gmm_evaluate, gmm_fit
from utils.ensemble_utils import load_ensemble, ensemble_forward_pass
from utils.eval_utils import model_load_name
from utils.train_utils import model_save_name, test_single_epoch
from utils.args import eval_args

# Temperature scaling
from utils.temperature_scaling import ModelWithTemperature

# Dataset params
dataset_num_classes = {"colonoscopy": 2, "colon_ood": 3}

dataset_loader = {"colonoscopy": colon_data, "colon_ood": colon_data}

# Mapping model name to model function
models = {"resnet50": resnet50}

model_to_num_dim = {"resnet50": 2048}


if __name__ == "__main__":

    args = eval_args().parse_args()
    curr_time = time.strftime("%Y-%m-%d_%H:%M:%S")

    # Checking if GPU is available
    cuda = torch.cuda.is_available()

    # Setting additional parameters
    print("Parsed args", args)
    print("Seed: ", args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if cuda else "cpu")

    # Taking input for the dataset
    num_classes = dataset_num_classes[args.dataset]

    test_loader = dataset_loader[args.dataset].get_test_loader(batch_size=args.batch_size, pin_memory=args.gpu)
    deploy_loader_1, deploy_loader_2, deploy_loader_3 = dataset_loader[args.dataset].get_deploy_loader(batch_size=args.batch_size, pin_memory=args.gpu)
    
    if args.ood_dataset == "colon_ood":
        ood_test_loader = dataset_loader[args.ood_dataset].get_ood_test_loader(batch_size=args.batch_size, pin_memory=args.gpu)
    else:
        ood_test_loader = dataset_loader[args.ood_dataset].get_test_loader(batch_size=args.batch_size, pin_memory=args.gpu)
    
    # Evaluating the models
    accuracies = []

    # Pre temperature scaling
    # m1 - Uncertainty/Confidence Metric 1
    #      for deterministic model: logsumexp, for ensemble: entropy
    # m2 - Uncertainty/Confidence Metric 2
    #      for deterministic model: entropy, for ensemble: MI
    eces = []
    m1_aurocs = []
    m1_auprcs = []
    m2_aurocs = []
    m2_auprcs = []

    # Post temperature scaling
    t_eces = []
    t_m1_aurocs = []
    t_m1_auprcs = []
    t_m2_aurocs = []
    t_m2_auprcs = []

    topt = None
    external_1 = {}
    external_2 = {}
    external_3 = {}

    for i in range(args.runs):
        print (f"Evaluating run: {(i+1)}")
        # Loading the model(s)
        if args.model_type == "ensemble":
            val_loaders = []
            for j in range(args.ensemble):
                train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
                    batch_size=args.batch_size, augment=args.data_aug, val_seed=(args.seed+(5*i)+j), val_size=0.1, pin_memory=args.gpu,
                )
                val_loaders.append(val_loader)
            # Evaluate an ensemble
            ensemble_loc = os.path.join(args.load_loc, ("Run" + str(i + 1)))
            net_ensemble = load_ensemble(
                ensemble_loc=ensemble_loc,
                model_name=args.model,
                device=device,
                num_classes=num_classes,
                spectral_normalization=args.sn,
                mod=args.mod,
                coeff=args.coeff,
                seed=(5*i+1)
            )

        else:
            train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
                batch_size=args.batch_size, augment=args.data_aug, val_seed=(args.seed+i), val_size=0.1, pin_memory=args.gpu,
            )
            # saved_model_name = os.path.join(
            #     args.load_loc,
            #     "Run" + str(i + 1),
            #     model_load_name(args.model, args.sn, args.mod, args.coeff, args.seed, i) + "_350.model",
            # )
            # ckpt = torch.load('modified_checkpoint.pth')
            saved_model_name = args.load_loc
            net = models[args.model](
                spectral_normalization=args.sn, mod=args.mod, coeff=args.coeff, num_classes=num_classes, temp=1.0,
            )
            if args.gpu:
                net.cuda()
                # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
                # cudnn.benchmark = True
            ckpt = torch.load(str(saved_model_name))
            new_state_dict = {}
            for key, value in ckpt.items():
                if key.startswith('module.'):
                    new_key = key[len('module.'):]
                else:
                    new_key = key
                new_state_dict[new_key] = value
            ckpt = new_state_dict
            net.load_state_dict(ckpt, strict = False)
            # print(net)
            net.eval()
            # print(net)

        # Evaluating the model(s)
        if args.model_type == "ensemble":
            print("Ensemble")
            (conf_matrix, accuracy, labels_list, predictions, confidences,) = test_classification_net_ensemble(
                net_ensemble, test_loader, device
            )
            ece = expected_calibration_error(confidences, predictions, labels_list, num_bins=15)

            (_, _, _), (_, _, _), m1_auroc, m1_auprc = get_roc_auc_ensemble(
                net_ensemble, test_loader, ood_test_loader, "mutual_information", device
            )
            (_, _, _), (_, _, _), m2_auroc, m2_auprc = get_roc_auc_ensemble(
                net_ensemble, test_loader, ood_test_loader, "entropy", device
            )

            # Temperature scale the ensemble
            t_ensemble = []
            for model, val_loader in zip(net_ensemble, val_loaders):
                t_model = ModelWithTemperature(model)
                t_model.set_temperature(val_loader)
                t_ensemble.append(t_model)

            (
                t_conf_matrix,
                t_accuracy,
                t_labels_list,
                t_predictions,
                t_confidences,
            ) = test_classification_net_ensemble(t_ensemble, test_loader, device)
            t_ece = expected_calibration_error(t_confidences, t_predictions, t_labels_list, num_bins=15)

            (_, _, _), (_, _, _), t_m1_auroc, t_m1_auprc = get_roc_auc_ensemble(
                t_ensemble, test_loader, ood_test_loader, "mutual_information", device
            )
            (_, _, _), (_, _, _), t_m2_auroc, t_m2_auprc = get_roc_auc_ensemble(
                t_ensemble, test_loader, ood_test_loader, "entropy", device
            )

        else:
            (conf_matrix, accuracy, labels_list, predictions, confidences,) = test_classification_net(
                net, test_loader, device
            )
            print(conf_matrix)
            print(f'test_classification_net_accuracy: {accuracy}')
            confidences_array = np.array(confidences)
            mean_confidence = np.mean(confidences_array)
            print(f'confidences: {mean_confidence}')
            ece = expected_calibration_error(confidences, predictions, labels_list, num_bins=15)
            (conf_matrix_1, accuracy_1, labels_list_1, predictions_1, confidences_1,) = test_classification_net(
                net, deploy_loader_1, device
            )
            (conf_matrix_2, accuracy_2, labels_list_2, predictions_2, confidences_2,) = test_classification_net(
                net, deploy_loader_2, device
            )
            (conf_matrix_3, accuracy_3, labels_list_3, predictions_3, confidences_3,) = test_classification_net(
                net, deploy_loader_3, device
            )

            print(f'test_classification_net_accuracy for Ewha: {accuracy_1}')
            confidences_array_1 = np.array(confidences_1)
            mean_confidence_1 = np.mean(confidences_array_1)
            print(f'confidences: {mean_confidence_1}')
            print(f'test_classification_net_accuracy for Yonsei: {accuracy_2}')
            confidences_array_2 = np.array(confidences_2)
            mean_confidence_2 = np.mean(confidences_array_2)
            print(f'confidences: {mean_confidence_2}')
            print(f'test_classification_net_accuracy for Asan: {accuracy_3}')
            confidences_array_3 = np.array(confidences_3)
            mean_confidence_3 = np.mean(confidences_array_3)
            print(f'confidences: {mean_confidence_3}')
            external_1[i] = [accuracy_1, float(mean_confidence_1), list(confidences_array_1)]
            external_2[i] = [accuracy_2, float(mean_confidence_2), list(confidences_array_2)]
            external_3[i] = [accuracy_3, float(mean_confidence_3), list(confidences_array_3)]

            temp_scaled_net = ModelWithTemperature(net)
            temp_scaled_net.set_temperature(val_loader)
            topt = temp_scaled_net.temperature

            (t_conf_matrix, t_accuracy, t_labels_list, t_predictions, t_confidences,) = test_classification_net(
                temp_scaled_net, test_loader, device
            )
            confidences_array = np.array(t_confidences)
            t_mean_confidence = np.mean(confidences_array)
            print(f'confidences: {t_mean_confidence}')
            t_ece = expected_calibration_error(t_confidences, t_predictions, t_labels_list, num_bins=15)

            if (args.model_type == "gmm"):
                # Evaluate a GMM model
                print("GMM Model")
                embeddings, labels = get_embeddings(
                    net,
                    train_loader,
                    num_dim=model_to_num_dim[args.model],
                    dtype=torch.double,
                    device=device,
                    storage_device=device,
                )

                try:
                    gaussians_model, jitter_eps = gmm_fit(embeddings=embeddings, labels=labels, num_classes=num_classes)
                    logits, labels = gmm_evaluate(
                        net, gaussians_model, test_loader, device=device, num_classes=num_classes, storage_device=device,
                    )

                    ood_logits, ood_labels = gmm_evaluate(
                        net, gaussians_model, ood_test_loader, device=device, num_classes=num_classes, storage_device=device,
                    )

                    (fpr, tpr, _), (precision, recall, _), m1_auroc, m1_auprc = get_roc_auc_logits(
                        logits, ood_logits, logsumexp, device, confidence=True
                    )
                    (fpr, tpr, _), (precision, recall, _), m2_auroc, m2_auprc = get_roc_auc_logits(logits, ood_logits, entropy, device)

                    t_m1_auroc = m1_auroc
                    t_m1_auprc = m1_auprc
                    t_m2_auroc = m2_auroc
                    t_m2_auprc = m2_auprc

                except RuntimeError as e:
                    print("Runtime Error caught: " + str(e))
                    continue

            else:
                # Evaluate a normal Softmax model
                print("Softmax Model")
                (fpr, tpr, _), (precision, recall, _), m1_auroc, m1_auprc = get_roc_auc(
                    net, test_loader, ood_test_loader, logsumexp, device, confidence=True
                )
                # print(f'logsumexp before tmp: {fpr}, {tpr}, {precision}, {recall}')
                (fpr, tpr, _), (precision, recall, _), m2_auroc, m2_auprc = get_roc_auc(net, test_loader, ood_test_loader, entropy, device)
                # print(f'entropy before tmp: {fpr}, {tpr}, {precision}, {recall}')
                (fpr, tpr, _), (precision, recall, _), t_m1_auroc, t_m1_auprc = get_roc_auc(
                    temp_scaled_net, test_loader, ood_test_loader, logsumexp, device, confidence=True,
                )
                # print(f'logsumexp after tmp: {fpr}, {tpr}, {precision}, {recall}')
                (fpr, tpr, _), (precision, recall, _), t_m2_auroc, t_m2_auprc = get_roc_auc(
                    temp_scaled_net, test_loader, ood_test_loader, entropy, device
                )
                # print(f'entropy after tmp: {fpr}, {tpr}, {precision}, {recall}')

        accuracies.append(accuracy)

        # Pre-temperature results
        eces.append(ece)
        m1_aurocs.append(m1_auroc)
        m1_auprcs.append(m1_auprc)
        m2_aurocs.append(m2_auroc)
        m2_auprcs.append(m2_auprc)

        # Post-temperature results
        t_eces.append(t_ece)
        t_m1_aurocs.append(t_m1_auroc)
        t_m1_auprcs.append(t_m1_auprc)
        t_m2_aurocs.append(t_m2_auroc)
        t_m2_auprcs.append(t_m2_auprc)

    accuracy_tensor = torch.tensor(accuracies)
    ece_tensor = torch.tensor(eces)
    m1_auroc_tensor = torch.tensor(m1_aurocs)
    m1_auprc_tensor = torch.tensor(m1_auprcs)
    m2_auroc_tensor = torch.tensor(m2_aurocs)
    m2_auprc_tensor = torch.tensor(m2_auprcs)

    t_ece_tensor = torch.tensor(t_eces)
    t_m1_auroc_tensor = torch.tensor(t_m1_aurocs)
    t_m1_auprc_tensor = torch.tensor(t_m1_auprcs)
    t_m2_auroc_tensor = torch.tensor(t_m2_aurocs)
    t_m2_auprc_tensor = torch.tensor(t_m2_auprcs)

    mean_accuracy = torch.mean(accuracy_tensor)
    mean_ece = torch.mean(ece_tensor)
    mean_m1_auroc = torch.mean(m1_auroc_tensor)
    mean_m1_auprc = torch.mean(m1_auprc_tensor)
    mean_m2_auroc = torch.mean(m2_auroc_tensor)
    mean_m2_auprc = torch.mean(m2_auprc_tensor)

    mean_t_ece = torch.mean(t_ece_tensor)
    mean_t_m1_auroc = torch.mean(t_m1_auroc_tensor)
    mean_t_m1_auprc = torch.mean(t_m1_auprc_tensor)
    mean_t_m2_auroc = torch.mean(t_m2_auroc_tensor)
    mean_t_m2_auprc = torch.mean(t_m2_auprc_tensor)

    std_accuracy = torch.std(accuracy_tensor) / math.sqrt(accuracy_tensor.shape[0])
    std_ece = torch.std(ece_tensor) / math.sqrt(ece_tensor.shape[0])
    std_m1_auroc = torch.std(m1_auroc_tensor) / math.sqrt(m1_auroc_tensor.shape[0])
    std_m1_auprc = torch.std(m1_auprc_tensor) / math.sqrt(m1_auprc_tensor.shape[0])
    std_m2_auroc = torch.std(m2_auroc_tensor) / math.sqrt(m2_auroc_tensor.shape[0])
    std_m2_auprc = torch.std(m2_auprc_tensor) / math.sqrt(m2_auprc_tensor.shape[0])

    std_t_ece = torch.std(t_ece_tensor) / math.sqrt(t_ece_tensor.shape[0])
    std_t_m1_auroc = torch.std(t_m1_auroc_tensor) / math.sqrt(t_m1_auroc_tensor.shape[0])
    std_t_m1_auprc = torch.std(t_m1_auprc_tensor) / math.sqrt(t_m1_auprc_tensor.shape[0])
    std_t_m2_auroc = torch.std(t_m2_auroc_tensor) / math.sqrt(t_m2_auroc_tensor.shape[0])
    std_t_m2_auprc = torch.std(t_m2_auprc_tensor) / math.sqrt(t_m2_auprc_tensor.shape[0])

    res_dict = {}
    res_dict["mean"] = {}
    res_dict["mean"]["accuracy"] = mean_accuracy.item()
    res_dict["mean"]["ece"] = mean_ece.item()
    res_dict["mean"]["m1_auroc"] = mean_m1_auroc.item()
    res_dict["mean"]["m1_auprc"] = mean_m1_auprc.item()
    res_dict["mean"]["m2_auroc"] = mean_m2_auroc.item()
    res_dict["mean"]["m2_auprc"] = mean_m2_auprc.item()
    res_dict["mean"]["t_ece"] = mean_t_ece.item()
    res_dict["mean"]["t_m1_auroc"] = mean_t_m1_auroc.item()
    res_dict["mean"]["t_m1_auprc"] = mean_t_m1_auprc.item()
    res_dict["mean"]["t_m2_auroc"] = mean_t_m2_auroc.item()
    res_dict["mean"]["t_m2_auprc"] = mean_t_m2_auprc.item()

    res_dict["std"] = {}
    res_dict["std"]["accuracy"] = std_accuracy.item()
    res_dict["std"]["ece"] = std_ece.item()
    res_dict["std"]["m1_auroc"] = std_m1_auroc.item()
    res_dict["std"]["m1_auprc"] = std_m1_auprc.item()
    res_dict["std"]["m2_auroc"] = std_m2_auroc.item()
    res_dict["std"]["m2_auprc"] = std_m2_auprc.item()
    res_dict["std"]["t_ece"] = std_t_ece.item()
    res_dict["std"]["t_m1_auroc"] = std_t_m1_auroc.item()
    res_dict["std"]["t_m1_auprc"] = std_t_m1_auprc.item()
    res_dict["std"]["t_m2_auroc"] = std_t_m2_auroc.item()
    res_dict["std"]["t_m2_auprc"] = std_t_m2_auprc.item()

    res_dict["mean"] = {}
    res_dict["mean"]["accuracy"] = mean_accuracy.item()
    res_dict["mean"]["ece"] = mean_ece.item()
    res_dict["mean"]["m1_auroc"] = mean_m1_auroc.item()
    res_dict["mean"]["m1_auprc"] = mean_m1_auprc.item()
    res_dict["mean"]["m2_auroc"] = mean_m2_auroc.item()
    res_dict["mean"]["m2_auprc"] = mean_m2_auprc.item()
    res_dict["mean"]["t_ece"] = mean_t_ece.item()
    res_dict["mean"]["t_m1_auroc"] = mean_t_m1_auroc.item()
    res_dict["mean"]["t_m1_auprc"] = mean_t_m1_auprc.item()
    res_dict["mean"]["t_m2_auroc"] = mean_t_m2_auroc.item()
    res_dict["mean"]["t_m2_auprc"] = mean_t_m2_auprc.item()

    res_dict["values"] = {}
    res_dict["values"]["accuracy"] = accuracies
    res_dict["values"]["ece"] = eces
    res_dict["values"]["m1_auroc"] = m1_aurocs
    res_dict["values"]["m1_auprc"] = m1_auprcs
    res_dict["values"]["m2_auroc"] = m2_aurocs
    res_dict["values"]["m2_auprc"] = m2_auprcs
    res_dict["values"]["t_ece"] = t_eces
    res_dict["values"]["t_m1_auroc"] = t_m1_aurocs
    res_dict["values"]["t_m1_auprc"] = t_m1_auprcs
    res_dict["values"]["t_m2_auroc"] = t_m2_aurocs
    res_dict["values"]["t_m2_auprc"] = t_m2_auprcs

    res_dict["info"] = vars(args)

    with open(
        "res_"
        + model_save_name(args.model, args.sn, args.mod, args.coeff, args.seed)
        + "_"
        + args.model_type
        + "_"
        + args.dataset
        + "_"
        + args.ood_dataset
        + "_"
        + curr_time
        + ".json",
        "w",
    ) as f:
        json.dump(res_dict, f)

    # with open(
    #     "res_"
    #     + model_save_name(args.model, args.sn, args.mod, args.coeff, args.seed)
    #     + "_"
    #     + args.model_type
    #     + "_"
    #     + args.dataset
    #     + "_"
    #     + "externals"
    #     + "_"
    #     + curr_time
    #     + ".json",
    #     "w",
    # ) as f:
    #     json.dump(external_1, f)
    #     json.dump(external_2, f)
    #     json.dump(external_3, f)
    
    with open(
        "res_"
        + model_save_name(args.model, args.sn, args.mod, args.coeff, args.seed)
        + "_"
        + args.model_type
        + "_"
        + args.dataset
        + "_"
        + "externals"
        + "_"
        + curr_time
        + ".txt",
        "w",
    ) as f:
        for key, value in external_1.items():
            f.write(f"{key}: {value}\n")
        for key, value in external_2.items():
            f.write(f"{key}: {value}\n")
        for key, value in external_3.items():
            f.write(f"{key}: {value}\n")