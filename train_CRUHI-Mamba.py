import os
# Set LD_LIBRARY_PATH
print(os.environ.get('LD_LIBRARY_PATH'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import torch
import random
import argparse
import numpy as np
import utils.data_loading as UHI_load
from utils.loss_func import head_loss, resize
from utils.evaluation import Evaluator
from utils.setup_logger import setup_logger
from utils.visual_predict import visualize_predict, lab_colors_005, lab_colors_006, lab_colors_019  # Use corresponding color labels for different datasets.
from model.CRUHIMamba import CRUHIMamba
from calflops import calculate_flops
import xarray as xr
torch.autograd.set_detect_anomaly(True)


def vis_a_image(gt_vis, pred_vis, save_single_predict_path, save_single_gt_path, only_vis_label=False, colors=lab_colors_005):
    # Set 'colors' to visualize the corresponding dataset images
    visualize_predict(gt_vis, pred_vis, save_single_predict_path, save_single_gt_path, only_vis_label=only_vis_label, colors=colors)
    visualize_predict(gt_vis, pred_vis, save_single_predict_path.replace('.png', '_mask.png'), save_single_gt_path, only_vis_label=True, colors=colors)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# random seed setting
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_index', type=int, default=0)
    parser.add_argument('--data_set_path', type=str, default='./data')
    parser.add_argument('--work_dir', type=str, default='./')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--train_samples', type=int, default=135)    # The number of samples in each class used for training.
    parser.add_argument('--val_samples', type=int, default=30)      # The number of samples in each class used for validation.
    parser.add_argument('--exp_name', type=str, default='Results')    # Naming conventions for folders used to store experimental results
    parser.add_argument('--record_computecost', type=bool, default=True)
    args = parser.parse_args()
    return args


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = get_parser()
record_computecost = args.record_computecost
exp_name = args.exp_name
max_epoch = args.max_epoch

net_name = 'CRUHI_005'
dataset_index = args.dataset_index
num_list = [args.train_samples, args.val_samples]
learning_rate = args.lr
seed_list = [0, 1, 2, 3, 4]     # List of random seeds: 5 experiments
paras_dict = {'net_name': net_name, 'dataset_index': dataset_index, 'num_list': num_list,
              'lr': learning_rate, 'seed_list': seed_list}
                      #   0        1          2
data_set_name_list = ['cr_005', 'cr_006', 'cr_019']
data_set_name = data_set_name_list[dataset_index]

if data_set_name in ['cr_005', 'cr_006', 'cr_019']:  # Based on the selected dataset name, determine whether the image needs to be cropped.
    split_image = True
else:
    split_image = False


if __name__ == '__main__':
    # Set up the experimental environment
    data_set_path = args.data_set_path
    work_dir = args.work_dir
    exp_name = args.exp_name
    dataset_name = '005_13530PCA50_e200_CRUHI'    # Custom filename
    # Create a Save folder
    save_folder = os.path.join(work_dir, exp_name, net_name, dataset_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print("makedirs {}".format(save_folder))

    # Set up the logger
    save_log_path = os.path.join(save_folder, 'train_tr{}_val{}.log'.format(num_list[0], num_list[1]))
    logger = setup_logger(name='{}'.format(dataset_name), logfile=save_log_path)

    # Clear the CUDA cache; record the experiment details
    torch.cuda.empty_cache()
    logger.info(save_folder)

    # Load data: data, ground truth(gt); retrieve the data's height, width, number of channels, and number of classes
    data, gt = UHI_load.load_data(data_set_name, data_set_path)
    height, width, channels = data.shape
    class_count = max(np.unique(gt))

    # Preprocessing
    gt_reshape = gt.reshape(-1)
    # `flag_list` and `ratio_list` are used to define the sampling strategies for the training and validation datasets.
    flag_list = [1, 0]      # "1" indicates sampling by count; "0" indicates sampling by proportion
    ratio_list = [0.1, 0.01]  # [train_ratio,val_ratio] Set the sampling ratio for each class in the training set and validation set.
    # Define the loss function: Use the cross-entropy loss function, ignoring the category with index -1 (background).
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)

    # Initialize evaluation metrics
    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    mIOU_ALL = []
    EACH_ACC_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []
    evaluator = Evaluator(num_class=class_count)

    for exp_idx, curr_seed in enumerate(seed_list):     # 5 random seeds, corresponding to 5 experiments.
        train_start_time = time.time()
        setup_seed(curr_seed)
        single_experiment_name = 'run{}_seed{}'.format(str(exp_idx), str(curr_seed))
        save_single_experiment_folder = os.path.join(save_folder, single_experiment_name)
        if not os.path.exists(save_single_experiment_folder):
            os.mkdir(save_single_experiment_folder)
        save_vis_folder = os.path.join(save_single_experiment_folder, 'vis')    # Create a folder to store the visualization results of the experiment
        if not os.path.exists(save_vis_folder):
            os.makedirs(save_vis_folder)
            print("makedirs {}".format(save_vis_folder))

        # Define a path to save model weights, experiment results, and visualizations of predictions and ground truth labels.
        save_weight_path = os.path.join(save_single_experiment_folder, "best_tr{}_val{}.pth".format(num_list[0], num_list[1]))
        results_save_path = os.path.join(save_single_experiment_folder, 'result_tr{}_val{}.txt'.format(num_list[0], num_list[1]))
        predict_save_path = os.path.join(save_single_experiment_folder, 'pred_vis_tr{}_val{}.png'.format(num_list[0], num_list[1]))
        gt_save_path = os.path.join(save_single_experiment_folder, 'gt_vis_tr{}_val{}.png'.format(num_list[0], num_list[1]))

        # Call the `UHI_load.sampling` function to sample data based on the specified ratios (`ratio_list`)
        # and quantities (`num_list`) in order to create indices for the training, validation, and test datasets.
        train_data_index, val_data_index, test_data_index, all_data_index = UHI_load.sampling(ratio_list, num_list, gt_reshape, class_count, flag_list[0])
        index = (train_data_index, val_data_index, test_data_index)
        # Call the `UHI_load.generate_image_iter` function to generate labels for the training, validation, and test datasets.
        train_label, val_label, test_label = UHI_load.generate_image_iter(data, height, width, gt_reshape, index)

        # Create a CRUHI-Mamba model instance by passing in the number of input channels (in_channels),
        # the number of classes (num_classes), and the hidden layer dimension (hidden_dim).
        net = CRUHIMamba(in_channels=channels, num_classes=class_count, hidden_dim=64)
        logger.info(paras_dict)     # Use the logger to record experimental parameters (paras_dict) and model information (net).
        logger.info(net)

        # Preprocessing of Raw Hyperspectral Data for Coral Reefs（350 bands）
        x = data / 255
        for data in x:                      # Check the data for “NaN” and “infinity” values
            print("check?")
            if np.isnan(data).any() or np.isinf(data).any():
                print("There are invalid values in the dataset")
        x = torch.from_numpy(x)
        print(x.shape)

        # # Processing PCA-dimension-reduced data
        # x = torch.from_numpy(data)

        # Input Preprocessing
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)
        x = x.to(device)
        train_label = train_label.to(device)
        val_label = val_label.to(device)
        test_label = test_label.to(device)

        net.to(device)
        # Initialize the loss and accuracy lists: Set the loss value to 100 and the accuracy to 0.
        train_loss_list = [100]
        train_acc_list = [0]
        val_loss_list = [100]
        val_acc_list = [0]
        # Set up the optimizer.
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

        logger.info(optimizer)      # Record optimizer information.
        best_loss = 999           # Initialize the optimal loss value.

        if record_computecost:    # Compute model parameters.
            net.eval()
            total_params, trainable_params = count_parameters(model=net)
            logger.info("Total parameters: {}, Trainable parameters: {}".format(total_params, trainable_params))

        best_val_acc = 0    # Set the initial optimal validation accuracy to 0.

        for epoch in range(max_epoch):
            y_train = train_label.unsqueeze(0)
            # Initialize the training accuracy accumulator, sample counter, batch counter, training loss accumulator, and timestamp.
            train_acc_sum, trained_samples_counter = 0.0, 0
            batch_counter, train_loss_sum = 0, 0
            loss_dict = {}

            # Start training: 'net.train()' sets the model to training mode.
            net.train()
            # Image splitting logic; if `split_image` is True, the image is split into multiple parts for processing.
            if split_image:
                total_loss = 0
                valid_count = 0
                num_parts = 30    # Divide the image into several parts
                part_width, remainder = divmod(x.shape[2], num_parts)

                # Split images and labels
                for i in range(num_parts):
                    start_idx = i * part_width
                    if i < num_parts - 1:
                        end_idx = (i + 1) * part_width + 100   # Overlap by 100 lines between each part
                    else:
                        end_idx = x.shape[2]

                    x_part = x[:, :, start_idx:end_idx, :]
                    y_part = y_train[:, start_idx:end_idx, :]
                    # Model Predictions
                    y_pred_part = net(x_part)
                    # Calculate the loss
                    ls = head_loss(loss_func, y_pred_part, y_part.long())
                    if not torch.isnan(ls).item():
                        print(ls)
                        total_loss += ls
                        valid_count += 1

                    # Gradient Resetting, Backpropagation, and Optimizer Steps.
                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()
                    torch.cuda.empty_cache()

                # Calculate the average loss
                torch.cuda.empty_cache()
                print(valid_count)
                total_loss /= valid_count
                logger.info('Iter:{}|loss:{}'.format(epoch, total_loss.detach().cpu().numpy()))

            torch.cuda.empty_cache()

            # evaluate stage
            net.eval()  # Set the model to validation (evaluation) mode.
            with torch.no_grad():
                evaluator.reset()

                # # Verify in chunks, then concatenate the results of each chunk.
                # num_parts = 30
                # all_blocks_pred = []
                # part_width, remainder = divmod(x.shape[2], num_parts)
                # for i in range(num_parts):
                #     start_idx = i * part_width
                #     if i < num_parts - 1:
                #         end_idx = (i + 1) * part_width
                #     else:
                #         end_idx = x.shape[2]
                #     x_pre = x[:, :, start_idx:end_idx, :]
                #     block_pred = net(x_pre)
                #     all_blocks_pred.append(block_pred)
                #     torch.cuda.empty_cache()
                # output_val = torch.cat(all_blocks_pred, dim=2)

                # Chunked Linear Weighted Averaging (CLWA)
                num_parts = 30
                overlap = 100
                part_width, remainder = divmod(x.shape[2], num_parts)
                # Initialize the final prediction tensor.
                total_output_val = torch.zeros(1, class_count, height, width, device=x.device)
                weight_map = torch.zeros_like(total_output_val)  # Weight map, used to record the weight of each pixel.

                for i in range(num_parts):
                    start_idx = i * part_width
                    if i < num_parts - 1:
                        end_idx = (i + 1) * part_width + overlap
                    else:
                        end_idx = x.shape[2]

                    x_pre = x[:, :, start_idx:end_idx, :]
                    block_pred = net(x_pre)
                    block_pred = resize(input=block_pred, size=(end_idx - start_idx, width), mode='bilinear', align_corners=True)
                    actual_width = block_pred.shape[2]

                    # Calculate weights
                    if i == 0:
                        weights = torch.linspace(1, 0.5, actual_width, device=block_pred.device)
                    elif i == num_parts - 1:
                        weights = torch.linspace(0.5, 1, actual_width, device=block_pred.device)
                    else:
                        weights = torch.linspace(0.5, 1, actual_width // 2, device=block_pred.device)
                        weights = torch.cat([weights, torch.linspace(1, 0.5, actual_width - actual_width // 2, device=block_pred.device)])

                    # Using weights
                    weights = weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                    block_pred = block_pred * weights
                    total_output_val[:, :, start_idx:start_idx + actual_width, :] += block_pred
                    weight_map[:, :, start_idx:start_idx + actual_width, :] += weights
                output_val = total_output_val / weight_map

                y_val = val_label.unsqueeze(0)
                # Resize the model's predictions to match the dimensions of the validation labels.
                seg_logits = resize(input=output_val, size=y_val.shape[1:], mode='bilinear', align_corners=True)
                predict = torch.argmax(seg_logits, dim=1).cpu().numpy()

                # Processing validation labels
                Y_val_np = val_label.cpu().numpy()
                Y_val_255 = np.where(Y_val_np == -1, 255, Y_val_np)

                # Calculate evaluation metrics
                evaluator.add_batch(np.expand_dims(Y_val_255, axis=0), predict)
                OA = evaluator.Overall_Accuracy()
                mIOU, IOU = evaluator.Mean_Intersection_over_Union()
                mAcc, Acc = evaluator.Pixel_Accuracy_Class()
                Kappa = evaluator.Kappa()
                logger.info('Evaluate {}|OA:{}|MACC:{}|Kappa:{}|MIOU:{}|IOU:{}|ACC:{}'.format(epoch, OA, mAcc, Kappa, mIOU, IOU, Acc))

                # Save the best model
                if OA >= best_val_acc:
                    best_epoch = epoch + 1
                    best_val_acc = OA
                    torch.save(net.state_dict(), save_weight_path)

                if (epoch+1) % 50 == 0:
                    save_single_predict_path = os.path.join(save_vis_folder, 'predict_{}.png'.format(str(epoch+1)))
                    save_single_gt_path = os.path.join(save_vis_folder, 'gt.png')
                    vis_a_image(gt, predict, save_single_predict_path, save_single_gt_path)
            torch.cuda.empty_cache()
        train_end_time = time.time()
        training_time = train_end_time - train_start_time
        Train_Time_ALL.append(training_time)
        torch.cuda.empty_cache()
        # This concludes the training and validation for all epochs in each experiment.

        # Testing begins
        logger.info("\n\n*****Starting model testing.*****\n")
        test_start_time = time.time()
        load_weight_path = save_weight_path
        # Create a copy of the best model
        best_net = CRUHIMamba(in_channels=channels, num_classes=class_count, hidden_dim=64)
        best_net.to(device)
        best_net.load_state_dict(torch.load(load_weight_path))
        best_net.eval()
        test_evaluator = Evaluator(num_class=class_count)
        with torch.no_grad():
            test_evaluator.reset()

            # # Verify in chunks, then concatenate the results of each chunk.
            # num_parts = 30
            # all_blocks_test = []
            # part_width, remainder = divmod(x.shape[2], num_parts)
            # for i in range(num_parts):
            #     start_idx = i * part_width
            #     if i < num_parts - 1:
            #         end_idx = (i + 1) * part_width
            #     else:
            #         end_idx = x.shape[2]
            #     x_test = x[:, :, start_idx:end_idx, :]
            #     block_test = best_net(x_test)
            #     all_blocks_test.append(block_test)
            #     torch.cuda.empty_cache()
            # output_test = torch.cat(all_blocks_test, dim=2)

            # Chunked Linear Weighted Averaging (CLWA)
            num_parts = 30
            overlap = 100
            part_width, remainder = divmod(x.shape[2], num_parts)
            total_output_test = torch.zeros(1, class_count, height, width, device=x.device)
            weight_map_test = torch.zeros_like(total_output_test)

            for i in range(num_parts):
                start_idx = i * part_width
                if i < num_parts - 1:
                    end_idx = (i + 1) * part_width + overlap
                else:
                    end_idx = x.shape[2]

                x_pre = x[:, :, start_idx:end_idx, :]

                best_pred = best_net(x_pre)
                best_pred = resize(input=best_pred, size=(end_idx - start_idx, width), mode='bilinear', align_corners=True)
                actual_width = best_pred.shape[2]

                if i == 0:
                    weights = torch.linspace(1, 0.5, actual_width, device=best_pred.device)
                elif i == num_parts - 1:
                    weights = torch.linspace(0.5, 1, actual_width, device=best_pred.device)
                else:  # 中间块有左右重叠
                    weights = torch.linspace(0.5, 1, actual_width // 2, device=best_pred.device)
                    weights = torch.cat([weights, torch.linspace(1, 0.5, actual_width - actual_width // 2, device=best_pred.device)])

                weights = weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                best_pred = best_pred * weights

                total_output_test[:, :, start_idx:start_idx + actual_width, :] += best_pred
                weight_map_test[:, :, start_idx:start_idx + actual_width, :] += weights

            output_test = total_output_test / weight_map_test
            y_test = test_label.unsqueeze(0)
            seg_logits_test = resize(input=output_test, size=y_test.shape[1:], mode='bilinear', align_corners=True)
            predict_test = torch.argmax(seg_logits_test, dim=1).cpu().numpy()
            Y_test_np = test_label.cpu().numpy()
            Y_test_255 = np.where(Y_test_np == -1, 255, Y_test_np)
            test_evaluator.add_batch(np.expand_dims(Y_test_255, axis=0), predict_test)
            OA_test = test_evaluator.Overall_Accuracy()
            mIOU_test, IOU_test = test_evaluator.Mean_Intersection_over_Union()
            mAcc_test, Acc_test = test_evaluator.Pixel_Accuracy_Class()
            Kappa_test = test_evaluator.Kappa()
            logger.info('Test {}|OA_t:{}|MACC_t:{}|Kappa_t:{}|MIOU_t:{}|IOU_t:{}|ACC_t:{}'.format(epoch, OA_test, mAcc_test,
                                                                                                  Kappa_test, mIOU_test, IOU_test, Acc_test))
            vis_a_image(gt, predict_test, predict_save_path, gt_save_path)

            # # Save the classification results to an ‘nc’ file
            # pre_lab_save_path = os.path.join(save_single_experiment_folder,'pred_lab_tr{}_val{}.nc'.format(num_list[0], num_list[1]))
            # pre_test_squeezed = np.squeeze(predict_test, axis=0)
            # pre_labels = xr.DataArray(pre_test_squeezed, dims=['Y', 'X'], name='labels')
            # pre_labels.to_netcdf(pre_lab_save_path)

            torch.cuda.empty_cache()
        test_end_time = time.time()
        testing_time = test_end_time - test_start_time
        Test_Time_ALL.append(testing_time)
        torch.cuda.empty_cache()

        # Used to record and preserve the results of a complete experiment.
        np.set_printoptions(precision=4)
        f = open(results_save_path, 'a+')
        str_results = '\n=================' \
                      + " exp_idx=" + str(exp_idx) \
                      + " seed=" + str(curr_seed) \
                      + " learning rate=" + str(learning_rate) \
                      + " epochs=" + str(max_epoch) \
                      + " train ratio=" + str(ratio_list[0]) \
                      + " val ratio=" + str(ratio_list[1]) \
                      + " valid_parts=" + str(valid_count) \
                      + " ================" \
                      + "\nOA=" + str(OA_test) \
                      + "\nAA=" + str(mAcc_test) \
                      + '\nkpp=' + str(Kappa_test) \
                      + '\nmIOU_test:' + str(mIOU_test) \
                      + "\nIOU_test:" + str(IOU_test) \
                      + "\nAcc_test:" + str(Acc_test) \
                      + "\ntraining_time:" + str(training_time) \
                      + "\ntesting_time:" + str(testing_time) + "\n"
        logger.info(str_results)
        f.write(str_results)
        f.close()
        torch.cuda.empty_cache()
        # The experiment has been successfully completed.

        # Cumulative results: Add the accuracy from each experiment to the results list.
        OA_ALL.append(OA_test)
        AA_ALL.append(mAcc_test)
        KPP_ALL.append(Kappa_test)
        mIOU_ALL.append(mIOU_test)
        EACH_ACC_ALL.append(Acc_test)
        torch.cuda.empty_cache()
    # That concludes the five experiments.

    # Convert the list of accumulated results into a NumPy array for further analysis.
    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    mIOU_ALL = np.array(mIOU_ALL)
    EACH_ACC_ALL = np.array(EACH_ACC_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)

    np.set_printoptions(precision=4)
    # Recording the Lab Log: Summary of Experimental Results.
    logger.info("\n====================Mean result of {} times runs =========================".format(len(seed_list)))
    logger.info('List of OA:', list(OA_ALL))
    logger.info('List of AA:', list(AA_ALL))
    logger.info('List of KPP:', list(KPP_ALL))
    logger.info('List of mIOU:', list(mIOU_ALL))
    # Calculate and record the mean and standard deviation; then multiply by 100 to convert them to percentages.
    logger.info('OA=', round(np.mean(OA_ALL) * 100, 2), '+-', round(np.std(OA_ALL) * 100, 2))
    logger.info('AA=', round(np.mean(AA_ALL) * 100, 2), '+-', round(np.std(AA_ALL) * 100, 2))
    logger.info('Kpp=', round(np.mean(KPP_ALL) * 100, 2), '+-', round(np.std(KPP_ALL) * 100, 2))
    logger.info('Acc per class=', np.round(np.mean(EACH_ACC_ALL, 0) * 100, decimals=2), '+-', np.round(np.std(EACH_ACC_ALL, 0) * 100, decimals=2))
    # Record the average training and testing times.
    logger.info("Average training time=", round(np.mean(Train_Time_ALL), 2), '+-', round(np.std(Train_Time_ALL), 3))
    logger.info("Average testing time=", round(np.mean(Test_Time_ALL) * 1000, 2), '+-', round(np.std(Test_Time_ALL) * 1000, 3))

    # Write the average results of the experiment to the text file “mean_result.txt”.
    mean_result_path = os.path.join(save_folder, 'mean_result.txt')
    f = open(mean_result_path, 'w')
    str_results = '\n\n***************Mean result of ' + str(len(seed_list)) + 'times runs ********************' \
                  + '\nList of OA:' + str(list(OA_ALL)) \
                  + '\nList of AA:' + str(list(AA_ALL)) \
                  + '\nList of KPP:' + str(list(KPP_ALL)) \
                  + '\nList of mIOU:' + str(list(mIOU_ALL)) \
                  + '\nList of Tr_Time:' + str(list(Train_Time_ALL)) \
                  + '\nList of Te_Time:' + str(list(Test_Time_ALL)) \
                  + '\nOA=' + str(round(np.mean(OA_ALL) * 100, 2)) + '+-' + str(round(np.std(OA_ALL) * 100, 2)) \
                  + '\nAA=' + str(round(np.mean(AA_ALL) * 100, 2)) + '+-' + str(round(np.std(AA_ALL) * 100, 2)) \
                  + '\nKpp=' + str(round(np.mean(KPP_ALL) * 100, 2)) + '+-' + str(round(np.std(KPP_ALL) * 100, 2)) \
                  + '\nmIOU=' + str(round(np.mean(mIOU_ALL) * 100, 2)) + '+-' + str(round(np.std(mIOU_ALL) * 100, 2)) \
                  + '\nAcc per class=\n' + str(np.round(np.mean(EACH_ACC_ALL, 0) * 100, 2)) + '+-' + str(np.round(np.std(EACH_ACC_ALL, 0) * 100, 2)) \
                  + "\nAverage training time=" + str(np.round(np.mean(Train_Time_ALL), decimals=2)) + '+-' + str(np.round(np.std(Train_Time_ALL), decimals=3)) \
                  + "\nAverage testing time=" + str(np.round(np.mean(Test_Time_ALL), decimals=2)) + '+-' + str(np.round(np.std(Test_Time_ALL), decimals=3))
    f.write(str_results)
    f.close()

    del net
