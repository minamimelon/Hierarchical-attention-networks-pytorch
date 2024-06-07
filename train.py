import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import get_max_lengths, get_evaluation
from src.dataset import MyDataset
from src.model import HierAttNet
from tensorboardX import SummaryWriter
import argparse
import shutil
import numpy as np
import pickle


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_epoches", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--word_hidden_size", type=int, default=50)
    parser.add_argument("--sent_hidden_size", type=int, default=50)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=5,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str, default="yelp/train.json")
    parser.add_argument("--val_set", type=str, default="yelp/val.json")
    parser.add_argument("--test_set", type=str, default="yelp/test.json")
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--word2vec_path", type=str, default="glove.840B.300d.txt")
    parser.add_argument("--log_path", type=str, default="tensorboard/han_voc_dbg")
    parser.add_argument("--saved_path", type=str, default="trained_models_dbg")
    args = parser.parse_args()
    return args

def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
    else:
        torch.manual_seed(1)
    os.makedirs(opt.saved_path, exist_ok=True)
    output_file = open(opt.saved_path + os.sep + "logs.txt", "w")
    output_file.write("Model's parameters: {}".format(vars(opt)))
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True}
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}

    # max_word_length, max_sent_length = get_max_lengths(opt.train_set)
    max_word_length, max_sent_length = 24,13
    print(max_word_length, max_sent_length)

    test_set_name = opt.test_set.split(".")[0]+".pkl"
    if os.path.isfile(test_set_name):
        with open(test_set_name, 'rb') as f:
            test_set = pickle.load(f)
    else:
        test_set = MyDataset(opt.test_set, opt.word2vec_path, max_sent_length, max_word_length)
        with open(test_set_name, "wb") as f:
            pickle.dump(test_set, f)
    test_generator = DataLoader(test_set, num_workers=10, **test_params)

    val_set_name = opt.val_set.split(".")[0]+".pkl"
    if os.path.isfile(val_set_name):
        with open(val_set_name, 'rb') as f:
            val_set = pickle.load(f)
    else:
        val_set = MyDataset(opt.val_set, opt.word2vec_path, max_sent_length, max_word_length)
        with open(val_set_name, "wb") as f:
            pickle.dump(val_set, f)
    val_generator = DataLoader(val_set, num_workers=10, **test_params)

    training_set_name = opt.train_set.split(".")[0]+".pkl"
    if os.path.isfile(training_set_name):
        with open(training_set_name, 'rb') as f:
            training_set = pickle.load(f)
    else:
        training_set = MyDataset(opt.train_set, opt.word2vec_path, max_sent_length, max_word_length)
        with open(training_set_name, "wb") as f:
            pickle.dump(training_set, f)
    training_generator = DataLoader(training_set, num_workers=10, **training_params)

    model = HierAttNet(opt.word_hidden_size, opt.sent_hidden_size, opt.batch_size, training_set.num_classes,
                       opt.word2vec_path, max_sent_length, max_word_length)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    # writer.add_graph(model, torch.zeros(opt.batch_size, max_sent_length, max_word_length))

    def write_log(file, x ,y):
        with open(opt.log_path+"/"+file+".txt", "a") as f:
            f.write("{},{}\n".format(x,y))

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, momentum=opt.momentum)
    best_loss = 1e5
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epoches):
        model.train()
        for iter, (feature, label) in enumerate(training_generator):
            if torch.cuda.is_available():
                feature = feature.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            model._init_hidden_state()
            predictions = model(feature)
            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()
            training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(), list_metrics=["accuracy"])
            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                iter + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss, training_metrics["accuracy"]))
            writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter)
            writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iter)

            write_log('Train_Loss', loss, epoch * num_iter_per_epoch + iter)
            write_log('Train_Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iter)

        if epoch % opt.test_interval == 0:
            model.eval()
            loss_ls = []
            te_label_ls = []
            te_pred_ls = []
            for te_feature, te_label in val_generator:
                num_sample = len(te_label)
                if torch.cuda.is_available():
                    te_feature = te_feature.cuda()
                    te_label = te_label.cuda()
                with torch.no_grad():
                    model._init_hidden_state(num_sample)
                    te_predictions = model(te_feature)
                te_loss = criterion(te_predictions, te_label)
                loss_ls.append(te_loss * num_sample)
                te_label_ls.extend(te_label.clone().cpu())
                te_pred_ls.append(te_predictions.clone().cpu())
            te_loss = sum(loss_ls) / test_set.__len__()
            te_pred = torch.cat(te_pred_ls, 0)
            te_label = np.array(te_label_ls)
            test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])
            output_file.write(
                "Epoch: {}/{} \nVal loss: {} Val accuracy: {} \nVal confusion matrix: \n{}\n\n".format(
                    epoch + 1, opt.num_epoches,
                    te_loss,
                    test_metrics["accuracy"],
                    test_metrics["confusion_matrix"]))
            print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                optimizer.param_groups[0]['lr'],
                te_loss, test_metrics["accuracy"]))
            writer.add_scalar('Val/Loss', te_loss, epoch)
            writer.add_scalar('Val/Accuracy', test_metrics["accuracy"], epoch)

            write_log('Val_Loss', te_loss, epoch)
            write_log('Val_Accuracy', test_metrics["accuracy"], epoch)

            if te_loss + opt.es_min_delta < best_loss:
                best_loss = te_loss
                best_epoch = epoch
                torch.save(model, opt.saved_path + os.sep + "whole_model_han")

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
                break
    
    model = torch.load(opt.saved_path + os.sep + "whole_model_han")
    model.eval()
    loss_ls = []
    te_label_ls = []
    te_pred_ls = []
    for te_feature, te_label in test_generator:
        num_sample = len(te_label)
        if torch.cuda.is_available():
            te_feature = te_feature.cuda()
            te_label = te_label.cuda()
        with torch.no_grad():
            model._init_hidden_state(num_sample)
            te_predictions = model(te_feature)
        te_loss = criterion(te_predictions, te_label)
        loss_ls.append(te_loss * num_sample)
        te_label_ls.extend(te_label.clone().cpu())
        te_pred_ls.append(te_predictions.clone().cpu())
    te_loss = sum(loss_ls) / test_set.__len__()
    te_pred = torch.cat(te_pred_ls, 0)
    te_label = np.array(te_label_ls)
    test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])
    output_file.write(
        "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
            epoch + 1, opt.num_epoches,
            te_loss,
            test_metrics["accuracy"],
            test_metrics["confusion_matrix"]))
    print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
        epoch + 1,
        opt.num_epoches,
        optimizer.param_groups[0]['lr'],
        te_loss, test_metrics["accuracy"]))
    writer.add_scalar('Test/Loss', te_loss, epoch)
    writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)


if __name__ == "__main__":
    opt = get_args()
    train(opt)
