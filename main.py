import sys
import os
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import math
from DGCNN_embedding import DGCNN
from mlp_dropout import MLPClassifier
from sklearn import metrics
from embedding import EmbedMeanField, EmbedLoopyBP
from util import cmd_args, load_data

import warnings

# 忽略所有的 DeprecationWarning 警告
warnings.filterwarnings('ignore')

sys.path.append(
    '%s/pytorch_structure2vec-master/s2v_lib' % os.path.dirname(
        os.path.realpath(__file__)))


cmd_args.batch_size=50
# cmd_args.data = 'ENZYMES'
# cmd_args.data = 'DD'
cmd_args.data = 'PROTEINS'
# cmd_args.data='COLLAB'
# cmd_args.data='IMDBBINARY'
# cmd_args.data='IMDBMULTI'
# cmd_args.data='MUTAG'

cmd_args.gm = 'DGCNN'
cmd_args.latent_dim = [64]
# cmd_args.mode = 'cuda:1'
cmd_args.mode = 'gpu'
cmd_args.gpu = 'cuda:7'
# cmd_args.mode = 'cpu'
cmd_args.number_iterations = 5
cmd_args.sortpooling_k=32

device=cmd_args.gpu
# cross_fold= 11
#训练10轮取平均
cross_fold=10
class Classifier(nn.Module):
    def __init__(self,lr, alpha):
        super(Classifier, self).__init__()
        if cmd_args.gm == 'mean_field':
            model = EmbedMeanField
        elif cmd_args.gm == 'loopy_bp':
            model = EmbedLoopyBP
        elif cmd_args.gm == 'DGCNN':
            model = DGCNN
        else:
            print('unknown gm %s' % cmd_args.gm)
            sys.exit()

        if cmd_args.gm == 'DGCNN':
            self.s2v = model(
                latent_dim=cmd_args.latent_dim,
                output_dim=cmd_args.out_dim,
                num_node_feats=cmd_args.feat_dim+cmd_args.attr_dim,
                num_edge_feats=0,
                k=cmd_args.sortpooling_k,device=device,number_iterations=cmd_args.number_iterations,lr=lr,alpha=alpha)
        else:
            self.s2v = model(
                latent_dim=cmd_args.latent_dim,
                output_dim=cmd_args.out_dim,
                num_node_feats=cmd_args.feat_dim,
                num_edge_feats=0,
                max_lv=cmd_args.max_lv)
        out_dim = cmd_args.out_dim
        if out_dim == 0:
            if cmd_args.gm == 'DGCNN':
                out_dim = self.s2v.dense_dim
            else:
                out_dim = cmd_args.latent_dim
        self.mlp = MLPClassifier(
            input_size=out_dim, hidden_size=cmd_args.hidden,
            num_class=cmd_args.num_class, with_dropout=cmd_args.dropout)

    def PrepareFeatureLabel(self, batch_graph,device='cpu'):
        labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            concat_tag = []
        else:
            node_tag_flag = False

        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            concat_feat = []
        else:
            node_feat_flag = False

        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            if node_tag_flag:
                concat_tag += batch_graph[i].node_tags
            if node_feat_flag:
                tmp = torch.from_numpy(
                    batch_graph[i].node_features).type('torch.FloatTensor')
                concat_feat.append(tmp)

        if node_tag_flag:
            concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
            node_tag = torch.zeros(n_nodes, cmd_args.feat_dim)
            node_tag.scatter_(1, concat_tag, 1)

        if node_feat_flag:
            node_feat = torch.cat(concat_feat, 0)

        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels)
            # with continuous node features
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag is False and node_tag_flag:
            node_feat = node_tag
        elif node_feat_flag and node_tag_flag is False:
            pass
        else:
            node_feat = torch.ones(n_nodes, 1)

        if cmd_args.mode == 'gpu':
            # node_feat = node_feat.cuda()
            # labels = labels.cuda()
            node_feat = node_feat.to(device=device)
            labels = labels.to(device=device)

        return node_feat, labels

    def forward(self, batch_graph):
        node_feat, labels = self.PrepareFeatureLabel(batch_graph,device=device)
        if self.training:
            embed, f_loss = self.s2v(batch_graph, node_feat, None)
        else:
            embed = self.s2v(batch_graph, node_feat, None)

        if self.training:
            logits, loss, acc = self.mlp(embed, labels)
            loss += f_loss
            return logits, loss, acc
        else:
            return self.mlp(embed, labels)

    def output_features(self, batch_graph):
        node_feat, labels = self.PrepareFeatureLabel(batch_graph,device=device)
        embed = self.s2v(batch_graph, node_feat, None)
        return embed, labels


def loop_dataset(g_list, classifier, sample_idxes, optimizer=None,
                 bsize=cmd_args.batch_size):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize # noqa
    pbar = tqdm(range(total_iters), unit='batch')
    all_targets = []
    all_scores = []

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize: (pos + 1) * bsize]

        batch_graph = [g_list[idx] for idx in selected_idx]
        targets = [g_list[idx].label for idx in selected_idx]
        all_targets += targets
        logits, loss, acc = classifier(batch_graph)
        all_scores.append(logits[:, 1].detach())  # for binary classification

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().numpy()
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc))

        total_loss.append(np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().numpy()

    # np.savetxt('test_scores.txt', all_scores)  # output test predictions

    all_targets = np.array(all_targets)
    fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    avg_loss = np.concatenate((avg_loss, [auc]))

    return avg_loss


if __name__ == '__main__':
    print(cmd_args)
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    # train_graphs, test_graphs = load_data()
    g_list = load_data()
    cnt=0
    for g in g_list:
        g.order=cnt
        g.dataset=cmd_args.data
        cnt+=1

    alphas = [0.5]
    lrs = [1.0]
    import time
    start_time = time.time()
    print('lrs:',lrs)
    print('alphas:',alphas)
    for lr in lrs:
        for alpha in alphas:
            print('lr:',lr,'alpha:',alpha)
            for i in range(1, cross_fold):
                train_idxes = np.loadtxt('./data/%s/10fold_idx/train_idx-%d.txt' % (cmd_args.data, i), dtype=np.int32).tolist()
                test_idxes = np.loadtxt('./data/%s/10fold_idx/test_idx-%d.txt' % (cmd_args.data, i), dtype=np.int32).tolist()
                train_graphs, test_graphs = [g_list[i] for i in train_idxes], [g_list[i] for i in test_idxes]
                print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

                if cmd_args.sortpooling_k <= 1:
                    num_nodes_list = sorted([
                        g.num_nodes for g in train_graphs + test_graphs])
                    cmd_args.sortpooling_k = num_nodes_list[
                        int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
                    cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
                    print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

                classifier = Classifier(lr=lr,alpha=alpha)
                if cmd_args.mode == 'gpu':
                    classifier=classifier.to(device=device)

                optimizer = optim.Adam(
                    classifier.parameters(), lr=cmd_args.learning_rate, amsgrad=True,
                    weight_decay=0.0008)

                train_idxes = list(range(len(train_graphs)))
                best_loss = None
                max_acc = 0.0
                best_epoch = 0
                for epoch in range(cmd_args.num_epochs):
                    random.shuffle(train_idxes)
                    classifier.train()
                    avg_loss = loop_dataset(
                        train_graphs, classifier, train_idxes, optimizer=optimizer)
                    if not cmd_args.printAUC:
                        avg_loss[2] = 0.0
                    print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2])) # noqa

                    classifier.eval()
                    test_loss = loop_dataset(
                        test_graphs, classifier, list(range(len(test_graphs))))
                    if not cmd_args.printAUC:
                        test_loss[2] = 0.0
                    print('\033[93maverage test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, test_loss[0], test_loss[1], test_loss[2])) # noqa
                    # max_acc = max(max_acc, test_loss[1])
                    if test_loss[1] > max_acc:
                        max_acc = test_loss[1]
                        best_loss = test_loss
                        best_epoch = epoch

                        features, labels = classifier.output_features(train_graphs)
                        labels = labels.type('torch.FloatTensor')
                        np.savetxt(
                            f'./results/{cmd_args.data}_K{cmd_args.sortpooling_k}_iter{cmd_args.number_iterations}_extracted_features_train.txt',
                            torch.cat(
                                [labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(),
                            '%.4f')
                        features, labels = classifier.output_features(test_graphs)
                        labels = labels.type('torch.FloatTensor')
                        np.savetxt(
                            f'./results/{cmd_args.data}_K{cmd_args.sortpooling_k}_iter{cmd_args.number_iterations}_extracted_features_test.txt',
                            torch.cat(
                                [labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(),
                            '%.4f')

                print(f'round: {i}, max_acc: {max_acc}, best_epoch: {best_epoch}')

                with open(f'./results/{cmd_args.data}_K{cmd_args.sortpooling_k}_iter{cmd_args.number_iterations}_lr{lr}_alpha{alpha}_acc_result.txt', 'a+') as f:
                    f.write('round: '+str(i)+', max_acc: '+str(max_acc)+', best_epoch: '+str(best_epoch))

                if cmd_args.printAUC:
                    with open(f'./results/{cmd_args.data}_K{cmd_args.sortpooling_k}_iter{cmd_args.number_iterations}_lr{lr}_alpha{alpha}_auc_results.txt', 'a+') as f:
                        f.write(str(test_loss[2]) + '\n')

                if cmd_args.extract_features:
                    features, labels = classifier.output_features(train_graphs)
                    labels = labels.type('torch.FloatTensor')
                    np.savetxt(f'./results/{cmd_args.data}_K{cmd_args.sortpooling_k}_iter{cmd_args.number_iterations}_lr{lr}_alpha{alpha}_extracted_features_train.txt', torch.cat(
                        [labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(),
                            '%.4f')
                    features, labels = classifier.output_features(test_graphs)
                    labels = labels.type('torch.FloatTensor')
                    np.savetxt(f'./results/{cmd_args.data}_K{cmd_args.sortpooling_k}_iter{cmd_args.number_iterations}_lr{lr}_alpha{alpha}_extracted_features_test.txt', torch.cat(
                        [labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(),
                            '%.4f')

    print('cost: ',time.time() - start_time)