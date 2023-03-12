'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import multiprocessing

import numpy as np
import torch

import model
import utils
import world
from utils import timer

CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"


def test_one_batch(X, top_bottom_items_list):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items, top_bottom_items_list)
    pre, recall, ndcg = [], [], []
    pre_top, recall_top, ndcg_top = [], [], []
    pre_bottom, recall_bottom, ndcg_bottom = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k, top_bottom_items_list)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k)[0])

        pre_top.append(ret['precision_top'])
        recall_top.append(ret['recall_top'])
        ndcg_top.append(utils.NDCGatK_r(groundTrue, r, k)[1])

        pre_bottom.append(ret['precision_bottom'])
        recall_bottom.append(ret['recall_bottom'])
        ndcg_bottom.append(utils.NDCGatK_r(groundTrue, r, k)[2])

    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg),
            'recall_top': np.array(recall_top),
            'precision_top': np.array(pre_top),
            'ndcg_top': np.array(ndcg_top),
            'recall_bottom': np.array(recall_bottom),
            'precision_bottom': np.array(pre_bottom),
            'ndcg_bottom': np.array(ndcg_bottom),
            }


def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    top_bottom_items_list = split_items(dataset)
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'precision_top': np.zeros(len(world.topks)),
               'precision_bottom': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'recall_top': np.zeros(len(world.topks)),
               'recall_bottom': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks)),
               'ndcg_top': np.zeros(len(world.topks)),
               'ndcg_bottom': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            # rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [
            #         utils.AUC(rating[i],
            #                   dataset,
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x, top_bottom_items_list[0]))
        scale = float(u_batch_size / len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']

            results['recall_top'] += result['recall_top']
            results['precision_top'] += result['precision_top']
            results['ndcg_top'] += result['ndcg_top']

            results['recall_bottom'] += result['recall_bottom']
            results['precision_bottom'] += result['precision_bottom']
            results['ndcg_bottom'] += result['ndcg_bottom']

        # recall_list, precision_list, ndcg_list = [], [], []
        # for recall, precision, ndcg in zip(list(results['recall']), list(results['precision']), list(results['ndcg'])):
        #     recall_list.append(recall/float(len(users)))
        #     precision_list.append(precision/float(len(users)))
        #     ndcg_list.append(ndcg/float(len(users)))
            # ndcg_list.append(' '.format(['{:.4f}'.format(r) for r in ndcg/float(len(users))]))

        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))

        results['recall_top'] /= float(len(users))
        results['precision_top'] /= float(len(users))
        results['ndcg_top'] /= float(len(users))

        results['recall_bottom'] /= float(len(users))
        results['precision_bottom'] /= float(len(users))
        results['ndcg_bottom'] /= float(len(users))

        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}', {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Recall_top@{world.topks}', {str(world.topks[i]): results['recall_top'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Recall_bottom@{world.topks}', {str(world.topks[i]): results['recall_bottom'][i] for i in range(len(world.topks))}, epoch)

            w.add_scalars(f'Test/Precision@{world.topks}', {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision_top@{world.topks}', {str(world.topks[i]): results['precision_top'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision_bottom@{world.topks}', {str(world.topks[i]): results['precision_bottom'][i] for i in range(len(world.topks))}, epoch)

            w.add_scalars(f'Test/NDCG@{world.topks}', {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG_top@{world.topks}', {str(world.topks[i]): results['ndcg_top'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG_bottom@{world.topks}', {str(world.topks[i]): results['ndcg_top'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print('Recall@5: {:.4f}\t{:.4f}\t{:.4f}'.format(results['recall'][0], results['recall_top'][0], results['recall_bottom'][0]))
        print('Recall@10: {:.4f}\t{:.4f}\t{:.4f}'.format(results['recall'][1], results['recall_top'][1], results['recall_bottom'][1]))
        print('Recall@20: {:.4f}\t{:.4f}\t{:.4f}'.format(results['recall'][2], results['recall_top'][2], results['recall_bottom'][2]))
        print('Recall@50: {:.4f}\t{:.4f}\t{:.4f}'.format(results['recall'][3], results['recall_top'][3], results['recall_bottom'][3]))
        print('NDCG@5: {:.4f}\t{:.4f}\t{:.4f}'.format(results['ndcg'][0], results['ndcg_top'][0], results['ndcg_bottom'][0]))
        print('NDCG@10: {:.4f}\t{:.4f}\t{:.4f}'.format(results['ndcg'][1], results['ndcg_top'][1], results['ndcg_bottom'][1]))
        print('NDCG@20: {:.4f}\t{:.4f}\t{:.4f}'.format(results['ndcg'][2], results['ndcg_top'][2], results['ndcg_bottom'][2]))
        print('NDCG@50: {:.4f}\t{:.4f}\t{:.4f}'.format(results['ndcg'][3], results['ndcg_top'][3], results['ndcg_bottom'][3]))
        return results

def split_items(data):
    item_degree_dict = {}
    for item_dict in [data.train_dict.items(), data.test_dict.items()]:
        # for key, values in data.train_dict.items():
        for key, values in item_dict:
            for v in values:
                if v in item_degree_dict.keys():
                    item_degree_dict[v] += 1
                else:
                    item_degree_dict[v] = 0
    sorted_item_degree_dict = sorted(item_degree_dict.items(), key=lambda x: x[1], reverse=True)
    items = []
    ratio_list = [0.2]
    for ratio in ratio_list:
        top_items = sorted_item_degree_dict[:int(ratio*len(item_degree_dict))]
        bottom_items = sorted_item_degree_dict[int(ratio*len(item_degree_dict)):]
        items.append([set(np.array(top_items)[:, 0]), set(np.array(bottom_items)[:, 0])])
    return items