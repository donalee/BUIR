import torch
import numpy as np
import copy 

from Utils.data_utils import to_np

def evaluate(model, data_loader, train_mat, valid_mat, test_mat):
    
    metrics = {'P10':[], 'P20':[], 'P50':[], 'R10':[], 'R20':[], 'R50':[], 'N10':[], 'N20':[], 'N50':[]}
    eval_results = {'valid':copy.deepcopy(metrics), 'test': copy.deepcopy(metrics)}
  
    u_online, u_target, i_online, i_target = model.get_embedding()
    score_mat_ui = torch.matmul(u_online, i_target.transpose(0, 1))
    score_mat_iu = torch.matmul(u_target, i_online.transpose(0, 1))
    score_mat = score_mat_ui + score_mat_iu
   
    sorted_mat = torch.argsort(score_mat.cpu(), dim=1, descending=True)
    
    for test_user in test_mat:
        sorted_list = list(to_np(sorted_mat[test_user]))
    
        for mode in ['valid', 'test']:
            sorted_list_tmp = []
            if mode == 'valid':
                gt_mat = valid_mat
                already_seen_items = set(train_mat[test_user].keys()) | set(test_mat[test_user].keys())
            elif mode == 'test':
                gt_mat = test_mat
                already_seen_items = set(train_mat[test_user].keys()) | set(valid_mat[test_user].keys())

            for item in sorted_list:
                if item not in already_seen_items:
                    sorted_list_tmp.append(item)
                if len(sorted_list_tmp) == 50: break
                
            hit_10 = len(set(sorted_list_tmp[:10]) & set(gt_mat[test_user].keys()))
            hit_20 = len(set(sorted_list_tmp[:20]) & set(gt_mat[test_user].keys()))
            hit_50 = len(set(sorted_list_tmp[:50]) & set(gt_mat[test_user].keys()))
        
            eval_results[mode]['P10'].append(hit_10 / min(10, len(gt_mat[test_user].keys())))
            eval_results[mode]['P20'].append(hit_20 / min(20, len(gt_mat[test_user].keys())))
            eval_results[mode]['P50'].append(hit_50 / min(50, len(gt_mat[test_user].keys())))

            eval_results[mode]['R10'].append(hit_10 / len(gt_mat[test_user].keys()))
            eval_results[mode]['R20'].append(hit_20 / len(gt_mat[test_user].keys()))
            eval_results[mode]['R50'].append(hit_50 / len(gt_mat[test_user].keys()))       
   
            # ndcg
            denom = np.log2(np.arange(2, 10 + 2))
            dcg_10 = np.sum(np.in1d(sorted_list_tmp[:10], list(gt_mat[test_user].keys())) / denom)
            idcg_10 = np.sum((1 / denom)[:min(len(list(gt_mat[test_user].keys())), 10)])
            
            denom = np.log2(np.arange(2, 20 + 2))
            dcg_20 = np.sum(np.in1d(sorted_list_tmp[:20], list(gt_mat[test_user].keys())) / denom)
            idcg_20 = np.sum((1 / denom)[:min(len(list(gt_mat[test_user].keys())), 20)])
            
            denom = np.log2(np.arange(2, 50 + 2))
            dcg_50 = np.sum(np.in1d(sorted_list_tmp[:50], list(gt_mat[test_user].keys())) / denom)
            idcg_50 = np.sum((1 / denom)[:min(len(list(gt_mat[test_user].keys())), 50)])
            
            eval_results[mode]['N10'].append(dcg_10 / idcg_10)
            eval_results[mode]['N20'].append(dcg_20 / idcg_20)
            eval_results[mode]['N50'].append(dcg_50 / idcg_50)
                
    for mode in ['test', 'valid']:
        for topk in [10, 20, 50]:
            eval_results[mode]['P' + str(topk)] = round(np.asarray(eval_results[mode]['P' + str(topk)]).mean(), 4)
            eval_results[mode]['R' + str(topk)] = round(np.asarray(eval_results[mode]['R' + str(topk)]).mean(), 4)   
            eval_results[mode]['N' + str(topk)] = round(np.asarray(eval_results[mode]['N' + str(topk)]).mean(), 4)

    return eval_results

def print_eval_results(eval_results):

    for mode in ['valid', 'test']:
        for topk in [10, 20, 50]:
            p = eval_results[mode]['P' + str(topk)]
            r = eval_results[mode]['R' + str(topk)] 
            n = eval_results[mode]['N' + str(topk)] 
            print('{:5s} P@{}: {:.4f}, R@{}: {:.4f}, N@{}: {:.4f}'.format(mode.upper(), topk, p, topk, r, topk, n))
        print()

