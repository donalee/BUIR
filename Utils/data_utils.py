import numpy as np
import scipy.sparse as sp
import os

def to_np(x):
	return x.data.cpu().numpy()

def dict_set(base_dict, user_id, item_id, val):
	if user_id in base_dict:
		base_dict[user_id][item_id] = val
	else:
		base_dict[user_id] = {item_id: val}

def list_to_dict(base_list):
	result = {}
	for user_id, item_id, value in base_list:
		dict_set(result, user_id, item_id, value)
	return result

def read_interaction_file(f):
    total_interactions = []
    for user_id, line in enumerate(f.readlines()):
        items = line.replace('\n', '').split(' ')[1:]
        for item in items:
            item_id = item
            total_interactions.append((user_id, item_id, 1))
    return total_interactions

def get_count_dict(total_interactions):
    user_count_dict, item_count_dict = {}, {}

    for interaction in total_interactions:
        user, item, rating = interaction

        if user not in user_count_dict:
            user_count_dict[user] = 0
        if item not in item_count_dict:
            item_count_dict[item] = 0
    
        user_count_dict[user] += 1
        item_count_dict[item] += 1

    return user_count_dict, item_count_dict

def filter_interactions(total_interaction_tmp, user_count_dict, item_count_dict, min_count=[5, 0]):
    total_interactions = []
    user_to_id, item_to_id = {}, {}
    user_count, item_count = 0, 0
    
    for line in total_interaction_tmp:
        user, item, rating = line

        # count filtering
        if user_count_dict[user] < min_count[0]:
            continue

        if item_count_dict[item] < min_count[1]:
            continue

        if user not in user_to_id:
            user_to_id[user] = user_count
            user_count += 1

        if item not in item_to_id:
            item_to_id[item] = item_count
            item_count += 1

        user_id = user_to_id[user]
        item_id = item_to_id[item]
        rating = 1.

        total_interactions.append((user_id, item_id, rating))

    return user_count, item_count, user_to_id, item_to_id, total_interactions

def load_dataset(path, filename, train_ratio=0.5, min_count=[0, 0], random_seed=0):
    np.random.seed(random_seed)
    test_ratio = (1. - train_ratio) / 2
    
    with open(os.path.join(path, filename), 'r') as f:
        total_interaction_tmp = read_interaction_file(f)
    
    user_count_dict, item_count_dict = get_count_dict(total_interaction_tmp)
    user_count, item_count, user_to_id, item_to_id, total_interactions = filter_interactions(total_interaction_tmp, user_count_dict, item_count_dict, min_count=min_count)

    total_mat = list_to_dict(total_interactions)
    train_mat, valid_mat, test_mat = {}, {}, {}

    for user in total_mat:
        items = list(total_mat[user].keys())
        np.random.shuffle(items)

        num_test_items = int(len(items) * test_ratio)
        test_items = items[:num_test_items]
        valid_items = items[num_test_items: num_test_items*2]
        train_items = items[num_test_items*2:]

        for item in test_items:
            dict_set(test_mat, user, item, 1)

        for item in valid_items:
            dict_set(valid_mat, user, item, 1)

        for item in train_items:
            dict_set(train_mat, user, item, 1)
           
    train_mat_t = {}

    for user in train_mat:
        for item in train_mat[user]:
            dict_set(train_mat_t, item, user, 1)
    
    for user in list(valid_mat.keys()):
        for item in list(valid_mat[user].keys()):
            if item not in train_mat_t:
                del valid_mat[user][item]
        if len(valid_mat[user]) == 0:
            del valid_mat[user]
            del test_mat[user]
            
    for user in list(test_mat.keys()):
        for item in list(test_mat[user].keys()):
            if item not in train_mat_t:
                del test_mat[user][item]
        if len(test_mat[user]) == 0:
            del test_mat[user]
            del valid_mat[user]
    
    return user_count, item_count, train_mat, valid_mat, test_mat

def build_adjmat(user_count, item_count, train_mat, selfloop_flag=True):
    R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
    for user in train_mat:
        for item in train_mat[user]:
            R[user, item] = 1
    R = R.tolil()
    
    adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
    adj_mat = adj_mat.tolil()

    adj_mat[:user_count, user_count:] = R
    adj_mat[user_count:, :user_count] = R.T
    adj_mat = adj_mat.todok()

    def mean_adj_single(adj):
        # D^-1 * A
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    def normalized_adj_single(adj):
        # D^-1/2 * A * D^-1/2
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    if selfloop_flag:
        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
    else:
        norm_adj_mat = normalized_adj_single(adj_mat)

    return norm_adj_mat.tocsr()
