from Models.BUIR_ID import BUIR_ID
from Models.BUIR_NB import BUIR_NB

from Utils.data_loaders import ImplicitFeedback
from Utils.data_utils import load_dataset, build_adjmat
from Utils.evaluation import evaluate, print_eval_results

import torch
from torch.utils.data import DataLoader

import numpy as np
import argparse
import os, time

def run(args):
    
    path = os.path.join(args.path, args.dataset)
    filename = 'users.dat'

    user_count, item_count, train_mat, valid_mat, test_mat = load_dataset(path, filename, train_ratio=args.train_ratio, random_seed=args.random_seed)
    train_dataset = ImplicitFeedback(user_count, item_count, train_mat)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    if args.model == 'buir-id':
        model = BUIR_ID(user_count, item_count, args.latent_size, args.momentum)
    elif args.model == 'buir-nb':
        norm_adjmat = build_adjmat(user_count, item_count, train_mat, selfloop_flag=False)
        model = BUIR_NB(user_count, item_count, args.latent_size, norm_adjmat, args.momentum, n_layers=args.n_layers, drop_flag=args.drop_flag)

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -np.inf
    early_stop_cnt = 0

    for epoch in range(args.max_epochs):
        tic1 = time.time()
        
        train_loss = []
        for batch in train_loader:
            batch = {key: value.cuda() for key, value in batch.items()} 

            # Forward
            model.train()
            output = model(batch)
            batch_loss = model.get_loss(output)
            train_loss.append(batch_loss)

            # Backward and optimize
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            model._update_target()

        train_loss = torch.mean(torch.stack(train_loss)).data.cpu().numpy()
        toc1 = time.time()

        if epoch % 10 == 0:
            is_improved = False

            model.eval()
            with torch.no_grad():
                tic2 = time.time()
                eval_results = evaluate(model, train_loader, train_mat, valid_mat, test_mat)
                toc2 = time.time()

            if eval_results['valid']['P50'] > best_score:
                is_improved = True
                best_score = eval_results['valid']['P50']
                valid_result = eval_results['valid']
                test_result = eval_results['test']
            
                print('Epoch [{}/{}]'.format(epoch, args.max_epochs))
                print('Training Loss: {:.4f}, Elapsed Time for Training: {:.2f}s, for Testing: {:.2f}s\n'.format(train_loss, toc1-tic1, toc2-tic2))
                print_eval_results(eval_results)

            else:
                early_stop_cnt += 1
                if early_stop_cnt == args.early_stop: break
                
    print('===== [FINAL PERFORMANCE] =====\n')
    print_eval_results({'valid': valid_result, 'test': test_result})

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Invalid boolean value')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='buir-id', help='buir-id | buir-nb')
    parser.add_argument('--latent_size', type=int, default=250)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--drop_flag', type=str2bool, default=False)

    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.995)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--early_stop', type=int, default=10)
    
    parser.add_argument('--path', type=str, default='./Data/')
    parser.add_argument('--dataset', type=str, default='toy-dataset')
    parser.add_argument('--train_ratio', type=float, default=0.5)
    parser.add_argument('--random_seed', type=int, default=0)

    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
     
    # GPU setting
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Random seed initialization
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    run(args)

