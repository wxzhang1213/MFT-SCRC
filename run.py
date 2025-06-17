import os
import gc
import time
import random
import logging
import torch
import argparse
import numpy as np
import pandas as pd
from models import model
from trains.train import TAN_train
from data.load_data import MMDataLoader

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run(args, dataloader):
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    args.model_save_path = os.path.join(args.model_save_dir, f'{args.modelName}-{args.datasetName}.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    # load models
    model = model.Model(args).to(device)
    def count_parameters(model):
        answer = 0
        for p in model.parameters():
            if p.requires_grad:
                answer += p.numel()
        return answer

    logger.info(f'The models has {count_parameters(model)} trainable parameters')

    atio = TAN_train(args)
    # do train
    atio.do_train(model, dataloader)
    # load pretrained models
    assert os.path.exists(args.model_save_path)
    model.load_state_dict(torch.load(args.model_save_path))
    model.to(device)
    # do test
    results = atio.do_test(model, dataloader['test'])

    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(5)
    return results


def run_normal(args):
    init_args = args
    model_results = []
    seeds = args.seeds

    args = init_args
    # load data
    dataloader = MMDataLoader(args)
    # run results
    for i, seed in enumerate(seeds):
        setup_seed(seed)
        args.seed = seed
        logger.info('Start running %s...' %(args.modelName))
        logger.info(args)
        # runnning
        args.cur_time = i+1
        test_results = run(args, dataloader)
        # restore results
        model_results.append(test_results)
        logger.info(f"==> Test results of seed {seed}:\n{test_results}")

    criterions = list(model_results[0].keys())
    # load other results
    save_path = os.path.join(args.res_save_dir, f'{args.datasetName}-{args.train_mode}.csv')
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Model"] + criterions)
    # save results
    res = [args.modelName]
    for c in criterions:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values)*100, 2)
        std = round(np.std(values)*100, 2)
        res.append((mean, std))
    df.loc[len(df)] = res
    df.to_csv(save_path, index=None)
    logger.info('Results are added to %s...' %(save_path))

    # detailed results
    import datetime
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(args.res_save_dir, f'{args.datasetName}-{args.train_mode}-detail.csv')
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Time", "Model", "Params", "Seed"] + criterions)
    # seed
    for i, seed in enumerate(seeds):
        res = [cur_time, args.modelName, str(args), f'{seed}']
        for c in criterions:
            val = round(model_results[i][c]*100, 2)
            res.append(val)
        df.loc[len(df)] = res
    # mean
    res = [cur_time, args.modelName, str(args), '<mean/std>']
    for c in criterions:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values)*100, 2)
        std = round(np.std(values)*100, 2)
        res.append((mean, std))
    df.loc[len(df)] = res
    # max
    res = [cur_time, args.modelName, str(args), '<max/seed>']
    for c in criterions:
        values = [r[c] for r in model_results]
        max_val = round(np.max(values)*100, 2)
        max_seed = seeds[np.argmax(values)]
        res.append((max_val, max_seed))
    df.loc[len(df)] = res
    # min
    res = [cur_time, args.modelName, str(args), '<min/seed>']
    for c in criterions:
        values = [r[c] for r in model_results]
        min_val = round(np.min(values)*100, 2)
        min_seed = seeds[np.argmin(values)]
        res.append((min_val, min_seed))
    df.loc[len(df)] = res
    df.to_csv(save_path, index=None)
    logger.info('Detailed results are added to %s...' %(save_path))


def set_log(args):
    res_dir = os.path.join(args.res_save_dir, 'normals')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    log_file_path = os.path.join(res_dir, f'{args.modelName}-{args.datasetName}.log')
    # set logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    for ph in logger.handlers:
        logger.removeHandler(ph)
    # add FileHandler to log file
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    # add StreamHandler to terminal outputs
    formatter_stream = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)
    return logger


def parse_args(datasetName):
    parser = argparse.ArgumentParser()
    parser.add_argument('--need_task_scheduling', type=bool, default=False,
                        help='use the task scheduling module.')
    parser.add_argument('--use_bert', type=bool, default=True,
                        help='use bert to encode text ?')
    parser.add_argument('--use_finetune', type=bool, default=True,
                        help='use finetune to update bert ?')
    parser.add_argument('--language', type=str, default='en', help='support en/cn')
    parser.add_argument('--bert_model_path', type=str, default='./data/pretrained_berts/bert_en')
    parser.add_argument('--learnable_pos_emb', type=bool, default=True)

    parser.add_argument('--modelName', type=str, default='MFT', help='support MFT')
    parser.add_argument('--train_mode', type=str, default="regression", help='regression')
    parser.add_argument('--dataPath', type=str, default='/home/cpss/workspace/MultiModal_1/data/Datasets/')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers of loading data')
    parser.add_argument('--model_save_dir', type=str, default='results/models',
                        help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default='results/results',
                        help='path to save results.')
    
    parser.add_argument('--seed', type=int, default=1111, help='start seed')
    parser.add_argument('--num_seeds', type=int, default=None, help='number of total seeds')
    parser.add_argument('--exp_name', type=str, default='', help='experiment name')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_audio_len', type=int, default=500, help='max length of audio sequence')
    parser.add_argument('--grad_clip', type=float, default=5.0)

    if datasetName == 'mosi':
        parser.add_argument('--datasetName', type=str, default='mosi')
        parser.add_argument('--weight_decay_bert', type=float, default=1e-6)
        parser.add_argument('--weight_decay_audio', type=float, default=1e-3)
        parser.add_argument('--weight_decay_video', type=float, default=1e-3)
        parser.add_argument('--weight_decay_other', type=float, default=1e-3)
        parser.add_argument('--weight_decay_class', type=float, default=1e-3)
        parser.add_argument('--learning_rate_bert', type=float, default=1e-5)
        parser.add_argument('--learning_rate_audio', type=float, default=1e-4)
        parser.add_argument('--learning_rate_video', type=float, default=1e-4)
        parser.add_argument('--learning_rate_other', type=float, default=1e-4)
        parser.add_argument('--learning_rate_class', type=float, default=1e-4)
        parser.add_argument('--update_epochs', type=int, default=20)
        parser.add_argument('--early_stop', type=int, default=10)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--a_lstm_hidden_size', type=int, default=10, help='hidden size of a_lstm')
        parser.add_argument('--a_lstm_layers', type=int, default=1, help='layer of v_lstm')
        parser.add_argument('--v_lstm_hidden_size', type=int, default=10, help='hidden size of v_lstm')
        parser.add_argument('--v_lstm_layers', type=int, default=1, help='layer of v_lstm')

        parser.add_argument('--feature_dims', type=int, default=[768, 5, 20], help='dim of input modality')
        parser.add_argument('--text_out', type=int, default=768, help='dim of text representation')
        parser.add_argument('--audio_out', type=int, default=40, help='dim of audio representation')
        parser.add_argument('--video_out', type=int, default=40, help='dim of video representation')
        parser.add_argument('--modal_dim', type=int, default=80, help='dim of modality representation, 80')
        parser.add_argument('--hid_dim', type=int, default=80, help='hidden dim of feature, 80')
        parser.add_argument('--post_fusion_dim', type=int, default=40, help='dim of post fusion representation, 40')
        parser.add_argument('--depth', type=int, default=1, help='depth of transformer')
        parser.add_argument('--alpha', type=float, default=0.1, help='0.1')
        parser.add_argument('--beta', type=float, default=0.08, help='0.08')
        parser.add_argument('--gamma', type=float, default=0.4)

        parser.add_argument('--att_drop', type=float, default=0.3, help='dropout rate of attention layer')
        parser.add_argument('--ff_drop', type=float, default=0.2, help='dropout rate of ff layer')
        parser.add_argument('--emb_drop', type=float, default=0.2, help='dropout rate of embedding')
        parser.add_argument('--post_fusion_drop', type=float, default=0.2, help='dropout rate of classify layer')
        parser.add_argument('--ff_expansion', type=int, default=2, help='expansion rate of ff layer')
        parser.add_argument('--KeyEval', type=str, default='MAE')
        
    elif datasetName == 'mosei':
        parser.add_argument('--datasetName', type=str, default='mosei')
        parser.add_argument('--weight_decay_bert', type=float, default=1e-6)
        parser.add_argument('--weight_decay_audio', type=float, default=1e-3)
        parser.add_argument('--weight_decay_video', type=float, default=1e-3)
        parser.add_argument('--weight_decay_other', type=float, default=1e-3)
        parser.add_argument('--weight_decay_class', type=float, default=1e-3)
        parser.add_argument('--learning_rate_bert', type=float, default=1e-5)
        parser.add_argument('--learning_rate_audio', type=float, default=1e-4)
        parser.add_argument('--learning_rate_video', type=float, default=1e-4)
        parser.add_argument('--learning_rate_other', type=float, default=1e-4)
        parser.add_argument('--learning_rate_class', type=float, default=1e-4)
        parser.add_argument('--update_epochs', type=int, default=4)
        parser.add_argument('--early_stop', type=int, default=3)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--a_lstm_hidden_size', type=int, default=40, help='hidden size of a_lstm')
        parser.add_argument('--a_lstm_layers', type=int, default=1, help='layer of a_lstm')
        parser.add_argument('--v_lstm_hidden_size', type=int, default=40, help='hidden size of v_lstm')
        parser.add_argument('--v_lstm_layers', type=int, default=1, help='layer of v_lstm')

        parser.add_argument('--feature_dims', type=int, default=[768, 74, 35], help='dim of input modality')
        parser.add_argument('--text_out', type=int, default=768, help='dim of text representation')
        parser.add_argument('--audio_out', type=int, default=120, help='dim of audio representation')
        parser.add_argument('--video_out', type=int, default=120, help='dim of video representation')
        parser.add_argument('--modal_dim', type=int, default=120, help='dim of modality representation')
        parser.add_argument('--hid_dim', type=int, default=120, help='hidden dim of feature')
        parser.add_argument('--post_fusion_dim', type=int, default=120, help='dim of post fusion representation,120')
        parser.add_argument('--depth', type=int, default=1, help='depth of transformer, 1')
        parser.add_argument('--alpha', type=float, default=0.4, help='0.4')
        parser.add_argument('--beta', type=float, default=0.08, help='0.08')
        parser.add_argument('--gamma', type=float, default=0.5)
    
        parser.add_argument('--att_drop', type=float, default=0.32, help='dropout rate of attention layer, 0.3')
        parser.add_argument('--ff_drop', type=float, default=0.2, help='dropout rate of ff layer, 0.2')
        parser.add_argument('--emb_drop', type=float, default=0.2, help='dropout rate of embedding, 0.2')
        parser.add_argument('--post_fusion_drop', type=float, default=0.2, help='dropout rate of classify layer, 0.2')
        parser.add_argument('--ff_expansion', type=int, default=2, help='expansion rate of ff layer')
        parser.add_argument('--KeyEval', type=str, default='MAE')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args('mosei')
    global logger; logger = set_log(args)
    args.seeds = [11111] if args.num_seeds is None else list(range(args.seed, args.seed + args.num_seeds))
    args.num_seeds = len(args.seeds)
    run_normal(args)
