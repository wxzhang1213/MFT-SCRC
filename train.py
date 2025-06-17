import logging
from tqdm import tqdm
import time
import torch
import torch.nn as nn
from torch import optim
from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop

logger = logging.getLogger('MSA')


class TAN_train():
    def __init__(self, args):
        assert args.train_mode == 'regression'
        self.args = args
        self.args.tasks = "M"
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)
        self.criterion = nn.L1Loss(reduction="mean")

    def do_train(self, model, dataloader):
        def count_parameters(model):
            answer = 0
            bert = 0
            audio = 0
            video = 0
            other = 0
            for n, p in model.named_parameters():
                answer += p.numel()
                if 'text_model' in n:
                    bert += p.numel()
                if 'audio_model' in n:
                    audio += p.numel()
                if 'video_model' in n:
                    video += p.numel()
                if 'text_model' not in n and 'audio_model' not in n and 'video_model' not in n:
                    other += p.numel()
            return answer, bert, audio, video, other

        answer, bert, audio, video, other = count_parameters(model)
        logger.info(f'The model has {answer} parameters.')
        logger.info(f'The model has {bert} bert parameters.')
        logger.info(f'The model has {audio} audio parameters.')
        logger.info(f'The model has {video} video parameters.')
        logger.info(f'The model has {other} other parameters.')

        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.text_model.named_parameters())
        audio_params = list(model.audio_model.named_parameters())
        video_params = list(model.video_model.named_parameters())

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        audio_params = [p for n, p in audio_params]
        video_params = [p for n, p in video_params]
        model_params_other = [p for n, p in list(model.named_parameters()) if 'text_model' not in n and \
                              'audio_model' not in n and 'video_model' not in n and 'discriminator' not in n]
        
        class_params = []
        for name, p in model.named_parameters():
            if p.requires_grad:
                if 'discriminator' in name:
                    class_params.append(p)

        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert,
             'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
            {'params': audio_params, 'weight_decay': self.args.weight_decay_audio, 'lr': self.args.learning_rate_audio},
            {'params': video_params, 'weight_decay': self.args.weight_decay_video, 'lr': self.args.learning_rate_video},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other, 'lr': self.args.learning_rate_other}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)

        optimizer_class_parameters = [
            {'params': class_params, 'weight_decay': self.args.weight_decay_class, 'lr': self.args.learning_rate_class}
        ]
        optimizer_class = optim.Adam(optimizer_class_parameters)

        # initilize results
        logger.info("Start training...")
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss', 'MAE'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        left_epochs = self.args.update_epochs
        # loop util earlystop
        while True:
            epochs += 1
            left_epochs -= 1
            # train
            y_pred, y_true = [], []
            model.train()
            train_loss = 0.0
            train_loss_1 = 0.0
            train_loss_2 = 0.0
            s_t = time.time()

            with tqdm(dataloader['train']) as td:
                for batch_idx, batch_data in enumerate(td, 1):
                    # complete view
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                    vision_lengths = batch_data['vision_lengths'].to(self.args.device)

                    optimizer_class.zero_grad()
                    results = model(text, audio, audio_lengths, vision, vision_lengths)
                    loss2 = results['sm'] 
                    loss2.backward()
                    optimizer_class.step()

                    # forward
                    optimizer.zero_grad()
                    results = model(text, audio, audio_lengths, vision, vision_lengths, labels)
                    # store results
                    y_pred.append(results['M'].cpu())
                    y_true.append(labels.cpu())
                    # compute loss
                    ## prediction loss
                    loss_pred = self.criterion(results['M'].view(-1), labels.view(-1))
                    ## total loss
                    loss = loss_pred + self.args.alpha * results['con'] + self.args.beta * results['im']
                    loss.backward()
                    # update parameters
                    optimizer.step()
                    train_loss += loss.item()
                    if results['con'] != 0:
                        train_loss_1 += results['con'].item()
                    train_loss_2 += results['im'].item()

            e_t = time.time()
            logger.info(f'One epoch time for training: {e_t - s_t:.3f}s.')
            train_loss = train_loss / len(dataloader['train'])
            train_loss_1 = train_loss_1 / len(dataloader['train'])
            train_loss_2 = train_loss_2 / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            log_infos = [''] * 8
            log_infos[0] = log_infos[-1] = '-' * 100

            # validation
            s_t = time.time()
            val_results = self.do_test(model, dataloader['valid'], epochs=epochs)
            test_results = self.do_test(model, dataloader['test'], epochs=epochs)

            e_t = time.time()
            logger.info(f'One epoch time for validation: {e_t - s_t:.3f}s.')

            cur_valid = val_results[self.args.KeyEval]
            # save best models
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                self.best_epoch = best_epoch
                # save models
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
                log_infos[5] = f'==> Note: achieve best [Val] results at epoch {best_epoch}'

            log_infos[1] = f"Seed {self.args.seed} ({self.args.seeds.index(self.args.seed) + 1}/{self.args.num_seeds}) " \
                           f"| Epoch {epochs} (early stop={epochs - best_epoch}) | Train Loss {train_loss:.4f} | Loss_1 {train_loss_1:.4f} | Loss_2 {train_loss_2:.4f}"
            log_infos[2] = f"[Train] {dict_to_str(train_results)}"
            log_infos[3] = f"  [Val] {dict_to_str(val_results)}"
            log_infos[4] = f"  [Test] {dict_to_str(test_results)}"

            # log information
            for log_info in log_infos:
                if log_info: logger.info(log_info)

            # early stop
            if epochs - best_epoch >= self.args.early_stop or left_epochs == 0:
                logger.info(
                    f"==> Note: since '{self.args.KeyEval}' does not improve in the past {self.args.early_stop} epochs, early stop the training process!")
                return

    def do_test(self, model, dataloader, epochs=None):
        if epochs is None:
            logger.info("=" * 30 + f"Start Test of Seed {self.args.seed}" + "=" * 30)
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        eval_loss_1 = 0.0
        eval_loss_2 = 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_idx, batch_data in enumerate(td, 1):
                    # complete view
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)

                    audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                    vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device).view(-1)
                    results = model(text, audio, audio_lengths, vision, vision_lengths, labels)

                    # compute loss
                    loss_pred = self.criterion(results['M'].view(-1), labels.view(-1))
                    ## total loss
                    loss = loss_pred
                    eval_loss += loss.item()
                    if results['con'] != 0:
                        eval_loss_1 += results['con'].item()
                    eval_loss_2 += results['im'].item()
                    y_pred.append(results['M'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        eval_loss_1 = eval_loss_1 / len(dataloader)
        eval_loss_2 = eval_loss_2 / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results['Loss'] = eval_loss
        eval_results['Loss_con'] = eval_loss_1
        eval_results['Loss_adv'] = eval_loss_2
        if epochs is None:  # for TEST
            logger.info(f"\n [Test] {dict_to_str(eval_results)}")
            logger.info(
                f"==> Note: achieve this results at epoch {self.best_epoch} (best [Val]) / {getattr(self, 'best_test_epoch', None)} (best [Test])")
        return eval_results
