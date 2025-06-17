import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.subNets.BertTextEncoder import BertTextEncoder
from models.subNets.my_FFTNet import Simple_Transformer
from models.subNets.loss import Con_Label_Loss, Con_loss, time_consistency, Feature_Diff


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        # unimodal encoders
        self.modal_dim = args.modal_dim
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)

        audio_in, video_in = args.feature_dims[1:]
        self.audio_model = AuViSubNet(audio_in, args.a_lstm_hidden_size, args.audio_out,
                            num_layers=args.a_lstm_layers, bidirectional=True, device=args.device)
        self.video_model = AuViSubNet(video_in, args.v_lstm_hidden_size, args.video_out,
                            num_layers=args.v_lstm_layers, bidirectional=True, device=args.device)

        # equalization
        self.text_I = nn.Sequential(
            nn.Linear(args.text_out, args.modal_dim, bias=True), nn.ReLU())
        self.audio_I = nn.Sequential(
            nn.Linear(args.audio_out, args.modal_dim, bias=True), nn.ReLU())
        self.video_I = nn.Sequential(
            nn.Linear(args.video_out, args.modal_dim, bias=True), nn.ReLU())
        
        self.discriminator = Discriminator(args.modal_dim, args.modal_dim//4)

        # fusion
        self.fusion = Simple_Transformer(device=args.device, 
                        depth=args.depth, modal_dim=args.modal_dim, hid_dim=args.hid_dim, 
                        attn_dropout=args.att_drop, ff_expansion=args.ff_expansion, ff_dropout=args.ff_drop, 
                        learnable_pos_emb=args.learnable_pos_emb,
                        emb_dropout=args.emb_drop, max_len=args.max_audio_len)

        # final prediction module
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_drop)
        self.post_fusion_layer_1 = nn.Linear(args.modal_dim*3, args.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim, 1)
    
    def forward_once(self, text, audio, audio_lengths, video, video_lengths, label=None):
        mask_len = torch.sum(text[:, 1, :], dim=1, keepdim=True).int().to(self.device)
        text_lengths = mask_len.squeeze().int().detach()
        audio_lengths = audio_lengths.int().detach()
        video_lengths = video_lengths.int().detach()

        # unimodal encoders
        text, _ = self.text_model(text)
        audio = self.audio_model(audio, audio_lengths)
        video = self.video_model(video, video_lengths)
        batch = text.size()[0]

        # get  mask
        modality_masks = [length_to_mask(length=seq_len, max_len=max_len)
                          for seq_len, max_len in zip([text_lengths, audio_lengths, video_lengths],
                                                      [text.shape[1], audio.shape[1], video.shape[1]])]
        
        text_mask = modality_masks[0].unsqueeze(-1).detach().to(self.device)
        audio_mask = modality_masks[1].unsqueeze(-1).detach().to(self.device)
        video_mask = modality_masks[2].unsqueeze(-1).detach().to(self.device)

        text = self.text_I(text) * text_mask
        audio = self.audio_I(audio) * audio_mask
        video = self.video_I(video) * video_mask
        diff_loss = (Feature_Diff(text) + Feature_Diff(audio) + Feature_Diff(video)) / 3.0 

        in_t = torch.sum(text, dim=1, keepdim=False) / text_lengths.unsqueeze(-1)
        in_a = torch.sum(audio, dim=1, keepdim=False) / audio_lengths.unsqueeze(-1)
        in_v = torch.sum(video, dim=1, keepdim=False) / video_lengths.unsqueeze(-1)
            
        if label is not None:
            con_loss = (Con_Label_Loss(in_t, in_a, label) + Con_Label_Loss(in_t, in_v, label)) / 2.0

        t_label = torch.zeros(batch).to(self.device).long().detach()
        a_label = (torch.zeros(batch) + 1).to(self.device).long().detach()
        v_label = (torch.zeros(batch) + 2).to(self.device).long().detach()
        m_label = (torch.zeros(batch) + 3).to(self.device).long().detach()
        loss_im = (self.discriminator(in_t, m_label) + \
            self.discriminator(in_a, m_label) + \
            self.discriminator(in_v, m_label)) / 3.0
        
        loss_sm = (self.discriminator(in_t, t_label) + \
            self.discriminator(in_a, a_label) + \
            self.discriminator(in_v, v_label)) / 3.0
        
        # fusion
        out_t, out_a, out_v  = \
            self.fusion([text, audio, video], 
                        [text_mask, audio_mask, video_mask],
                        [text_lengths, audio_lengths, video_lengths])

        t_h = torch.sum(out_t, dim=1, keepdim=False) / text_lengths.unsqueeze(-1)
        a_h = torch.sum(out_a, dim=1, keepdim=False) / audio_lengths.unsqueeze(-1)
        v_h = torch.sum(out_v, dim=1, keepdim=False) / video_lengths.unsqueeze(-1)

        # final prediction module
        fusion_h = torch.cat([t_h, a_h, v_h], dim=-1)
        fusion_h = self.post_fusion_dropout(fusion_h)
        fusion_h = self.post_fusion_layer_1(fusion_h)
        x_f = F.relu(fusion_h, inplace=False)
        output_fusion = self.post_fusion_layer_2(x_f)

        if label is not None:
            res = {
                'con': con_loss,
                'im': loss_im,
                'sm': loss_sm,
                'M': output_fusion
                }
        else:
            res = {
                'im': loss_im,
                'sm': loss_sm,
                'M': output_fusion
                }
        return res

    def forward(self, text, audio, audio_lengths, video, video_lengths, label=None):
        res = self.forward_once(text, audio, audio_lengths, video, video_lengths, label)
        return res


class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size=None, num_layers=1, bidirectional=True, device='cpu'):
        super().__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.hidden_size = hidden_size
        feature_size = hidden_size * 2 if bidirectional else hidden_size
        self.linear_1 = nn.Sequential(nn.Linear(feature_size, out_size), nn.ReLU()) \
            if feature_size != out_size and out_size is not None else nn.Identity()
        self.device = device

    def forward(self, x, lengths):
        batch_size = x.size()[0]
        seq = x.size()[1]
        h0, c0 = rnn_zero_state(batch_size, self.hidden_size, num_layers=1, bidirectional=True)
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)
        packed_sequence = pack_padded_sequence(x, lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        packed_last_hidden_state, (_, _) = self.rnn(packed_sequence, (h0, c0))
        h, _ = pad_packed_sequence(packed_last_hidden_state, batch_first=True, total_length=seq)
        y_1 = self.linear_1(h)
        return y_1


def length_to_mask(length, max_len=None):
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    return mask


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0, c0


class Discriminator(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hid_dim, 4, bias=True))
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x, y):
        ym = self.mlp(x)
        loss = self.criterion(ym, y)
        return loss
