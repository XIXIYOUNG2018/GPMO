import torch
import torch.nn as nn
import copy
import random
import numpy as np
from models.transformer.module.positional_encoding import PositionalEncoding
from models.transformer.module.positionwise_feedforward import PositionwiseFeedForward
from models.transformer.module.multi_headed_attention import MultiHeadedAttention
from models.transformer.module.embeddings import Embeddings
from models.transformer.encode_decode.encoder import Encoder
from models.transformer.encode_decode.decoder import Decoder
from models.transformer.encode_decode.encoder_layer import EncoderLayer
from models.transformer.encode_decode.decoder_layer import DecoderLayer
from models.transformer.module.generator import Generator
import torch.nn.functional as F
from models.transformer.encode_decode.model import EncoderDecoderold

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    """

    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.models=self.load_from_file('./experiments/train_transformer/checkpointpretrain/model.pt')
        # self.encode = self.models.encode
        # self.decode = self.models.decode
        self.src_embed = self.models.src_embed
        self.tgt_embed = self.models.tgt_embed
        self.generator = self.models.generator
        self.pad='pad'
        self.tau = 0.1
        self.pos_eps = 3.0
        self.neg_eps = 3.0
        set_seed(123)
        device = torch.device('cuda:0')
        if device.type.startswith('cuda'):
            torch.cuda.set_device(device.index or 0)
        self.projectione = nn.Sequential(nn.Linear(256, 128),
                                        nn.ReLU())
        self.projectiond = nn.Sequential(nn.Linear(256, 128),
                                        nn.ReLU())
        self.models.generator.eval()
        for p in self.models.generator.parameters():
            p.requires_grad=False
    def forward(self, src, tgt, src_mask, tgt_mask, attention_mask,decoder_attention_mask,adv,lm_labels):
        "Take in and process masked src and target sequences."
        hidden_states=self.models.encode(src, src_mask)
        sequence_output=self.models.decode(self.models.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)
        #with torch.no_grad():
        lm_logits=self.models.generator(sequence_output)
        #print(attention_mask.shape,src.shape,decoder_attention_mask.shape,tgt.shape)
        sequence_output=sequence_output*(256**-0.5)
        if adv:
            proj_enc_h = self.projectione(hidden_states)
            proj_dec_h = self.projectiond(sequence_output)
            avg_doc = self.avg_pool(proj_enc_h, attention_mask)
            avg_abs = self.avg_pool(proj_dec_h, decoder_attention_mask)
            
            cos = nn.CosineSimilarity(dim=-1)
            cont_crit = nn.CrossEntropyLoss()
            sim_matrix = cos(avg_doc.unsqueeze(1),
                             avg_abs.unsqueeze(0))
            perturbed_dec = self.generate_adv(sequence_output,
                                              lm_labels,self.models.generator)  # [n,b,t,d] or [b,t,d]
            batch_size = src.size(0)

            proj_pert_dec_h = self.projectiond(perturbed_dec)
            avg_pert = self.avg_pool(proj_pert_dec_h,
                                     decoder_attention_mask)

            adv_sim = cos(avg_doc, avg_pert).unsqueeze(1)  # [b,1]

            pos_dec_hidden = self.generate_cont_adv(hidden_states, attention_mask,
                                                    sequence_output, decoder_attention_mask,
                                                    lm_logits,
                                                    self.tau, self.pos_eps,self.models.generator)
            avg_pos_dec = self.avg_pool(self.projectiond(pos_dec_hidden),
                                        decoder_attention_mask)
            #print(pos_dec_hidden.shape)
            pos_sim = cos(avg_doc, avg_pos_dec).unsqueeze(-1)  # [b,1]
            logits = torch.cat([sim_matrix, adv_sim], 1) / self.tau

            identity = torch.eye(batch_size, device=src.device)
            pos_sim = identity * pos_sim
            neg_sim = sim_matrix.masked_fill(identity == 1, 0)
            new_sim_matrix = pos_sim + neg_sim
            new_logits = torch.cat([new_sim_matrix, adv_sim], 1)

            labels = torch.arange(batch_size,
                                  device=src.device)

            cont_loss = cont_crit(logits, labels)
            new_cont_loss = cont_crit(new_logits, labels)

            cont_loss = 0.5 * (cont_loss + new_cont_loss)

            return self.models.decode(self.models.encode(src, src_mask), src_mask,
                           tgt, tgt_mask), cont_loss, self.models

        else:
            return self.models.decode(self.models.encode(src, src_mask), src_mask,
                           tgt, tgt_mask),self.models

    def generate_adv(self, dec_hiddens, lm_labels,decoder_fc):
        dec_hiddens = dec_hiddens.detach()
        dec_hiddens.requires_grad = True
        #with torch.no_grad():
        lm_logits = decoder_fc(dec_hiddens)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        loss = criterion(lm_logits.reshape(-1, lm_logits.size(-1)),
                         lm_labels.reshape(-1))
        loss.backward()
        dec_grad = dec_hiddens.grad.detach()
        l2_norm = torch.norm(dec_grad, dim=-1)
        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)
        perturbed_dec = dec_hiddens + self.neg_eps * dec_grad.detach()
        perturbed_dec = perturbed_dec  # [b,t,d]
        self.zero_grad()
        return perturbed_dec

    def generate_cont_adv(self, enc_hiddens, enc_mask,
                          dec_hiddens, dec_mask, lm_logits,
                          tau, eps,decoder_fc):
        enc_hiddens = enc_hiddens.detach()
        dec_hiddens = dec_hiddens.detach()
        lm_logits = lm_logits.detach()
        dec_hiddens.requires_grad = True
        
        avg_enc = self.avg_pool(self.projectione(enc_hiddens),enc_mask)
        avg_dec = self.avg_pool(self.projectiond(dec_hiddens),
                                dec_mask)

        cos = nn.CosineSimilarity(dim=-1)
        logits = cos(avg_enc.unsqueeze(1), avg_dec.unsqueeze(0)) / tau

        cont_crit = nn.CrossEntropyLoss()
        labels = torch.arange(avg_enc.size(0),
                              device=enc_hiddens.device)
        loss = cont_crit(logits, labels)
        loss.backward()
        
        dec_grad = dec_hiddens.grad.detach()
        l2_norm = torch.norm(dec_grad, dim=-1)
        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturb_dec_hidden = dec_hiddens + eps * dec_grad
        perturb_dec_hidden = perturb_dec_hidden.detach()
        perturb_dec_hidden.requires_grad = True
        perturb_logits = decoder_fc(perturb_dec_hidden)

        true_probs = F.softmax(lm_logits, -1)
        true_probs = true_probs * dec_mask.unsqueeze(-1).float()

        perturb_log_probs = F.log_softmax(perturb_logits, -1)

        kl_crit = nn.KLDivLoss(reduction="sum")
        vocab_size = lm_logits.size(-1)

        kl = kl_crit(perturb_log_probs.view(-1, vocab_size),
                     true_probs.view(-1, vocab_size))
        kl = kl / torch.sum(dec_mask).float() 
        kl.backward()

        kl_grad = perturb_dec_hidden.grad.detach()

        l2_norm = torch.norm(kl_grad, dim=-1)

        kl_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturb_dec_hidden = perturb_dec_hidden - eps * kl_grad

        
        return perturb_dec_hidden

    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()
        mask = mask.unsqueeze(2)
        hidden = hidden_states.masked_fill(mask == 0, 0.0)
        avg_hidden = torch.sum(hidden, 1) / length

        return avg_hidden

    #@classmethod
    # def load_from_file(clf,opt, vocab_size,file_path):
    #     # Load model
    #     pretrain_model =EncoderDecoderold.make_model(vocab_size, vocab_size, N=opt.N,
    #                                       d_model=opt.d_model, d_ff=opt.d_ff, h=opt.H, dropout=opt.dropout)
    #     pretrain_model.load_state_dict(torch.load(file_path, map_location='cuda:0'))
    @classmethod
    def load_from_file(cls, file_path):
        # Load model
        checkpoint = torch.load(file_path, map_location='cuda:0')
        para_dict = checkpoint['model_parameters']
        vocab_size = para_dict['vocab_size']
        model = EncoderDecoderold.make_model(vocab_size, vocab_size, para_dict['N'],
                                  para_dict['d_model'], para_dict['d_ff'],
                                  para_dict['H'], para_dict['dropout'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model        
        
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
