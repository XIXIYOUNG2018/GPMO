import torch
from torch.autograd import Variable
import torch.nn.functional as F
from models.transformer.module.subsequent_mask import subsequent_mask


def decode(model, src, src_mask, max_len, type):
    ys = torch.ones(1)
    ys = ys.repeat(src.shape[0], 1).view(src.shape[0], 1).type_as(src.data)
    # ys shape [batch_size, 1]
    encoder_outputs = model.encode(src, src_mask)
    break_condition = torch.zeros(src.shape[0], dtype=torch.bool)
    for i in range(max_len-1):
        with torch.no_grad():
            out = model.decode(encoder_outputs, src_mask, Variable(ys),
                                      Variable(subsequent_mask(ys.size(1)).type_as(src.data)))

            log_prob = model.generator(out[:, -1])
            log_prob=F.log_softmax(log_prob, dim=-1)#在预训练的版本，模型使用的是在generator中做log-softmax,但在adv的时候
            #adv中的有些是需要log-softmax，有些是不需要log-softmax的因此在adv时，将generator中的log-softmax去掉，因此在生成的时候加入
            prob = torch.exp(log_prob)

            if type == 'greedy':
                _, next_word = torch.max(prob, dim = 1)
                ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)  # [batch_size, i]
            elif type == 'multinomial':
                next_word = torch.multinomial(prob, 1)
                ys = torch.cat([ys, next_word], dim=1) #[batch_size, i]
                next_word = torch.squeeze(next_word)

            break_condition = (break_condition | (next_word.to('cpu')==2))
            if all(break_condition): # end token
                break

    return ys

