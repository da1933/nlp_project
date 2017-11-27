'''
baseline model for Stanford natural language inference
'''
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
logger_name = "mylog"
logger = logging.getLogger(logger_name)
logger.setLevel(logging.INFO)

logger.info('cuda?')
logger.info(use_cuda)
logger.info(dtype)

class encoder(nn.Module):

    def __init__(self, num_embeddings, embedding_size, hidden_size, para_init):
        super(encoder, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.para_init = para_init

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_size)
        self.input_linear = nn.Linear(
            self.embedding_size, self.hidden_size, bias=False)  # linear transformation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, self.para_init)
                # m.bias.data.uniform_(-0.01, 0.01)

    def forward(self, sent1, sent2):
        '''
               sent: batch_size x length (Long tensor)
        '''
        batch_size = sent1.size(0)
        sent1 = self.embedding(sent1)
        sent2 = self.embedding(sent2)

        sent1 = sent1.view(-1, self.embedding_size)
        sent2 = sent2.view(-1, self.embedding_size)

        sent1_linear = self.input_linear(sent1).view(
            batch_size, -1, self.hidden_size)
        sent2_linear = self.input_linear(sent2).view(
            batch_size, -1, self.hidden_size)

        return sent1_linear, sent2_linear

class LSTMTagger(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.p_gru = nn.GRU(self.hidden_dim, self.hidden_dim, bidirectional=False).type(dtype)
        self.h_gru = nn.GRU(self.hidden_dim, self.hidden_dim, bidirectional=False).type(dtype)
        self.out = nn.Linear(self.hidden_dim, self.output_dim).type(dtype)
        
        # Attention Parameters
        self.W_y = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim).cuda()) if use_cuda else nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))  # n_dim x n_dim
        self.register_parameter('W_y', self.W_y)
        self.W_h = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim).cuda()) if use_cuda else nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))  # n_dim x n_dim
        self.register_parameter('W_h', self.W_h)
        self.W_r = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim).cuda()) if use_cuda else nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))  # n_dim x n_dim
        self.register_parameter('W_r', self.W_r)
        self.W_alpha = nn.Parameter(torch.randn(self.hidden_dim, 1).cuda()) if use_cuda else nn.Parameter(torch.randn(self.hidden_dim, 1))  # n_dim x 1
        self.register_parameter('W_alpha', self.W_alpha)

        # Final combination Parameters
        self.W_x = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim).cuda()) if use_cuda else nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))  # n_dim x n_dim
        self.register_parameter('W_x', self.W_x)
        self.W_p = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim).cuda()) if use_cuda else nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))  # n_dim x n_dim
        self.register_parameter('W_p', self.W_p)

    def init_hidden(self, batch_size):
        hidden_p = Variable(torch.zeros(1, batch_size, self.hidden_dim).type(dtype))
        return hidden_p

    def mask_mult(self, o_t, o_tm1, mask_t):
        '''
            o_t : batch x n
            o_tm1 : batch x n
            mask_t : batch x 1
        '''
        # return (mask_t.expand(*o_t.size()) * o_t) + ((1. - mask_t.expand(*o_t.size())) * (o_tm1))
        return (o_t * mask_t) + (o_tm1 * (1. - mask_t))

    def _gru_forward(self, gru, encoded_s, mask_s, h_0):
        '''
        inputs :
            gru : The GRU unit for which the forward pass is to be computed
            encoded_s : T x batch x n_embed
            mask_s : T x batch
            h_0 : 1 x batch x n_dim
        outputs :
            o_s : T x batch x n_dim
            h_n : 1 x batch x n_dim
        '''
        seq_len = encoded_s.size(0)
        batch_size = encoded_s.size(1)
        o_s = Variable(torch.zeros(seq_len, batch_size, self.hidden_dim).type(dtype))
        h_tm1 = h_0.squeeze(0)  # batch x n_dim
        o_tm1 = None

        for ix, (x_t, mask_t) in enumerate(zip(encoded_s, mask_s)):
            '''
                x_t : batch x n_embed
                mask_t : batch,
            '''
            o_t, h_t = gru(x_t.unsqueeze(0), h_tm1.unsqueeze(0))  # o_t : 1 x batch x n_dim
                                                                  # h_t : 1 x batch x n_dim
            mask_t = mask_t.unsqueeze(1)  # batch x 1
            h_t = self.mask_mult(h_t[0], h_tm1, mask_t)  # batch x n_dim

            if o_tm1 is not None:
                o_t = self.mask_mult(o_t[0], o_tm1, mask_t)
            o_tm1 = o_t[0] if o_tm1 is None else o_t
            h_tm1 = h_t
            o_s[ix] = o_t

        return o_s, h_t.unsqueeze(0)
    
    def _attention_forward(self, Y, mask_Y, h, r_tm1=None, index=None):
        '''
        Computes the Attention Weights over Y using h (and r_tm1 if given)
        Returns an attention weighted representation of Y, and the alphas
        inputs:
            Y : T x batch x n_dim
            mask_Y : T x batch
            h : batch x n_dim
            r_tm1 : batch x n_dim
            index : int : The timestep
        params:
            W_y : n_dim x n_dim
            W_h : n_dim x n_dim
            W_r : n_dim x n_dim
            W_alpha : n_dim x 1
        outputs :
            r = batch x n_dim
            alpha : batch x T
        '''
        Y = Y.transpose(1, 0)  # batch x T x n_dim
        mask_Y = mask_Y.transpose(1, 0)  # batch x T

        Wy = torch.bmm(Y, self.W_y.unsqueeze(0).expand(Y.size(0), *self.W_y.size()))  # batch x T x n_dim
        Wh = torch.mm(h, self.W_h)  # batch x n_dim
        if r_tm1 is not None:
            W_r_tm1 = self.batch_norm_r_r(torch.mm(r_tm1, self.W_r), index) if hasattr(self, 'batch_norm_r_r') else torch.mm(r_tm1, self.W_r)
            Wh = self.batch_norm_h_r(Wh, index) if hasattr(self, 'batch_norm_h_r') else Wh
            Wh += W_r_tm1
        M = torch.tanh(Wy + Wh.unsqueeze(1).expand(Wh.size(0), Y.size(1), Wh.size(1)))  # batch x T x n_dim
        alpha = torch.bmm(M, self.W_alpha.unsqueeze(0).expand(Y.size(0), *self.W_alpha.size())).squeeze(-1)  # batch x T
        alpha = alpha + (-1000.0 * (1. - mask_Y))  # To ensure probability mass doesn't fall on non tokens
        alpha = F.softmax(alpha)
        if r_tm1 is not None:
            r = torch.bmm(alpha.unsqueeze(1), Y).squeeze(1) + F.tanh(torch.mm(r_tm1, self.W_t))  # batch x n_dim
        else:
            r = torch.bmm(alpha.unsqueeze(1), Y).squeeze(1)  # batch x n_dim
        return r, alpha
    
    def _combine_last(self, r, h_t):
        '''
        inputs:
            r : batch x n_dim
            h_t : batch x n_dim (this is the output from the gru unit)
        params :
            W_x : n_dim x n_dim
            W_p : n_dim x n_dim
        out :
            h_star : batch x n_dim
        '''

        W_p_r = torch.mm(r, self.W_p)  # batch x n_dim
        W_x_h = torch.mm(h_t, self.W_x)  # batch x n_dim
        h_star = F.tanh(W_p_r + W_x_h)  # batch x n_dim

        return h_star
    
    def forward(self, premise, hypothesis, return_attn=False, p_dropout=0.2):
        '''
        inputs:
            premise : batch x T
            hypothesis : batch x T
        outputs :
            pred : batch x num_classes
        '''
        batch_size = premise.size(0)

        mask_p = Variable(torch.ones((premise.size(0),premise.size(1))).type(dtype))
        mask_h = Variable(torch.ones((hypothesis.size(0),hypothesis.size(1))).type(dtype))

        encoded_p = F.dropout(premise, p=p_dropout).type(dtype)
        encoded_h = F.dropout(hypothesis, p=p_dropout).type(dtype)

        encoded_p = encoded_p.transpose(1, 0)  # T x batch x n_embed
        encoded_h = encoded_h.transpose(1, 0)  # T x batch x n_embed

        mask_p = mask_p.transpose(1, 0)  # T x batch
        mask_h = mask_h.transpose(1, 0)  # T x batch
        
        
        h_0 = self.init_hidden(batch_size)  # 1 x batch x n_dim
        #return(self.p_gru, encoded_p, mask_p, h_0)
        o_p, h_n = self._gru_forward(self.p_gru, encoded_p, mask_p, h_0)  # o_p : T x batch x n_dim
                                                                          # h_n : 1 x batch x n_dim

        o_h, h_n = self._gru_forward(self.h_gru, encoded_h, mask_h, h_n)  # o_h : T x batch x n_dim
                                                                          # h_n : 1 x batch x n_dim

        r, alpha = self._attention_forward(o_p, mask_p, o_h[-1])  # r : batch x n_dim
                                                                      # alpha : batch x T

        h_star = self._combine_last(r, o_h[-1])  # batch x n_dim
        h_star = self.out(h_star)  # batch x num_classes
        h_star = F.tanh(h_star)  # Non linear projection
        pred = F.log_softmax(h_star)
        
        if return_attn:
            return pred, alpha
        else:
            return pred