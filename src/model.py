import torch
import torch.nn.init as init
from torch.autograd import Variable
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

import gc
import numpy as np
from utils import *
from lm_lstm import *
from noise import NoiseLayer
import json

class MlpAttn(nn.Module):
  def __init__(self, hparams):
    super(MlpAttn, self).__init__()
    self.hparams = hparams
    self.dropout = nn.Dropout(hparams.dropout)
    self.w_trg = nn.Linear(self.hparams.d_model, self.hparams.d_model)
    self.w_att = nn.Linear(self.hparams.d_model, 1)
    # if self.hparams.cuda:
    #   self.w_trg = self.w_trg.cuda()
    #   self.w_att = self.w_att.cuda()

  def forward(self, q, k, v, attn_mask=None):
    batch_size, d_q = q.size()
    batch_size, len_k, d_k = k.size()
    batch_size, len_v, d_v = v.size()
    # v is bi-directional encoding of source
    assert d_k == d_q
    #assert 2*d_k == d_v
    assert len_k == len_v
    # (batch_size, len_k, d_k)
    att_src_hidden = torch.tanh(k + self.w_trg(q).unsqueeze(1))
    # (batch_size, len_k)
    att_src_weights = self.w_att(att_src_hidden).squeeze(2)
    #if not attn_mask is None:
    #  att_src_weights.data.masked_fill_(attn_mask, -self.hparams.inf)
    att_src_weights = F.softmax(att_src_weights, dim=-1)
    att_src_weights = self.dropout(att_src_weights)
    ctx = torch.bmm(att_src_weights.unsqueeze(1), v).squeeze(1)
    return ctx

class Encoder(nn.Module):
  def __init__(self, hparams, *args, **kwargs):
    super(Encoder, self).__init__()

    self.hparams = hparams
    self.word_emb = nn.Embedding(self.hparams.src_vocab_size,
                                 self.hparams.d_word_vec,
                                 padding_idx=hparams.pad_id)

    self.layer = nn.LSTM(self.hparams.d_word_vec,
                         self.hparams.d_model,
                         batch_first=True,
                         bidirectional=True,
                         dropout=hparams.dropout)

    # bridge from encoder state to decoder init state
    self.bridge = nn.Linear(hparams.d_model * 2, hparams.d_model, bias=False)

    self.dropout = nn.Dropout(self.hparams.dropout)

  def forward(self, x_train, x_len, gumbel_softmax=False):
    """Performs a forward pass.
    Args:
      x_train: Torch Tensor of size [batch_size, max_len]
      x_mask: Torch Tensor of size [batch_size, max_len]. 1 means to ignore a
        position.
      x_len: [batch_size,]
    Returns:
      enc_output: Tensor of size [batch_size, max_len, d_model].
    """

    if not gumbel_softmax:
      batch_size, max_len = x_train.size()
      # x_train = x_train.transpose(0, 1)
      # [batch_size, max_len, d_word_vec]
      word_emb = self.word_emb(x_train)
      #print("1 embedding shape is " , word_emb.shape, " x_train shape is ",x_train.shape)


    else:
      batch_size, max_len, _ = x_train.size()
      word_emb = x_train @ self.word_emb.weight
      #print("2 embedding shape is " , word_emb.shape, " x_train shape is ",x_train.shape)

    word_emb = self.dropout(word_emb)
    packed_word_emb = pack_padded_sequence(word_emb, x_len, batch_first=True)
    enc_output, (ht, ct) = self.layer(packed_word_emb)
    enc_output, _ = pad_packed_sequence(enc_output, batch_first=True,
      padding_value=self.hparams.pad_id)
    #enc_output, (ht, ct) = self.layer(word_emb)
    # enc_output = enc_output.permute(1, 0, 2)

    # max pooling
    if self.hparams.max_pool_k_size > 1:
      enc_output = F.max_pool1d(enc_output.permute(0, 2, 1), kernel_size=self.hparams.max_pool_k_size, padding=(self.hparams.max_pool_k_size // 2)).permute(0, 2, 1)
    dec_init_cell = self.bridge(torch.cat([ct[0], ct[1]], 1))
    dec_init_state = F.tanh(dec_init_cell)
    dec_init = (dec_init_state, dec_init_cell)

    return enc_output, dec_init

class Decoder(nn.Module):
  def __init__(self, hparams, word_emb):
    super(Decoder, self).__init__()
    self.hparams = hparams

    #self.attention = DotProdAttn(hparams)
    self.attention = MlpAttn(hparams)
    # transform [ctx, h_t] to readout state vectors before softmax
    self.ctx_to_readout = nn.Linear(hparams.d_model * 2 + hparams.d_model, hparams.d_model, bias=False)
    self.readout = nn.Linear(hparams.d_model, hparams.src_vocab_size, bias=False)
    self.word_emb = word_emb
    self.attr_emb = nn.Embedding(self.hparams.trg_vocab_size,
                                 self.hparams.d_word_vec,
                                 padding_idx=hparams.trg_pad_id)

    self.len_embedding = nn.Embedding(self.hparams.max_len,
                                 self.hparams.d_len_vec,
                                 padding_idx=hparams.pad_id)
    # input: [y_t-1, input_feed]
    lstm_dim = self.hparams.d_word_vec * 1 + self.hparams.d_model * 2
    if self.hparams.decode_on_y:
      lstm_dim += self.hparams.d_word_vec *1
    if  self.hparams.len_control:
      lstm_dim +=  self.hparams.d_len_vec * 2 #hparams len and remaining (2*xx)
    if self.hparams.fl_len_control:
      lstm_dim += 2
    if self.hparams.reverse_len_control: #reverse comes on top of normal. It gets added
      lstm_dim += 1


    self.layer = nn.LSTMCell(lstm_dim,
                             hparams.d_model)
    self.dropout = nn.Dropout(hparams.dropout)

  def forward(self, x_enc, x_enc_k, dec_init, x_mask, y_train, y_mask, y_len, x_train, x_len):
    # get decoder init state and cell, use x_ct
    """
    x_enc: [batch_size, max_x_len, d_model * 2]
    """
    #print("x_train_0",x_train[0])
    batch_size_x = x_enc.size()[0]
    batch_size, x_max_len = x_train.size()
    assert batch_size_x == batch_size
    hidden = dec_init
    input_feed = torch.zeros((batch_size, self.hparams.d_model * 2),
        requires_grad=False, device=self.hparams.device)
    # if self.hparams.cuda:
    #   input_feed = input_feed.cuda()

    #
    # [batch_size, y_len, d_word_vec]
    x_emb = self.word_emb(x_train[:, :-1])

    pre_readouts = []
    logits = []

    # init with attr emb
    attr_emb =self.attr_emb(y_train)
    #attr_emb = attr_emb.sum(dim=1) / y_len.unsqueeze(1)
    attr_emb = attr_emb.sum(dim=1)
    x_len_t = torch.tensor(x_len, dtype=torch.long, device=self.hparams.device)

    remain_len_t = torch.ones_like(x_len_t) #ones
    if self.hparams.len_control: #d dim len control
      overall_len_emb = self.len_embedding(x_len_t) 
    elif self.hparams.fl_len_control:
      overall_len_emb = torch.unsqueeze(x_len_t.detach(), 1)
      remain_len_t = torch.unsqueeze(remain_len_t, 1)
    if self.hparams.reverse_len_control:
      real_remain_len_t =  torch.unsqueeze(x_len_t.detach().clone(), 1)
    

    len_ctrl_flag = self.hparams.len_control or self.hparams.fl_len_control
    for t in range(x_max_len-1):
      if self.hparams.decode_on_y and  (not len_ctrl_flag):
        x_emb_tm1 = torch.cat([x_emb[:, t, :], attr_emb], dim=1)
      
      elif self.hparams.decode_on_y and (len_ctrl_flag):
        
        remain_len_m = remain_len_t * (t)
        
        if (self.hparams.len_control):
          remain_len_embed = self.decoder.len_embedding(remain_len_m)
        elif (self.hparams.fl_len_control):
          remain_len_embed = remain_len_m.detach()
        if (self.hparams.reverse_len_control):
          real_remain_len_tensor = torch.where(real_remain_len_t-torch.ones_like(real_remain_len_t)*t> 0, real_remain_len_t-torch.ones_like(real_remain_len_t)*t,torch.zeros_like(real_remain_len_t) )

        if not self.hparams.reverse_len_control:
          x_emb_tm1 = torch.cat([x_emb[:, t, :], attr_emb, remain_len_embed, overall_len_emb], dim=1)
        else:
          x_emb_tm1 = torch.cat([x_emb[:, t, :], attr_emb, remain_len_embed, real_remain_len_tensor, overall_len_emb], dim=1)

      
      elif  (not self.hparams.decode_on_y) and (len_ctrl_flag):
        if t == 0:
          
          remain_len_m = remain_len_t * (0)
          if (self.hparams.len_control):
            remain_len_embed = self.len_embedding(remain_len_m)
          elif (self.hparams.fl_len_control):
            remain_len_embed = remain_len_m.detach()
          if (self.hparams.reverse_len_control):
            real_remain_len_tensor = torch.where(real_remain_len_t-torch.ones_like(real_remain_len_t)*t> 0, real_remain_len_t-torch.ones_like(real_remain_len_t)*t,torch.zeros_like(real_remain_len_t) )

          if not self.hparams.reverse_len_control:
            x_emb_tm1 = torch.cat([attr_emb, remain_len_embed, overall_len_emb], dim=-1) 
          else:
            x_emb_tm1 = torch.cat([attr_emb, remain_len_embed,real_remain_len_tensor, overall_len_emb], dim=-1) 
  
        else:
          remain_len_m = remain_len_t * (t) 
          if (self.hparams.len_control):
            remain_len_embed = self.len_embedding(remain_len_m)
          elif (self.hparams.fl_len_control):
            remain_len_embed = remain_len_m.detach()
          if (self.hparams.reverse_len_control):
            real_remain_len_tensor = torch.where(real_remain_len_t-torch.ones_like(real_remain_len_t)*t> 0, real_remain_len_t-torch.ones_like(real_remain_len_t)*t,torch.zeros_like(real_remain_len_t) )

          if not self.hparams.reverse_len_control:
            x_emb_tm1 = torch.cat([x_emb[:, t, :], remain_len_embed, overall_len_emb], dim=-1) 
          else:
            x_emb_tm1 = torch.cat([x_emb[:, t, :], remain_len_embed, real_remain_len_tensor, overall_len_emb], dim=-1) 
      
      else:
        if t == 0:
          x_emb_tm1 = attr_emb
        else:
          x_emb_tm1 = x_emb[:, t, :]
      x_input = torch.cat([x_emb_tm1, input_feed], dim=1)

      h_t, c_t = self.layer(x_input, hidden)
      ctx = self.attention(h_t, x_enc_k, x_enc, attn_mask=x_mask)
      pre_readout = F.tanh(self.ctx_to_readout(torch.cat([h_t, ctx], dim=1)))
      pre_readout = self.dropout(pre_readout)
      pre_readouts.append(pre_readout)

      input_feed = ctx
      hidden = (h_t, c_t)

    # [len_y, batch_size, trg_vocab_size]
    logits = self.readout(torch.stack(pre_readouts)).transpose(0, 1).contiguous()
    return logits

  def step(self, x_enc, x_enc_k, x_mask, y_tm1, dec_state, ctx_t, data):
    #y_emb_tm1 = self.word_emb(y_tm1)
    y_emb_tm1 = y_tm1
    y_input = torch.cat([y_emb_tm1, ctx_t], dim=1)
    h_t, c_t = self.layer(y_input, dec_state)
    ctx = self.attention(h_t, x_enc_k, x_enc, attn_mask=x_mask)
    pre_readout = F.tanh(self.ctx_to_readout(torch.cat([h_t, ctx], dim=1)))
    logits = self.readout(pre_readout)

    return logits, (h_t, c_t), ctx

class Seq2Seq(nn.Module):

  def __init__(self, hparams, data):
    super(Seq2Seq, self).__init__()
    self.encoder = Encoder(hparams)
    self.decoder = Decoder(hparams, self.encoder.word_emb)
    self.data = data
    # transform encoder state vectors into attention key vector
    self.enc_to_k = nn.Linear(hparams.d_model * 2, hparams.d_model, bias=False)
    self.hparams = hparams
    self.noise = NoiseLayer(hparams.word_blank, hparams.word_dropout,
        hparams.word_shuffle, hparams.pad_id, hparams.unk_id, hparams.eos_id)
    
    if self.hparams.vocab_boost:
        with open(self.hparams.vocab_weights, 'r') as f:
          vocab_weights_dict = json.load(f)
        data = [float(vocab_weights_dict[key]) for key in vocab_weights_dict.keys()]
        vocab_wights_array = np.array(data)  * -1.0* self.hparams.boost_w
        self.vocab_weights_tensor = torch.LongTensor(vocab_wights_array)
        assert self.vocab_weights_tensor.shape[0] == self.hparams.src_vocab_size
        self.vocab_weights_tensor.to(self.hparams.device)
        self.vocab_weights_tensor.requires_grad  = False
       
    if  self.hparams.vocab_boost_union:
      data_list = []
      for i in range (hparams.no_styles): 
        with open(self.hparams.vocab_weights+"{}".format(str(i+1))+".json", 'r') as f:
          vocab_weights_dict = json.load(f)
          data = [ 1.0/(float(vocab_weights_dict[key])**(1/float(5)) ) if vocab_weights_dict[key]!= 0 else 1.0 for key in vocab_weights_dict.keys()] 

          #data = [1.0/(float(vocab_weights_dict[key])**(1/float(20)) )for key in vocab_weights_dict.keys()] 
          #max_el = max(data)
          #data = [el/max_el for el in data]
          data_list.append(data)
      vocab_wights_array = np.array(data_list)  * 1.0* self.hparams.boost_w
      self.vocab_weights_tensor = torch.LongTensor(vocab_wights_array)
      assert self.vocab_weights_tensor.shape[1] == self.hparams.src_vocab_size
      self.vocab_weights_tensor.to(self.hparams.device)
      self.vocab_weights_tensor.requires_grad  = False
        #print(self.vocab_weights_tensor)
        #print(self.vocab_weights_tensor.shape)

    if hparams.lm:
      if hparams.automated_multi_domain:
        
      
        self.LM = torch.nn.ModuleList([])
        for i in range (hparams.no_styles):
          self.LM.append(torch.load(hparams.lm_style[i]))

        for LM in self.LM:
          for param in LM.parameters():
            param.requires_grad = False
          LM.eval()

      elif self.hparams.one_lm:
        self.LM = torch.nn.ModuleList([])
        self.LM.append(torch.load(self.hparams.lm_style[self.hparams.no_styles]))
        
        for param in self.LM[0].parameters():
          param.requires_grad = False
        self.LM[0].eval()

      else:
        self.LM0 = torch.load(hparams.lm_style0)
        self.LM1 = torch.load(hparams.lm_style1)

        for param in self.LM0.parameters():
            param.requires_grad = False

        for param in self.LM1.parameters():
            param.requires_grad = False

        self.LM0.eval()
        self.LM1.eval()

        self.LM = torch.nn.ModuleList([self.LM0, self.LM1])
    else:
      self.LM = None

  def set_lm(self):

    if self.hparams.lm:
      
      if self.hparams.automated_multi_domain:
        
      
        self.LM = torch.nn.ModuleList([])
        for i in range (self.hparams.no_styles):
          self.LM.append(torch.load(self.hparams.lm_style[i]))

        for LM in self.LM:
          for param in LM.parameters():
            param.requires_grad = False
          LM.eval()

      elif self.hparams.one_lm:
          self.LM = torch.nn.ModuleList([])
          self.LM.append(torch.load(self.hparams.lm_style[self.hparams.no_styles]))
          print("loading language model from ", self.hparams.lm_style[self.hparams.no_styles])

          for param in self.LM[0].parameters():
            param.requires_grad = False
          self.LM[0].eval()

      else:
        self.LM0 = torch.load(self.hparams.lm_style0)
        self.LM1 = torch.load(self.hparams.lm_style1)

        for param in self.LM0.parameters():
            param.requires_grad = False

        for param in self.LM1.parameters():
            param.requires_grad = False

        self.LM0.eval()
        self.LM1.eval()

        self.LM = torch.nn.ModuleList([self.LM0, self.LM1])
    else:
      self.LM = None


  def forward(self, x_train, x_mask, x_len, x_pos_emb_idxs, y_train, y_mask,
    y_len, y_pos_emb_idxs, y_sampled, y_sampled_mask, y_sampled_len,
    eval=False, temperature=None):

    if temperature is None:
        temperature = self.hparams.temperature
    y_len = torch.tensor(y_len, dtype=torch.float, device=self.hparams.device, requires_grad=False)
    y_sampled_len = torch.tensor(y_sampled_len, dtype=torch.float, device=self.hparams.device,
        requires_grad=False)
    #print("y_train is", y_train)
    #print(eval)
    #print(self.hparams.shuffle_train)
    #print(self.hparams.decode)
    #print(x_len)
    #exit(0)
    # first translate based on y_sampled
    # get_translation is to get translation one by one so there is no length order concern
    # index is a list which represents original position in this batch after reordering
    # translated sentences
    # on-the-fly back translation
    lm_flag = False
    if self.hparams.bt:
      if eval or (not self.hparams.gumbel_softmax):
        with torch.no_grad():
          

          if eval: 
            z_s = torch.ones_like(y_sampled)*self.hparams.transfer_to##y_sampled
          elif  not self.hparams.random_mix:
            z_s = torch.ones_like(y_sampled)*self.hparams.transfer_to#self.hparams.no_styles
          else: 
            z_s = torch.randint(0, self.hparams.no_styles, y_sampled.shape).to(y_sampled.device)
          #print(y_sampled.type())
          x_trans, x_trans_mask, x_trans_len, index = self.get_translations(x_train, x_mask, x_len, z_s, y_sampled_mask, y_sampled_len, temperature,  y_src=y_train)
        index = torch.tensor(index.copy(), dtype=torch.long, requires_grad=False, device=self.hparams.device)
      else:
        # with torch.no_grad():
        lm_flag = True
     

        if eval:
          z_s = torch.ones_like(y_sampled)*self.hparams.transfer_to##y_sampled
        elif  not self.hparams.random_mix:
          z_s = torch.ones_like(y_sampled)*self.hparams.transfer_to#self.hparams.no_styles
        else: 
          z_s = torch.randint(0, self.hparams.no_styles, y_sampled.shape).to(y_sampled.device)
        x_trans, x_trans_mask, x_trans_len, index, org_index, neg_entropy = self.get_soft_translations(x_train, x_mask, x_len,
          z_s, y_sampled_mask, y_sampled_len, y_src=y_train)
      trans_length = sum(x_trans_len)
    else:
      trans_length = 0.

    if self.hparams.lm:
      if not lm_flag:
        #: domain z
        
        if eval:
          z_s = torch.ones_like(y_sampled)*self.hparams.transfer_to##y_sampled
        elif not self.hparams.random_mix:
          z_s = torch.ones_like(y_sampled)*self.hparams.transfer_to#self.hparams.no_styles
        else: 
          z_s = torch.randint(0, self.hparams.no_styles, y_sampled.shape).to(y_sampled.device)
        x_trans_lm, x_trans_mask_lm, x_trans_len_lm, index_lm, org_index_lm, neg_entropy \
          = self.get_soft_translations(x_train, x_mask, x_len, z_s,
            y_sampled_mask, y_sampled_len,  y_src=y_train)
      else:
        x_trans_lm = x_trans
        x_trans_mask_lm = x_trans_mask
        x_trans_len_lm = x_trans_len
        org_index_lm = org_index

      lm_length = sum(x_trans_len_lm)
      y_sampled_reorder = torch.index_select(y_sampled, 0, org_index_lm)
      # E_{x ~ q(z|x, y)}[log p(z|y)]
      if self.hparams.automated_multi_domain and (not self.hparams.strike_out_max) and (not self.hparams.element_wise_all_kl) and (not self.hparams.xor_lm) and (not self.hparams.xor_lm_union):
        log_prior = self.log_prior_automated(x_trans_lm, x_trans_mask_lm, x_trans_len_lm, y_sampled_reorder)

        if self.hparams.dual:
          KL_loss = 0. - log_prior
        else:
          #print("here")
          KL_loss = neg_entropy*self.hparams.no_styles - log_prior 

      elif self.hparams.one_lm:
        log_prior = self.log_prior_one_lm(x_trans_lm, x_trans_mask_lm, x_trans_len_lm, y_sampled_reorder)
        if self.hparams.dual:
          KL_loss = 0. - log_prior
        else:
          #print("here")
          KL_loss = neg_entropy - log_prior 

      elif  self.hparams.strike_out_max:
        log_prior = self.log_prior_strike_out_max(x_trans_lm, x_trans_mask_lm, x_trans_len_lm,y_train)
        if self.hparams.dual:
          KL_loss = 0. - log_prior
        else:
          #print("here")
          KL_loss = (self.hparams.no_styles-1) *neg_entropy - log_prior 
      
      elif  self.hparams.xor_lm:
        log_prior = self.log_prior_xor(x_trans_lm, x_trans_mask_lm, x_trans_len_lm,y_train)
        if self.hparams.dual:
          KL_loss = 0. - log_prior
        else:
          #print("here")
          #print("log_prior is ", log_prior)
          KL_loss = neg_entropy - log_prior 
      
      elif  self.hparams.xor_lm_union:
        log_prior = self.log_prior_xor_union(x_trans_lm, x_trans_mask_lm, x_trans_len_lm,y_train)
        if self.hparams.dual:
          KL_loss = 0. - log_prior
        else:
          #print("here")
          
          KL_loss = neg_entropy - log_prior #

      elif   self.hparams.element_wise_all_kl : #works for both automated and not
        log_prior = self.log_prior_element_wise_all_kl(x_trans_lm, x_trans_mask_lm, x_trans_len_lm,y_train)
        if self.hparams.dual:
          KL_loss = 0. - log_prior
        else:
          #print("here")
          KL_loss = args.no_styles *neg_entropy - log_prior #(self.hparams.no_styles)


      else: 
        log_prior = self.log_prior(x_trans_lm, x_trans_mask_lm, x_trans_len_lm, y_sampled_reorder)
        log_prior_reverse = self.log_prior_reverse(x_trans_lm, x_trans_mask_lm, x_trans_len_lm, y_sampled_reorder)
        # KL = E_{x ~ q(z|x, y)}[log q(z|x, y) - log p(z|y)]
        if self.hparams.dual:
          KL_loss = 0. - log_prior
        else:
          #print("here")
          if self.hparams.no_reverse_kl_term:
            KL_loss = (neg_entropy - log_prior)
          else:
            KL_loss = ((neg_entropy - log_prior) + (neg_entropy - log_prior_reverse) )

      if self.hparams.avg_len:
          x_trans_len_t = torch.tensor(x_trans_len_lm, dtype=torch.float, requires_grad=False, device=self.hparams.device)
          x_trans_len_t = x_trans_len_t - 1
          KL_loss = KL_loss / x_trans_len_t
    else:
      KL_loss = None
      lm_length = 0

    if self.hparams.bt:
      # back-translation
      if self.hparams.bt_stop_grad and self.hparams.gumbel_softmax:
        x_trans = x_trans.detach()

      x_trans_enc, x_trans_init = self.encoder(x_trans, x_trans_len, gumbel_softmax=(not eval and self.hparams.gumbel_softmax))
      x_trans_enc = torch.index_select(x_trans_enc, 0, index)
      new_x_trans_init = []
      new_x_trans_init.append(torch.index_select(x_trans_init[0], 0, index))
      new_x_trans_init.append(torch.index_select(x_trans_init[1], 0, index))
      x_trans_init = (new_x_trans_init[0], new_x_trans_init[1])

      x_trans_enc_k = self.enc_to_k(x_trans_enc)
      
      
      #z_s = torch.ones_like(y_train)*self.hparams.no_styles 
    trans_logits = self.decoder(x_trans_enc, x_trans_enc_k, x_trans_init, x_trans_mask, y_train, y_mask, y_len, x_train, x_len)

    # then denoise encode
    if self.hparams.noise_flag:
      noise_logits = self.denoise_ae(x_train, x_mask, x_len, y_train, y_mask, y_len)
    else:
      noise_logits = None

    if not self.hparams.bt:
      trans_logits = noise_logits.new_zeros(noise_logits.size())

    # KL_loss = None

    return trans_logits, noise_logits, KL_loss, lm_length, trans_length

  def denoise_ae(self, x_train, x_mask, x_len, y_train, y_mask, y_len):
    # [batch_size, x_len, d_model * 2]
    x_noise, x_noise_mask, x_noise_len, index  = self.add_noise(x_train, x_mask, x_len)
    x_noise_enc, x_noise_init = self.encoder(x_noise, x_noise_len)
    x_noise_enc = torch.index_select(x_noise_enc, 0, index)
    new_x_noise_init = []
    new_x_noise_init.append(torch.index_select(x_noise_init[0], 0, index))
    new_x_noise_init.append(torch.index_select(x_noise_init[1], 0, index))
    x_noise_init = (new_x_noise_init[0], new_x_noise_init[1])

    x_noise_enc_k = self.enc_to_k(x_noise_enc)
    # [batch_size, y_len-1, trg_vocab_size]
    noise_logits = self.decoder(x_noise_enc, x_noise_enc_k, x_noise_init,
      x_noise_mask, y_train, y_mask, y_len, x_train, x_len)

    return noise_logits

  def log_prior(self, x, x_mask, x_len, y_sampled):
    x_mask = x_mask.float()
    y_sampled = y_sampled.squeeze(-1).float()

    # remove start symbol
    tgt = x[:, 1:]

    logits_0 = self.LM[0].compute_gumbel_logits(x, x_len)
    logits_1 = self.LM[1].compute_gumbel_logits(x, x_len)

    x_mask = x_mask[:, 1:]

    log_p0 = F.log_softmax(logits_0, dim=2)
    log_p1 = F.log_softmax(logits_1, dim=2)

    ll0 = ((log_p0 * tgt).sum(dim=2) * (1. - x_mask)).sum(dim=1)
    ll1 = ((log_p1 * tgt).sum(dim=2) * (1. - x_mask)).sum(dim=1)

    return (1. - y_sampled) * ll0 + y_sampled * ll1


  def log_prior_reverse(self, x, x_mask, x_len, y_sampled):
    x_mask = x_mask.float()
    y_sampled = y_sampled.squeeze(-1).float()

    # remove start symbol
    tgt = x[:, 1:]

    logits_0 = self.LM[1].compute_gumbel_logits(x, x_len) 
    logits_1 = self.LM[0].compute_gumbel_logits(x, x_len)

    x_mask = x_mask[:, 1:]

    log_p0 = F.log_softmax(logits_0, dim=2)
    log_p1 = F.log_softmax(logits_1, dim=2)

    ll0 = ((log_p0 * tgt).sum(dim=2) * (1. - x_mask)).sum(dim=1)
    ll1 = ((log_p1 * tgt).sum(dim=2) * (1. - x_mask)).sum(dim=1)

    return (1. - y_sampled) * ll0 + y_sampled * ll1

  def log_prior_automated(self, x, x_mask, x_len, y_sampled):
    x_mask = x_mask.float()
    y_sampled = y_sampled.squeeze(-1).float()

    # remove start symbol
    tgt = x[:, 1:]

    logits = []
    for i in range (self.hparams.no_styles):
      logits.append(self.LM[i].compute_gumbel_logits(x, x_len))
    #print(logits[0].shape) ([32, 21, 9659]

    x_mask = x_mask[:, 1:]

    #print(x_mask.shape) ([32, 21])
    #print(tgt.shape) torch.Size([32, 21, 9659])
    log_ps = []
    for i in range (self.hparams.no_styles):
      log_ps.append(F.log_softmax(logits[i], dim=2))        

    #print(log_ps[0].shape) #[32, 21, 9659])
    lls = []
    for i in range(self.hparams.no_styles):
      lls.append(((log_ps[i] * tgt).sum(dim=2) * (1. - x_mask)).sum(dim=1))

    #print(lls[0].shape) [32]

    sums =lls[0].clone()
    for i in range(1,self.hparams.no_styles):
      sums = sums+ lls[i] 

    
    return sums

  def log_prior_automated_strip(self, x, x_mask, x_len, y_sampled):
    x_mask = x_mask.float()
    y_sampled = y_sampled.squeeze(-1).float()

    # remove start symbol
    tgt = x[:, 1:]

    logits = []
    for i in range (self.hparams.no_styles):
      logits.append(self.LM[i].compute_gumbel_logits(x, x_len))


    x_mask = x_mask[:, 1:]

    log_ps = []
    for i in range (self.hparams.no_styles):
      log_ps.append(F.log_softmax(logits[i], dim=2))

    
    lls = []
    for i in range(self.hparams.no_styles):
      lls.append(((log_ps[i] * tgt).sum(dim=2) * (1. - x_mask)).sum(dim=1))


    indicator = lambda targ, lm : torch.where(torch.ones_like(targ)*lm==targ, torch.zeros_like(targ), torch.ones_like(targ))
    sums =lls[0] * indicator(y_sampled,0)
    for i in range(1,self.hparams.no_styles):
      sums += lls[i]*indicator(y_sampled, i)
    return sums


  def log_prior_one_lm(self, x, x_mask, x_len, y_sampled):
    x_mask = x_mask.float()
    

    # remove start symbol
    tgt = x[:, 1:]

    logits = []
    
    if self.LM[0].hparams.use_discriminator:
      logits.append(self.LM[0].compute_gumbel_logits(x, x_len)[0])
    else :
      logits.append(self.LM[0].compute_gumbel_logits(x, x_len))

    x_mask = x_mask[:, 1:]

    log_ps = []

    log_ps.append(F.log_softmax(logits[0], dim=2))


    lls = []
    
    lls.append(((log_ps[0] * tgt).sum(dim=2) * (1. - x_mask)).sum(dim=1))
    
    sums =lls[0]

    return sums

  def log_prior_strike_out_max(self, x, x_mask, x_len, y_train):
    x_mask = x_mask.float()
   

    # remove start symbol
    tgt = x[:, 1:]

    logits = []
    for i in range (self.hparams.no_styles):
      logits.append(self.LM[i].compute_gumbel_logits(x, x_len))


    x_mask = x_mask[:, 1:]

    log_ps = []
    for i in range (self.hparams.no_styles):
      log_ps.append(F.log_softmax(logits[i], dim=2))

    if self.hparams.no_styles == 3:
      max_1 =  torch.max(log_ps[0], log_ps[1])
      max_2 =  torch.max(log_ps[1], log_ps[2])
      max_logp = torch.max(max_1, max_2)

    elif self.hparams.no_styles ==2:
      max_logp = torch.max(log_ps[0], log_ps[1])

    sums =log_ps[0].clone()
    for i in range(1,self.hparams.no_styles):
      sums = sums + log_ps[i] 
    
    sums = sums - max_logp


    #lls = []
    #for i in range(self.hparams.no_styles):
      #lls.append(((log_ps[i] * tgt).sum(dim=2) * (1. - x_mask)))
      # lls.append(((log_ps[i] * tgt).sum(dim=2) * (1. - x_mask)).sum(dim=1))
    sums = (sums * tgt).sum(dim=2) * (1. - x_mask)


    #if self.hparams.no_styles == 2:
    #  max_logp = torch.max(lls[0], lls[1]

    #print(id(sums))
    #print(id(lls[0]))
    #print(id(lls[1]))
    #print(id(lls[2]))
    #print("lls 0 main", lls[0])
    #print("lls 1 main", lls[1])
    #print("lls 2 main", lls[2])
    #print(sums)
    #print("********************")
    #sums.register_hook(lambda grad: print("sums", grad))
    #lls[0].register_hook(lambda grad: print("ll0",grad)) 
    #lls[1].register_hook(lambda grad: print("ll1", grad)) 
    #lls[2].register_hook(lambda grad: print("ll2",grad)) 
    #print("*******")
    #print("y_train",y_train)

    
    return sums.sum(dim=1)#(sums-max_logp).sum(dim=1) #lls[0]+lls[1]+lls[2]

  def log_prior_xor(self, x, x_mask, x_len, y_train):
    x_mask = x_mask.float()
   

    # remove start symbol
    tgt = x[:, 1:]

    logits = []
    for i in range (self.hparams.no_styles):
      logits.append(self.LM[i].compute_gumbel_logits(x, x_len))


    x_mask = x_mask[:, 1:]

    log_ps = []
    ps = []
    for i in range (self.hparams.no_styles):
      log_ps.append(F.log_softmax(logits[i], dim=2))
      ps.append(F.softmax(logits[i], dim=2))

    if self.hparams.no_styles == 3:
      max_1 =  torch.max(log_ps[0], log_ps[1])
      max_2 =  torch.max(log_ps[1], log_ps[2])
      max_logp = torch.max(max_1, max_2)
      
      min_1 =  torch.min(log_ps[0], log_ps[1])
      min_2 =  torch.min(log_ps[1], log_ps[2])
      min_logp = torch.min(min_1, min_2)


    elif self.hparams.no_styles ==2:
      max_logp = torch.max(log_ps[0], log_ps[1])
      min_logp = torch.min(log_ps[0], log_ps[1])
      max_p = torch.max(ps[0], ps[1])
      min_p = torch.min(ps[0], ps[1])

    #tab 5   -min_logp , multi synth xo
    #tab 1  -min_logp , synthesized xor
    #tab 4  -min_logp,  lr 0.0001, kl 0.15
    #tab 3   -min_logp, lr0.001, kl0.04
    
    #sums = torch.where( (max_logp - min_logp)>0.005,max_logp - min_logp, torch.zeros_like(max_logp - min_logp)) - 0.2*(torch.max(max_logp, -0.5*torch.ones_like(max_logp)) + torch.max(min_logp, -0.5*torch.ones_like(min_logp))) #/torch.abs(max_logp)
    #sums = torch.where( (max_logp - min_logp)>1,max_logp - min_logp, torch.zeros_like(max_logp - min_logp)) - 0.2*(max_logp + min_logp) #/torch.abs(max_logp)
    #sums = torch.where( (max_logp - min_logp)>0.005,max_logp - min_logp, torch.zeros_like(max_logp - min_logp)) - 0.5*(min_logp) #/torch.abs(max_logp)
    #sums = torch.where( (max_p - min_p)>0.95,max_p - min_p, torch.zeros_like(max_p - min_p)) - 1*torch.max((min_p), 0.1*torch.ones_like(min_p))#/torch.abs(max_logp)
    #sums = (max_p - min_p )- 0.25*(min_p+max_p)#/torch.abs(max_logp)

    sums = min_logp #+ 0.0001*max_p

    #sums = torch.where( (max_logp - min_logp)>0.05,max_logp - min_logp, torch.zeros_like(max_logp - min_logp)) - 0.15*torch.where( (max_logp + min_logp)<1.25,max_logp + min_logp, 1.25*torch.ones_like(max_logp + min_logp))#0.1*(max_logp + min_logp) #/torch.abs(max_logp)

    sums2 = (sums * tgt).sum(dim=2) * (1. - x_mask)

 
    return sums2.sum(dim=1)


  def log_prior_xor_union(self, x, x_mask, x_len, y_train):  #token level max
    x_mask = x_mask.float()


    # remove start symbol
    tgt = x[:, 1:]

    logits = []
    for i in range (self.hparams.no_styles):
      logits.append(self.LM[i].compute_gumbel_logits(x, x_len))


    x_mask = x_mask[:, 1:]

    log_ps = []
    ps = []
    for i in range (self.hparams.no_styles):
      log_ps.append(F.log_softmax(logits[i], dim=2))
      ps.append(F.softmax(logits[i], dim=2))

    if self.hparams.no_styles == 3:
      max_1 =  torch.max(log_ps[0], log_ps[1])
      max_2 =  torch.max(log_ps[1], log_ps[2])
      max_logp = torch.max(max_1, max_2)
      
      min_1 =  torch.min(log_ps[0], log_ps[1])
      min_2 =  torch.min(log_ps[1], log_ps[2])
      min_logp = torch.min(min_1, min_2)




    elif self.hparams.no_styles ==2:
      max_logp = torch.max(log_ps[0], log_ps[1])
      min_logp = torch.min(log_ps[0], log_ps[1])
      max_p = torch.max(ps[0], ps[1])
      min_p = torch.min(ps[0], ps[1])

    #tab 1  avg
    #tab 3 min sent (union)
    #tab 4 min_logp-0.1*max_logp
    sums =  min_logp-0.55*max_logp#0.1*max_logp


    sums2 = (sums * tgt).sum(dim=2) * (1. - x_mask)



    return sums2.sum(dim=1)


  def log_prior_xor_union2(self, x, x_mask, x_len, y_sampled):  #sent level min
    x_mask = x_mask.float()
    y_sampled = y_sampled.squeeze(-1).float()

    # remove start symbol
    tgt = x[:, 1:]

    logits = []
    for i in range (self.hparams.no_styles):
      logits.append(self.LM[i].compute_gumbel_logits(x, x_len))
    #print(logits[0].shape) ([32, 21, 9659]

    x_mask = x_mask[:, 1:]

    #print(x_mask.shape) ([32, 21])
    #print(tgt.shape) torch.Size([32, 21, 9659])
    log_ps = []
    for i in range (self.hparams.no_styles):
      log_ps.append(F.log_softmax(logits[i], dim=2))        

    #print(log_ps[0].shape) #[32, 21, 9659])
    lls = []
    for i in range(self.hparams.no_styles):
      lls.append(((log_ps[i] * tgt).sum(dim=2) * (1. - x_mask)).sum(dim=1))

    #print(lls[0].shape) [32]

    sums =lls[0].clone()
    for i in range(1,self.hparams.no_styles):
      sums = torch.min(sums, lls[i])

    
    return sums




  def log_prior_element_wise_all_kl(self, x, x_mask, x_len, y_train):
    x_mask = x_mask.float()
   

    # remove start symbol
    tgt = x[:, 1:]

    logits = []
    for i in range (self.hparams.no_styles):
      logits.append(self.LM[i].compute_gumbel_logits(x, x_len))


    x_mask = x_mask[:, 1:]

    log_ps = []
    for i in range (self.hparams.no_styles):
      log_ps.append(F.log_softmax(logits[i], dim=2))

    sums =log_ps[0].clone()
    for i in range(1,self.hparams.no_styles):
      sums = sums + log_ps[i] 

    sums = (sums * tgt).sum(dim=2) * (1. - x_mask)


    
    return sums.sum(dim=1)#(sums-max_logp).sum(dim=1) #lls[0]+lls[1]+lls[2]


  def get_translations(self, x_train, x_mask, x_len, y_sampled, y_sampled_mask, y_sampled_len, temperature=None,  y_src=None):
    # list
    translated_x = self.translate(x_train, x_mask, x_len, y_sampled, y_sampled_mask, y_sampled_len,
        temperature=temperature, sampling=True,  y_src=y_src)
    translated_x = [[self.hparams.bos_id]+x+[self.hparams.eos_id] for x in translated_x]
    

    translated_x = np.array(translated_x)

    trans_len = [len(i) for i in translated_x]
    index = np.argsort(trans_len)
    index = index[::-1]
    translated_x = translated_x[index].tolist()
    reverse_index = [-1 for _ in range(len(index))]
    for i, idx in enumerate(index):
      reverse_index[idx] = i

    x_trans, x_mask, x_count, x_len, _ = self.data._pad(translated_x, self.hparams.pad_id)
    
    return x_trans, x_mask, x_len, reverse_index

  def get_soft_translations(self, x_train, x_mask, x_len,
                            y_sampled, y_sampled_mask, y_sampled_len, max_len=20, y_src=None):
    batch_size = x_train.size(0)
   
    # x_enc: (batch, seq_len, 2 * d_model)
    x_enc, dec_init = self.encoder(x_train, x_len)

    # (batch, seq_len, d_model)
    x_enc_k = self.enc_to_k(x_enc)
    length = 0
    input_feed = torch.zeros((batch_size, self.hparams.d_model * 2),
      requires_grad=False, device=self.hparams.device)

    x_len_t = torch.tensor(x_len, dtype=torch.long, device=self.hparams.device)
    remain_len_t = torch.ones_like(x_len_t) #ones
    if self.hparams.len_control:
      overall_len_emb = self.decoder.len_embedding(x_len_t) 
    elif self.hparams.fl_len_control:
      overall_len_emb = torch.unsqueeze(x_len_t.detach(), 1)
      remain_len_t = torch.unsqueeze(remain_len_t, 1)
    if self.hparams.reverse_len_control:
      real_remain_len_t =  torch.unsqueeze(x_len_t.detach().clone(), 1)


    attr_emb = self.decoder.attr_emb(y_sampled).sum(1) / y_sampled_len.unsqueeze(1) #this division is not important, /1
    #print("shape", self.decoder.attr_emb(y_sampled).shape) # it is 16,1,128 -> 16,128
    #print ("shape", x_len_t.shape) #[16]]
    mask = torch.ones((batch_size), dtype=torch.uint8, device=self.hparams.device)
    end_symbol = torch.zeros((batch_size, self.hparams.src_vocab_size),
      dtype=torch.float, requires_grad=False, device=self.hparams.device)
    end_symbol[:, self.hparams.eos_id] = 1

    pad_vec = torch.zeros((1, self.hparams.src_vocab_size),
      dtype=torch.float, requires_grad=False, device=self.hparams.device)
    pad_vec[:, self.hparams.pad_id] = 1

    bos_vec = torch.zeros((1, self.hparams.src_vocab_size),
      dtype=torch.float, requires_grad=False, device=self.hparams.device)
    bos_vec[:, self.hparams.bos_id] = 1

    eos_vec = torch.zeros((1, self.hparams.src_vocab_size),
      dtype=torch.float, requires_grad=False, device=self.hparams.device)
    eos_vec[:, self.hparams.eos_id] = 1

    decoder_input = bos_vec.expand(batch_size, self.hparams.src_vocab_size)
    hyp = Hyp(state=dec_init, y=decoder_input, ctx_tm1=input_feed, score=0.)

    decoded_batch = [[bos_vec.clone()] for _ in range(batch_size)]
    trans_len = [1 for _ in range(batch_size)]

    end_flag = [0 for _ in range(batch_size)]

    # E_{x~q}[log q], used to compute KL term in VAE
    neg_entropy = 0.

    stack_logits = []
    stack_sample = []
    

    max_len_cut = max(x_len)-2 if self.hparams.hard_len_stop else max_len
    condition = lambda  leng:  leng < max_len_cut #leng < max(x_len) if self.hparams.len_control else

    len_ctrl_flag = self.hparams.len_control or self.hparams.fl_len_control

    while mask.sum().item() != 0 and condition(length):
      length += 1
      if self.hparams.decode_on_y and  (not len_ctrl_flag ):
        y_tm1 = hyp.y
        y_tm1 = y_tm1 @ self.decoder.word_emb.weight
        y_tm1 = torch.cat([y_tm1, attr_emb], dim=-1)

      elif self.hparams.decode_on_y and (len_ctrl_flag):
        y_tm1 = hyp.y
        y_tm1 = y_tm1 @ self.decoder.word_emb.weight
        remain_len_m = remain_len_t * (length-1)
        if (self.hparams.len_control):
          remain_len_embed = self.decoder.len_embedding(remain_len_m)
        elif (self.hparams.fl_len_control):
          remain_len_embed = remain_len_m.detach()
        if (self.hparams.reverse_len_control):
          t =   (length-1)
          real_remain_len_tensor = torch.where(real_remain_len_t-torch.ones_like(real_remain_len_t)*t> 0, real_remain_len_t-torch.ones_like(real_remain_len_t)*t,torch.zeros_like(real_remain_len_t) )

        if not self.hparams.reverse_len_control:
          y_tm1 = torch.cat([y_tm1, attr_emb, remain_len_embed, overall_len_emb], dim=-1)
        else: 
          y_tm1 = torch.cat([y_tm1, attr_emb, remain_len_embed, real_remain_len_tensor, overall_len_emb], dim=-1) 
 


      elif  (not self.hparams.decode_on_y) and (len_ctrl_flag):
        if length == 1:
          y_tm1 = attr_emb
          remain_len_m = remain_len_t * (0)
          if (self.hparams.len_control):
            remain_len_embed = self.decoder.len_embedding(remain_len_m)
          elif (self.hparams.fl_len_control):
            remain_len_embed = remain_len_m.detach()
          if (self.hparams.reverse_len_control):
            t =   (length-1)
            real_remain_len_tensor = torch.where(real_remain_len_t-torch.ones_like(real_remain_len_t)*t> 0, real_remain_len_t-torch.ones_like(real_remain_len_t)*t,torch.zeros_like(real_remain_len_t) )

          if not self.hparams.reverse_len_control:
            y_tm1 = torch.cat([attr_emb, remain_len_embed, overall_len_emb], dim=-1) 
          else: 
            y_tm1 = torch.cat([attr_emb, remain_len_embed, real_remain_len_tensor ,overall_len_emb], dim=-1) 


        else:
          y_tm1 = hyp.y
          # (batch, d_word_vec)
          remain_len_m = remain_len_t * (length-1)
          if (self.hparams.len_control):
            remain_len_embed = self.decoder.len_embedding(remain_len_m)
          elif (self.hparams.fl_len_control):
            remain_len_embed = remain_len_m.detach()
          if (self.hparams.reverse_len_control):
            t =   (length-1)
            real_remain_len_tensor = torch.where(real_remain_len_t-torch.ones_like(real_remain_len_t)*t> 0, real_remain_len_t-torch.ones_like(real_remain_len_t)*t,torch.zeros_like(real_remain_len_t) )


          
          y_tm1 = y_tm1 @ self.decoder.word_emb.weight
          if not self.hparams.reverse_len_control:
            y_tm1 = torch.cat([y_tm1, remain_len_embed, overall_len_emb], dim=-1) 
          else:
            y_tm1 = torch.cat([y_tm1, remain_len_embed, real_remain_len_tensor ,overall_len_emb], dim=-1) 

      else:
        if length == 1:
          y_tm1 = attr_emb
        else:
          y_tm1 = hyp.y
          # (batch, d_word_vec)
          y_tm1 = y_tm1 @ self.decoder.word_emb.weight

      # logits: (batch_size, vocab_size)
      logits, dec_state, ctx = self.decoder.step(x_enc, x_enc_k, x_mask, y_tm1, hyp.state, hyp.ctx_tm1, self.data)
      if self.hparams.vocab_boost:
        logits = logits + self.vocab_weights_tensor.to(self.hparams.device)
      if self.hparams.vocab_boost_union:
        diffs = (torch.ones(logits.shape[0], self.hparams.no_styles).to(self.hparams.device)- 1*F.one_hot(y_src, self.hparams.no_styles).squeeze(1).to(self.hparams.device))
        matmul = torch.matmul( diffs,self.vocab_weights_tensor.type(torch.FloatTensor).to(self.hparams.device) )
        logits = logits + matmul
      #APPLY HERE
      hyp.state = dec_state
      hyp.ctx_tm1 = ctx

      # FloatTensor: (batch_size, vocab_size)
      sampled_y = F.gumbel_softmax(logits, tau=self.hparams.gs_temp, hard=(not self.hparams.gs_soft))
      hyp.y = sampled_y
      #print("SHAPE IS sampled y ", sampled_y.shape, sampled_y)

      if self.hparams.lm:
        stack_logits.append(logits.unsqueeze(1))
        stack_sample.append(torch.mul(sampled_y, mask.float().unsqueeze(1)).unsqueeze(1))
        # neg_entropy = neg_entropy + ((F.log_softmax(logits, dim=1) * sampled_y).sum(dim=1) * mask.float()).detach()

      if self.hparams.gs_soft:
        mask = (length < (x_len_t-1)).float()

      for i in range(batch_size):
        if mask[i].item():
          trans_len[i] += 1
          decoded_batch[i].append(sampled_y[i].unsqueeze(0))
          if sampled_y[i, self.hparams.eos_id].item() == 1:
            end_flag[i] = 1

        elif length == (x_len[i] -1) and self.hparams.gs_soft:
          decoded_batch[i].append(eos_vec.clone())

          trans_len[i] += 1
        else:
          decoded_batch[i].append(pad_vec.clone())

      if not self.hparams.gs_soft:
        mask = torch.mul((sampled_y != end_symbol).sum(1) > 0, mask)

      

    if not self.hparams.gs_soft:
      if length >= max_len and mask.sum().item() != 0:
        for i, batch in enumerate(decoded_batch):
          if end_flag[i] == 0:
            batch.append(eos_vec.clone())
            trans_len[i] += 1
          else:
            batch.append(pad_vec.clone())

    for i in range(batch_size):
      decoded_batch[i] = torch.cat(decoded_batch[i], dim=0).unsqueeze(0)

    # (batch_size, seq_len, vocab)
    x_trans = torch.cat(decoded_batch, dim=0)

    # assert(max(trans_len) == x_trans.size(1))

    index = np.argsort(trans_len)
    index = index[::-1]
    reverse_index = [-1 for _ in range(len(index))]
    for i, idx in enumerate(index):
      reverse_index[idx] = i

    index_t = torch.tensor(index.copy(), dtype=torch.long, requires_grad=False, device=self.hparams.device)
    reverse_index = torch.tensor(reverse_index.copy(), dtype=torch.long, requires_grad=False, device=self.hparams.device)

    x_trans = torch.index_select(x_trans, dim=0, index=index_t)
    x_len = [trans_len[i] for i in index]
    max_len = x_trans.size(1)

    x_mask = [[0 if i < length else 1 for i in range(max_len)] for length in x_len]
    x_mask = torch.tensor(x_mask, dtype=torch.uint8, requires_grad=False, device=self.hparams.device)

    x_count = sum(x_len)
    #print("SHAPE IS xtrans", x_trans.shape)
    if self.hparams.lm:
      stack_logits = torch.cat(stack_logits, dim=1)
      stack_sample = torch.cat(stack_sample, dim=1)
      neg_entropy = (F.log_softmax(stack_logits, dim=2) * stack_sample).sum(dim=2).sum(dim=1)
 
    return x_trans, x_mask, x_len, reverse_index, index_t, neg_entropy

  def add_noise(self, x_train, x_mask, x_len):
    """
    Args:
      x_train: (batch, seq_len, dim)
      x_mask: (batch, seq_len)
      x_len: a list of lengths
    Returns: x_train, mask, x_len, index
      index: a numpy array to show the original position before reordering
    """
    x_train = x_train.transpose(0, 1)
    x_train, x_len = self.noise(x_train, x_len)
    x_train = x_train.transpose(0, 1)

    index = np.argsort(x_len)
    index = index[::-1]
    reverse_index = [-1 for _ in range(len(index))]
    for i, idx in enumerate(index):
      reverse_index[idx] = i
    index = torch.tensor(index.copy(), dtype=torch.long, requires_grad=False, device=self.hparams.device)
    reverse_index = torch.tensor(reverse_index.copy(), dtype=torch.long, requires_grad=False, device=self.hparams.device)

    x_train = torch.index_select(x_train, 0, index)
    x_len = [x_len[i] for i in index]

    bs, max_len = x_train.size()
    mask = [[0] * x_len[i] + [1] * (max_len - x_len[i]) for i in range(bs)]
    mask = torch.tensor(mask, dtype=torch.uint8, requires_grad=False, device=self.hparams.device)

    return x_train, mask, x_len, reverse_index
    # index = torch.tensor(np.arange(x_train.size(0)), dtype=torch.long, requires_grad=False, device=self.hparams.device)
    # return x_train, x_mask, x_len, index

  def translate(self, x_train, x_mask, x_len, y, y_mask, y_len, max_len=100, beam_size=2, poly_norm_m=0, temperature=None, sampling=False,  y_src=None):
    if sampling:
        hyps = self.sampling_translate(x_train, x_mask, x_len, y, y_mask, y_len, max_len=max_len, temperature=temperature,  y_src=y_src)
        return hyps

    if beam_size == 1:
        hyps = self.sampling_translate(x_train, x_mask, x_len, y, y_mask, y_len, max_len=max_len, greedy=True,  y_src=y_src)

        return hyps

    hyps = []
    batch_size = x_train.size(0)
    for i in range(batch_size):
      x = x_train[i,:].unsqueeze(0)
      mask = x_mask[i,:].unsqueeze(0)
      y_i = y[i,:].unsqueeze(0)
      y_i_mask = y_mask[i,:].unsqueeze(0)
      y_i_len = torch.tensor([y_len[i]], dtype=torch.float, device=self.hparams.device).unsqueeze(0)
      # if sampling:
      #   hyp = self.sampling_translate(x, mask, y_i, y_i_mask, y_i_len, max_len=max_len)
      # else:
      hyp = self.translate_sent(x, mask, [x_len[i]], y_i, y_i_mask, y_i_len, max_len=max_len, beam_size=beam_size, poly_norm_m=poly_norm_m, y_src=y_src[i].unsqueeze(0))[0]
      hyps.append(hyp.y[1:-1])
    return hyps

  def sampling_translate(self, x_train, x_mask, x_len, y, y_mask, y_len, max_len=100, temperature=None, greedy=False,  y_src=None):
    if temperature is None:
      temperature = self.hparams.temperature
    batch_size = x_train.size(0)

    if isinstance(y_len, list):
      y_len = torch.tensor(y_len, dtype=torch.float, device=self.hparams.device, requires_grad=False)

    # x_enc: (batch, seq_len, 2 * d_model)
    x_enc, dec_init = self.encoder(x_train, x_len)

    # (batch, seq_len, d_model)
    x_enc_k = self.enc_to_k(x_enc)
    length = 0
    input_feed = torch.zeros((batch_size, self.hparams.d_model * 2),
      requires_grad=False, device=self.hparams.device)
    decoder_input = torch.tensor([self.hparams.bos_id] * batch_size,
      dtype=torch.long, device=self.hparams.device)
    hyp = Hyp(state=dec_init, y=decoder_input, ctx_tm1=input_feed, score=0.)
    #attr_emb = self.decoder.attr_emb(y).sum(1) / y_len.unsqueeze(1)
    #print("attr emb is: ", hyp)
    x_len_t = torch.tensor(x_len, dtype=torch.long, device=self.hparams.device)
    attr_emb = self.decoder.attr_emb(y).sum(1)

    remain_len_t = torch.ones_like(x_len_t) #ones
    if self.hparams.len_control:
      overall_len_emb = self.decoder.len_embedding(x_len_t)
    elif self.hparams.fl_len_control:
      overall_len_emb = torch.unsqueeze(x_len_t.detach(), 1)
      remain_len_t = torch.unsqueeze(remain_len_t, 1)
    if self.hparams.reverse_len_control:
      real_remain_len_t =  torch.unsqueeze(x_len_t.detach().clone(), 1)

  
    mask = torch.ones((batch_size), dtype=torch.uint8, device=self.hparams.device)
    end_symbol = torch.tensor([self.hparams.eos_id] * batch_size,
      dtype=torch.long, device=self.hparams.device)

    decoded_batch = [[] for _ in range(batch_size)]
    
    max_len_cut = max(x_len)-2 if self.hparams.hard_len_stop else max_len
    condition = lambda  leng:  leng < max_len_cut #leng < max(x_len) if self.hparams.len_control else
    
    len_ctrl_flag = self.hparams.len_control or self.hparams.fl_len_control

    while mask.sum().item() != 0 and condition(length):
      length += 1
      y_tm1 = hyp.y
      if self.hparams.decode_on_y and (not len_ctrl_flag):
        # y_tm1 = torch.tensor([y_tm1], dtype=torch.long,
        #   requires_grad=False, device=self.hparams.device)
        y_tm1 = self.decoder.word_emb(y_tm1)
        y_tm1 = torch.cat([y_tm1, attr_emb], dim=-1)
            
      elif self.hparams.decode_on_y and (len_ctrl_flag):
        
        remain_len_m = remain_len_t * (length-1)
        if (self.hparams.len_control):
          remain_len_embed = self.decoder.len_embedding(remain_len_m)
        elif (self.hparams.fl_len_control):
          remain_len_embed = remain_len_m.detach()
        if (self.hparams.reverse_len_control):
          t =   (length-1)
          real_remain_len_tensor = torch.where(real_remain_len_t-torch.ones_like(real_remain_len_t)*t> 0, real_remain_len_t-torch.ones_like(real_remain_len_t)*t,torch.zeros_like(real_remain_len_t) )


        y_tm1 = self.decoder.word_emb(y_tm1)
        if not self.hparams.reverse_len_control:
          y_tm1 = torch.cat([y_tm1, attr_emb, remain_len_embed, overall_len_emb], dim=-1)
        else:
          y_tm1 = torch.cat([y_tm1, attr_emb, remain_len_embed, real_remain_len_tensor ,overall_len_emb], dim=-1)

      
      elif  (not self.hparams.decode_on_y) and (len_ctrl_flag):
        
        if length == 1:
          y_tm1 = attr_emb
          remain_len_m = remain_len_t * (0)
          if (self.hparams.len_control):
            remain_len_embed = self.decoder.len_embedding(remain_len_m)
          elif (self.hparams.fl_len_control):
            remain_len_embed = remain_len_m.detach()
          if (self.hparams.reverse_len_control):
            t =   (length-1)
            real_remain_len_tensor = torch.where(real_remain_len_t-torch.ones_like(real_remain_len_t)*t> 0, real_remain_len_t-torch.ones_like(real_remain_len_t)*t,torch.zeros_like(real_remain_len_t) )

          if not self.hparams.reverse_len_control:
            y_tm1 = torch.cat([attr_emb, remain_len_embed, overall_len_emb], dim=-1) 
          else : 
            y_tm1 = torch.cat([attr_emb, remain_len_embed, real_remain_len_tensor ,overall_len_emb], dim=-1) 


        else:

          y_tm1 = self.decoder.word_emb(y_tm1)
          remain_len_m = remain_len_t * (length-1)
          if (self.hparams.len_control):
            remain_len_embed = self.decoder.len_embedding(remain_len_m)
          elif (self.hparams.fl_len_control):
            remain_len_embed = remain_len_m.detach()
          if (self.hparams.reverse_len_control):
            t =   (length-1)
            real_remain_len_tensor = torch.where(real_remain_len_t-torch.ones_like(real_remain_len_t)*t> 0, real_remain_len_t-torch.ones_like(real_remain_len_t)*t,torch.zeros_like(real_remain_len_t) )

          #y_tm1 = y_tm1 @ self.decoder.word_emb.weight
          if not self.hparams.reverse_len_control:
            y_tm1 = torch.cat([y_tm1, remain_len_embed, overall_len_emb], dim=-1) 

          else:
            y_tm1 = torch.cat([y_tm1, remain_len_embed,real_remain_len_tensor,  overall_len_emb], dim=-1) 


      else:
        # (batch, d_word_vec) --> (batch, 2 * d_word_vec)
        #y_tm1 = self.decoder.word_emb(y_tm1)
        if length == 1:
          # ave attr emb
          #y_tm1 = self.decoder.attr_emb(y).sum(1) / y_len.float()
          y_tm1 = attr_emb
        else:
          y_tm1 = self.decoder.word_emb(y_tm1)
      logits, dec_state, ctx = self.decoder.step(x_enc, x_enc_k, x_mask, y_tm1, hyp.state, hyp.ctx_tm1, self.data)
      if self.hparams.vocab_boost:
        logits = logits + self.vocab_weights_tensor.to(self.hparams.device)
      if self.hparams.vocab_boost_union:
        diffs = (torch.ones(logits.shape[0], self.hparams.no_styles).to(self.hparams.device)- 1 *F.one_hot(y_src, self.hparams.no_styles).squeeze(1).to(self.hparams.device))
        matmul = torch.matmul( diffs,self.vocab_weights_tensor.type(torch.FloatTensor).to(self.hparams.device) )
        logits = logits + matmul
      #APPLY HERE
      hyp.state = dec_state
      hyp.ctx_tm1 = ctx

      if greedy:
        #print("greedy")
        sampled_y = torch.argmax(logits, dim=1)
      else:
        #print("sampling")
        logits = logits / temperature
        sampled_y = torch.distributions.Categorical(logits=logits).sample()
        #print("SHAPE IS eval sampled y", sampled_y.shape, sampled_y)
        #print("logit shape is ", logits.shape)

      hyp.y = sampled_y

      mask = torch.mul((sampled_y != end_symbol), mask)

      for i in range(batch_size):
        if mask[i].item():
          decoded_batch[i].append(sampled_y[i].item())
    
    return decoded_batch

  def translate_sent(self, x_train, x_mask, x_len, y, y_mask, y_len, max_len=100, beam_size=5, poly_norm_m=0, y_src=None):
    x_enc, dec_init = self.encoder(x_train, x_len)
    x_enc_k = self.enc_to_k(x_enc)
    length = 0
    completed_hyp = []
    input_feed = torch.zeros((1, self.hparams.d_model * 2),
      requires_grad=False, device=self.hparams.device)
    active_hyp = [Hyp(state=dec_init, y=[self.hparams.bos_id], ctx_tm1=input_feed, score=0.)]
    #attr_emb = self.decoder.attr_emb(y).sum(1) / y_len

    x_len_t = torch.tensor(x_len, dtype=torch.long, device=self.hparams.device)


    attr_emb = self.decoder.attr_emb(y).sum(1)
    remain_len_t = torch.ones_like(x_len_t) #ones
    if self.hparams.len_control:
      overall_len_emb = self.decoder.len_embedding(x_len_t) 
    elif self.hparams.fl_len_control:
      overall_len_emb = torch.unsqueeze(x_len_t.detach(), 1)
      remain_len_t = torch.unsqueeze(remain_len_t, 1)
    if self.hparams.reverse_len_control:
      real_remain_len_t =  torch.unsqueeze(x_len_t.detach().clone(), 1)
    

    max_len_cut = max(x_len)-2 if self.hparams.hard_len_stop else max_len
    condition = lambda  leng:  leng < max_len_cut #leng < max(x_len) if self.hparams.len_control else

    len_ctrl_flag = self.hparams.len_control or self.hparams.fl_len_control

    while len(completed_hyp) < beam_size and condition(length):
      length += 1
      new_hyp_score_list = []
      for i, hyp in enumerate(active_hyp):
        if self.hparams.decode_on_y and  (not len_ctrl_flag):
          y_tm1 = torch.tensor([int(hyp.y[-1])], dtype=torch.long,
            requires_grad=False, device=self.hparams.device)
          y_tm1 = self.decoder.word_emb(y_tm1)
          y_tm1 = torch.cat([y_tm1, attr_emb], dim=-1)
        
        elif self.hparams.decode_on_y and (len_ctrl_flag):
          y_tm1 = torch.tensor([int(hyp.y[-1])], dtype=torch.long,
            requires_grad=False, device=self.hparams.device)
          y_tm1 = self.decoder.word_emb(y_tm1)
          remain_len_m = remain_len_t * (length-1)
          
          if (self.hparams.len_control):
            remain_len_embed = self.decoder.len_embedding(remain_len_m)
          elif (self.hparams.fl_len_control):
            remain_len_embed = remain_len_m.detach()
          if (self.hparams.reverse_len_control):
            t =   (length-1)
            real_remain_len_tensor = torch.where(real_remain_len_t-torch.ones_like(real_remain_len_t)*t> 0, real_remain_len_t-torch.ones_like(real_remain_len_t)*t,torch.zeros_like(real_remain_len_t) )
          
          if not self.hparams.reverse_len_control:
            y_tm1 = torch.cat([y_tm1, attr_emb, remain_len_embed, overall_len_emb], dim=-1)
          else: 
            y_tm1 = torch.cat([y_tm1, attr_emb, remain_len_embed,real_remain_len_tensor, overall_len_emb], dim=-1)
        
        elif  (not self.hparams.decode_on_y) and (len_ctrl_flag):
          if length == 1:
            remain_len_m = remain_len_t * (0)
            if (self.hparams.len_control):
              remain_len_embed = self.decoder.len_embedding(remain_len_m)
            elif (self.hparams.fl_len_control):
              remain_len_embed = remain_len_m.detach()
            if (self.hparams.reverse_len_control):
              t =   (length-1)
              real_remain_len_tensor = torch.where(real_remain_len_t-torch.ones_like(real_remain_len_t)*t> 0, real_remain_len_t-torch.ones_like(real_remain_len_t)*t,torch.zeros_like(real_remain_len_t) )
            
            if not self.hparams.reverse_len_control:
              y_tm1 = torch.cat([attr_emb, remain_len_embed, overall_len_emb], dim=-1) 
            else: 
              y_tm1 = torch.cat([attr_emb, remain_len_embed, real_remain_len_tensor ,overall_len_emb], dim=-1) 

            # ave attr emb
            #y_tm1 = self.decoder.attr_emb(y).sum(1) / y_len.float()
            #y_tm1 = attr_emb
          else:
            #y_tm1 = torch.LongTensor([int(hyp.y[-1])], device=self.hparams.device)
            y_tm1 = Variable(torch.LongTensor([int(hyp.y[-1])]))
            if self.hparams.cuda:
              y_tm1 = y_tm1.cuda()
            y_tm1 = self.decoder.word_emb(y_tm1)
            remain_len_m = remain_len_t * (length-1)
            if (self.hparams.len_control):
              remain_len_embed = self.decoder.len_embedding(remain_len_m)
            elif (self.hparams.fl_len_control):
              remain_len_embed = remain_len_m.detach()
            if (self.hparams.reverse_len_control):
              t =   (length-1)
              real_remain_len_tensor = torch.where(real_remain_len_t-torch.ones_like(real_remain_len_t)*t> 0, real_remain_len_t-torch.ones_like(real_remain_len_t)*t,torch.zeros_like(real_remain_len_t) )
            
            if not self.hparams.reverse_len_control:
              y_tm1 = torch.cat([y_tm1, remain_len_embed, overall_len_emb], dim=-1) 
            else:
              y_tm1 = torch.cat([y_tm1, remain_len_embed,  real_remain_len_t ,overall_len_emb], dim=-1) 



        else:
          if length == 1:
            # ave attr emb
            #y_tm1 = self.decoder.attr_emb(y).sum(1) / y_len.float()
            y_tm1 = attr_emb
          else:
            #y_tm1 = torch.LongTensor([int(hyp.y[-1])], device=self.hparams.device)
            y_tm1 = Variable(torch.LongTensor([int(hyp.y[-1])]))
            if self.hparams.cuda:
              y_tm1 = y_tm1.cuda()
            y_tm1 = self.decoder.word_emb(y_tm1)

        logits, dec_state, ctx = self.decoder.step(x_enc, x_enc_k, x_mask, y_tm1, hyp.state, hyp.ctx_tm1, self.data)
        if self.hparams.vocab_boost:
          logits = logits + self.vocab_weights_tensor.to(self.hparams.device)
        if self.hparams.vocab_boost_union:
          diffs = (torch.ones(logits.shape[0], self.hparams.no_styles).to(self.hparams.device)- 1*F.one_hot(y_src, self.hparams.no_styles).squeeze(1).to(self.hparams.device))
          matmul = torch.matmul( diffs,self.vocab_weights_tensor.type(torch.FloatTensor).to(self.hparams.device) )
          logits = logits + matmul
                #APPLY HERE
        hyp.state = dec_state
        hyp.ctx_tm1 = ctx

        p_t = F.log_softmax(logits, -1).data
        if poly_norm_m > 0 and length > 1:
          new_hyp_scores = (hyp.score * pow(length-1, poly_norm_m) + p_t) / pow(length, poly_norm_m)
        else:
          new_hyp_scores = hyp.score + p_t
        new_hyp_score_list.append(new_hyp_scores.cpu())
      live_hyp_num = beam_size - len(completed_hyp)
      new_hyp_scores = np.concatenate(new_hyp_score_list).flatten()
      new_hyp_pos = (-new_hyp_scores).argsort()[:live_hyp_num]
      prev_hyp_ids = new_hyp_pos / self.hparams.src_vocab_size
      word_ids = new_hyp_pos % self.hparams.src_vocab_size
      new_hyp_scores = new_hyp_scores[new_hyp_pos]

      new_hypotheses = []
      for prev_hyp_id, word_id, hyp_score in zip(prev_hyp_ids, word_ids, new_hyp_scores):
        prev_hyp = active_hyp[int(prev_hyp_id)]
        hyp = Hyp(state=prev_hyp.state, y=prev_hyp.y+[word_id], ctx_tm1=prev_hyp.ctx_tm1, score=hyp_score)
        if word_id == self.hparams.eos_id:
          completed_hyp.append(hyp)
        else:
          new_hypotheses.append(hyp)
        #print(word_id, hyp_score)
      #exit(0)
      active_hyp = new_hypotheses

    if len(completed_hyp) == 0:
      completed_hyp.append(active_hyp[0])
    return sorted(completed_hyp, key=lambda x: x.score, reverse=True)

class Hyp(object):
  def __init__(self, state, y, ctx_tm1, score):
    self.state = state
    self.y = y
    self.ctx_tm1 = ctx_tm1
    self.score = score
