'''
TransGrowGenerator model consisting of CNN Encoder, Transformer, CNN Decoder
'''

import torch
import timm
import numpy as np
from torch.autograd import Variable
from models.giraffe_generator import Generator, GiraffeGen1, GiraffeGen2
from models.lightweight_gan_generator import LWGenerator
from models.pos_enc import PositionalEncoding
from einops import rearrange, repeat

class TransGrowGenerator(torch.nn.Module):
    def __init__(self, data_shape, g_enc_net, g_enc_net_pretrained, g_dec_net, g_dec_net_pretrained, pe_max_len, pe_fusion_type, dim_img, dim_pe, dim_z, dim, depth, heads, dropout_trf, dropout_emb, prd_token_type):
        super(TransGrowGenerator, self).__init__()
        self.data_shape = data_shape
        self.s = data_shape[0] 
        self.s_in = self.s-1
        self.c = data_shape[1]
        self.w = data_shape[2]
        self.h = data_shape[3]
        self.pe_max_len = pe_max_len
        self.pe_fusion_type = pe_fusion_type
        self.dim_img = dim_img
        self.dim_pe = dim_pe
        self.dim_z = dim_z
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dropout_trf = dropout_trf
        self.dropout_emb = dropout_emb
        self.prd_token_type = prd_token_type
        
        # # Encoder
        if g_enc_net=='res18':
            self.encoder = timm.create_model('resnet18', pretrained=g_enc_net_pretrained)
            self.linear_enc = torch.nn.Linear(512,self.dim_img)
        elif g_enc_net=='res50':
            self.encoder = timm.create_model('resnet50', pretrained=g_enc_net_pretrained)
            self.linear_enc = torch.nn.Linear(2048,self.dim_img)
        else:
            print('Wrong g_enc_net')
        # # remove linear classification layer
        self.encoder = torch.nn.Sequential(*list(self.encoder.children())[:-1])

        # # Decoder (pretraining not available so far)
        if g_dec_net=='giraffe':
            if self.w<=64:
                self.decoder = GiraffeGen1(self.dim_img, self.w)
            else:
                self.decoder = GiraffeGen2(self.dim_img//4, self.w)
        elif g_dec_net=='giraffe_base_dec':
            self.decoder = Generator(self.dim_img, self.w, nf_end=16, nf_start=512)
        elif g_dec_net=='lightweight':
            self.decoder = LWGenerator(self.w, latent_dim=self.dim_img)
        else:
            print('Wrong d_enc_net')
        
        # # Transformer and all what belongs to it
        # # pred_token
        self.pred_token = torch.nn.Parameter(torch.randn(1, 1, self.dim_img))
        # # positional encoding
        self.pos_enc = PositionalEncoding(self.dim_pe, max_len=self.pe_max_len)
        # self.pos_enc = torch.nn.Parameter(torch.randn(1, self.s, self.dim)) # # use this for learnable pos_enc
        # # dropout_emb
        self.dropout_emb = torch.nn.Dropout(self.dropout_emb)
        # # transformer encoder (temporal)
        encoder_layer = torch.nn.TransformerEncoderLayer(self.dim, self.heads, dropout=self.dropout_trf, activation="gelu")
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, self.depth)
        # # z: Tensor declaration for creation of z Tensor
        cuda = True if torch.cuda.is_available() else False
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        
        # # prints
        print('Generator: Encoder CNN params:', sum(p.numel() for p in self.encoder.parameters()))
        print('Generator: Encoder LIN params:', sum(p.numel() for p in self.linear_enc.parameters()))
        print('Generator: Transformer params:', sum(p.numel() for p in self.transformer.parameters()))
        print('Generator: Decoder CNN params:', sum(p.numel() for p in self.decoder.parameters()))
    
    
    def forward(self, img_in, timedelta_in, timedelta_target, mask=None, z=None):
        '''
        encoding block: CNN + Linear
        [B,S,C,W,H] -> [B,S,dim_img]
        '''
        # # rearrange input for convolution [B,S,C,W,H] -> [B*S,C,W,H] 
        x = torch.reshape(img_in, (img_in.shape[0]*img_in.shape[1], img_in.shape[2], img_in.shape[3], img_in.shape[4]))
        # x = (torch.ones(x.shape)*0.5).cuda()
        # # convoltuion: [B*S,C,W,H] -> [B*S,enc_dim]
        x = self.encoder(x)
        # # linear encoding [B*S,enc_dim] -> [B*S,dim_img]
        x = self.linear_enc(x)
        # # [B*S,dim_img] -> [B,S,dim_img]
        x = torch.reshape(x, (img_in.shape[0],img_in.shape[1], self.dim_img))
    
        '''
        add pred_token
        [B,S,dim_img] -> [B,1+S,dim_img]
        '''
        # # Learnable pred_token
        # # repeat pred token according to batch_size [1,1,dim] -> [B,1,dim]
        pred_tokens = repeat(self.pred_token, '() n d -> b n d', b = img_in.shape[0])

            
        # # Interpolated pred_token **under development**
        # # ATTENTION: THIS DOES NOT WORK IN PREDCITION IF MANUALL INSERTING TARGET TIMES FOR BATCHES > 1 (z:B. Bei batchsize=8 wird Sample 0 stimmen, Sample 1-7 aber sind falsch, da sich nur der erste Eintrag [0,:] angeguckt wird um die Interpolation zu berechnen)
        # dt_target_in = timedelta_target.unsqueeze(dim=1)-timedelta_in
        
        # # for case of future extrapolation
        # if torch.min(dt_target_in[0,:])>=0:
        #     target_idx = self.s_in
        # else:
        #     for i in range(self.s_in):
        #         # if extrapolation in the future (target_idx = self.s it will never enter the if-clause)
        #         if dt_target_in[0,i]<0:
        #             target_idx = i
        #             break
                
        # dt_in = torch.diff(timedelta_in)
        # dt_in[dt_in==0]=1 # just to avoid division by 0 -> nan values
        # if target_idx==0:
        #     latent_1 = x[:,0,:]
        #     latent_2 = x[:,1,:]
        #     latent_21 = latent_1-latent_2
        #     alpha = (abs(dt_target_in[:,0])/dt_in[:,0]).unsqueeze(dim=1)
        #     # Note: here Vektor 21 added to latent 1 (going into the past) (fake extrapolation into the past)
        #     pred_tokens = (latent_1 + alpha*latent_21).unsqueeze(dim=1)
        # if target_idx==3:
        #     latent_1 = x[:,-2,:]
        #     latent_2 = x[:,-1,:]
        #     latent_12 = latent_2-latent_1
        #     alpha = (dt_target_in[:,-1]/dt_in[:,-1]).unsqueeze(dim=1)
        #     # Note: here Vektor 12 added to latent 1 (going into the future) (true interpolation)
        #     pred_tokens = (latent_2 + alpha*latent_12).unsqueeze(dim=1)
        # else:
        #     latent_1 = x[:,target_idx-1,:]
        #     latent_2 = x[:,target_idx,:]
        #     latent_12 = latent_2-latent_1
        #     alpha = (abs(dt_target_in[:,target_idx-1])/dt_in[:,target_idx-1]).unsqueeze(dim=1)
        #     # Note: here Vektor 12 added to latent 2 (going into the future) (fake extrapolation into the future)
        #     pred_tokens = (latent_1 + alpha*latent_12).unsqueeze(dim=1)
        
        # # stack pred token to img sequence: cat([B,1,dim],[B,S,dim]) = [B,1+S,dim]
        x = torch.cat((pred_tokens, x), dim=1)
        
        '''
        add z
        [B,1+S,dim_img] -> [B,1+S,dim_img]
        '''
        if z == None:
            # # different z per image in sequence
            # z = Variable(self.Tensor(np.random.normal(0, 1, (x.shape[0],x.shape[1],self.dim_z))))
            # z = z.repeat(1, 1, int(self.dim_img/self.dim_z))
            # # same z per image in sequence
            z = Variable(self.Tensor(np.random.normal(0, 1, (x.shape[0],1,self.dim_z))))
            z = z.repeat(1, x.shape[1], int(self.dim_img/self.dim_z))
        # # add z to x
        x += z
        
        '''
        add/concat pe [B,1+S,dim_img] -> [B,1+S,dim]
        if add: dim = dim_img = dim_pe
        if concat: dim = dim_img + dim_pe
        '''
        # # pe for target [B,1,dim_pe]
        pos_enc_target = self.pos_enc.get_posEnc(timedelta_target)
        # # cat with pe for input [B,1+S,dim_pe]
        pos_enc_comb = pos_enc_target
        # pos_enc_comb_0 = torch.zeros(pos_enc_comb.shape).cuda()
        for i in range(0,timedelta_in.shape[1]):
            pos_enc_comb = torch.cat((pos_enc_comb, self.pos_enc.get_posEnc(timedelta_in[:,i])), dim=1)
            # pos_enc_comb = torch.cat((pos_enc_comb, pos_enc_comb_0), dim=1)
        if self.pe_fusion_type == 'add':
            x += pos_enc_comb
        elif self.pe_fusion_type == 'cat':
            x = torch.cat((x, pos_enc_comb), dim=2)   
        else:
            print('Wrong pe_fusion_type:', self.pe_fusion_type)
        # x += self.pos_enc[:,:(self.s)] # # use this for learnable pos_enc  
        
        '''
        add dropout layer
        [B,1+S,dim] -> [B,1+S,dim]
        '''
        # # apply dropout
        x = self.dropout_emb(x)
        
        '''
        transformer
        [B,1+S,dim] -> [B,1+S,dim]
        '''
        # # [B,1+S,dim] -> [1+S,B,dim] 
        x = x.permute(1,0,2)
        # # TRANSFORMER [1+S,B,dim] -> [1+S,B,dim]
        x = self.transformer(x,mask)
        # # [1+S,B,dim] -> [B,1+S,dim]
        x = x.permute(1,0,2)
        
        '''
        get prd_token from out_sequence
        [B,1+S,dim] -> [B,dim]
        '''
        # # either compute mean or take the first sample [B,1+S,dim] -> [B, dim]
        x = x.mean(dim = 1) if self.prd_token_type == 'mean' else x[:,0,:]
        
        '''
        remove pe_dim in case of concatenation
        [B,dim] -> [B,dim_img]
        '''
        x = x[:,:self.dim_img]
        
        '''
        decoding block: Linear + CNN
        [B,dim_img] -> [B,C,W,H]
        '''
        # # convolutional decoding [B,dim_img] -> [B,dim_img,1,1]
        x = x.view(x.shape[0],x.shape[1],1,1)
        # # [B,pooldim,1,1] -> [B,C,W,H]
        x = self.decoder(x)

        return x