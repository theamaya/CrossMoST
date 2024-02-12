from timm.models.layers import DropPath
from models.pointbert.dvae import Group
from models.pointbert.dvae import Encoder
from models.pointbert.dvae import Decoder
from models.pointbert.dvae import DiscreteVAE
from models.pointbert.logger import print_log

from models.pointbert.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

import random

import math


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.args = kwargs["args"]

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        # self.load_model_from_ckpt('/export/home/repos/SLIP/pretrained_models/point_transformer_8192.pt')
        if not self.args.evaluate_3d:
            self.load_model_from_ckpt('./data/initialize_models/point_bert_pretrained.pt')

        # self.cls_head_finetune = nn.Sequential(
        #     nn.Linear(self.trans_dim * 2, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, self.cls_dim)
        # )

        # self.build_loss_func()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, pred, gt, smoothing=True):
        # import pdb; pdb.set_trace()
        gt = gt.contiguous().view(-1).long()

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = self.loss_ce(pred, gt.long())

        pred = pred.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))

        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger='Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger='Transformer'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger='Transformer'
            )

        print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')

    def forward(self, pts):
        # divide the point cloud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  # B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        # ret = self.cls_head_finetune(concat_f)
        return concat_f


class MaskTransformerMUST(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.encoder_dims = config.dvae_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self._prepare_encoder(self.config.dvae_config.ckpt)

        # self.projection = torch.nn.Linear(768, 512)

        # self.pc_projection = nn.Parameter(torch.empty(768, 512))
        # nn.init.normal_(self.pc_projection, std=512 ** -0.5)

        self.group_size = config.dvae_config.group_size
        self.num_group = config.dvae_config.num_group

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)


        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.cls_dim = config.transformer_config.cls_dim
        self.replace_pob = config.transformer_config.replace_pob
        self.num_heads = config.transformer_config.num_heads

        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)
        try:
            self.mask_rand = config.mask_rand
        except:
            self.mask_rand = False

        # define the learnable tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        # pos embedding for each patch
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        # define the transformer blocks
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        # layer norm
        self.norm = nn.LayerNorm(self.trans_dim)
        # head for token classification
        self.num_tokens = config.dvae_config.num_tokens

        # initialize the learnable tokens
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        trunc_normal_(self.mask_token, std=.02)
        
        self.apply(self._init_weights)
        
        self.proj = nn.Parameter(1 * torch.randn(384, 512))
        self.lm_head = nn.Linear(self.trans_dim, self.trans_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _prepare_encoder(self, dvae_ckpt):
        ckpt = torch.load(dvae_ckpt, map_location='cpu')
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        encoder_ckpt = {k.replace("encoder.", ""): v for k, v in base_ckpt.items() if 'encoder' in k}

        self.encoder.load_state_dict(encoder_ckpt, strict=True)
        print_log(f'[Encoder] Successful Loading the ckpt for encoder from {dvae_ckpt}', logger='dVAE BERT')
        self.encoder.requires_grad = False

    def _mask_center(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio[1] == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = random.uniform(self.mask_ratio[0], self.mask_ratio[1])
            mask_num = int(ratio * len(idx))

            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio[1] == 0:
            return torch.zeros(center.shape[:2]).bool()

        ratio = random.random() * (self.mask_ratio[1] - self.mask_ratio[0]) + self.mask_ratio[0]
        bool_masked_pos = (torch.rand(center.shape[:2]) < ratio).bool().to(center.device)

        return bool_masked_pos

    def _random_replace(self, group_input_tokens, bool_masked_pos, noaug=False):
        '''
            group_input_tokens : B G C
            bool_masked_pos : B G
            -----------------
            replaced_group_input_tokens: B G C
        '''
        # skip replace
        if noaug or self.replace_pob == 0:
            return group_input_tokens, bool_masked_pos

        replace_mask = (torch.rand(group_input_tokens.shape[:2]) < self.replace_pob).to(bool_masked_pos.device).bool()
        replace_mask = (replace_mask & ~bool_masked_pos)  # do not replace the mask pos

        overall_mask = (replace_mask + bool_masked_pos).bool().to(bool_masked_pos.device)  # True for flake input

        detached_group_input_tokens = group_input_tokens.detach()
        flatten_group_input_tokens = detached_group_input_tokens.reshape(
            detached_group_input_tokens.size(0) * detached_group_input_tokens.size(1),
            detached_group_input_tokens.size(2))
        idx = torch.randperm(flatten_group_input_tokens.shape[0])
        shuffled_group_input_tokens = flatten_group_input_tokens[idx].reshape(detached_group_input_tokens.size(0),
                                                                              detached_group_input_tokens.size(1),
                                                                              detached_group_input_tokens.size(2))

        replace_mask = replace_mask.unsqueeze(-1).type_as(detached_group_input_tokens)
        replaced_group_input_tokens = group_input_tokens * (
                    1 - replace_mask) + shuffled_group_input_tokens * replace_mask
        return replaced_group_input_tokens, overall_mask

    def forward(self, pts: torch.Tensor, mask=None):
        neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  # B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)

        batch_size= neighborhood.shape[0]
        # prepare cls and mask
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cls_pos = self.cls_pos.expand(batch_size, -1, -1)
        # add pos embedding
        pos = self.pos_embed(center)

        if mask is not None:
            # generate mask
            if self.mask_rand:
                bool_masked_pos = self._mask_center_rand(center)  # B G
            else:
                bool_masked_pos = self._mask_center(center)  # B G

            # replace the point
            replaced_group_input_tokens, overall_mask = self._random_replace(group_input_tokens, bool_masked_pos.clone())
            _, seq_len, _ = replaced_group_input_tokens.size()

            mask_token = self.mask_token.expand(batch_size, seq_len, -1)

            # mask the input tokens
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
            masked_group_input_tokens = replaced_group_input_tokens * (1 - w) + mask_token * w

            # final inputs
            x = torch.cat((cls_tokens, masked_group_input_tokens), dim=1)
            pos = torch.cat((cls_pos, pos), dim=1)

            # transformer
            x = self.blocks(x, pos)
            x = self.norm(x)

            # COMBINED FEATURE
            out_feature = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)

        else:
            x = torch.cat((cls_tokens, group_input_tokens), dim=1)
            pos = torch.cat((cls_pos, pos), dim=1)

            # transformer
            x = self.blocks(x, pos)
            x = self.norm(x)

            # COMBINED FEATURE
            out_feature = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)

        # out = self.projection(out_feature)
        # out = out_feature @ self.pc_projection
        features = out_feature
        # logits = 100 * out @ self.classifier

        if mask is not None:
            # MPM reconstruction loss

            # loss_mim = F.l1_loss(x[:, 1:] * w, masked_group_input_tokens * w, reduction='none')
            loss_mim = F.l1_loss(self.lm_head(x[:, 1:]) * w, masked_group_input_tokens * w, reduction='none')
            loss_mim = loss_mim.sum() / w.sum() / x.shape[0]

            x_cls = F.normalize(x[:, 0] @ self.proj , dim=-1)
            x_mask = F.normalize(x[:, 1:] @ self.proj , dim=-1)
            loss_align = torch.sum((x_mask - x_cls.unsqueeze(1)).pow(2), dim=-1, keepdim=True)
            loss_align = loss_align[w.bool()]

        if mask is not None:
            return features, loss_mim, loss_align  # reduce the Batch dim
            # return features, loss_mim, x_cls, x_mask, w  # reduce the Batch dim
        return features

class MaskTransformerMUST_withdvaeloss(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.encoder_dims = config.dvae_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self._prepare_encoder(self.config.dvae_config.ckpt)

        self.dvae = DiscreteVAE(config.dvae_config)
        self._prepare_dvae(self.config.dvae_config.ckpt)

        # self.decoder_dims = config.dvae_config.decoder_dims
        # self.decoder = Decoder(encoder_channel=self.decoder_dims)
        # self._prepare_decoder(self.config.dvae_config.ckpt)

        self.group_size = config.dvae_config.group_size
        self.num_group = config.dvae_config.num_group

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)


        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.cls_dim = config.transformer_config.cls_dim
        self.replace_pob = config.transformer_config.replace_pob
        self.num_heads = config.transformer_config.num_heads

        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)
        try:
            self.mask_rand = config.mask_rand
        except:
            self.mask_rand = False

        # define the learnable tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        # pos embedding for each patch
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        # define the transformer blocks
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        # layer norm
        self.norm = nn.LayerNorm(self.trans_dim)
        # head for token classification
        self.num_tokens = config.dvae_config.num_tokens

        # initialize the learnable tokens
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        trunc_normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

        self.proj = nn.Parameter(1 * torch.randn(384, 512))

        self.lm_head = nn.Linear(self.trans_dim, self.num_tokens)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _prepare_encoder(self, dvae_ckpt):
        ckpt = torch.load(dvae_ckpt, map_location='cpu')
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        encoder_ckpt = {k.replace("encoder.", ""): v for k, v in base_ckpt.items() if 'encoder' in k}

        self.encoder.load_state_dict(encoder_ckpt, strict=True)
        print_log(f'[Encoder] Successful Loading the ckpt for encoder from {dvae_ckpt}', logger='dVAE BERT')
        self.encoder.requires_grad = False

    def _prepare_dvae(self, dvae_ckpt):
        ckpt = torch.load(dvae_ckpt, map_location='cpu')
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        self.dvae.load_state_dict(base_ckpt, strict=True)
        print_log(f'[dVAE] Successful Loading the ckpt for dvae from {dvae_ckpt}', logger ='Point_BERT')
        self.dvae.requires_grad = False

    def _prepare_decoder(self, dvae_ckpt):
        ckpt = torch.load(dvae_ckpt, map_location='cpu')
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        decoder_ckpt = {k.replace("decoder.", ""): v for k, v in base_ckpt.items() if 'decoder' in k}

        self.decoder.load_state_dict(decoder_ckpt, strict=True)
        print_log(f'[Encoder] Successful Loading the ckpt for encoder from {dvae_ckpt}', logger='dVAE BERT')
        self.decoder.requires_grad = False

    def _mask_center(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio[1] == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = random.uniform(self.mask_ratio[0], self.mask_ratio[1])
            mask_num = int(ratio * len(idx))

            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio[1] == 0:
            return torch.zeros(center.shape[:2]).bool()

        ratio = random.random() * (self.mask_ratio[1] - self.mask_ratio[0]) + self.mask_ratio[0]
        bool_masked_pos = (torch.rand(center.shape[:2]) < ratio).bool().to(center.device)

        return bool_masked_pos

    def _random_replace(self, group_input_tokens, bool_masked_pos, noaug=False):
        '''
            group_input_tokens : B G C
            bool_masked_pos : B G
            -----------------
            replaced_group_input_tokens: B G C
        '''
        # skip replace
        if noaug or self.replace_pob == 0:
            return group_input_tokens, bool_masked_pos

        replace_mask = (torch.rand(group_input_tokens.shape[:2]) < self.replace_pob).to(bool_masked_pos.device).bool()
        replace_mask = (replace_mask & ~bool_masked_pos)  # do not replace the mask pos

        overall_mask = (replace_mask + bool_masked_pos).bool().to(bool_masked_pos.device)  # True for flake input

        detached_group_input_tokens = group_input_tokens.detach()
        flatten_group_input_tokens = detached_group_input_tokens.reshape(
            detached_group_input_tokens.size(0) * detached_group_input_tokens.size(1),
            detached_group_input_tokens.size(2))
        idx = torch.randperm(flatten_group_input_tokens.shape[0])
        shuffled_group_input_tokens = flatten_group_input_tokens[idx].reshape(detached_group_input_tokens.size(0),
                                                                              detached_group_input_tokens.size(1),
                                                                              detached_group_input_tokens.size(2))

        replace_mask = replace_mask.unsqueeze(-1).type_as(detached_group_input_tokens)
        replaced_group_input_tokens = group_input_tokens * (
                    1 - replace_mask) + shuffled_group_input_tokens * replace_mask
        return replaced_group_input_tokens, overall_mask

    def forward(self, pts: torch.Tensor, mask=None):
        neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  # B G N

        gt_logits = self.dvae.dgcnn_1(group_input_tokens, center)  # B G N
        dvae_label = gt_logits.argmax(-1).long()

        group_input_tokens = self.reduce_dim(group_input_tokens)

        batch_size= neighborhood.shape[0]
        # prepare cls and mask
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cls_pos = self.cls_pos.expand(batch_size, -1, -1)
        # add pos embedding
        pos = self.pos_embed(center)

        if mask is not None:
            # generate mask
            if self.mask_rand:
                bool_masked_pos = self._mask_center_rand(center)  # B G
            else:
                bool_masked_pos = self._mask_center(center)  # B G

            # replace the point
            replaced_group_input_tokens, overall_mask = self._random_replace(group_input_tokens, bool_masked_pos.clone())
            _, seq_len, _ = replaced_group_input_tokens.size()

            mask_token = self.mask_token.expand(batch_size, seq_len, -1)

            # mask the input tokens
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
            masked_group_input_tokens = replaced_group_input_tokens * (1 - w) + mask_token * w

            # final inputs
            x = torch.cat((cls_tokens, masked_group_input_tokens), dim=1)
            pos = torch.cat((cls_pos, pos), dim=1)

            # transformer
            x = self.blocks(x, pos)
            x = self.norm(x)

            # COMBINED FEATURE
            out_feature = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)

        else:
            x = torch.cat((cls_tokens, group_input_tokens), dim=1)
            pos = torch.cat((cls_pos, pos), dim=1)

            # transformer
            x = self.blocks(x, pos)
            x = self.norm(x)

            # COMBINED FEATURE
            out_feature = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)

        # out = self.projection(out_feature)
        # out = out_feature @ self.pc_projection
        features = out_feature
        # logits = 100 * out @ self.classifier

        if mask is not None:
            # MPM reconstruction loss
            # loss_mim = F.l1_loss(x[:, 1:] * w, masked_group_input_tokens * w, reduction='none')
            # loss_mim = loss_mim.sum() / w.sum() / x.shape[0]

            logits = self.lm_head(x[:, 1:])
            loss_mim = F.cross_entropy(logits.reshape(-1, logits.size(-1)), dvae_label.reshape(-1,))

            x_cls = F.normalize(x[:, 0] @ self.proj , dim=-1)
            x_mask = F.normalize(x[:, 1:] @ self.proj , dim=-1)
            loss_align = torch.sum((x_mask - x_cls.unsqueeze(1)).pow(2), dim=-1, keepdim=True)
            loss_align = loss_align[w.bool()]

        if mask is not None:
            return features, loss_mim, loss_align  # reduce the Batch dim
            # return features, loss_mim, x_cls, x_mask, w  # reduce the Batch dim
        return features
