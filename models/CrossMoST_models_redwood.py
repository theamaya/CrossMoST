'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''

# Modified from github.com/openai/CLIP
from collections import OrderedDict

import timm
from torch import nn
from models.pointnet2.pointnet2 import Pointnet2_Ssg
from data.dataset_3d import *

from models import losses
from torch.nn.parameter import Parameter
from easydict import EasyDict
from models.pointbert.point_encoder import MaskTransformerMUST, MaskTransformerMUST_withdvaeloss
from models.must_clip.model import VisionTransformer_MIM
from typing import Any, Union, List
import torch
import torch.nn as nn
import json
from tqdm import tqdm

import hashlib
import os
import urllib
import warnings

import torch.nn.functional as F

_MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True,
                  unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
         jit: bool = False, download_root: str = None, mask: bool = False):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict(), mask=mask).to(device)
        if str(device) == "cpu":
            model.float()
        return model

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model


def load_state_dict(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
                    jit: bool = False, download_root: str = None, mask: bool = False):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    # try:
    #     # loading JIT archive
    #     model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
    #     state_dict = None
    #
    # except RuntimeError:
    #     # loading saved state dict
    #     if jit:
    #         warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
    #         jit = False
    state_dict = torch.load(model_path, map_location="cpu")
    return state_dict


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class CrossMoST_multiview(nn.Module):
    def __init__(self, args, config, point_encoder, **kwargs):
        super().__init__()
        kwargs = EasyDict(kwargs)
        self.context_length = kwargs.context_length
        self.vision_width = kwargs.vision_width
        self.visual = kwargs.vision_model
        self.classes = kwargs.classes
        self.templates = kwargs.templates
        self.tokenizer = kwargs.tokenizer

        self.transformer = Transformer(
            width=kwargs.transformer_width,
            layers=kwargs.transformer_layers,
            heads=kwargs.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = kwargs.vocab_size
        self.token_embedding = nn.Embedding(kwargs.vocab_size, kwargs.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, kwargs.transformer_width))
        self.ln_final = LayerNorm(kwargs.transformer_width)

        self.image_projection = nn.Parameter(torch.empty(kwargs.vision_width, kwargs.embed_dim))
        self.text_projection = nn.Parameter(torch.empty(kwargs.transformer_width, kwargs.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

        self.point_encoder = point_encoder

        self.pc_projection = nn.Parameter(torch.empty(kwargs.pc_feat_dims, 512))
        nn.init.normal_(self.pc_projection, std=512 ** -0.5)

        self.ulip = config.ulip

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask


    def init_classifier(self, args):
        text_features = []
        # prompts = best_prompt_weight[
        #     '{}_{}_test_prompts'.format('modelnet40_crossmost', 'vit_b16')]
        # prompts = best_prompt_weight[
        #     '{}_{}_test_prompts'.format('scanobjectnn', 'vit_b16')]
        classes = list(self.classes.keys())
        for i in range(len(classes)):
            l = classes[i]
            texts = [t.format(l) for t in self.templates]
            # texts = []
            # texts.append(prompts[i])
            # texts.append(prompts[i])
            texts = self.tokenizer(texts).cuda(args.gpu, non_blocking=True)
            with torch.no_grad():
                class_embeddings = self.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(class_embeddings)

        self.zeroshot_weights = torch.stack(text_features, dim=0)
        self.classifier = nn.Parameter(self.zeroshot_weights.T.to(args.device))

        del self.transformer, self.token_embedding, self.positional_embedding, self.ln_final, self.text_projection #, self.logit_scale
        return

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def encode_text(self, text, mask=None):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def encode_image(self, image, Mask):
        x = self.visual(image, Mask)
        return x

    def encode_pc(self, pc, Mask):
        pc_feat = self.point_encoder(pc, Mask)
        return pc_feat

    def forward(self, image, pc, text=None, Mask=None, val=False):
        pc_embed = self.encode_pc(pc, Mask)

        if self.ulip:
            image_embed = self.encode_image(image, Mask)
            if val:
                pc_embed = pc_embed @ self.pc_projection
                image_embed = image_embed @ self.image_projection

                return {'pc_embed': pc_embed,
                        'image_embed': image_embed}

            text_embed_all = []
            for i in range(text.shape[0]):
                text_for_one_sample = text[i]
                text_embed = self.encode_text(text_for_one_sample)
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
                text_embed = text_embed.mean(dim=0)
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
                text_embed_all.append(text_embed)

            text_embed_all = torch.stack(text_embed_all)

            pc_embed = pc_embed @ self.pc_projection
            image_embed = image_embed @ self.image_projection

            return {'text_embed': text_embed_all,
                    'pc_embed': pc_embed,
                    'image_embed': image_embed,
                    'logit_scale': self.logit_scale.exp()}

        elif Mask is None:
            img = image.view(1, -1, *(image.size()[2:])).squeeze(0)
            image_embed = self.encode_image(img, Mask)
            # image_embed = image_embed.view(image.shape[0], -1, *image_embed.shape[1:]).mean(1)
            image_logits = self.logit_scale.exp() * F.normalize(image_embed @ self.image_projection, dim=-1, p=2) @ self.classifier
            image_logits = image_logits.view(image.shape[0], -1, image_logits.shape[-1]).mean(1)

            # a = pc_embed @ self.pc_projection
            # a = a / a.norm(dim=-1, keepdim=True)
            # pc_logits = a @ self.classifier

            pc_logits = self.logit_scale.exp() * F.normalize(pc_embed @ self.pc_projection, dim=-1, p=2) @ self.classifier

            return image_logits, pc_logits

        else:
            picked = torch.randint(0, 9, (image.shape[0],))
            image_embed = self.encode_image(image[:,picked].squeeze(1), Mask)
            pc_features = F.normalize(pc_embed[0] @ self.pc_projection, dim=-1, p=2)
            pc_logits, pc_mim, pc_align = self.logit_scale.exp() * pc_features @ self.classifier, pc_embed[1], pc_embed[2]

            # pc_logits, pc_mim, x_cls, x_mask, w = 100 * pc_features @ self.classifier, pc_embed[1], pc_embed[2], pc_embed[3], pc_embed[4]
            #
            # x_mask = F.normalize(x_mask @ self.pc_projection, dim=-1)
            # x_cls = F.normalize(x_cls @ self.pc_projection, dim=-1, p=2)
            # loss_align = torch.sum((x_mask - x_cls.unsqueeze(1)).pow(2), dim=-1, keepdim=True)
            # pc_align = loss_align[w.bool()]


            image_features = F.normalize(image_embed[0] @ self.image_projection, dim=-1, p=2)
            image_logits, image_mim, x_patch_image = self.logit_scale.exp() * image_features @ self.classifier, image_embed[1], image_embed[2]

            x_mask = F.normalize(x_patch_image @ self.image_projection, dim=-1)
            x_cls = F.normalize(image_embed[0] @ self.image_projection, dim=-1, p=2)
            loss_align = torch.sum((x_mask - x_cls.unsqueeze(1)).pow(2), dim=-1, keepdim=True)
            w = Mask.flatten(1).unsqueeze(-1)
            image_align = loss_align[w.bool()].view(w.size(0), -1)

            pc_image_align_logits = pc_features @ image_features.T
            image_pc_align_logits = image_features @ pc_features.T

            return image_logits, pc_logits, image_mim, pc_mim, image_align, pc_align, pc_image_align_logits, image_pc_align_logits


class CrossMoST(nn.Module):
    def __init__(self, args, config, point_encoder, **kwargs):
        super().__init__()
        kwargs = EasyDict(kwargs)
        self.context_length = kwargs.context_length
        self.vision_width = kwargs.vision_width
        self.visual = kwargs.vision_model
        self.classes = kwargs.classes
        self.templates = kwargs.templates
        self.tokenizer = kwargs.tokenizer

        self.transformer = Transformer(
            width=kwargs.transformer_width,
            layers=kwargs.transformer_layers,
            heads=kwargs.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = kwargs.vocab_size
        self.token_embedding = nn.Embedding(kwargs.vocab_size, kwargs.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, kwargs.transformer_width))
        self.ln_final = LayerNorm(kwargs.transformer_width)

        self.image_projection = nn.Parameter(torch.empty(kwargs.vision_width, kwargs.embed_dim))
        self.text_projection = nn.Parameter(torch.empty(kwargs.transformer_width, kwargs.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

        self.point_encoder = point_encoder

        self.pc_projection = nn.Parameter(torch.empty(kwargs.pc_feat_dims, 512))
        nn.init.normal_(self.pc_projection, std=512 ** -0.5)

        self.ulip = config.ulip

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def init_classifier(self, args):
        text_features = []
        for l in self.classes.keys():
            texts = [t.format(l) for t in self.templates]
            texts = self.tokenizer(texts).cuda(args.gpu, non_blocking=True)
            # if len(texts.shape) < 2:
            #     texts = texts[None, ...]
            with torch.no_grad():
                class_embeddings = self.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(class_embeddings)

        self.zeroshot_weights = torch.stack(text_features, dim=0)
        self.classifier = nn.Parameter(self.zeroshot_weights.T.to(args.device))

        del self.transformer, self.token_embedding, self.positional_embedding, self.ln_final, self.text_projection #, self.logit_scale
        return

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def encode_text(self, text, mask=None):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def encode_image(self, image, Mask):
        x = self.visual(image, Mask)
        return x

    def encode_pc(self, pc, Mask):
        pc_feat = self.point_encoder(pc, Mask)
        return pc_feat

    def forward(self, image, pc, text=None, Mask=None, val=False):
        pc_embed = self.encode_pc(pc, Mask)
        image_embed = self.encode_image(image, Mask)

        if self.ulip:
            if val:
                pc_embed = pc_embed @ self.pc_projection
                image_embed = image_embed @ self.image_projection

                return {'pc_embed': pc_embed,
                        'image_embed': image_embed}

            text_embed_all = []
            for i in range(text.shape[0]):
                text_for_one_sample = text[i]
                text_embed = self.encode_text(text_for_one_sample)
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
                text_embed = text_embed.mean(dim=0)
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
                text_embed_all.append(text_embed)

            text_embed_all = torch.stack(text_embed_all)

            pc_embed = pc_embed @ self.pc_projection
            image_embed = image_embed @ self.image_projection

            return {'text_embed': text_embed_all,
                    'pc_embed': pc_embed,
                    'image_embed': image_embed,
                    'logit_scale': self.logit_scale.exp()}

        elif Mask is None:
            image_logits = self.logit_scale.exp() * F.normalize(image_embed @ self.image_projection, dim=-1, p=2) @ self.classifier

            # a = pc_embed @ self.pc_projection
            # a = a / a.norm(dim=-1, keepdim=True)
            # pc_logits = a @ self.classifier

            pc_logits = self.logit_scale.exp() * F.normalize(pc_embed @ self.pc_projection, dim=-1, p=2) @ self.classifier

            return image_logits, pc_logits

        else:
            pc_features = F.normalize(pc_embed[0] @ self.pc_projection, dim=-1, p=2)
            pc_logits, pc_mim, pc_align = self.logit_scale.exp() * pc_features @ self.classifier, pc_embed[1], pc_embed[2]
            image_features = F.normalize(image_embed[0] @ self.image_projection, dim=-1, p=2)
            image_logits, image_mim, x_patch_image = self.logit_scale.exp() * image_features @ self.classifier, image_embed[1], \
                                                     image_embed[2]

            x_mask = F.normalize(x_patch_image @ self.image_projection, dim=-1)
            x_cls = F.normalize(image_embed[0] @ self.image_projection, dim=-1, p=2)
            loss_align = torch.sum((x_mask - x_cls.unsqueeze(1)).pow(2), dim=-1, keepdim=True)
            w = Mask.flatten(1).unsqueeze(-1)
            image_align = loss_align[w.bool()].view(w.size(0), -1)

            pc_image_align_logits = pc_features @ image_features.T
            image_pc_align_logits = image_features @ pc_features.T

            return image_logits, pc_logits, image_mim, pc_mim, image_align, pc_align, pc_image_align_logits, image_pc_align_logits


def get_loss(args):
    return losses.ULIPWithImageLoss()


def get_metric_names(model):
    return ['loss', 'ulip_loss', 'ulip_pc_image_acc', 'ulip_pc_text_acc']


def CrossMoST_PointBERT(args, classes, templates, tokenizer):
    state_dict = load_state_dict(args.clip_model).state_dict()

    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len(
        [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_heads = vision_width // 64
    vision_model = VisionTransformer_MIM(
        input_resolution=image_resolution,
        patch_size=vision_patch_size,
        width=vision_width,
        layers=vision_layers,
        heads=vision_heads,
        output_dim=embed_dim,
        mask=args.mask
    )
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    # config_addr = './models/pointbert/MaskTransformerMUST.yaml'
    config_addr = args.config
    config = cfg_from_yaml_file(config_addr)
    point_encoder = MaskTransformerMUST(config.model)
    # point_encoder = MaskTransformerMUST_withdvaeloss(config.model)
    pc_feat_dims = 768
    # =====================================================================

    model = CrossMoST(args, config, embed_dim=embed_dim, vision_width=vision_width, point_encoder=point_encoder,
                      vision_model=vision_model,
                      context_length=context_length, vocab_size=vocab_size, transformer_width=transformer_width,
                      transformer_heads=transformer_heads,
                      transformer_layers=transformer_layers, pc_feat_dims=pc_feat_dims, classes=classes,
                      templates=templates, tokenizer=tokenizer)

    needed = []
    for name, v in model.named_parameters():
        needed.append(name)

    other_needed = ["point_encoder.encoder.first_conv.1.running_mean", "point_encoder.encoder.first_conv.1.running_var",
                    "point_encoder.encoder.second_conv.1.running_mean",
                    "point_encoder.encoder.second_conv.1.running_var"]
    needed.extend(other_needed)

    clip_model = load_state_dict(args.clip_model).state_dict()
    avail = list(clip_model.keys())
    temp = {}
    for i in needed:
        if i in avail:
            temp[i] = clip_model[i]
        else:
            continue
    temp['image_projection'] = clip_model['visual.proj']
    model.load_state_dict(temp, strict=False)

    if args.ulip:
        for k, p in model.named_parameters():
            if 'point_encoder' in k:
                p.requires_grad = True
            elif 'pc_projection' in k or 'image_projection' in k or 'text_projection' in k:
                p.requires_grad = True
            else:
                p.requires_grad = False

    if not args.from_scratch:
        pretrained_ulip_model = torch.load('./outputs/3modal_ULIP_10june/checkpoint_best.pt',
                                           map_location=torch.device('cpu'))
        pretrained_ulip_model = pretrained_ulip_model['state_dict']
        temp = {}
        for key, value in pretrained_ulip_model.items():
            r = ("module.", "")
            k = key.replace(*r)
            if k != key:
                temp[k] = value
        model.load_state_dict(temp, strict=False)

    return model
