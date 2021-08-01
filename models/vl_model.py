import torch
import torch.nn as nn
from .fakeTransformer import FakeTransformer
from .bert import Bert
import torch.nn.functional as F
import timm
import numpy as np
import sys


class ImgLearnableEncoder(nn.Module):
    def __init__(self, model_cfg):
        super(ImgLearnableEncoder, self).__init__()

        self.backbone = timm.create_model(model_cfg.CNN, pretrained=True)
        self.model_cfg = model_cfg

        self.learnable = nn.ModuleDict()
        self.learnable['imgFC'] = FakeTransformer(model_cfg.IMG_FEATURE_DIM, model_cfg.IMG_FEATURE_DIM, model_cfg.IMG_FEATURE_DIM)
        img_encoder_layer = nn.TransformerEncoderLayer(d_model=model_cfg.IMG_FEATURE_DIM, nhead=model_cfg.IMG_TRANSFORMER_HEAD)
        self.learnable['imgAtt'] = nn.TransformerEncoder(img_encoder_layer, num_layers=model_cfg.IMG_TRANSFORMER_LAYER)

        self.learnable['max_pool'] = nn.Sequential(
                                                    nn.Conv2d(model_cfg.IMG_FEATURE_DIM, model_cfg.IMG_FEATURE_DIM, kernel_size=1),
                                                    nn.AvgPool2d(model_cfg.GRID_SIZE, stride=1)
                                                ) 

        self.init_param()

    def init_param(self):
        for name, param in self.backbone.named_parameters():
            condition = 'blocks.6' not in name and 'blocks.5' not in name and 'blocks.4' not in name and 'blocks.3' not in name
            if condition:
                param.requires_grad = False
            else:
                param.requires_grad = True
        sys.stdout.flush()        

    def roi_grid_pool(self, spatial_features_2d, rois):
        """
        Args:
            rois: (B, num_rois, 4)
            spatial_features_2d: (B, C, H, W)
        Returns:
            pooled_features : (B, num_rois, C)

        """
        batch_size = spatial_features_2d.size(0)
        rois = rois.detach()
        height, width = spatial_features_2d.size(2), spatial_features_2d.size(3)

        down_sample_ratio = self.model_cfg.IMG_SIZE / height

        pooled_features_list = []
        torch.backends.cudnn.enabled = False
        for b_id in range(batch_size):
            # Map global boxes coordinates to feature map coordinates
            x1 = rois[b_id, :, 0] / down_sample_ratio
            y1 = rois[b_id, :, 1] / down_sample_ratio
            x2 = rois[b_id, :, 2] / down_sample_ratio
            y2 = rois[b_id, :, 3] / down_sample_ratio

            angle = torch.zeros((1), device=spatial_features_2d.device)
            cosa = torch.cos(angle)
            sina = torch.sin(angle)

            theta = torch.stack((
                (x2 - x1) / (width - 1) * cosa, (x2 - x1) / (width - 1) * (-sina), (x1 + x2 - width + 1) / (width - 1),
                (y2 - y1) / (height - 1) * sina, (y2 - y1) / (height - 1) * cosa, (y1 + y2 - height + 1) / (height - 1)
            ), dim=1).view(-1, 2, 3).float()

            grid_size = self.model_cfg.GRID_SIZE
            grid = nn.functional.affine_grid(
                theta,
                torch.Size((rois.size(1), spatial_features_2d.size(1), grid_size, grid_size))
            )

            pooled_features = nn.functional.grid_sample(
                spatial_features_2d[b_id].unsqueeze(0).expand(rois.size(1), spatial_features_2d.size(1), height, width),
                grid
            )
            pooled_features = self.learnable['max_pool'](pooled_features)
            pooled_features_list.append(pooled_features.squeeze())

        torch.backends.cudnn.enabled = True
        pooled_features = torch.stack(pooled_features_list, dim=0)

        return pooled_features

    def forward(self, imgFea, maskImages, image_boxs):
        feature_map = self.backbone.forward_features(imgFea)
        imgFea = self.roi_grid_pool(feature_map, image_boxs)

        imgFea = F.normalize(imgFea, p=2, dim=-1)
        imgFea = self.learnable['imgAtt'](imgFea.transpose(0, 1), src_key_padding_mask=(maskImages == 0)).transpose(0,1)

        tmpMask = torch.where(maskImages == 1, torch.tensor([1.0], device=maskImages.device),
                              torch.tensor([0.0], device=maskImages.device))
        
        imgFea = (imgFea * tmpMask.unsqueeze(-1)).sum(dim=1) / tmpMask.sum(dim=1).unsqueeze(-1)  # (bs, dim)
        imgFea = self.learnable['imgFC'](imgFea)
        
        return imgFea


class TextLearnableEncoder(nn.Module):
    def __init__(self, model_cfg):
        super(TextLearnableEncoder, self).__init__()

        self.backbone = Bert(model_cfg)
        self.model_cfg = model_cfg

        self.learnable = nn.ModuleDict()
        self.learnable['textFC'] = FakeTransformer(model_cfg.TEXT_FEATURE_DIM, model_cfg.IMG_FEATURE_DIM, model_cfg.IMG_FEATURE_DIM)
        text_encoder_layer = nn.TransformerEncoderLayer(d_model=model_cfg.TEXT_FEATURE_DIM, nhead=model_cfg.TEXT_TRANSFORMER_HEAD)
        self.learnable['textAtt'] = nn.TransformerEncoder(text_encoder_layer, num_layers=model_cfg.TEXT_TRANSFORMER_LAYER)

        self.init_param()

    def init_param(self):
        for name, param in self.backbone.named_parameters():
            if 'large' not in self.model_cfg.ENCODER:
                if 'layer.11' not in name and 'layer.10' not in name and 'layer.9' not in name and 'layer.8' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                if 'layer.21' not in name and 'layer.22' not in name and 'layer.23' not in name and 'layer.20' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        sys.stdout.flush()

    def forward(self, textFea, maskTexts):
        textFea = self.backbone(textFea)

        textFea = F.normalize(textFea, p=2, dim=-1)
        textFea = self.learnable['textAtt'](textFea.transpose(0, 1), src_key_padding_mask=(maskTexts == 0)).transpose(0,1)

        tmpMask = torch.where(maskTexts == 1, torch.tensor([1.0], device=maskTexts.device),
                              torch.tensor([0.0], device=maskTexts.device))
        
        textFea = (textFea * tmpMask.unsqueeze(-1)).sum(dim=1) / tmpMask.sum(dim=1).unsqueeze(-1)  # (bs, dim)
        textFea = self.learnable['textFC'](textFea)

        return textFea


class VL_model(nn.Module):
    def __init__(self, model_cfg):
        super(VL_model, self).__init__()
        self.model_cfg = model_cfg

        self.learnable = nn.ModuleDict()
        self.learnable['imgencoder'] = ImgLearnableEncoder(model_cfg)
        self.learnable['imgencoder_mom'] = ImgLearnableEncoder(model_cfg)
        self.learnable['textencoder'] = TextLearnableEncoder(model_cfg)
        self.learnable['textencoder_mom'] = TextLearnableEncoder(model_cfg)

        ############ add new params in .yml config file
        self.K = model_cfg.QUEUE_SIZE
        self.m = model_cfg.MOMENTUM
        self.T = model_cfg.TEMPERATURE
        self.topk = model_cfg.TOPK
        self.multi_label = False
        ############ add new params in .yml config file

        # init the parameter of two models
        self.init_param() 
        # create the img queue 
        self.register_buffer("img_queue", torch.randn(model_cfg.IMG_FEATURE_DIM, self.K))
        self.img_queue = nn.functional.normalize(self.img_queue, dim=0)
        self.register_buffer("img_queue_ptr", torch.zeros(1, dtype=torch.long)) # image queue points
        # create the text queue
        self.register_buffer("text_queue", torch.randn(model_cfg.IMG_FEATURE_DIM, self.K))
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        self.register_buffer("text_queue_ptr", torch.zeros(1, dtype=torch.long)) # text queue points

    def init_param(self):
        for param_q, param_k in zip(self.learnable['imgencoder'].parameters(), self.learnable['imgencoder_mom'].parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.learnable['textencoder'].parameters(), self.learnable['textencoder_mom'].parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def forward(self, imgFea, texts, maskImages, maskTexts, image_boxs):
        imgFea = self.learnable['imgencoder'](imgFea, maskImages, image_boxs) # <bsz, img_dim>
        textFea = self.learnable['textencoder'](texts, maskTexts) # <bsz, img_dim>
        
        imgFea = F.normalize(imgFea, p=2, dim=-1)
        textFea = F.normalize(textFea, p=2, dim=-1)

        retrieval_feat_group = {}
        retrieval_feat_group['img_text'] = (imgFea, textFea)

        return retrieval_feat_group


