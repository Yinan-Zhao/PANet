import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import numpy as np
from . import resnet, resnext, mobilenet, hrnet
import time
import pdb

affine_par = True
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_Deeplab(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet_Deeplab, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x, return_feature_maps=True):
        # important: do not optimize the RESNET backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feat_layer2 = x
        feat_layer3 = self.layer3(x)
        x=torch.cat([feat_layer2,feat_layer3],dim=1)
        if return_feature_maps:
            return [x]
        else:
            return x

class ResNet_Deeplab_Objectness(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet_Deeplab_Objectness, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x, return_feature_maps=True):
        # important: do not optimize the RESNET backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
       
        if return_feature_maps:
            return [x]
        else:
            return x

def load_resnet50_param(model, stop_layer='layer4'):
    resnet50 = torchvision.models.resnet50(pretrained=True)
    saved_state_dict = resnet50.state_dict()
    new_params = model.state_dict().copy()
    # copy params from resnet50, except layers after stop_layer
    for i in saved_state_dict:  
        i_parts = i.split('.')
        if not i_parts[0] == stop_layer:
            new_params['.'.join(i_parts)] = saved_state_dict[i]
        else:
            break
    model.load_state_dict(new_params)
    return model

def load_resnet101_param(model, stop_layer='layer4'):
    resnet101 = torchvision.models.resnet101(pretrained=True)
    saved_state_dict = resnet101.state_dict()
    new_params = model.state_dict().copy()
    # copy params from resnet50, except layers after stop_layer
    for i in saved_state_dict:  
        i_parts = i.split('.')
        if not i_parts[0] == stop_layer:
            new_params['.'.join(i_parts)] = saved_state_dict[i]
        else:
            break
    model.load_state_dict(new_params)
    return model


def ResNet50_Deeplab(pretrained=True, **kwargs):
    model = ResNet_Deeplab(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model=load_resnet50_param(model)
    return model

def ResNet101_Deeplab(pretrained=True, **kwargs):
    model = ResNet_Deeplab(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model=load_resnet101_param(model)
    return model

def ResNet50_Deeplab_Objectness(pretrained=True, **kwargs):
    model = ResNet_Deeplab_Objectness(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model=load_resnet50_param(model, stop_layer='fc')
    return model


class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, deep_sup_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale

    def forward(self, feed_dict, *, segSize=None):
        # training
        if segSize is None:
            if self.deep_sup_scale is not None: # use deep supervision technique
                (pred, pred_deepsup) = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))
            else:
                pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))

            loss = self.crit(pred, feed_dict['seg_label'])
            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                loss = loss + loss_deepsup * self.deep_sup_scale

            acc = self.pixel_acc(pred, feed_dict['seg_label'])
            return loss, acc
        # inference
        else:
            pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True), segSize=segSize)
            return pred

class SegmentationAttentionSeparateModule(SegmentationModuleBase):
    def __init__(self, net_enc_query, net_enc_memory, net_att_query, net_att_memory, net_dec, net_projection, net_objectness, net_objectness_decoder, crit, deep_sup_scale=None, zero_memory=False, random_memory_bias=False, random_memory_nobias=False, random_scale=1.0, zero_qval=False, normalize_key=False, p_scalar=40., memory_feature_aggregation=False, memory_noLabel=False, mask_feat_downsample_rate=1, att_mat_downsample_rate=1, objectness_feat_downsample_rate=1., segm_downsampling_rate=8., mask_foreground=False, global_pool_read=False, average_memory_voting=False, average_memory_voting_nonorm=False, mask_memory_RGB=False, linear_classifier_support=False, decay_lamb=1.0, linear_classifier_support_only=False, qread_only=False, feature_as_key=False, objectness_multiply=False, debug=False):
        super(SegmentationAttentionSeparateModule, self).__init__()
        self.encoder_query = net_enc_query
        self.encoder_memory = net_enc_memory
        self.attention_query = net_att_query
        self.attention_memory = net_att_memory
        self.decoder = net_dec
        self.projection = net_projection
        self.objectness = net_objectness
        self.objectness_decoder = net_objectness_decoder
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale
        self.zero_memory = zero_memory
        self.random_memory_bias = random_memory_bias
        self.random_memory_nobias = random_memory_nobias
        self.random_scale = random_scale
        self.zero_qval = zero_qval
        self.normalize_key = normalize_key
        self.p_scalar = p_scalar
        self.memory_feature_aggregation = memory_feature_aggregation
        self.memory_noLabel = memory_noLabel
        self.mask_feat_downsample_rate = mask_feat_downsample_rate
        self.att_mat_downsample_rate = att_mat_downsample_rate
        self.objectness_feat_downsample_rate = objectness_feat_downsample_rate
        self.segm_downsampling_rate = segm_downsampling_rate
        self.mask_foreground = mask_foreground
        self.global_pool_read = global_pool_read
        self.average_memory_voting = average_memory_voting
        self.average_memory_voting_nonorm = average_memory_voting_nonorm
        self.mask_memory_RGB = mask_memory_RGB
        self.linear_classifier_support = linear_classifier_support
        self.decay_lamb = decay_lamb
        self.linear_classifier_support_only = linear_classifier_support_only
        self.qread_only = qread_only
        self.feature_as_key = feature_as_key
        self.objectness_multiply = objectness_multiply

        self.debug = debug

    def maskRead(self, qkey, qmask, mkey, mval, mmask, output_shape, debug=False):
        '''
        read for *mask area* of query from *mask area* of memory
        '''
        B, Dk, _, H, W = mkey.size()
        _, Dv, _, _, _ = mval.size()

        qread = torch.zeros(output_shape).cuda()
        # key: b,dk,t,h,w
        # value: b,dv,t,h,w
        # mask: b,1,t,h,w
        for b in range(B):
            # exceptions
            if qmask[b,0].sum() == 0 or mmask[b,0].sum() == 0: 
                # print('skipping read', qmask[b,0].sum(), mmask[b,0].sum())
                # no query or mask pixels -> skip read

                #qk_b = None
                #mk_b = None
                #mv_b = None
                #p = None
                #qread = None
                continue
                #pass
            qk_b = qkey[b,:,qmask[b,0]] # dk, Nq
            mk_b = mkey[b,:,mmask[b,0]] # dk, Nm
            mv_b = mval[b,:,mmask[b,0]] # dv, Nm 
            # print(mv_b.shape)

            p = self.p_scalar*torch.mm(torch.transpose(mk_b, 0, 1), qk_b) # Nm, Nq
            #p = p / math.sqrt(Dk)
            p = F.softmax(p, dim=0)

            qread[b,:,qmask[b,0]] = torch.mm(mv_b, p) # dv, Nq
            # qval[b,:,qmask[b,0]] = read # dv, Nq
            #qread[b,:,qmask[b,0]] = qread[b,:,qmask[b,0]] + read # dv, Nq

        if debug:
            return qk_b, mk_b, mv_b, p, qread
        else:
            return qread

    def memoryAverageVoting(self, qkey, qmask, mkey, mmask, output_shape, debug=False):
        '''
        read for *mask area* of query from *mask area* of memory
        '''
        B, Dk, _, H, W = mkey.size()

        qread = torch.zeros(output_shape).cuda()
        # key: b,dk,t,h,w
        # value: b,dv,t,h,w
        # mask: b,1,t,h,w
        for b in range(B):
            # exceptions
            if qmask[b,0].sum() == 0 or mmask[b,0].sum() == 0: 
                # print('skipping read', qmask[b,0].sum(), mmask[b,0].sum())
                # no query or mask pixels -> skip read
                continue
            qk_b = qkey[b,:,qmask[b,0]] # dk, Nq
            mk_b = mkey[b,:,mmask[b,0]] # dk, Nm

            p = self.p_scalar*torch.mm(torch.transpose(mk_b, 0, 1), qk_b) # Nm, Nq
            p = F.softmax(p, dim=1)

            p_mean = torch.mean(p, dim=0, keepdim=True)

            qread[b,:,qmask[b,0]] = p_mean # dv, Nq
            # qval[b,:,qmask[b,0]] = read # dv, Nq
            #qread[b,:,qmask[b,0]] = qread[b,:,qmask[b,0]] + read # dv, Nq

        if debug:
            return qk_b, mk_b, None, p, qread
        else:
            return qread

    def memoryAverageVoting_noNorm(self, qkey, qmask, mkey, mmask, output_shape, debug=False):
        '''
        read for *mask area* of query from *mask area* of memory
        '''
        B, Dk, _, H, W = mkey.size()

        qread = torch.zeros(output_shape).cuda()
        # key: b,dk,t,h,w
        # value: b,dv,t,h,w
        # mask: b,1,t,h,w
        for b in range(B):
            # exceptions
            if qmask[b,0].sum() == 0 or mmask[b,0].sum() == 0: 
                # print('skipping read', qmask[b,0].sum(), mmask[b,0].sum())
                # no query or mask pixels -> skip read
                continue
            qk_b = qkey[b,:,qmask[b,0]] # dk, Nq
            mk_b = mkey[b,:,mmask[b,0]] # dk, Nm

            p = torch.mm(torch.transpose(mk_b, 0, 1), qk_b) # Nm, Nq

            p_mean = torch.mean(p, dim=0, keepdim=True)

            qread[b,:,qmask[b,0]] = p_mean # dv, Nq
            # qval[b,:,qmask[b,0]] = read # dv, Nq
            #qread[b,:,qmask[b,0]] = qread[b,:,qmask[b,0]] + read # dv, Nq

        if debug:
            return qk_b, mk_b, None, p, qread
        else:
            return qread

    def memoryEncode(self, encoder, img_refs, return_feature_maps=True):
        # encoding into memory
        feat_ = []
        batch_size, _, num_frames, height, width = img_refs.size()
        for t in range(num_frames):
            feat = encoder(img_refs[:,:,t], return_feature_maps=return_feature_maps)
            feat_.append(feat)

        feats = []
        for i in range(len(feat_[0])):
            tmp = []
            for j in range(len(feat_)):
                tmp.append(feat_[j][i])
            feats.append(torch.stack(tmp, dim=2))
        return feats

    def memoryAttention(self, att_module, feat):
        key_ = []
        val_ = []
        batch_size, _, num_frames, height, width = feat[-1].size()
        for t in range(num_frames):
            key, val = att_module([feat_item[:,:,t] for feat_item in feat])
            key_.append(key)
            val_.append(val)

        keys = torch.stack(key_, dim=2)
        vals = torch.stack(val_, dim=2)
        return keys, vals

    def memoryProjection(self, proj_module, feat):
        key_ = []
        batch_size, _, num_frames, height, width = feat[-1].size()
        for t in range(num_frames):
            key = proj_module([feat_item[:,:,t] for feat_item in feat])
            key_.append(key)

        keys = torch.stack(key_, dim=2)
        return keys

    def linear_classify(self, qproj, mproj, mmask, output_shape, debug=False):
        B, Dk, T, H, W = mproj.size()
        B_q, Dk_q, H_q, W_q = qproj.size()
        assert B_q==B and Dk==Dk_q and H==H_q and W==W_q

        qread = torch.zeros(output_shape).cuda()
        # key: b,dk,t,h,w
        # value: b,dv,t,h,w
        # mask: b,1,t,h,w
        for b in range(B):
            # exceptions
            if mmask[b,0].sum() == 0: 
                # print('skipping read', qmask[b,0].sum(), mmask[b,0].sum())
                # no query or mask pixels -> skip read
                continue            
            m_x = torch.reshape(mproj[b], (Dk,-1)) # dk, Nm
            m_y = torch.reshape(mmask[b], (1,-1))
            w = torch.mm(torch.inverse(torch.mm(m_x, torch.t(m_x))+self.decay_lamb), torch.mm(m_x, torch.t(m_y)))
            q_x = torch.reshape(qproj[b], (Dk,-1)) # dk, Nq
            q_y = torch.mm(torch.t(w), q_x)

            qread[b] = torch.reshape(q_y, (q_y.shape[0],H,W))

        if debug:
            return w, qread
        else:
            return qread

    def downsample_5d(self, feat, downsample_rate, mode='bilinear'):
        # feat: b,dk,t,h,w
        feat_downsample = torch.zeros(feat.shape[0], feat.shape[1], feat.shape[2], feat.shape[3]//downsample_rate, feat.shape[4]//downsample_rate).cuda()
        for t in range(feat.shape[2]):
            feat_downsample[:,:,t,:,:] = nn.functional.interpolate(feat[:,:,t,:,:], 
                size=(feat.shape[3]//downsample_rate, feat.shape[4]//downsample_rate), 
                mode=mode)
        return feat_downsample

    def forward(self, feed_dict, *, segSize=None):
        # training
        if segSize is None:
            if self.deep_sup_scale is not None: # use deep supervision technique
                raise Exception('deep_sup_scale is not implemented') 
            else:
                feature_enc = self.encoder_query(feed_dict['img_data'], return_feature_maps=True)    
         
                qkey, qval = self.attention_query(feature_enc)

                if self.mask_memory_RGB:
                    for i in range(3):
                        feed_dict['img_refs_rgb'][:,i] = feed_dict['img_refs_rgb'][:,i]*feed_dict['img_refs_mask'][:,1]
                
                feature_memory = self.memoryEncode(self.encoder_query, feed_dict['img_refs_rgb'], return_feature_maps=True)
                
                mkey, mval_rgb = self.memoryAttention(self.attention_query, feature_memory)
                
                mask_feature_memory = self.memoryEncode(self.encoder_memory, feed_dict['img_refs_mask'], return_feature_maps=True)
                
                _, mval = self.memoryAttention(self.attention_memory, mask_feature_memory)

                if self.att_mat_downsample_rate != 1:
                    output_shape = (qval.shape[0], qval.shape[1], qval.shape[2]//self.att_mat_downsample_rate, qval.shape[3]//self.att_mat_downsample_rate)
                    qkey = nn.functional.interpolate(qkey, 
                        size=(qkey.shape[2]//self.att_mat_downsample_rate, 
                            qkey.shape[3]//self.att_mat_downsample_rate), 
                        mode='bilinear')

                    mkey = self.downsample_5d(mkey, downsample_rate=self.att_mat_downsample_rate)

                    if self.memory_feature_aggregation:
                        mval_rgb = self.downsample_5d(mval_rgb, downsample_rate=self.att_mat_downsample_rate)
                else:
                    output_shape = qval.shape

                if self.mask_feat_downsample_rate != 1:
                    mval = self.downsample_5d(mval, downsample_rate=self.mask_feat_downsample_rate)

                qmask = torch.ones_like(qkey)[:,0:1] > 0.
                if self.mask_foreground:
                    mmask = self.downsample_5d(feed_dict['img_refs_mask'][:,1:2,:,:,:], downsample_rate=self.segm_downsampling_rate, mode='nearest') > 0.5
                else:
                    mmask = torch.ones_like(mkey)[:,0:1] > 0.

                if self.feature_as_key:
                    qkey = feature_enc[-1]
                    mkey = feature_memory[-1]

                if self.normalize_key:
                    qkey = F.normalize(qkey, p=2, dim=1)
                    mkey = F.normalize(mkey, p=2, dim=1)

                if self.memory_feature_aggregation:
                    if self.memory_noLabel:
                        mval = mval_rgb
                        if self.debug:
                            qk_b, mk_b, mv_b, p, qread = self.maskRead(qkey, qmask, mkey, mval, mmask, output_shape, self.debug)
                        else:
                            qread = self.maskRead(qkey, qmask, mkey, mval, mmask, output_shape)
                    else:
                        if self.debug:
                            qk_b, mk_b, mv_b, p, qread_label = self.maskRead(qkey, qmask, mkey, mval, mmask, output_shape, self.debug)
                            qk_b, mk_b, mv_b, p, qread_rgb = self.maskRead(qkey, qmask, mkey, mval_rgb, mmask, output_shape, self.debug)
                        else:
                            qread_label = self.maskRead(qkey, qmask, mkey, mval, mmask, output_shape)
                            qread_rgb = self.maskRead(qkey, qmask, mkey, mval_rgb, mmask, output_shape)
                        qread = torch.cat((qread_label, qread_rgb), dim=1)
                elif self.average_memory_voting:
                    output_shape = (qval.shape[0], 1, qval.shape[2], qval.shape[3])
                    if self.debug:
                        qk_b, mk_b, mv_b, p, qread = self.memoryAverageVoting(qkey, qmask, mkey, mmask, output_shape, self.debug)
                    else:
                        qread = self.memoryAverageVoting(qkey, qmask, mkey, mmask, output_shape)
                elif self.average_memory_voting_nonorm:
                    output_shape = (qval.shape[0], 1, qval.shape[2], qval.shape[3])
                    if self.debug:
                        qk_b, mk_b, mv_b, p, qread = self.memoryAverageVoting_noNorm(qkey, qmask, mkey, mmask, output_shape, self.debug)
                    else:
                        qread = self.memoryAverageVoting_noNorm(qkey, qmask, mkey, mmask, output_shape)
                else:
                    if self.debug:
                        qk_b, mk_b, mv_b, p, qread = self.maskRead(qkey, qmask, mkey, mval, mmask, output_shape, self.debug)
                    else:
                        qread = self.maskRead(qkey, qmask, mkey, mval, mmask, output_shape)

                if self.att_mat_downsample_rate != 1:
                    qread = nn.functional.interpolate(qread, 
                        size=(qread.shape[2]*self.att_mat_downsample_rate, 
                            qread.shape[3]*self.att_mat_downsample_rate), 
                        mode='bilinear')

                if self.global_pool_read:
                    support_mask = self.downsample_5d(feed_dict['img_refs_mask'][:,1:2,:,:,:], downsample_rate=self.segm_downsampling_rate, mode='nearest')
                    for p in range(support_mask.shape[2]):
                        support_mask_slice = support_mask[:,:,p,:,:]
                        h,w = mval_rgb.shape[-2], mval_rgb.shape[-1]

                        area = F.avg_pool2d(support_mask_slice, mval_rgb.shape[-2:]) * h * w + 0.0005
                        z = support_mask_slice * mval_rgb[:,:,p,:,:]
                        z = F.avg_pool2d(input=z,
                                         kernel_size=mval_rgb.shape[-2:]) * h * w / area
                        if p == 0:
                            z_final = z
                        else:
                            z_final += z
                    z = z_final / support_mask.shape[2]
                    #qread = z.expand(-1, -1, mval_rgb.shape[-2], mval_rgb.shape[-1])  # tile for cat
                    qread = z.expand(-1, -1, qval.shape[-2], qval.shape[-1])

                if self.linear_classifier_support:
                    output_linear_shape = (qval.shape[0], 1, qval.shape[2], qval.shape[3])
                    m_y = self.downsample_5d(feed_dict['img_refs_mask'][:,1:2,:,:,:], downsample_rate=self.segm_downsampling_rate, mode='nearest')
                    mproj = self.memoryProjection(self.projection, feature_memory)
                    qproj = self.projection(feature_enc)
                    qread_linear = self.linear_classify(qproj, mproj, m_y, output_linear_shape, debug=False)
                    if self.linear_classifier_support_only:
                        qread = qread_linear
                    else:
                        qread = torch.cat((qread_linear, qread), dim=1)

                if self.objectness and self.objectness_decoder:
                    feat_objectness = self.objectness(feed_dict['img_data'], return_feature_maps=True)
                    pred_objectness = self.objectness_decoder(feat_objectness, return_softmax_noresize=True)
                    if self.objectness_feat_downsample_rate != 1.:
                        pred_objectness = nn.functional.interpolate(pred_objectness, 
                            size=(pred_objectness.shape[2]//self.objectness_feat_downsample_rate, 
                                pred_objectness.shape[3]//self.objectness_feat_downsample_rate), 
                            mode='bilinear')
                    if self.objectness_multiply:
                        objectness_prob = pred_objectness[:,1:2]
                    else:
                        qread = torch.cat((pred_objectness[:,1:2], qread), dim=1)

                if self.zero_qval:
                    qval = torch.zeros_like(qval)

                if self.zero_memory:
                    feature = torch.cat((qval, torch.zeros_like(qread)), dim=1)
                elif self.random_memory_bias:
                    feature = torch.cat((qval, self.random_scale*torch.rand_like(qread)), dim=1)
                elif self.random_memory_nobias:
                    feature = torch.cat((qval, self.random_scale*(torch.rand_like(qread)-0.5)), dim=1)
                else:
                    feature = torch.cat((qval, qread), dim=1)
                if self.qread_only:
                    feature = qread

                if self.objectness_multiply:
                    feature = torch.mul(feature, objectness_prob)
                pred = self.decoder([feature])

            loss = self.crit(pred, feed_dict['seg_label'])
            '''if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                loss = loss + loss_deepsup * self.deep_sup_scale'''

            acc = self.pixel_acc(pred, feed_dict['seg_label'])
            if self.debug:
                return loss, acc, qread, qval, qk_b, mk_b, mv_b, p, feature_enc, feature_memory
            else:
                return loss, acc
        # inference
        else:
            feature_enc = self.encoder_query(feed_dict['img_data'], return_feature_maps=True)                
            qkey, qval = self.attention_query(feature_enc)
            
            feature_memory = self.memoryEncode(self.encoder_query, feed_dict['img_refs_rgb'], return_feature_maps=True)
            mkey, mval_rgb = self.memoryAttention(self.attention_query, feature_memory)

            mask_feature_memory = self.memoryEncode(self.encoder_memory, feed_dict['img_refs_mask'], return_feature_maps=True)
            _, mval = self.memoryAttention(self.attention_memory, mask_feature_memory)

            if self.att_mat_downsample_rate != 1:
                output_shape = (qval.shape[0], qval.shape[1], qval.shape[2]//self.att_mat_downsample_rate, qval.shape[3]//self.att_mat_downsample_rate)
                qkey = nn.functional.interpolate(qkey, 
                    size=(qkey.shape[2]//self.att_mat_downsample_rate, 
                        qkey.shape[3]//self.att_mat_downsample_rate), 
                    mode='bilinear')

                mkey = self.downsample_5d(mkey, downsample_rate=self.att_mat_downsample_rate)

                if self.memory_feature_aggregation:
                    mval_rgb = self.downsample_5d(mval_rgb, downsample_rate=self.att_mat_downsample_rate)
            else:
                output_shape = qval.shape

            if self.mask_feat_downsample_rate != 1:
                mval = self.downsample_5d(mval, downsample_rate=self.mask_feat_downsample_rate)

            qmask = torch.ones_like(qkey)[:,0:1] > 0.
            if self.mask_foreground:
                mmask = self.downsample_5d(feed_dict['img_refs_mask'][:,1:2,:,:,:], downsample_rate=self.segm_downsampling_rate, mode='nearest') > 0.5
            else:
                mmask = torch.ones_like(mkey)[:,0:1] > 0.

            if self.feature_as_key:
                qkey = feature_enc[-1]
                mkey = feature_memory[-1]

            if self.normalize_key:
                qkey = F.normalize(qkey, p=2, dim=1)
                mkey = F.normalize(mkey, p=2, dim=1)
            
            if self.memory_feature_aggregation:
                if self.memory_noLabel:
                    mval = mval_rgb
                    if self.debug:
                        qk_b, mk_b, mv_b, p, qread = self.maskRead(qkey, qmask, mkey, mval, mmask, output_shape, self.debug)
                    else:
                        qread = self.maskRead(qkey, qmask, mkey, mval, mmask, output_shape)
                else:
                    if self.debug:
                        qk_b, mk_b, mv_b, p, qread_label = self.maskRead(qkey, qmask, mkey, mval, mmask, output_shape, self.debug)
                        qk_b, mk_b, mv_b, p, qread_rgb = self.maskRead(qkey, qmask, mkey, mval_rgb, mmask, output_shape, self.debug)
                    else:
                        qread_label = self.maskRead(qkey, qmask, mkey, mval, mmask, output_shape)
                        qread_rgb = self.maskRead(qkey, qmask, mkey, mval_rgb, mmask, output_shape)
                    qread = torch.cat((qread_label, qread_rgb), dim=1)
            elif self.average_memory_voting:
                output_shape = (qval.shape[0], 1, qval.shape[2], qval.shape[3])
                if self.debug:
                    qk_b, mk_b, mv_b, p, qread = self.memoryAverageVoting(qkey, qmask, mkey, mmask, output_shape, self.debug)
                else:
                    qread = self.memoryAverageVoting(qkey, qmask, mkey, mmask, output_shape)
            elif self.average_memory_voting_nonorm:
                output_shape = (qval.shape[0], 1, qval.shape[2], qval.shape[3])
                if self.debug:
                    qk_b, mk_b, mv_b, p, qread = self.memoryAverageVoting_noNorm(qkey, qmask, mkey, mmask, output_shape, self.debug)
                else:
                    qread = self.memoryAverageVoting_noNorm(qkey, qmask, mkey, mmask, output_shape)
            else:
                if self.debug:
                    qk_b, mk_b, mv_b, p, qread = self.maskRead(qkey, qmask, mkey, mval, mmask, output_shape, self.debug)
                else:
                    qread = self.maskRead(qkey, qmask, mkey, mval, mmask, output_shape)

            if self.att_mat_downsample_rate != 1:
                qread = nn.functional.interpolate(qread, 
                    size=(qread.shape[2]*self.att_mat_downsample_rate, 
                        qread.shape[3]*self.att_mat_downsample_rate), 
                    mode='bilinear')

            if self.global_pool_read:
                support_mask = self.downsample_5d(feed_dict['img_refs_mask'][:,1:2,:,:,:], downsample_rate=self.segm_downsampling_rate, mode='nearest')
                for p in range(support_mask.shape[2]):
                    support_mask_slice = support_mask[:,:,p,:,:]
                    h,w = mval_rgb.shape[-2], mval_rgb.shape[-1]

                    area = F.avg_pool2d(support_mask_slice, mval_rgb.shape[-2:]) * h * w + 0.0005
                    z = support_mask_slice * mval_rgb[:,:,p,:,:]
                    z = F.avg_pool2d(input=z,
                                     kernel_size=mval_rgb.shape[-2:]) * h * w / area
                    if p == 0:
                        z_final = z
                    else:
                        z_final += z
                z = z_final / support_mask.shape[2]
                #qread = z.expand(-1, -1, mval_rgb.shape[-2], mval_rgb.shape[-1])  # tile for cat
                qread = z.expand(-1, -1, qval.shape[-2], qval.shape[-1])

            if self.linear_classifier_support:
                output_linear_shape = (qval.shape[0], 1, qval.shape[2], qval.shape[3])
                m_y = self.downsample_5d(feed_dict['img_refs_mask'][:,1:2,:,:,:], downsample_rate=self.segm_downsampling_rate, mode='nearest')
                mproj = self.memoryProjection(self.projection, feature_memory)
                qproj = self.projection(feature_enc)
                qread_linear = self.linear_classify(qproj, mproj, m_y, output_linear_shape, debug=False)
                if self.linear_classifier_support_only:
                    qread = qread_linear
                else:
                    qread = torch.cat((qread_linear, qread), dim=1)

            if self.objectness and self.objectness_decoder:
                feat_objectness = self.objectness(feed_dict['img_data'], return_feature_maps=True)
                pred_objectness = self.objectness_decoder(feat_objectness, return_softmax_noresize=True)
                if self.objectness_feat_downsample_rate != 1.:
                    pred_objectness = nn.functional.interpolate(pred_objectness, 
                        size=(pred_objectness.shape[2]//self.objectness_feat_downsample_rate, 
                            pred_objectness.shape[3]//self.objectness_feat_downsample_rate), 
                        mode='bilinear')
                if self.objectness_multiply:
                    objectness_prob = pred_objectness[:,1:2]
                else:
                    qread = torch.cat((pred_objectness[:,1:2], qread), dim=1)

            if self.zero_qval:
                qval = torch.zeros_like(qval)

            if self.zero_memory:
                feature = torch.cat((qval, torch.zeros_like(qread)), dim=1)
            elif self.random_memory_bias:
                feature = torch.cat((qval, self.random_scale*torch.rand_like(qread)), dim=1)
            elif self.random_memory_nobias:
                feature = torch.cat((qval, self.random_scale*(torch.rand_like(qread)-0.5)), dim=1)
            else:
                feature = torch.cat((qval, qread), dim=1)
            if self.qread_only:
                feature = qread

            if self.objectness_multiply:
                feature = torch.mul(feature, objectness_prob)
            pred = self.decoder([feature], segSize=segSize)

            if self.debug:
                return pred, qread, qval, qk_b, mk_b, mv_b, p, feature_enc, feature_memory
            else:
                return pred


class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)


    @staticmethod
    def build_encoder_memory_separate(arch='resnet50dilated', fc_dim=512, weights='', num_class=150, RGB_mask_combine_val=False, segm_downsampling_rate=8, pretrained=True):
        arch = arch.lower()
        if arch == 'resnet18dilated':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated_Memory_Separate(orig_resnet, dilate_scale=8, num_class=num_class, RGB_mask_combine_val=RGB_mask_combine_val)
        elif arch == 'resnet50dilated':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated_Memory_Separate(orig_resnet, dilate_scale=8, num_class=num_class, RGB_mask_combine_val=RGB_mask_combine_val)
        elif arch == 'resnet101dilated':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated_Memory_Separate(orig_resnet, dilate_scale=8, num_class=num_class, RGB_mask_combine_val=RGB_mask_combine_val)
        elif arch == 'resnet18dilated_nobn':
            orig_resnet = resnet.__dict__['resnet18_noBN']()
            net_encoder = ResnetDilated_Memory_Separate_noBN(orig_resnet, dilate_scale=8, num_class=num_class, RGB_mask_combine_val=RGB_mask_combine_val)
        elif arch == 'c1':
            net_encoder = C1_Encoder_Memory(num_class=num_class, fc_dim=fc_dim, segm_downsampling_rate=segm_downsampling_rate, RGB_mask_combine_val=RGB_mask_combine_val)
        elif arch == 'c3':
            net_encoder = C3_Encoder_Memory(num_class=num_class, fc_dim=fc_dim, segm_downsampling_rate=segm_downsampling_rate, RGB_mask_combine_val=RGB_mask_combine_val)
        elif arch == 'hrnetv2':
            if RGB_mask_combine_val:
                net_encoder = hrnet.__dict__['hrnetv2'](pretrained=pretrained, input_dim=3+1+num_class)
            else:
                net_encoder = hrnet.__dict__['hrnetv2'](pretrained=pretrained, input_dim=1+num_class)
        else:
            print(arch)
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        if not pretrained:
            net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_encoder(arch='resnet50_deeplab', fc_dim=512, weights='', fix_encoder=True):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        if arch == 'resnet50_deeplab':
            net_encoder = ResNet50_Deeplab(pretrained=pretrained)
        elif arch == 'resnet101_deeplab':
            net_encoder = ResNet101_Deeplab(pretrained=pretrained)
        elif arch == 'resnet50_deeplab_layer4':
            net_encoder = ResNet50_Deeplab_Objectness(pretrained=pretrained)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet_middle(orig_resnet)
        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)

        if fix_encoder:
            for param in net_encoder.parameters():
                param.requires_grad = False
        return net_encoder

    @staticmethod
    def build_objectness(arch='resnet50_deeplab', weights='', fix_encoder=True):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        if arch == 'resnet50_deeplab':
            net_encoder = ResNet50_Deeplab_Objectness(pretrained=pretrained)
        elif arch == 'hrnetv2':
            net_encoder = hrnet.__dict__['hrnetv2'](pretrained=pretrained, input_dim=3)
        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)

        if fix_encoder:
            for param in net_encoder.parameters():
                param.requires_grad = False
        return net_encoder

    @staticmethod
    def build_attention(arch='attention', input_dim=512, fc_dim=512, weights=''):
        arch = arch.lower()
        if arch == 'attention':
            net_encoder = AttModule(input_dim=input_dim, fc_dim=fc_dim)
        elif arch == 'attention_double':
            net_encoder = AttModule_Double(input_dim=input_dim, fc_dim=fc_dim)
        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_att_query')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_projection(arch='projection', input_dim=512, fc_dim=8, weights=''):
        arch = arch.lower()
        if arch == 'projection':
            net_encoder = ProjModule(input_dim=input_dim, fc_dim=fc_dim)
        elif arch == '':
            return None
        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_att_query')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_decoder(arch='ppm_deepsup', input_dim=512,
                      fc_dim=512, ppm_dim=256, num_class=150,
                      weights='', dropout_rate=0.5, use_dropout=False,
                      use_softmax=False):
        arch = arch.lower()
        if arch == 'c1_deepsup':
            net_decoder = C1DeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c1':
            net_decoder = C1(
                num_class=num_class,
                input_dim=input_dim,
                fc_dim=fc_dim,
                dropout_rate=dropout_rate,
                use_dropout=use_dropout,
                use_softmax=use_softmax)
        elif arch == 'c1_nodropout':
            net_decoder = C1_NoDropout(
                num_class=num_class,
                input_dim=input_dim,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm':
            net_decoder = PPM(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_few_shot':
            net_decoder = PPM_Few_Shot(
                num_class=num_class,
                input_dim=input_dim,
                fc_dim=ppm_dim,
                dropout_rate=dropout_rate,
                use_dropout=use_dropout,
                use_softmax=use_softmax)
        elif arch == 'ppm_few_shot_dropout':
            net_decoder = PPM_Few_Shot_Dropout(
                num_class=num_class,
                input_dim=input_dim,
                fc_dim=ppm_dim,
                dropout_rate=dropout_rate,
                use_dropout=use_dropout,
                use_softmax=use_softmax)
        elif arch == 'ppm_few_shot_dropout_nobn':
            net_decoder = PPM_Few_Shot_Dropout_noBN(
                num_class=num_class,
                input_dim=input_dim,
                fc_dim=ppm_dim,
                dropout_rate=dropout_rate,
                use_dropout=use_dropout,
                use_softmax=use_softmax)
        elif arch == 'aspp_few_shot':
            net_decoder = ASPP_Few_Shot(
                num_class=num_class,
                input_dim=input_dim,
                fc_dim=ppm_dim,
                dropout_rate=dropout_rate,
                use_dropout=use_dropout,
                use_softmax=use_softmax)
        elif arch == 'aspp_few_shot_bn':
            net_decoder = ASPP_Few_Shot_BN(
                num_class=num_class,
                input_dim=input_dim,
                fc_dim=ppm_dim,
                dropout_rate=dropout_rate,
                use_dropout=use_dropout,
                use_softmax=use_softmax)
        elif arch == 'ppm_deepsup':
            net_decoder = PPMDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'upernet_lite':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=256)
        elif arch == 'upernet':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )

def conv3x3_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.ReLU(inplace=True),
            )


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = torch.cat([conv_out[-2],conv_out[-1]],dim=1)

        if return_feature_maps:
            return conv_out
        return [x]

class Resnet_middle(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet_middle, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        layer2_downsample = nn.functional.interpolate(conv_out[-2], 
            size=(x.shape[-2], x.shape[-1]), 
            mode='bilinear')
        x = torch.cat([layer2_downsample,conv_out[-1]],dim=1)


        if return_feature_maps:
            return [x]
        else:
            return x

class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = torch.cat([conv_out[-2],conv_out[-1]],dim=1)

        if return_feature_maps:
            return conv_out
        return [x]

class ResnetDilated_Memory_Separate(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8, num_class=150, RGB_mask_combine_val=False):
        super(ResnetDilated_Memory_Separate, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        #self.conv1 = orig_resnet.conv1
        if RGB_mask_combine_val:
            self.conv1 = conv3x3(3+1+num_class, 64, stride=2)
        else:
            self.conv1 = conv3x3(1+num_class, 64, stride=2)

        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        nn.init.kaiming_normal_(self.conv1.weight.data)
        if RGB_mask_combine_val:
            self.conv1.weight.data[:,0:3,:,:] = orig_resnet.conv1.weight.data

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]

class ResnetDilated_Memory_Separate_noBN(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8, num_class=150, RGB_mask_combine_val=False):
        super(ResnetDilated_Memory_Separate_noBN, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        #self.conv1 = orig_resnet.conv1
        if RGB_mask_combine_val:
            self.conv1 = conv3x3(3+1+num_class, 64, stride=2)
        else:
            self.conv1 = conv3x3(1+num_class, 64, stride=2)
        
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        nn.init.kaiming_normal_(self.conv1.weight.data)

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]

class C1_Encoder_Memory(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, segm_downsampling_rate=8, RGB_mask_combine_val=False):
        super(C1_Encoder_Memory, self).__init__()
        if RGB_mask_combine_val:
            self.cbr = conv3x3_relu(3+1+num_class, fc_dim, 1)
        else:
            self.cbr = conv3x3_relu(1+num_class, fc_dim, 1)
        self.segm_downsampling_rate = segm_downsampling_rate

    def forward(self, x, return_feature_maps=False):
        x_downsample = nn.functional.interpolate(x, 
            size=(x.shape[-2]//self.segm_downsampling_rate, x.shape[-1]//self.segm_downsampling_rate), 
            mode='nearest')
        x = self.cbr(x_downsample)
        if return_feature_maps:
            return [x]
        else:
            return x

class C3_Encoder_Memory(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, segm_downsampling_rate=8, RGB_mask_combine_val=False):
        super(C3_Encoder_Memory, self).__init__()
        if RGB_mask_combine_val:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3+1+num_class, out_channels=fc_dim//4, kernel_size=3, stride=2, padding=1,
                      bias=True),
                nn.ReLU(inplace=True)
                )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1+num_class, out_channels=fc_dim//4, kernel_size=3, stride=2, padding=1,
                      bias=True),
                nn.ReLU(inplace=True),
            )
        self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=fc_dim//4, out_channels=fc_dim//2, kernel_size=3, stride=2, padding=1,
                      bias=True),
                nn.ReLU(inplace=True)
                )
        self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels=fc_dim//2, out_channels=fc_dim, kernel_size=3, stride=2, padding=1,
                      bias=True),
                nn.ReLU(inplace=True)
                )

        self.segm_downsampling_rate = segm_downsampling_rate

    def forward(self, x, return_feature_maps=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if return_feature_maps:
            return [x]
        else:
            return x

class MobileNetV2Dilated(nn.Module):
    def __init__(self, orig_net, dilate_scale=8):
        super(MobileNetV2Dilated, self).__init__()
        from functools import partial

        # take pretrained mobilenet features
        self.features = orig_net.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if dilate_scale == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif dilate_scale == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        if return_feature_maps:
            conv_out = []
            for i in range(self.total_idx):
                x = self.features[i](x)
                if i in self.down_idx:
                    conv_out.append(x)
            conv_out.append(x)
            return conv_out

        else:
            return [self.features(x)]


# last conv, deep supervision
class C1DeepSup(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1DeepSup, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# last conv
class C1(nn.Module):
    def __init__(self, num_class=150, input_dim=1024, fc_dim=2048, dropout_rate=0.5, use_dropout=False, use_softmax=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax
        self.use_dropout = use_dropout

        self.cbr = conv3x3_bn_relu(input_dim, fc_dim // 2, 1)
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 2, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.conv_last(x)

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x
        
# last conv
class C1_NoDropout(nn.Module):
    def __init__(self, num_class=150, input_dim=1024, fc_dim=2048, use_softmax=False):
        super(C1_NoDropout, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(input_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None, return_softmax_noresize=False):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)

        if return_softmax_noresize:
            x = nn.functional.softmax(x, dim=1)
            return x

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x

class AttModule(nn.Module):
    def __init__(self, input_dim=512, fc_dim=2048):
        super(AttModule, self).__init__()

        self.key_conv = nn.Conv2d(input_dim, fc_dim//8, kernel_size=1,
                      stride=1, padding=0, bias=False)
        self.value_conv = nn.Conv2d(input_dim, fc_dim//2, kernel_size=1,
                      stride=1, padding=0, bias=False)
        self.out_dim = fc_dim//2

    def forward(self, conv_out):
        conv5 = conv_out[-1]
        key = self.key_conv(conv5)
        value = self.value_conv(conv5)

        return key, value

class AttModule_Double(nn.Module):
    def __init__(self, input_dim=512, fc_dim=2048):
        super(AttModule_Double, self).__init__()

        self.key_conv = nn.Conv2d(input_dim, fc_dim//4, kernel_size=1,
                      stride=1, padding=0, bias=False)
        self.value_conv = nn.Conv2d(input_dim, fc_dim, kernel_size=1,
                      stride=1, padding=0, bias=False)
        self.out_dim = fc_dim

    def forward(self, conv_out):
        conv5 = conv_out[-1]
        key = self.key_conv(conv5)
        value = self.value_conv(conv5)

        return key, value

class ProjModule(nn.Module):
    def __init__(self, input_dim=512, fc_dim=8):
        super(ProjModule, self).__init__()

        self.key_conv = nn.Conv2d(input_dim, fc_dim, kernel_size=1,
                      stride=1, padding=0, bias=False)

    def forward(self, conv_out):
        conv5 = conv_out[-1]
        projection = self.key_conv(conv5)

        return projection

# pyramid pooling
class PPM(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x

# pyramid pooling
class PPM_Few_Shot(nn.Module):
    def __init__(self, num_class=150, input_dim=1024, fc_dim=256, dropout_rate=0.5,
                 use_dropout=False, use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPM_Few_Shot, self).__init__()
        self.use_softmax = use_softmax
        self.use_dropout = use_dropout
        self.fc_dim = fc_dim

        self.ppm = []

        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, fc_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fc_dim),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        if self.use_dropout:
            self.conv1 = nn.Sequential(
                conv3x3_bn_relu(input_dim, fc_dim, 1),
                nn.Dropout2d(dropout_rate)
            )
            self.conv_last = nn.Sequential(
                nn.Conv2d(fc_dim+len(pool_scales)*fc_dim, fc_dim,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fc_dim),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate),
                nn.Conv2d(fc_dim, num_class, kernel_size=1)
            )
        else:
            self.conv1 = nn.Sequential(
                conv3x3_bn_relu(input_dim, fc_dim, 1)
            )
            self.conv_last = nn.Sequential(
                nn.Conv2d(fc_dim+len(pool_scales)*fc_dim, fc_dim,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fc_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(fc_dim, num_class, kernel_size=1)
            )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        x = self.conv1(conv5)
        ppm_out = [x]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(x[:,:self.fc_dim]),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x

class PPM_Few_Shot_Dropout(nn.Module):
    def __init__(self, num_class=150, input_dim=1024, fc_dim=256, dropout_rate=0.5,
                 use_dropout=False, use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPM_Few_Shot_Dropout, self).__init__()
        self.use_softmax = use_softmax
        self.use_dropout = use_dropout
        self.fc_dim = fc_dim

        self.ppm = []

        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, fc_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fc_dim),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_rate)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        if self.use_dropout:
            self.conv1 = nn.Sequential(
                conv3x3_bn_relu(input_dim, fc_dim, 1),
                nn.Dropout2d(dropout_rate)
            )
            self.conv_last = nn.Sequential(
                nn.Conv2d(fc_dim+len(pool_scales)*fc_dim, fc_dim,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fc_dim),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate),
                nn.Conv2d(fc_dim, num_class, kernel_size=1)
            )
        else:
            self.conv1 = nn.Sequential(
                conv3x3_bn_relu(input_dim, fc_dim, 1)
            )
            self.conv_last = nn.Sequential(
                nn.Conv2d(fc_dim+len(pool_scales)*fc_dim, fc_dim,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fc_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(fc_dim, num_class, kernel_size=1)
            )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        x = self.conv1(conv5)
        ppm_out = [x]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(x[:,:self.fc_dim]),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x

class PPM_Few_Shot_Dropout_noBN(nn.Module):
    def __init__(self, num_class=150, input_dim=1024, fc_dim=256, dropout_rate=0.5,
                 use_dropout=False, use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPM_Few_Shot_Dropout_noBN, self).__init__()
        self.use_softmax = use_softmax
        self.use_dropout = use_dropout
        self.fc_dim = fc_dim

        self.ppm = []

        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, fc_dim, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_rate)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        if self.use_dropout:
            self.conv1 = nn.Sequential(
                conv3x3_relu(input_dim, fc_dim, 1),
                nn.Dropout2d(dropout_rate)
            )
            self.conv_last = nn.Sequential(
                nn.Conv2d(fc_dim+len(pool_scales)*fc_dim, fc_dim,
                          kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate),
                nn.Conv2d(fc_dim, num_class, kernel_size=1)
            )
        else:
            self.conv1 = nn.Sequential(
                conv3x3_relu(input_dim, fc_dim, 1)
            )
            self.conv_last = nn.Sequential(
                nn.Conv2d(fc_dim+len(pool_scales)*fc_dim, fc_dim,
                          kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(fc_dim, num_class, kernel_size=1)
            )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        x = self.conv1(conv5)
        ppm_out = [x]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(x[:,:self.fc_dim]),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x

class ASPP_Few_Shot(nn.Module):
    def __init__(self, num_class=150, input_dim=1024, fc_dim=256, dropout_rate=0.5,
                 use_dropout=False, use_softmax=False):
        super(ASPP_Few_Shot, self).__init__()
        self.use_softmax = use_softmax
        self.use_dropout = use_dropout
        self.fc_dim = fc_dim

        self.conv1 = nn.Sequential(
            conv3x3_relu(input_dim, fc_dim, 1),
            nn.Dropout2d(p=dropout_rate)
        )
        self.layer6_0 = nn.Sequential(
            nn.Conv2d(fc_dim , fc_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate),
        )
        self.layer6_1 = nn.Sequential(
            nn.Conv2d(fc_dim , fc_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate),
            )
        self.layer6_2 = nn.Sequential(
            nn.Conv2d(fc_dim , fc_dim , kernel_size=3, stride=1, padding=6,dilation=6, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate)
            )
        self.layer6_3 = nn.Sequential(
            nn.Conv2d(fc_dim , fc_dim, kernel_size=3, stride=1, padding=12, dilation=12, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate)
            )
        self.layer6_4 = nn.Sequential(
            nn.Conv2d(fc_dim , fc_dim , kernel_size=3, stride=1, padding=18, dilation=18, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate)
            )
        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+4*fc_dim, fc_dim,
                      kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(fc_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None, return_softmax_noresize=False):
        conv5 = conv_out[-1]

        x = self.conv1(conv5)
        feature_size = x.shape[-2:]

        global_feature = F.avg_pool2d(x, kernel_size=feature_size)
        global_feature = self.layer6_0(global_feature)
        global_feature = global_feature.expand(-1,-1,feature_size[0],feature_size[1])
        out = torch.cat([global_feature, self.layer6_1(x), self.layer6_2(x), self.layer6_3(x), self.layer6_4(x)], dim=1)

        x = self.conv_last(out)

        if return_softmax_noresize:
            x = nn.functional.softmax(x, dim=1)
            return x

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=True)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x

class ASPP_Few_Shot_BN(nn.Module):
    def __init__(self, num_class=150, input_dim=1024, fc_dim=256, dropout_rate=0.5,
                 use_dropout=False, use_softmax=False):
        super(ASPP_Few_Shot_BN, self).__init__()
        self.use_softmax = use_softmax
        self.use_dropout = use_dropout
        self.fc_dim = fc_dim

        self.conv1 = nn.Sequential(
            conv3x3_bn_relu(input_dim, fc_dim, 1),
            nn.Dropout2d(p=dropout_rate)
        )
        self.layer6_0 = nn.Sequential(
            nn.Conv2d(fc_dim , fc_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(fc_dim),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate),
        )
        self.layer6_1 = nn.Sequential(
            nn.Conv2d(fc_dim , fc_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(fc_dim),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate),
            )
        self.layer6_2 = nn.Sequential(
            nn.Conv2d(fc_dim , fc_dim , kernel_size=3, stride=1, padding=6,dilation=6, bias=True),
            nn.BatchNorm2d(fc_dim),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate)
            )
        self.layer6_3 = nn.Sequential(
            nn.Conv2d(fc_dim , fc_dim, kernel_size=3, stride=1, padding=12, dilation=12, bias=True),
            nn.BatchNorm2d(fc_dim),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate)
            )
        self.layer6_4 = nn.Sequential(
            nn.Conv2d(fc_dim , fc_dim , kernel_size=3, stride=1, padding=18, dilation=18, bias=True),
            nn.BatchNorm2d(fc_dim),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate)
            )
        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+4*fc_dim, fc_dim,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(fc_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.conv1(conv5)
        feature_size = x.shape[-2:]

        global_feature = F.avg_pool2d(x, kernel_size=feature_size)
        global_feature = self.layer6_0(global_feature)
        global_feature = global_feature.expand(-1,-1,feature_size[0],feature_size[1])
        out = torch.cat([global_feature, self.layer6_1(x), self.layer6_2(x), self.layer6_3(x), self.layer6_4(x)], dim=1)

        x = self.conv_last(out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


# pyramid pooling, deep supervision
class PPMDeepsup(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMDeepsup, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# upernet
class UPerNet(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:   # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)

        return x

