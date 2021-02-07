import math, time
import datetime
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
from base import BaseModel
from utils.helpers import set_trainable
from utils.losses import *
from models.decoders import *
from models.encoder import Encoder
from utils.losses import CE_loss

class CCT(BaseModel):
    def __init__(self, num_classes, conf, sup_loss=None, cons_w_unsup=None, ignore_index=None, testing=False,
            pretrained=True, use_weak_lables=False,unsupervised_mode=None, weakly_loss_w=0.4):

        if not testing:
            assert (ignore_index is not None) and (sup_loss is not None) and (cons_w_unsup is not None)

        super(CCT, self).__init__()
        assert int(conf['supervised']) + int(conf['semi']) == 1, 'one mode only'
        if conf['supervised']:
            self.mode = 'supervised'
        else:
            self.mode = 'semi'

        # Supervised and unsupervised losses
        self.ignore_index = ignore_index
        if conf['un_loss'] == "KL":
        	self.unsuper_loss = softmax_kl_loss
        elif conf['un_loss'] == "MSE":
        	self.unsuper_loss = softmax_mse_loss
        elif conf['un_loss'] == "JS":
        	self.unsuper_loss = softmax_js_loss
        else:
        	raise ValueError(f"Invalid supervised loss {conf['un_loss']}")

        self.unsup_loss_w = cons_w_unsup
        self.sup_loss_w = conf['supervised_w']
        self.softmax_temp = conf['softmax_temp']
        self.sup_loss = sup_loss
        self.sup_type = conf['sup_loss']
        self.unsupervised_mode = unsupervised_mode

        # Use weak labels
        self.use_weak_lables = use_weak_lables
        self.weakly_loss_w = weakly_loss_w
        # pair wise loss (sup mat)
        self.aux_constraint = conf['aux_constraint']
        self.aux_constraint_w = conf['aux_constraint_w']
        # confidence masking (sup mat)
        self.confidence_th = conf['confidence_th']
        self.confidence_masking = conf['confidence_masking']

        # Create the model
        self.encoder = Encoder(pretrained=pretrained)

        # The main encoder
        upscale = 8
        num_out_ch = 2048
        decoder_in_ch = num_out_ch // 4
        self.main_decoder = MainDecoder(upscale, decoder_in_ch, num_classes=num_classes)

        # The auxilary decoders
        if self.mode == 'semi' or self.mode == 'weakly_semi':
            if 'seq' in unsupervised_mode:
                vat_decoder_seq = [VATDecoder(upscale, decoder_in_ch, num_classes, xi=conf['xi'],
                                                                    eps=conf['eps']) for _ in range(conf['vat'])]
                self.seq_decoder = nn.ModuleList([*vat_decoder_seq])

            elif 'pert' in unsupervised_mode:
                vat_decoder = [VATDecoder(upscale, decoder_in_ch, num_classes, xi=conf['xi'],
                                                                    eps=conf['eps']) for _ in range(conf['vat'])]
                drop_decoder = [DropOutDecoder(upscale, decoder_in_ch, num_classes,
                                                                    drop_rate=conf['drop_rate'], spatial_dropout=conf['spatial'])
                                                                    for _ in range(conf['drop'])]
                cut_decoder = [CutOutDecoder(upscale, decoder_in_ch, num_classes, erase=conf['erase'])
                                                                    for _ in range(conf['cutout'])]
                context_m_decoder = [ContextMaskingDecoder(upscale, decoder_in_ch, num_classes)
                                                                    for _ in range(conf['context_masking'])]
                object_masking = [ObjectMaskingDecoder(upscale, decoder_in_ch, num_classes)
                                                                    for _ in range(conf['object_masking'])]
                feature_drop = [FeatureDropDecoder(upscale, decoder_in_ch, num_classes)
                                                                    for _ in range(conf['feature_drop'])]
                feature_noise = [FeatureNoiseDecoder(upscale, decoder_in_ch, num_classes,
                                                                    uniform_range=conf['uniform_range'])
                                                                    for _ in range(conf['feature_noise'])]

                self.aux_decoders = nn.ModuleList([*vat_decoder, *drop_decoder, *cut_decoder,
                                        *context_m_decoder, *object_masking, *feature_drop, *feature_noise])
                ''' 
                temp = [DropOutDecoder(upscale, decoder_in_ch, num_classes,
                                                                    drop_rate=conf['drop_rate'], spatial_dropout=conf['spatial'])
                                                                    for _ in range(conf['drop'])]
                self.seq_decoder = MainDecoder(upscale, decoder_in_ch, num_classes=num_classes)
                '''



    #def forward(self, x_l=None, target_l=None, x_ul=None, target_ul=None, curr_iter=None, epoch=None, img_id_l=None, img_id_ul=None):
    def forward(self, x_l=None, target_l=None, x_ul=None, target_ul=None, curr_iter=None, epoch=None, x_seq_triplet=None, unsupervised_mode=None):
        '''x_seq_triplet is a list with the lan being equal to the len of the seq
        each element in the list is a tuple with the following format
        (imgs:torch.Size([5, 3, 320, 320]), annotations:torch.Size([5, 320, 320, 3]), [list of the img_names])
        5 in this case if the batch size'''
        if not self.training:
            return self.main_decoder(self.encoder(x_l))

        def get_date(x):
            date_ = x.split('.')[0]
            y = '00'
            if date_[3] == '1':
                y = '12'
            elif date_[3] == '0':
                y = '11'
            m = date_[4:6]
            d = date_[6:8]
            date_ = m + d + y
            return datetime.datetime.strptime(date_, '%m%d%y')

        # We compute the losses in the forward pass to avoid problems encountered in muti-gpu 

        # Forward pass the labels example
        input_size = (x_l.size(2), x_l.size(3))
        output_l = self.main_decoder(self.encoder(x_l))
        if output_l.shape != x_l.shape:
            output_l = F.interpolate(output_l, size=input_size, mode='bilinear', align_corners=True)

        # Supervised loss
        if self.sup_type == 'CE':
            loss_sup = self.sup_loss(output_l, target_l, ignore_index=self.ignore_index, temperature=self.softmax_temp) * self.sup_loss_w
        elif self.sup_type == 'FL':
            loss_sup = self.sup_loss(output_l,target_l) * self.sup_loss_w
        else:
            loss_sup = self.sup_loss(output_l, target_l, curr_iter=curr_iter, epoch=epoch, ignore_index=self.ignore_index) * self.sup_loss_w

        # If supervised mode only, return
        if self.mode == 'supervised':
            curr_losses = {'loss_sup': loss_sup}
            outputs = {'sup_pred': output_l}
            total_loss = loss_sup
            return total_loss, curr_losses, outputs

        # If semi supervised mode
        elif self.mode == 'semi':
            # Get sequence predictions
            if unsupervised_mode == 'seq' or unsupervised_mode == 'pertAndSeq':
                seqs = []
                for i in range(len(x_seq_triplet)): 
                    '''a batch of ith imgs in the seq. If the batch_size ==5:
                        (imgs:torch.Size([5, 3, 320, 320]), annotations:torch.Size([5, 320, 320, 3]), [list of the img_names])'''
                    x_seq = self.encoder(x_seq_triplet[i][0])

                    output_seq = self.main_decoder(x_seq)
                    seqs.append((x_seq, output_seq))

                # Get auxiliary predictions
                seq_losses = [] 
                ''' seq_losses has as many elements as the number of imgs in each seq.
                the ith element in seq_losses is the unsupervised loss calculated for 
                the outputs of the maindecoder and the seqdecoder for the batch of ith imgs in seq'''
                all_outputs_ul = []
                '''the ith element in all_outputs_ul is the outputs of the seqdecoder for the batch of the ith imgs in the seq'''
                for pair in seqs:
                    ''' the number of pairs is the number of the images in each seq
                    each pair hold (the output of the encoder for the batch , the output of the maindecoder on pair[0])
                    and outputs_ul is the output of seqdecoder for the batch'''
                    #outputs_ul = [aux_decoder(pair[0], pair[1].detach()) for aux_decoder in self.aux_decoders]
                    #outputs_ul = [aux_decoder(pair[0]) for aux_decoder in [self.seq_decoder]]
                    outputs_ul = [aux_decoder(pair[0], pair[1].detach()) for aux_decoder in self.seq_decoder]
                    targets = F.softmax(pair[1].detach(), dim=1)
                    all_outputs_ul.append(outputs_ul)

                    # Compute unsupervised loss
                    # this loss is between the output of the main decoder and the seqdecoder's
                    loss_seq = sum([self.unsuper_loss(inputs=u, targets=targets, conf_mask=self.confidence_masking,
                        threshold=self.confidence_th, use_softmax=False) for u in outputs_ul])
                    loss_seq = (loss_seq / len(outputs_ul))
                    seq_losses.append(loss_seq)

                loss_seq = np.sum(seq_losses)/len(seqs)
                loss_seq_accross_decoder = []
                for i in range(len(all_outputs_ul)):
                    for j in range(i+1,len(all_outputs_ul)):
                        pert_loss = 0
                        for z in range(len(all_outputs_ul[0])): #because we have 2 decoders in seqdecoder 
                            pert_loss += self.unsuper_loss(inputs=all_outputs_ul[i][z], targets=all_outputs_ul[j][z],
                                    conf_mask=self.confidence_masking,threshold=self.confidence_th, use_softmax=False)
                        pert_loss = pert_loss / len(all_outputs_ul[0])
                        loss_seq_accross_decoder.append(pert_loss)
                loss_seq += np.sum(loss_seq_accross_decoder) /len(loss_seq_accross_decoder)

                output_seqs = []
                for i in range(len(seqs)):
                    temp_x_seq = seqs[i][0]
                    temp_output_seq = seqs[i][1]
                    if temp_output_seq.shape != temp_x_seq.shape:
                        temp_output_seq = F.interpolate(temp_output_seq, size=input_size, mode='bilinear', align_corners=True)
                    output_seqs.append(temp_output_seq)

                # TODO: Figure out why we only return the first output from the main decoder.
                outputs = {'sup_pred': output_l, 'unsup_pred': output_seqs[0]}
                #loss_seq = (loss_seq / 3)
                weight_u = self.unsup_loss_w(epoch=epoch, curr_iter=curr_iter)
                loss_seq = loss_seq * weight_u

            if unsupervised_mode != 'seq':
                # Get main prediction
                x_ul = self.encoder(x_ul)
                output_ul = self.main_decoder(x_ul)
                # Get auxiliary predictions
                outputs_ul = [aux_decoder(x_ul, output_ul.detach()) for aux_decoder in self.aux_decoders]
                targets = F.softmax(output_ul.detach(), dim=1)
                # Compute unsupervised loss
                loss_unsup = sum([self.unsuper_loss(inputs=u, targets=targets, conf_mask=self.confidence_masking,
                    threshold=self.confidence_th, use_softmax=False) for u in outputs_ul])
                loss_unsup = (loss_unsup / len(outputs_ul))

                if output_ul.shape != x_l.shape:
                    output_ul = F.interpolate(output_ul, size=input_size, mode='bilinear', align_corners=True)
                outputs = {'sup_pred': output_l, 'unsup_pred': output_ul}

                weight_u = self.unsup_loss_w(epoch=epoch, curr_iter=curr_iter)
                loss_unsup = loss_unsup * weight_u


            curr_losses = {'loss_sup': loss_sup}
            if unsupervised_mode == 'seq':
                curr_losses['loss_unsup'] = loss_seq 
                total_loss = loss_seq + loss_sup
            elif unsupervised_mode == 'pertAndSeq':
                curr_losses['loss_unsup'] = loss_seq + loss_unsup 
                total_loss = loss_seq + loss_unsup + loss_sup
            else:
                curr_losses['loss_unsup'] = loss_unsup 
                total_loss = loss_unsup + loss_sup

            # In case we're using weak lables, add the weak loss term with a 
            #weight (self.weakly_loss_w)
            if self.use_weak_lables:
                target_ul = target_ul[:,:,:,0]
                weight_w = (weight_u / self.unsup_loss_w.final_w) * self.weakly_loss_w
                loss_weakly = sum([CE_loss(outp, target_ul, ignore_index=self.ignore_index) 
                    for outp in outputs_ul]) / len(outputs_ul)
                loss_weakly = loss_weakly * weight_w
                curr_losses['loss_weakly'] = loss_weakly
                total_loss += loss_weakly

            # Pair-wise loss
            if self.aux_constraint:
                pair_wise = pair_wise_loss(outputs_ul) * self.aux_constraint_w
                curr_losses['pair_wise'] = pair_wise
                loss_unsup += pair_wise

            return total_loss, curr_losses, outputs

    def get_backbone_params(self):
        return self.encoder.get_backbone_params()

    def get_other_params(self):
        if self.mode == 'semi':
            if self.unsupervised_mode == 'pertAndSeq':
                return chain(self.encoder.get_module_params(), self.main_decoder.parameters(), 
                            self.aux_decoders.parameters(), self.seq_decoder.parameters())
            elif self.unsupervised_mode == 'seq':
                return chain(self.encoder.get_module_params(), self.main_decoder.parameters(), 
                             self.seq_decoder.parameters())
            elif self.unsupervised_mode == 'pert':
                return chain(self.encoder.get_module_params(), self.main_decoder.parameters(),
                            self.aux_decoders.parameters())
        return chain(self.encoder.get_module_params(), self.main_decoder.parameters())
