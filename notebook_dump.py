
--- Cell 1 ---
# Only run on Kaggle if packages are missing
!pip install pyyaml scikit-learn progress openslide-python h5py -q

--- Cell 3 ---
import sys
import math
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from collections import defaultdict

import torch
from torch import nn

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def adjust_learning_rate(n_epoch_warmup, n_epoch, max_lr, optimizer, dloader, step):
    """
    Set learning rate according to cosine schedule
    """

    max_steps = int(n_epoch * len(dloader))
    warmup_steps = int(n_epoch_warmup * len(dloader))
    
    if step < warmup_steps:
        lr = max_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = max_lr * 0.001
        lr = max_lr * q + end_lr * (1 - q)

    optimizer.param_groups[0]['lr'] = lr

def shuffle_batch(x, shuffle_idx=None):
    """ shuffles each instance in batch the same way """
    
    if not torch.is_tensor(shuffle_idx):
        seq_len = x.shape[1]
        shuffle_idx = torch.randperm(seq_len)
    x = x[:, shuffle_idx]
    
    return x, shuffle_idx

def shuffle_instance(x, axis, shuffle_idx=None):
    """ shuffles each instance in batch in a different way """

    if not torch.is_tensor(shuffle_idx):
        # get permutation indices
        shuffle_idx = torch.rand(x.shape[:axis+1], device=x.device).argsort(axis)  
    
    idx_expand = shuffle_idx.clone().to(x.device)
    for _ in range(x.ndim-axis-1):
        idx_expand.unsqueeze_(-1)
    # reformat for gather operation
    idx_expand = idx_expand.repeat(*[1 for _ in range(axis+1)], *(x.shape[axis+1:]))  
    
    x = x.gather(axis, idx_expand)

    return x, shuffle_idx

class Logger(nn.Module):
    ''' Stores and computes statistiscs of losses and metrics '''

    def __init__(self, task_dict):
        super().__init__()

        self.task_dict = task_dict
        self.losses_it = defaultdict(list)
        self.losses_epoch = defaultdict(list)
        self.y_preds = defaultdict(list)
        self.y_trues = defaultdict(list)
        self.metrics = defaultdict(list)

    def update(self, next_loss, next_y_pred, next_y_true):

        for task in self.task_dict.values():
            t, t_metr = task['name'], task['metric']
            self.losses_it[t].append(next_loss[t])
            
            if t_metr == 'accuracy':
                y_pred = np.argmax(next_y_pred[t], axis=-1)
            elif t_metr in ['multilabel_accuracy', 'auc']:
                y_pred = next_y_pred[t].tolist()
            self.y_preds[t].extend(y_pred)
            
            self.y_trues[t].extend(next_y_true[t])

    def compute_metric(self):

        for task in self.task_dict.values():
            t = task['name']
            losses = self.losses_it[t]
            self.losses_epoch[t].append(np.mean(losses))

            current_metric = task['metric']
            if current_metric == 'accuracy':
                metric = accuracy_score(self.y_trues[t], self.y_preds[t])
                self.metrics[t].append(metric)
            elif current_metric == 'multilabel_accuracy':
                y_pred = np.array(self.y_preds[t])
                y_true = np.array(self.y_trues[t])
                
                y_pred = np.where(y_pred >= 0.5, 1., 0.)
                correct = np.all(y_pred == y_true, axis=-1).sum()
                total = y_pred.shape[0]
                
                self.metrics[t].append(correct / total)
            elif current_metric == 'auc':
                y_pred = np.array(self.y_preds[t])
                y_true = np.array(self.y_trues[t])
                auc = roc_auc_score(y_true, y_pred)
                self.metrics[t].append(auc)

            # reset per iteration losses, preds and labels
            self.losses_it[t] = []
            self.y_preds[t] = []
            self.y_trues[t] = []


    def print_stats(self, epoch, train, **kwargs):

        print_str = 'Train' if train else 'Test'
        print_str +=  " Epoch: {} \n".format(epoch+1)

        avg_loss = 0
        for task in self.task_dict.values():
            t = task['name']
            metric_name = task['metric']
            mean_loss = self.losses_epoch[t][epoch]
            metric = self.metrics[t][epoch]
           
            avg_loss += mean_loss
 
            print_str += "task: {}, mean loss: {:.5f}, {}: {:.5f}, ".format(t, mean_loss, metric_name, metric)

        avg_loss /= len(self.task_dict.values())
        print_str += "avg. loss over tasks: {:.5f}".format(avg_loss)

        for k, v in kwargs.items():
            print_str += ", {}: {}".format(k, v)
        print_str += "\n"

        print(print_str)

--- Cell 6 ---
import math

import torch
from torch import nn

def pos_enc_1d(D, len_seq):
    
    if D % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(D))
    pe = torch.zeros(len_seq, D)
    position = torch.arange(0, len_seq).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, D, 2, dtype=torch.float) *
                         -(math.log(10000.0) / D)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

class ScaledDotProductAttention(nn.Module):
    ''' Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def compute_attn(self, q, k):
        
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = self.dropout(torch.softmax(attn, dim=-1))

        return attn

    def forward(self, q, k, v):
        
        attn = self.compute_attn(q, k)
        output = torch.matmul(attn, v)

        return output

class MultiHeadCrossAttention(nn.Module):
    ''' Multi-head cross-attention module '''

    def __init__(self, n_token, H, D, D_k, D_v, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        
        self.n_token = n_token
        self.H = H
        self.D_k = D_k
        self.D_v = D_v

        self.q = nn.Parameter(torch.empty((1, n_token, D)))
        q_init_val = math.sqrt(1 / D_k)
        nn.init.uniform_(self.q, a=-q_init_val, b=q_init_val)

        self.q_w = nn.Linear(D, H * D_k, bias=False)
        self.k_w = nn.Linear(D, H * D_k, bias=False)
        self.v_w = nn.Linear(D, H * D_v, bias=False)
        self.fc = nn.Linear(H * D_v, D, bias=False)

        self.attention = ScaledDotProductAttention(
            temperature=D_k ** 0.5,
            attn_dropout=attn_dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(D, eps=1e-6)

    def get_attn(self, x):
        
        D_k, H, n_token = self.D_k, self.H, self.n_token
        B, len_seq = x.shape[:2]

        q = self.q_w(self.q).view(1, n_token, H, D_k)
        k = self.k_w(x).view(B, len_seq, H, D_k)

        q, k = q.transpose(1, 2), k.transpose(1, 2)

        attn = self.attention.compute_attn(q, k)

        return attn

    def forward(self, x):
        
        D_k, D_v, H, n_token = self.D_k, self.D_v, self.H, self.n_token
        B, len_seq = x.shape[:2]

        # project and separate heads
        q = self.q_w(self.q).view(1, n_token, H, D_k)
        k = self.k_w(x).view(B, len_seq, H, D_k)
        v = self.v_w(x).view(B, len_seq, H, D_v)

        # transpose for attention dot product: B x H x len_seq x D_k or D_v
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # cross-attention
        x = self.attention(q, k, v)

        # transpose again: B x n_token x H x D_v
        # concat heads: B x n_token x (H * D_v)
        x = x.transpose(1, 2).contiguous().view(B, n_token, -1)
        # combine heads
        x = self.dropout(self.fc(x))
        # residual connection + layernorm
        x += self.q
        x = self.layer_norm(x)

        return x

class MLP(nn.Module):
    ''' MLP consisting of two feed-forward layers '''

    def __init__(self, D, D_inner, dropout=0.1):
        super().__init__()
        
        self.w_1 = nn.Linear(D, D_inner)
        self.w_2 = nn.Linear(D_inner, D)
        self.layer_norm = nn.LayerNorm(D, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(torch.relu(self.w_1(x)))
        x = self.dropout(x)
        
        x += residual
        x = self.layer_norm(x)

        return x

class Transformer(nn.Module):
    """ Cross-attention based transformer module """

    def __init__(self, n_token, H, D, D_k, D_v, D_inner, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        
        self.crs_attn = MultiHeadCrossAttention(n_token, H, D, D_k, D_v, attn_dropout=attn_dropout, dropout=dropout)
        self.mlp = MLP(D, D_inner, dropout=dropout)
    
    def get_scores(self, x):

        attn = self.crs_attn.get_attn(x)
        # Average scores over heads and tasks
        # Average over tasks is only required for multi-task learning (mnist).
        return attn.mean(dim=1).transpose(1, 2).mean(-1)

    def forward(self, x):

        return self.mlp(self.crs_attn(x))

--- Cell 8 ---
import sys
import math

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights



class IPSNet(nn.Module):
    """
    Net that runs all the main components:
    patch encoder, IPS, patch aggregator and classification head
    """

    def get_conv_patch_enc(self, enc_type, pretrained, n_chan_in, n_res_blocks):
        # Get architecture for patch encoder
        if enc_type == 'resnet18': 
            res_net_fn = resnet18
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        elif enc_type == 'resnet50':
            res_net_fn = resnet50
            # Resnet50 pretrained weights not used in experiments
            weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None        

        res_net = res_net_fn(weights=weights)

        if n_chan_in == 1:
            # Standard resnet uses 3 input channels
            res_net.conv1 = nn.Conv2d(n_chan_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Compose patch encoder
        layer_ls = []
        layer_ls.extend([
            res_net.conv1,
            res_net.bn1,
            res_net.relu,
            res_net.maxpool,
            res_net.layer1,
            res_net.layer2
        ])

        if n_res_blocks == 4:
            layer_ls.extend([
                res_net.layer3,
                res_net.layer4
            ])
        
        layer_ls.append(res_net.avgpool)

        return nn.Sequential(*layer_ls)

    def get_projector(self, n_chan_in, D):
        return nn.Sequential(
            nn.LayerNorm(n_chan_in, eps=1e-05, elementwise_affine=False),
            nn.Linear(n_chan_in, D),
            nn.BatchNorm1d(D),
            nn.ReLU()
        )   

    def get_output_layers(self, tasks):
        """
        Create an output layer for each task according to task definition
        """

        D = self.D
        n_class = self.n_class

        output_layers = nn.ModuleDict()
        for task in tasks.values():
            if task['act_fn'] == 'softmax':
                act_fn = nn.Softmax(dim=-1)
            elif task['act_fn'] == 'sigmoid':
                act_fn = nn.Sigmoid()
            
            layers = [
                nn.Linear(D, n_class),
                act_fn
            ]
            output_layers[task['name']] = nn.Sequential(*layers)

        return output_layers

    def __init__(self, device, conf):
        super().__init__()

        self.device = device
        self.n_class = conf.n_class
        self.M = conf.M
        self.I = conf.I
        self.D = conf.D 
        self.use_pos = conf.use_pos
        self.tasks = conf.tasks
        self.shuffle = conf.shuffle
        self.shuffle_style = conf.shuffle_style
        self.is_image = conf.is_image

        if self.is_image:
            self.encoder = self.get_conv_patch_enc(conf.enc_type, conf.pretrained,
                conf.n_chan_in, conf.n_res_blocks)
        else:
            self.encoder = self.get_projector(conf.n_chan_in, self.D)

        # Define the multi-head cross-attention transformer
        self.transf = Transformer(conf.n_token, conf.H, conf.D, conf.D_k, conf.D_v,
            conf.D_inner, conf.attn_dropout, conf.dropout)

        # Optionally use standard 1d sinusoidal positional encoding
        if conf.use_pos:
            self.pos_enc = pos_enc_1d(conf.D, conf.N).unsqueeze(0).to(device)
        else:
            self.pos_enc = None
        
        # Define an output layer for each task
        self.output_layers = self.get_output_layers(conf.tasks)

    def do_shuffle(self, patches, pos_enc):
        """
        Shuffles patches and pos_enc so that patches that have an equivalent score
        are sampled uniformly
        """

        shuffle_style = self.shuffle_style
        if shuffle_style == 'batch':
            patches, shuffle_idx = shuffle_batch(patches)
            if torch.is_tensor(pos_enc):
                pos_enc, _ = shuffle_batch(pos_enc, shuffle_idx)
        elif shuffle_style == 'instance':
            patches, shuffle_idx = shuffle_instance(patches, 1)
            if torch.is_tensor(pos_enc):
                pos_enc, _ = shuffle_instance(pos_enc, 1, shuffle_idx)
        
        return patches, pos_enc

    def score_and_select(self, emb, emb_pos, M, idx):
        """ 
        Scores embeddings and selects the top-M embeddings
        """
        D = emb.shape[2]

        emb_to_score = emb_pos if torch.is_tensor(emb_pos) else emb

        # Obtain scores from transformer
        attn = self.transf.get_scores(emb_to_score) # (B, M+I)

        # Get indixes of top-scoring patches
        top_idx = torch.topk(attn, M, dim = -1)[1] # (B, M)
        
        # Update memory buffers
        # Note: Scoring is based on `emb_to_score`, selection is based on `emb`
        mem_emb = torch.gather(emb, 1, top_idx.unsqueeze(-1).expand(-1,-1,D))
        mem_idx = torch.gather(idx, 1, top_idx)

        return mem_emb, mem_idx

    def get_preds(self, embeddings):
        preds = {}
        for task in self.tasks.values():
            t_name, t_id = task['name'], task['id']
            layer = self.output_layers[t_name]

            emb = embeddings[:,t_id]
            preds[t_name] = layer(emb)            

        return preds

    # IPS runs in no-gradient mode
    @torch.no_grad()
    def ips(self, patches):
        """ Iterative Patch Selection """

        # Get useful variables
        M = self.M
        I = self.I
        D = self.D  
        device = self.device
        shuffle = self.shuffle
        use_pos = self.use_pos
        pos_enc = self.pos_enc
        patch_shape = patches.shape
        B, N = patch_shape[:2]

        # Shortcut: IPS not required when memory is larger than total number of patches
        if M >= N:
            # Batchify pos enc
            pos_enc = pos_enc.expand(B, -1, -1) if use_pos else None
            return patches.to(device), pos_enc 

        # IPS runs in evaluation mode
        if self.training:
            self.encoder.eval()
            self.transf.eval()

        # Batchify positional encoding
        if use_pos:
            pos_enc = pos_enc.expand(B, -1, -1)

        # Shuffle patches (i.e., randomize when patches obtain identical scores)
        if shuffle:
            patches, pos_enc = self.do_shuffle(patches, pos_enc)

        # Init memory buffer
        # Put patches onto GPU in case it is not there yet (lazy loading).
        # `to` will return self in case patches are located on GPU already (eager loading)
        init_patch = patches[:,:M].to(device) 
        
        ## Embed
        mem_emb = self.encoder(init_patch.reshape(-1, *patch_shape[2:]))
        mem_emb = mem_emb.view(B, M, -1)
        
        # Init memory indixes in order to select patches at the end of IPS.
        idx = torch.arange(N, dtype=torch.int64, device=device).unsqueeze(0).expand(B, -1)
        mem_idx = idx[:,:M]

        # Apply IPS for `n_iter` iterations
        n_iter = math.ceil((N - M) / I)
        for i in range(n_iter):
            # Get next patches
            start_idx = i * I + M
            end_idx = min(start_idx + I, N)

            iter_patch = patches[:, start_idx:end_idx].to(device)
            iter_idx = idx[:, start_idx:end_idx]

            # Embed
            iter_emb = self.encoder(iter_patch.reshape(-1, *patch_shape[2:]))
            iter_emb = iter_emb.view(B, -1, D)
            
            # Concatenate with memory buffer
            all_emb = torch.cat((mem_emb, iter_emb), dim=1)
            all_idx = torch.cat((mem_idx, iter_idx), dim=1)
            # When using positional encoding, also apply it during patch selection
            if use_pos:
                all_pos_enc = torch.gather(pos_enc, 1, all_idx.view(B, -1, 1).expand(-1, -1, D))
                all_emb_pos = all_emb + all_pos_enc
            else:
                all_emb_pos = None

            # Select Top-M patches according to cross-attention scores
            mem_emb, mem_idx = self.score_and_select(all_emb, all_emb_pos, M, all_idx)

        # Select patches
        n_dim_expand = len(patch_shape) - 2
        mem_patch = torch.gather(patches, 1, 
            mem_idx.view(B, -1, *(1,)*n_dim_expand).expand(-1, -1, *patch_shape[2:]).to(patches.device)
        ).to(device)

        if use_pos:
            mem_pos = torch.gather(pos_enc, 1, mem_idx.unsqueeze(-1).expand(-1, -1, D))
        else:
            mem_pos = None

        # Set components back to training mode
        # Although components of `self` that are relevant for IPS have been set to eval mode,
        # self is still in training mode at training time, i.e., we can use it here.
        if self.training:
            self.encoder.train()
            self.transf.train()
    
        # Return selected patch and corresponding positional embeddings
        return mem_patch, mem_pos

    def forward(self, mem_patch, mem_pos=None):
        """
        After M patches have been selected during IPS, encode and aggregate them.
        The aggregated embedding is input to a classification head.
        """

        patch_shape = mem_patch.shape
        B, M = patch_shape[:2]

        mem_emb = self.encoder(mem_patch.reshape(-1, *patch_shape[2:]))
        mem_emb = mem_emb.view(B, M, -1)        

        if torch.is_tensor(mem_pos):
            mem_emb = mem_emb + mem_pos

        image_emb = self.transf(mem_emb)

        preds = self.get_preds(image_emb)
        
        return preds

--- Cell 11 ---
# data/megapixel_mnist/make_mnist.py
# (preprocessing script — not run during training, only needed to generate the dataset)
# Wrapped to avoid shadowing MegapixelMNIST loader defined above

if True:  # set to True and run standalone to generate the dataset
    # Adapted from https://github.com/idiap/attention-sampling


    
    import os
    import argparse
    import json
    
    import numpy as np
    from keras.datasets import mnist
    
    class MegapixelMNIST:
        """
        Class to create an artificial megapixel mnist dataset
        """
    
        class Sample(object):
            def __init__(self, dataset, idxs, positions, noise_positions,
                         noise_patterns):
                self._dataset = dataset
                self._idxs = idxs
                self._positions = positions
                self._noise_positions = noise_positions
                self._noise_patterns = noise_patterns
    
                self._img = None
    
            @property
            def noise_positions_and_patterns(self):
                return zip(self._noise_positions, self._noise_patterns)
    
            def _get_slice(self, pos, s=28, scale=1, offset=(0, 0)):
                pos = (int(pos[0]*scale-offset[0]), int(pos[1]*scale-offset[1]))
                s = int(s)
                return (
                    slice(max(0, pos[0]), max(0, pos[0]+s)),
                    slice(max(0, pos[1]), max(0, pos[1]+s)),
                    0
                )
    
            def create_img(self):
                if self._img is None:
                    size = self._dataset._H, self._dataset._W
                    img = np.zeros(size + (1,), dtype=np.uint8)
                    
                    if self._dataset._should_add_noise:
                        for p, i in self.noise_positions_and_patterns:
                            img[self._get_slice(p)] = \
                                255*self._dataset._noise[i]
                    
                    for p, i in zip(self._positions, self._idxs):
                        img[self._get_slice(p)] = \
                            255*self._dataset._images[i]
                    
                    self._img = img
                return self._img
    
        def __init__(self, N=5000, W=1500, H=1500, train=True,
                     noise=True, n_noise=50, seed=0):
            # Load the images
            x, y = mnist.load_data()[0 if train else 1]
            x = x.astype(np.float32) / 255.
    
            self._W, self._H = W, H
            self._images = x
    
            # Generate the dataset
            try:
                random_state = np.random.get_state()
                np.random.seed(seed + int(train))
                self._nums, self._targets, self._digits, self._max_targets = self._get_numbers(N, y)
                self._pos = self._get_positions(N, W, H)
                self._top_targets = self._get_top_targets()
    
                self._noise, self._noise_positions, self._noise_patterns = \
                    self._create_noise(N, W, H, n_noise)
            finally:
                np.random.set_state(random_state)
    
            # Boolean whether to add noise
            self._should_add_noise = noise
    
        def _create_noise(self, N, W, H, n_noise):
            """
            Create some random scribble noise of straight lines
            """
            angles = np.tan(np.random.rand(n_noise)*np.pi/2.5)
            A = np.zeros((n_noise, 28, 28))
            for i in range(n_noise):
                m = min(27.49, 27.49/angles[i])
                x = np.linspace(0, m, 56)
                y = angles[i]*x
                A[i, np.round(x).astype(int), np.round(y).astype(int)] = 1.
            B = np.array(A)
            np.random.shuffle(B)
            flip_x = np.random.rand(n_noise) < 0.33
            flip_y = np.random.rand(n_noise) < 0.33
            B[flip_x] = np.flip(B[flip_x], 2)
            B[flip_y] = np.flip(B[flip_y], 2)
            noise = ((A + B) > 0).astype(float)
            noise *= np.random.rand(n_noise, 28, 28)*0.2 + 0.8
            noise = noise.astype(np.float32)
    
            # Randomly assign noise to all images
            positions = (np.random.rand(N, n_noise, 2)*[H-56, W-56] + 28).astype(int)
            patterns = (np.random.rand(N, n_noise)*n_noise).astype(int)
    
            return noise, positions, patterns
    
        def _get_numbers(self, N, y):
            """
            Method to get numbers from the dataset
    
            Parameters:
            N (int): Number of samples (megapixel images) to create
            y (numpy array): Labels of standard mnist images
    
            Returns:
            numpy array: Array of indexes of the selected samples
            numpy array: Array of targets for the selected samples (task majority)
            numpy array: Array of digits for the selected samples (task multilabel)
            numpy array: Array of maximum digits for the selected samples (task max)
            """
            # Initialize empty lists for the output arrays
            nums = []
            targets = []
            max_targets = []
            all_digits = []
            # Get all indexes of the dataset
            all_idxs = np.arange(len(y))
            # Loop over the required number of samples
            for _ in range(N):
                # Get a random digit
                target = int(np.random.rand()*10)
                # Get three indexes where the target digit is present
                positive_idxs = np.random.choice(all_idxs[y == target], 3)
                # Get two negative indexes where the target digit is not present
                neg_idxs = np.random.choice(all_idxs[y != target], 2)
    
                # Concatenate the positive and negative indexes
                pos_neg_idxs = np.concatenate([positive_idxs, neg_idxs])
                # Get the digits from the concatenated indexes
                digits = y[pos_neg_idxs]
                # Get the maximum digit from the digits array
                max_target = np.max(digits)
    
                # Append the outputs to their respective lists
                nums.append(pos_neg_idxs)
                targets.append(target)
                all_digits.append(digits)
                max_targets.append(max_target)
    
            # Convert the lists to numpy arrays and return
            return np.array(nums), np.array(targets), np.array(all_digits), np.array(max_targets)
    
        def _get_positions(self, N, W, H):
            """
            Generates random positions of 5 digits in an image
            with size (H, W)
    
            Arguments:
                N: number of images to generate positions for
                W: width of the image
                H: height of the image
    
            Returns:
                np.array with shape (N, 5, 2) containing positions of 5 digits
            """
            def overlap(positions, pos):
                """
                Check if the new position 'pos' overlaps with
                any of the existing positions in 'positions'
                """
                if len(positions) == 0:
                    return False
                distances = np.abs(
                    np.asarray(positions) - np.asarray(pos)[np.newaxis]
                )
                axis_overlap = distances < 28
                return np.logical_and(axis_overlap[:, 0], axis_overlap[:, 1]).any()
    
            positions = []
            for _ in range(N):
                position = []
                for _ in range(5):
                    while True:
                        pos = np.round(np.random.rand(2)*[H-28, W-28]).astype(int)
                        if not overlap(position, pos):
                            break
                    position.append(pos)
                positions.append(position)
    
            return np.array(positions)
        
        def _get_top_targets(self):
            """
            Get digit that is topmost in each image
            """
            # `pos` is a numpy array with shape (n_img, digits, height and width)
            pos_height = self._pos[:,:,0] 
            
            # Get the index of the digit with the minimum height, i.e. top-most
            top_pos_idx = np.argmin(pos_height, axis=-1)
            
            # Get the digit with the minimum height for each image
            N = self._digits.shape[0]
            top_targets = self._digits[np.arange(N), top_pos_idx]
    
            return top_targets
    
        def __len__(self):
            return len(self._nums)
    
        def __getitem__(self, i):
            if len(self) <= i:
                raise IndexError()
            # Create a new sample
            sample = self.Sample(
                self,
                self._nums[i],
                self._pos[i],
                self._noise_positions[i],
                self._noise_patterns[i]
            )
            x = sample.create_img().astype(np.float32) / 255
            # Obtain labels for all tasks
            y = self._targets[i]
            y_max = self._max_targets[i]
            y_top = self._top_targets[i]
            y_multi = np.eye(10)[self._digits[i]].sum(0).clip(0,1)
    
            return x, y, y_max, y_top, y_multi
    
    
    def sparsify(dataset):
        """
        Store non-zero values and their indixes only to save memory
        """
        def to_sparse(x):
            x = x.ravel()
            indices = np.where(x != 0)
            values = x[indices]
            return (indices, values)
    
        print("Sparsifying dataset")
        data = []
        for i, (x, y_maj, y_max, y_top, y_multi) in enumerate(dataset):
            print(
                "\u001b[1000DProcessing {:5d} /  {:5d}".format(i+1, len(dataset)),
                end="",
                flush=True
            )
    
            data.append({
                'input': to_sparse(x),
                'majority': y_maj,
                'max': y_max,
                'top': y_top,
                'multi': y_multi
            })
        print()
        return data
    
    def main(argv):
        parser = argparse.ArgumentParser(
            description="Create the Megapixel MNIST dataset"
        )
        parser.add_argument(
            "--n_train",
            type=int,
            default=5000,
            help="How many images to create for training set"
        )
        parser.add_argument(
            "--n_test",
            type=int,
            default=1000,
            help="How many images to create for test set"
        )
        parser.add_argument(
            "--width",
            type=int,
            default=1500,
            help="Set the width for the image"
        )
        parser.add_argument(
            "--height",
            type=int,
            default=1500,
            help="Set the height for the image"
        )
        parser.add_argument(
            "--no_noise",
            action="store_false",
            dest="noise",
            help="Do not use noise in the dataset"
        )
        parser.add_argument(
            "--n_noise",
            type=int,
            default=50,
            help="Set the number of noise patterns per image"
        )
        parser.add_argument(
            "--dataset_seed",
            type=int,
            default=0,
            help="Choose the random seed for the dataset"
        )
        parser.add_argument(
            "output_directory",
            help="The directory to save the dataset into"
        )
    
        args = parser.parse_args(argv)
    
        if not os.path.exists(args.output_directory):
            os.makedirs(args.output_directory)
    
        with open(os.path.join(args.output_directory, "parameters.json"), "w") as f:
            json.dump(
                {
                    "n_train": args.n_train,
                    "n_test": args.n_test,
                    "width": args.width,
                    "height": args.height,
                    "noise": args.noise,
                    "n_noise": args.n_noise,
                    "seed": args.dataset_seed
                },
                f,
                indent=4
            )
        
        # Write the training set
        training = MegapixelMNIST(
            N=args.n_train,
            train=True,
            W=args.width,
            H=args.height,
            noise=args.noise,
            n_noise=args.n_noise,
            seed=args.dataset_seed
        )
        data = sparsify(training)
        np.save(os.path.join(args.output_directory, "train.npy"), data)
    
        # Write the test set
        test = MegapixelMNIST(
            N=args.n_test,
            train=False,
            W=args.width,
            H=args.height,
            noise=args.noise,
            n_noise=args.n_noise,
            seed=args.dataset_seed
        )
        data = sparsify(test)
        np.save(os.path.join(args.output_directory, "test.npy"), data)
    
    # Usage example: python make_mnist.py --width 1500 --height 1500 dsets/megapixel_mnist_1500
    if __name__ == "__main__":
        main(None)

    pass

--- Cell 12 ---
# Adapted from https://github.com/idiap/attention-sampling



import os
import argparse
import json

import numpy as np
from keras.datasets import mnist

class MegapixelMNIST:
    """
    Class to create an artificial megapixel mnist dataset
    """

    class Sample(object):
        def __init__(self, dataset, idxs, positions, noise_positions,
                     noise_patterns):
            self._dataset = dataset
            self._idxs = idxs
            self._positions = positions
            self._noise_positions = noise_positions
            self._noise_patterns = noise_patterns

            self._img = None

        @property
        def noise_positions_and_patterns(self):
            return zip(self._noise_positions, self._noise_patterns)

        def _get_slice(self, pos, s=28, scale=1, offset=(0, 0)):
            pos = (int(pos[0]*scale-offset[0]), int(pos[1]*scale-offset[1]))
            s = int(s)
            return (
                slice(max(0, pos[0]), max(0, pos[0]+s)),
                slice(max(0, pos[1]), max(0, pos[1]+s)),
                0
            )

        def create_img(self):
            if self._img is None:
                size = self._dataset._H, self._dataset._W
                img = np.zeros(size + (1,), dtype=np.uint8)
                
                if self._dataset._should_add_noise:
                    for p, i in self.noise_positions_and_patterns:
                        img[self._get_slice(p)] = \
                            255*self._dataset._noise[i]
                
                for p, i in zip(self._positions, self._idxs):
                    img[self._get_slice(p)] = \
                        255*self._dataset._images[i]
                
                self._img = img
            return self._img

    def __init__(self, N=5000, W=1500, H=1500, train=True,
                 noise=True, n_noise=50, seed=0):
        # Load the images
        x, y = mnist.load_data()[0 if train else 1]
        x = x.astype(np.float32) / 255.

        self._W, self._H = W, H
        self._images = x

        # Generate the dataset
        try:
            random_state = np.random.get_state()
            np.random.seed(seed + int(train))
            self._nums, self._targets, self._digits, self._max_targets = self._get_numbers(N, y)
            self._pos = self._get_positions(N, W, H)
            self._top_targets = self._get_top_targets()

            self._noise, self._noise_positions, self._noise_patterns = \
                self._create_noise(N, W, H, n_noise)
        finally:
            np.random.set_state(random_state)

        # Boolean whether to add noise
        self._should_add_noise = noise

    def _create_noise(self, N, W, H, n_noise):
        """
        Create some random scribble noise of straight lines
        """
        angles = np.tan(np.random.rand(n_noise)*np.pi/2.5)
        A = np.zeros((n_noise, 28, 28))
        for i in range(n_noise):
            m = min(27.49, 27.49/angles[i])
            x = np.linspace(0, m, 56)
            y = angles[i]*x
            A[i, np.round(x).astype(int), np.round(y).astype(int)] = 1.
        B = np.array(A)
        np.random.shuffle(B)
        flip_x = np.random.rand(n_noise) < 0.33
        flip_y = np.random.rand(n_noise) < 0.33
        B[flip_x] = np.flip(B[flip_x], 2)
        B[flip_y] = np.flip(B[flip_y], 2)
        noise = ((A + B) > 0).astype(float)
        noise *= np.random.rand(n_noise, 28, 28)*0.2 + 0.8
        noise = noise.astype(np.float32)

        # Randomly assign noise to all images
        positions = (np.random.rand(N, n_noise, 2)*[H-56, W-56] + 28).astype(int)
        patterns = (np.random.rand(N, n_noise)*n_noise).astype(int)

        return noise, positions, patterns

    def _get_numbers(self, N, y):
        """
        Method to get numbers from the dataset

        Parameters:
        N (int): Number of samples (megapixel images) to create
        y (numpy array): Labels of standard mnist images

        Returns:
        numpy array: Array of indexes of the selected samples
        numpy array: Array of targets for the selected samples (task majority)
        numpy array: Array of digits for the selected samples (task multilabel)
        numpy array: Array of maximum digits for the selected samples (task max)
        """
        # Initialize empty lists for the output arrays
        nums = []
        targets = []
        max_targets = []
        all_digits = []
        # Get all indexes of the dataset
        all_idxs = np.arange(len(y))
        # Loop over the required number of samples
        for _ in range(N):
            # Get a random digit
            target = int(np.random.rand()*10)
            # Get three indexes where the target digit is present
            positive_idxs = np.random.choice(all_idxs[y == target], 3)
            # Get two negative indexes where the target digit is not present
            neg_idxs = np.random.choice(all_idxs[y != target], 2)

            # Concatenate the positive and negative indexes
            pos_neg_idxs = np.concatenate([positive_idxs, neg_idxs])
            # Get the digits from the concatenated indexes
            digits = y[pos_neg_idxs]
            # Get the maximum digit from the digits array
            max_target = np.max(digits)

            # Append the outputs to their respective lists
            nums.append(pos_neg_idxs)
            targets.append(target)
            all_digits.append(digits)
            max_targets.append(max_target)

        # Convert the lists to numpy arrays and return
        return np.array(nums), np.array(targets), np.array(all_digits), np.array(max_targets)

    def _get_positions(self, N, W, H):
        """
        Generates random positions of 5 digits in an image
        with size (H, W)

        Arguments:
            N: number of images to generate positions for
            W: width of the image
            H: height of the image

        Returns:
            np.array with shape (N, 5, 2) containing positions of 5 digits
        """
        def overlap(positions, pos):
            """
            Check if the new position 'pos' overlaps with
            any of the existing positions in 'positions'
            """
            if len(positions) == 0:
                return False
            distances = np.abs(
                np.asarray(positions) - np.asarray(pos)[np.newaxis]
            )
            axis_overlap = distances < 28
            return np.logical_and(axis_overlap[:, 0], axis_overlap[:, 1]).any()

        positions = []
        for _ in range(N):
            position = []
            for _ in range(5):
                while True:
                    pos = np.round(np.random.rand(2)*[H-28, W-28]).astype(int)
                    if not overlap(position, pos):
                        break
                position.append(pos)
            positions.append(position)

        return np.array(positions)
    
    def _get_top_targets(self):
        """
        Get digit that is topmost in each image
        """
        # `pos` is a numpy array with shape (n_img, digits, height and width)
        pos_height = self._pos[:,:,0] 
        
        # Get the index of the digit with the minimum height, i.e. top-most
        top_pos_idx = np.argmin(pos_height, axis=-1)
        
        # Get the digit with the minimum height for each image
        N = self._digits.shape[0]
        top_targets = self._digits[np.arange(N), top_pos_idx]

        return top_targets

    def __len__(self):
        return len(self._nums)

    def __getitem__(self, i):
        if len(self) <= i:
            raise IndexError()
        # Create a new sample
        sample = self.Sample(
            self,
            self._nums[i],
            self._pos[i],
            self._noise_positions[i],
            self._noise_patterns[i]
        )
        x = sample.create_img().astype(np.float32) / 255
        # Obtain labels for all tasks
        y = self._targets[i]
        y_max = self._max_targets[i]
        y_top = self._top_targets[i]
        y_multi = np.eye(10)[self._digits[i]].sum(0).clip(0,1)

        return x, y, y_max, y_top, y_multi


def sparsify(dataset):
    """
    Store non-zero values and their indixes only to save memory
    """
    def to_sparse(x):
        x = x.ravel()
        indices = np.where(x != 0)
        values = x[indices]
        return (indices, values)

    print("Sparsifying dataset")
    data = []
    for i, (x, y_maj, y_max, y_top, y_multi) in enumerate(dataset):
        print(
            "\u001b[1000DProcessing {:5d} /  {:5d}".format(i+1, len(dataset)),
            end="",
            flush=True
        )

        data.append({
            'input': to_sparse(x),
            'majority': y_maj,
            'max': y_max,
            'top': y_top,
            'multi': y_multi
        })
    print()
    return data

def main(argv):
    parser = argparse.ArgumentParser(
        description="Create the Megapixel MNIST dataset"
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=5000,
        help="How many images to create for training set"
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=1000,
        help="How many images to create for test set"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1500,
        help="Set the width for the image"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1500,
        help="Set the height for the image"
    )
    parser.add_argument(
        "--no_noise",
        action="store_false",
        dest="noise",
        help="Do not use noise in the dataset"
    )
    parser.add_argument(
        "--n_noise",
        type=int,
        default=50,
        help="Set the number of noise patterns per image"
    )
    parser.add_argument(
        "--dataset_seed",
        type=int,
        default=0,
        help="Choose the random seed for the dataset"
    )
    parser.add_argument(
        "output_directory",
        help="The directory to save the dataset into"
    )

    args = parser.parse_args(argv)

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    with open(os.path.join(args.output_directory, "parameters.json"), "w") as f:
        json.dump(
            {
                "n_train": args.n_train,
                "n_test": args.n_test,
                "width": args.width,
                "height": args.height,
                "noise": args.noise,
                "n_noise": args.n_noise,
                "seed": args.dataset_seed
            },
            f,
            indent=4
        )
    
    # Write the training set
    training = MegapixelMNIST(
        N=args.n_train,
        train=True,
        W=args.width,
        H=args.height,
        noise=args.noise,
        n_noise=args.n_noise,
        seed=args.dataset_seed
    )
    data = sparsify(training)
    np.save(os.path.join(args.output_directory, "train.npy"), data)

    # Write the test set
    test = MegapixelMNIST(
        N=args.n_test,
        train=False,
        W=args.width,
        H=args.height,
        noise=args.noise,
        n_noise=args.n_noise,
        seed=args.dataset_seed
    )
    data = sparsify(test)
    np.save(os.path.join(args.output_directory, "test.npy"), data)

# Usage example: python make_mnist.py --width 1500 --height 1500 dsets/megapixel_mnist_1500
if __name__ == "__main__":
    main(None)


--- Cell 14 ---
import os
import json
import numpy as np
import torch

class MegapixelMNIST(torch.utils.data.Dataset):
    """ Loads the Megapixel MNIST dataset """

    def __init__(self, conf, train=True):
        with open(os.path.join(conf.data_dir, "parameters.json")) as f:
            self.parameters = json.load(f)

        self.patch_size = conf.patch_size
        self.patch_stride = conf.patch_stride
        self.tasks = conf.tasks

        filename = "train.npy" if train else "test.npy"
        W = self.parameters["width"]
        H = self.parameters["height"]

        self._img_shape = (H, W, 1)
        self._data = np.load(os.path.join(conf.data_dir, filename), allow_pickle=True)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError()

        patch_size = self.patch_size
        patch_stride = self.patch_stride

        # Placeholders
        img = np.zeros(self._img_shape, dtype=np.float32).ravel()

        # Fill the sparse representations
        data = self._data[i]
        img[data['input'][0]] = data['input'][1]

        # Reshape to final shape        
        img = img.reshape(self._img_shape)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)

        # Extract patches
        patches = img.unfold(
            1, patch_size[0], patch_stride[0]
        ).unfold(
            2, patch_size[1], patch_stride[1]
        ).permute(1, 2, 0, 3, 4)
        
        patches = patches.reshape(-1, *patches.shape[2:])

        data_dict = {'input': patches}
        for task in self.tasks.values():
            data_dict[task['name']] = data[task['name']] 

        return data_dict

--- Cell 16 ---
import os
from os import path
import sys
import hashlib
from functools import partial
from collections import namedtuple
import urllib.request
import zipfile
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

### Adapted from: https://github.com/sara-nl/attention-sampling-pytorch

def check_file(filepath, md5sum):
    """Check a file against an md5 hash value.
    Returns
    -------
        True if the file exists and has the given md5 sum False otherwise
    """

    try:
        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(partial(f.read, 4096), b""):
                md5.update(chunk)
        return md5.hexdigest() == md5sum
    except FileNotFoundError:
        return False

def ensure_dataset_exists(directory, tries=1, progress_file=sys.stderr):
    """Ensure that the dataset is downloaded and is correct.
    Correctness is checked only against the annotations files.
    """

    set1_url = ("http://www.isy.liu.se/cvl/research/trafficSigns"
                "/swedishSignsSummer/Set1/Set1Part0.zip")
    set1_annotations_url = ("http://www.isy.liu.se/cvl/research/trafficSigns"
                            "/swedishSignsSummer/Set1/annotations.txt")
    set1_annotations_md5 = "9106a905a86209c95dc9b51d12f520d6"
    set2_url = ("http://www.isy.liu.se/cvl/research/trafficSigns"
                "/swedishSignsSummer/Set2/Set2Part0.zip")
    set2_annotations_url = ("http://www.isy.liu.se/cvl/research/trafficSigns"
                            "/swedishSignsSummer/Set2/annotations.txt")
    set2_annotations_md5 = "09debbc67f6cd89c1e2a2688ad1d03ca"

    integrity = (
        check_file(
            path.join(directory, "Set1", "annotations.txt"),
            set1_annotations_md5
        ) and check_file(
            path.join(directory, "Set2", "annotations.txt"),
            set2_annotations_md5
        )
    )

    if integrity:
        return

    if tries <= 0:
        raise RuntimeError(("Cannot download dataset or dataset download "
                            "is corrupted"))

    print("Downloading Set1", file=progress_file)
    download_file(set1_url, path.join(directory, "Set1.zip"),
                  progress_file=progress_file)
    print("Extracting...", file=progress_file)
    with zipfile.ZipFile(path.join(directory, "Set1.zip")) as archive:
        archive.extractall(path.join(directory, "Set1"))
    print("Getting annotation file", file=progress_file)
    download_file(
        set1_annotations_url,
        path.join(directory, "Set1", "annotations.txt"),
        progress_file=progress_file
    )
    print("Downloading Set2", file=progress_file)
    download_file(set2_url, path.join(directory, "Set2.zip"),
                  progress_file=progress_file)
    print("Extracting...", file=progress_file)
    with zipfile.ZipFile(path.join(directory, "Set2.zip")) as archive:
        archive.extractall(path.join(directory, "Set2"))
    print("Getting annotation file", file=progress_file)
    download_file(
        set2_annotations_url,
        path.join(directory, "Set2", "annotations.txt"),
        progress_file=progress_file
    )

    return ensure_dataset_exists(
        directory,
        tries=tries - 1,
        progress_file=progress_file
    )

def download_file(url, destination, progress_file=sys.stderr):
    """Download a file with progress."""
    
    response = urllib.request.urlopen(url)
    n_bytes = response.headers.get("Content-Length")
    if n_bytes == "":
        n_bytes = 0
    else:
        n_bytes = int(n_bytes)

    message = "\rReceived {} / {}"
    cnt = 0
    with open(destination, "wb") as dst:
        while True:
            print(message.format(cnt, n_bytes), file=progress_file,
                  end="", flush=True)
            data = response.read(65535)
            if len(data) == 0:
                break
            dst.write(data)
            cnt += len(data)
    print(file=progress_file)

class Sign(namedtuple("Sign", ["visibility", "bbox", "type", "name"])):
    """A sign object. Useful for making ground truth images as well as making
    the dataset."""

    @property
    def x_min(self):
        
        return self.bbox[2]

    @property
    def x_max(self):
        
        return self.bbox[0]

    @property
    def y_min(self):
        
        return self.bbox[3]

    @property
    def y_max(self):
        
        return self.bbox[1]

    @property
    def area(self):

        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    @property
    def center(self):

        return [
            (self.y_max - self.y_min) / 2 + self.y_min,
            (self.x_max - self.x_min) / 2 + self.x_min
        ]

    @property
    def visibility_index(self):

        visibilities = ["VISIBLE", "BLURRED", "SIDE_ROAD", "OCCLUDED"]
        return visibilities.index(self.visibility)

    def pixels(self, scale, size):

        return zip(*(
            (i, j)
            for i in range(round(self.y_min * scale), round(self.y_max * scale) + 1)
            for j in range(round(self.x_min * scale), round(self.x_max * scale) + 1)
            if i < round(size[0] * scale) and j < round(size[1] * scale)
        ))

    def __lt__(self, other):

        if not isinstance(other, Sign):
            raise ValueError("Signs can only be compared to signs")

        if self.visibility_index != other.visibility_index:
            return self.visibility_index < other.visibility_index

        return self.area > other.area


class STS:
    """The STS class reads the annotations and creates the corresponding
    Sign objects."""

    def __init__(self, directory, train=True, seed=0):

        cwd = os.getcwd().replace('dataset', '')
        directory = path.join(cwd, directory)
        ensure_dataset_exists(directory)

        self._directory = directory
        self._inner = "Set{}".format(1 + ((seed + 1 + int(train)) % 2))
        self._data = self._load_signs(self._directory, self._inner)

    def _load_files(self, directory, inner):

        files = set()
        with open(path.join(directory, inner, "annotations.txt")) as f:
            for l in f:
                files.add(l.split(":", 1)[0])

        return sorted(files)

    def _read_bbox(self, parts):

        def _float(x):

            try:
                return float(x)
            except ValueError:
                if len(x) > 0:
                    return _float(x[:-1])
                raise

        return [_float(x) for x in parts]

    def _load_signs(self, directory, inner):

        with open(path.join(directory, inner, "annotations.txt")) as f:
            lines = [l.strip() for l in f]
        keys, values = zip(*(l.split(":", 1) for l in lines))
        all_signs = []
        for v in values:
            signs = []
            for sign in v.split(";"):
                if sign == [""] or sign == "":
                    continue
                parts = [s.strip() for s in sign.split(",")]
                if parts[0] == "MISC_SIGNS":
                    continue
                signs.append(Sign(
                    visibility=parts[0],
                    bbox=self._read_bbox(parts[1:5]),
                    type=parts[5],
                    name=parts[6]
                ))
            all_signs.append(signs)
        images = [path.join(directory, inner, f) for f in keys]

        return list(zip(images, all_signs))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]

class TrafficSigns(Dataset):
    """ Loads images from the traffic signs dataset as
    a filtered version of the STS dataset.
    Arguments
    ---------
        directory: str, The directory that the dataset already is or is going
                   to be downloaded in
        train: bool, Select the training or testing sets
        seed: int, The prng seed for the dataset
    """

    LIMITS = ["50_SIGN", "70_SIGN", "80_SIGN"]
    CLASSES = ["EMPTY", *LIMITS]
    IMG_SIZE = (1200, 1600)

    def __init__(self, conf, train=True):

        self.patch_size = conf.patch_size
        self.patch_stride = conf.patch_stride
        self.tasks = conf.tasks
        
        self._data = self._filter(STS(conf.data_dir, train, conf.seed))
        
        transform_list = [
            transforms.Resize([*self.IMG_SIZE])
        ]

        if train:
            transform_list += [
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                transforms.RandomAffine(degrees=0, translate=(100 / self.IMG_SIZE[1], 100 / self.IMG_SIZE[0])),
            ]
        
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
           
        self.transform = transforms.Compose(transform_list)

    def _filter(self, data):

        filtered = []
        for image, signs in data:
            signs, acceptable = self._acceptable(signs)
            if acceptable:
                if not signs:
                    filtered.append((image, 0))
                else:
                    filtered.append((image, self.CLASSES.index(signs[0].name)))
        return filtered

    def _acceptable(self, signs):

        # Keep it as empty
        if not signs:
            return signs, True

        # Filter just the speed limits and sort them wrt visibility
        signs = sorted(s for s in signs if s.name in self.LIMITS)

        # No speed limit but many other signs
        if not signs:
            return None, False

        # Not visible sign so skip
        if signs[0].visibility != "VISIBLE":
            return None, False

        return signs, True

    def __len__(self):

        return len(self._data)

    def __getitem__(self, i):

        patch_size = self.patch_size
        patch_stride = self.patch_stride

        img, category = self._data[i]
        img = Image.open(img)
        img = self.transform(img)

        # Extract patches
        patches = img.unfold(
            1, patch_size[0], patch_stride[0]
        ).unfold(
            2, patch_size[1], patch_stride[1]
        ).permute(1, 2, 0, 3, 4)

        patches = patches.reshape(-1, *patches.shape[2:])

        data_dict = {'input': patches}
        for task in self.tasks.values():
            data_dict[task['name']] = category

        return data_dict

--- Cell 18 ---
import os
import csv
from collections import defaultdict, OrderedDict, namedtuple
from typing import Tuple, Sequence, Any

from PIL import Image
import openslide
import xml.etree.ElementTree as Xml

from data.camelyon.cam_utils import Point, get_relative_polygon, draw_polygon, find_files

_RawAnnotation = namedtuple('RawAnnotation', 'name type_ part_of_group color polygon')

class Annotation:
    """Annotation class to provide access to a tumor annotation.

    Annotations can be displayed as an image with the annotation polygon put over the
    annotated section.


    Attributes
    ----------
    slide : Slide
        Slide the annotation belongs to.

    name : str
        Name of the annotation.

    type_ : str
        The type of the annotation specified in the annotation file.

    part_of_group: str
        The group of the annotation specified in the annotation file.

    color : tuple of int or str
        Annotation color as specified in the annotation file.

    polygon : sequence of Point
        A sequence of points annotating the tumor area.
    """

    def __init__(self, slide: 'Slide', name: str, type_: str, part_of_group: str,
                 color: Any, polygon: Sequence[Point]):
        """

        Parameters
        ----------
        slide : Slide
            Slide the annotation belongs to.

        name : str
            Name of the annotation.

        type_ : str
            The type of the annotation specified in the annotation file.

        part_of_group: str
            The group of the annotation specified in the annotation file.

        color : tuple of int or str
            Annotation color as specified in the annotation file.

        polygon : Sequence of Point
            A sequence of points annotating the tumor area.


        See Also
        --------
        PIL.ImageColor
        """
        self.slide = slide
        self.name = name
        self.type = type_
        self.part_of_group = part_of_group
        self.color = color
        self.polygon = polygon

    def __repr__(self):
        return '{}({!r}, {!r}, {!r}, {!r}, {!r}, {!r})'.format(
            type(self).__name__,
            self.slide,
            self.name,
            self.type,
            self.part_of_group,
            self.color,
            self.polygon
        )

    def __str__(self):
        return '{}(slide={!r}, name={!r}, polygon size={!r})'.format(
            type(self).__name__,
            self.slide.name,
            self.name,
            len(self.polygon)
        )

    def get_boundaries(self, level, padding=0):
        """
        Return the annotation boundaries.

        Parameters
        ----------
        level : int
            Layer

        padding : int, optional
            Add additional pixels to the boundaries of the Annotation. (Default: 0)


        Returns
        -------
        origin : (int, int)
            Coordinates of the top left corner of the annotation on the specified layer.

        size : (int, int)
            Annotation width and height on the specified layer.

        """
        x = int(min([p.x for p in self.polygon]) - padding)
        y = int(min([p.y for p in self.polygon]) - padding)
        width = int(max([p.x for p in self.polygon]) - x + padding)
        height = int(max([p.y for p in self.polygon]) - y + padding)

        downsample = self.slide.level_downsamples[level]

        origin = Point(x, y)
        size = (int(width / downsample), int(height / downsample))

        return origin, size

    def get_image(self, *, level=4, padding=100, fill=(50, 50, 50, 80)) -> Image.Image:
        """
        Create an image of the annotated tissue section overlayed with the annotation polygon.

        The polygon's outline `color` will be set to the color attribute of the
        `Annotation` itself. The `fill` color can be specified via the parameter `fill`.

        Parameters
        ----------
        level : int, optional
            Slide level/layer used to create the image.

        padding : int, optional
            Padding added to either side of the image in pixel. Padding is added on layer
            0 and will be downsacled if a `level` higher than 0 is passed.

        fill : tuple of int or str, optional
            Annotation color used to fill the polygon.
            (Default: (50, 50, 50, 80), a dark gray).

        Returns
        -------
        Image.Image
            Image picturing the annotated section from the slide with annotation overlay.

        See Also
        --------
        PIL.ImageColor
        """
        origin, image_size = self.get_boundaries(level, padding)
        downsample = self.slide.level_downsamples[level]

        return draw_polygon(self.slide.read_region(origin, level, image_size),
                            get_relative_polygon(self.polygon, origin,
                                                 downsample),
                            fill=fill,
                            outline=self.color)

def _get_raw_annotations(filename):
    """
    Read all annotation data from an ASAP XML file.

    Parameters
    ----------
    filename : str
        File name of the annotation XML-File.

    Returns
    -------
    Tuple[_RawAnnotation]
        Parsed annotation form XML-File.
    """
    #logger.debug('Reading annotation data from {}', filename)
    tree = Xml.parse(filename)
    root = tree.getroot()
    annotations = []

    for annotation in root.iter('Annotation'):
        # all annotation points sorted by the `Order` attribute
        polygon = (Point(float(c.attrib['X']), float(c.attrib['Y'])) for c in
                   sorted(annotation.iter('Coordinate'),
                          key=lambda x: int(x.attrib['Order'])))

        annotations.append(_RawAnnotation(
            annotation.attrib['Name'].replace(' ', ''),
            annotation.attrib['Type'],
            annotation.attrib['PartOfGroup'],
            annotation.attrib['Color'],
            tuple(polygon)
        ))

    return tuple(annotations)

class Slide(openslide.OpenSlide):
    """
    Wrapper class for openslide.OpenSlide.

    In addition to the OpenSlide itself this class holds information like name and
    possible annotations and stage of the slide itself.

    Attributes
    ----------
    name : str
        Name of the slide.

    stage : str or None
        pN-stage of the slide (None for CAMELYON16 slides).

    has_tumor : bool
        True if the slide has annotations or a non negative pN-stage.

    is_annotated : bool
        True if the slide has annotation.

    See Also
    --------
    openslide.OpenSlide
    """

    def __init__(self, name, filename, annotation_filename=None, stage=None,
                 otsu_thresholds=None):
        """
        Parameters
        ----------
        name : str
            Slide name. Usually the filename without extension.

        filename : str
            Relative or absolute path to slide file.

        annotation_filename : str or None, optional
            Relative or absolute path to an annotation XML file. (Default: None)

        stage : str or None, optional
            nP-stage for CAMELYON17 slides. Leave `None` for CAMELYON16 slides.
            (Default: None)

        otsu_thresholds : dict of float or None, optional
            Dictionary with otsu thresholds for each level. (Default: None)
            Dictionary does not have to be exhaustive e.g.: {0: 6.333, 5: 7.0}
        """
        super().__init__(filename)
        self.name = name
        self._filename = filename
        self._annotation_filename = annotation_filename
        self.stage = stage
        self.is_annotated = self._annotation_filename is not None
        self.has_tumor = self.is_annotated or (
            self.stage is not None and self.stage != 'negative')
        self._otsu_thresholds = otsu_thresholds if otsu_thresholds is not None else {}
        self._annotations = None

    @property
    def annotations(self) -> Tuple[Annotation]:
        """
        Return a tuple of all annotations.

        Returns
        -------
        tuple of Annotation
            All annotations belonging to this instance of `Slide` as a tuple.
        """
        if self._annotations is None:
            if self.is_annotated:
                raw_annotations = _get_raw_annotations(self._annotation_filename)
                self._annotations = tuple(Annotation(self, *x) for x in raw_annotations)
            else:
                self._annotations = ()

        return self._annotations

    def get_full_slide(self, level) -> Image.Image:
        """
        Return the full image of a slide layer.

        Returns
        -------
        Image.Image
            Complete slide on layer `level`.
        """
        return self.read_region((0, 0), level, self.level_dimensions[level])

    def get_otsu_threshold(self, level):
        """
        Return pre-calculated otsu threshold of a layer.

        Parameters
        ----------
        level : int
            Slide layer

        Returns
        -------
        otsu_threshold: float or None
            Otsu threshold of layer `level` or None if not pre-calculated.
        """
        if level in self._otsu_thresholds:
            return self._otsu_thresholds[level]
        else:
            return None

    def __repr__(self):
        if self.is_annotated:
            repr_str = "{}({!r}, {!r}, {!r}, {!r})"
        else:
            repr_str = "{}({!r}, {!r}, {!r})"

        return repr_str.format(type(self).__name__,
                               self.name,
                               self._filename,
                               self.stage,
                               self._annotation_filename)

class SlideManager:
    """
    Provide access to slices from CAMELYON16.

    Attributes
    ----------
    negative_slides : tuple of Slide
        All slides that do not have annotations.

    annotated_slides : tuple of Slide
        All slides that have annotations.
    """

    def __init__(self, *, data_dir, otsu_fname):
        """
        Initialize the CAMELYON data set.

        Parameters
        ----------
        data_dir : str
            Path to the CAMELYON16 directory.
        """

        self._slides = OrderedDict()
        self.slide_paths = OrderedDict()
        self.annotation_paths = OrderedDict()
        self.stages = OrderedDict()
        self.negative_slides = tuple()
        self.annotated_slides = tuple()
        self.test_slides = tuple()

        self.num_positive_train = 0
        self.num_negative_train = 0

        data_dir = os.path.expanduser(data_dir)
        self._path = {
            'dir': data_dir,
            'negative': os.path.join(data_dir, 'training/normal'),
            'positive': os.path.join(data_dir, 'training/tumor'),
            'annotations': os.path.join(data_dir, 'training/lesion_annotations'),
            'test': os.path.join(data_dir, 'testing/images'),
            'test_annotations': os.path.join(data_dir, 'testing/lesion_annotations'),
            'otsu': os.path.join(data_dir, otsu_fname)
        }
        self.__load_data()

    def __load_data(self):
        """Load slides."""

        # Negative slides
        self.otsu_thresholds = defaultdict(dict)
        try:
            with open(self._path['otsu'], 'r') as f:
                reader = csv.DictReader(f)
                for line in reader:
                    self.otsu_thresholds[line['name']][int(line['level'])] = float(
                        line['threshold'])
        except FileNotFoundError:
            print('No pre-calculated otsu thresholds found.')

        slide_files = find_files('*.tif', self._path['negative'])
        for file_name, slide_path in sorted(slide_files.items()):
            slide_name, _, _ = file_name.partition('.')
            slide = Slide(slide_name, slide_path,
                          otsu_thresholds=self.otsu_thresholds[slide_name])

            if slide_name in self._slides:
                raise RuntimeError(f'Slide "{slide_name}" already exists! ({slide_path})')

            self._slides[slide_name] = slide
            self.slide_paths[slide_name] = slide_path
            self.negative_slides += (slide,)
            self.num_negative_train += 1

        # Positive (tumor) slides
        slide_files = find_files('*.tif', self._path['positive'])
        for file_name, slide_path in sorted(slide_files.items()):
            slide_name, _, _ = file_name.partition('.')
            annotation_path = os.path.join(self._path['annotations'],
                                           f'{slide_name}.xml')
            if not os.path.exists(annotation_path):
                raise FileNotFoundError(annotation_path)
            slide = Slide(slide_name, slide_path, otsu_thresholds=self.otsu_thresholds[slide_name],
                annotation_filename=annotation_path)

            if slide_name in self._slides:
                raise RuntimeError(f'Slide "{slide_name}" already exists! ({slide_path})')

            self._slides[slide_name] = slide
            self.slide_paths[slide_name] = slide_path
            self.annotation_paths[slide_name] = annotation_path
            self.annotated_slides += (slide,)
    
            self.num_positive_train += 1

        # test slides
        slide_files = find_files('*.tif', self._path['test'])
        for file_name, slide_path in sorted(slide_files.items()):
            slide_name, _, _ = file_name.partition('.')
            annotation_path = os.path.join(self._path['test_annotations'],
                                           f'{slide_name}.xml')
            if not os.path.exists(annotation_path):
                slide = Slide(slide_name, slide_path,
                          otsu_thresholds=self.otsu_thresholds[slide_name])
            else:
                slide = Slide(slide_name, slide_path, otsu_thresholds=self.otsu_thresholds[slide_name], 
                    annotation_filename=annotation_path)
                
                self.annotation_paths[slide_name] = annotation_path

            if slide_name in self._slides:
                raise RuntimeError(f'Slide "{slide_name}" already exists! ({slide_path})')

            self._slides[slide_name] = slide
            self.slide_paths[slide_name] = slide_path
            self.test_slides += (slide,)


    @property
    def slides(self) -> Tuple[Slide]:
        """
        Return all slides as tuple.

        Returns
        -------
        tuple of Slide
            All slides managed by the instance of `SlideManager`.
        """
        return tuple(self._slides.values())

    @property
    def slide_names(self) -> Tuple[str]:
        """
        Return slide names as tuple.

        Returns
        -------
        tuple of str
            Slide names of all slides managed by the instance of `SlideManager`.
        """
        return tuple(self._slides.keys())

    def get_slide_names_subset(self, train=True) -> Tuple[str]:
        """
        Return slide names as tuple.

        Returns
        -------
        tuple of str
            Slide names of all slides managed by the instance of `SlideManager`.
        """
        if train:
            names = tuple(name for name in self._slides.keys() if 'test' not in name)
        else:
            names = tuple(name for name in self._slides.keys() if 'test' in name)

        return names

    def get_slide(self, name) -> Slide:
        """
        Retrieve a slide by its name.

        Parameters
        ----------
        name : str
            Slide name.


        Returns
        -------
        Slide
            Slide-Object with the name passed.
        """
        return self._slides[name]

    def __repr__(self):
        return '{}(cam16_dir={!r}, cam17_dir={!r})'.format(type(self).__name__,
                                                           self._path['cam16']['dir'],
                                                           self._path['cam17']['dir'])

    def __str__(self):
        return 'SlideManager contains: {} Slides ({} annotated; {} negative)'.format(
            len(self.slides),
            len(self.annotated_slides),
            len(self.negative_slides))

--- Cell 20 ---
import math
import numpy as np
import datetime

from skimage.draw import polygon as ski_polygon
from skimage.measure import label as ski_label

from data.camelyon.datamodel import Slide
from data.camelyon.cam_utils import ProgressBar

def remove_alpha_channel(image: np.ndarray) -> np.ndarray:
    """
    Remove the alpha channel of an image.

    Parameters
    ----------
    image : np.ndarray
        RGBA image as numpy array with W×H×C dimensions.

    Returns
    -------
    np.ndarray
        RGB image as numpy array
    """
    if len(image.shape) == 3 and image.shape[2] == 4:
        return image[::, ::, 0:3:]
    else:
        return image

def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB color image to a custom gray scale for HE-stained WSI

    Parameters
    ----------
    rgb : np.ndarray
        Color image.

    Returns
    -------
    np.ndarray
        Gray scale image as float64 array.
    """
    gray = 1.0 * rgb[::, ::, 0] + rgb[::, ::, 2] - (
        (1.0 * rgb[::, ::, 0] + rgb[::, ::, 1] + rgb[::, ::, 2])
        / 1.5)
    gray[gray < 0] = 0
    gray[gray > 255] = 255
    return gray

def create_otsu_mask_by_threshold(image: np.ndarray, threshold) -> np.ndarray:
    """
    Create a binary mask separating fore and background based on the otsu threshold.

    Parameters
    ----------
    image : np.ndarray
        Gray scale image as array W×H dimensions.

    threshold : float
        Upper Otsu threshold value.

    Returns
    -------
    np.ndarray
        The generated binary masks has value 1 in foreground areas and 0s everywhere
        else (background)
    """
    otsu_mask = image > threshold
    otsu_mask2 = image > threshold * 0.25

    otsu_mask2_labeled = ski_label(otsu_mask2)
    for i in range(1, otsu_mask2_labeled.max()):
        if otsu_mask[otsu_mask2_labeled == i].sum() == 0:
            otsu_mask2_labeled[otsu_mask2_labeled == i] = 0
    otsu_mask3 = otsu_mask2_labeled
    otsu_mask3[otsu_mask3 > 0] = 1

    return otsu_mask3.astype(np.uint8)

def _otsu_by_hist(hist, bin_centers) -> float:
    """
    Return threshold value based on Otsu's method using an images histogram.

    Based on skimage's threshold_otsu method without histogram generation.

    Parameters
    ----------
    hist : np.ndarray
        Histogram of a gray scale input image.

    bin_centers: np.ndarray
        Centers of the histogram's bins.

    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.

    References
    ----------
    Wikipedia, http://en.wikipedia.org/wiki/Otsu's_Method

    See Also
    --------
    skimage.filters.threshold_otsu
    """
    hist = hist.astype(float)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold

def add_dict(left, right):
    """
    Merge two dictionaries by adding common items.

    Parameters
    ----------
    left: dict
        Left dictionary.

    right
        Right dictionary

    Returns
    -------
    dict
        Resulting dictionary
    """
    return {k: left.get(k, 0) + right.get(k, 0) for k in left.keys() | right.keys()}

def get_otsu_threshold(slide: Slide, level=0, step_size=1000) -> float:
    """
    Calculate the otsu threshold by reading in the slide in chunks.

    To avoid memory overflows the slide image will be loaded in by chunks of the size
    $slide width × `step_size`$. A histogram will be generated of these chunks that will
    be used to calculate the otsu threshold based on skimage's `threshold_otsu` function.

    Parameters
    ----------
    slide : Slide
        Whole slide image slide

    level : int
        Level/layer of the `slide` to be used. Use of level ≠ 0 is not advised, see notes.

    step_size : int
        Each chunk loaded will have the size $slide-width × `step_size`$ on the level 0
        slide. For higher levels the step will be downsampled accordingly (e.g.: with a
        `step_size` of 1000 and `level` of 1 and a downsample factor of 2 the actual size
        of each chunk is $level-1-slide width × 500$.

    Returns
    -------
    otsu_threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.
    """

    size = slide.level_dimensions[0]
    downsample = slide.level_downsamples[level]

    # dictionary with all unique values and counts of the whole slide
    slide_count_dict = {}
    for i, y in enumerate(range(0, size[1], step_size)):

        # check if next step exceeds the image height and adjust it if needed
        cur_step = step_size if size[1] - y > step_size else size[1] - y

        # read in the image and transform to gray scale
        start, cut_size = (0, y), (int(size[0] / downsample), int(cur_step / downsample))
        a_img_cut = np.asarray(slide.read_region(start, level, cut_size))
        a_img_cut = rgb2gray(a_img_cut)

        # get unique values and their count
        chunk_count_dict = dict(zip(*np.unique(a_img_cut, return_counts=True)))

        # add those values and count to the dictionary
        slide_count_dict = add_dict(slide_count_dict, chunk_count_dict)

    # transform dictionary back to a arrays and calculate otsu threshold
    unique_values, counts = tuple(np.asarray(x) for x in zip(*slide_count_dict.items()))
    threshold = _otsu_by_hist(counts, unique_values)

    return threshold

def create_tumor_mask(slide: Slide, level, bounds=None):
    """Create a tumor mask for a slide or slide section.

    If `bounds` is given the tumor mask of only the section of the slide will be
    calculated.


    Parameters
    ----------
    slide : Slide
        Tissue slide.

    level : int
        Slide layer.

    bounds : tuple, optional
        Boundaries of a section as: ((x, y), (width, height))
        Where x and y are coordinates of the top left corner of the slide section on
        layer 0 and width and height the dimensions of the section on the specific
        layer `level`.  (Default: None)


    Returns
    -------
    tumor_mask : np.ndarray
        Binary tumor mask of the specified section. Healthy tissue is represented by 0,
        cancerous by 1.
    """
    if bounds is None:
        start_pos = (0, 0)
        size = slide.level_dimensions[level]
    else:
        start_pos, size = bounds

    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    downsample = slide.level_downsamples[level]

    for i, annotation in enumerate(slide.annotations):
        c_values, r_values = list(zip(*annotation.polygon))
        r = np.array(r_values, dtype=np.float32)
        r -= start_pos[1]
        r /= downsample
        r = np.array(r + 0.5, dtype=np.int32)

        c = np.array(c_values, dtype=np.float32)
        c -= start_pos[0]
        c /= downsample
        c = np.array(c + 0.5, dtype=np.int32)

        rr, cc = ski_polygon(r, c, shape=mask.shape)
        mask[rr, cc] = 1

    return mask

def split_slide(slide: Slide, lvl, otsu_threshold,
                fg_perc_thresh, tile_size, overlap):
    """
    Create tiles from a slide.

    Iterator over the slide in `tile_size`×`tile_size` Tiles. For every tile an otsu mask
    is created and summed up. Only tiles with sums over the percental threshold
    `fg_perc_thresh` will be yield.

    Parameters
    ----------
    slide : Slide
        Input Slide.

    lvl : int
        Layer to produce tiles from.

    otsu_threshold : float
        Otsu threshold of the whole slide on layer `level`.

    fg_perc_thresh : float, optional
        Minimum percentage, 0 to 1, of pixels with tissue per tile. (Default 0.01; 1%)

    tile_size : int
        Pixel size of one side of a square tile the image will be split into.
        (Default: 256)

    overlap : int, optional
        Count of pixel overlapping between two tiles. (Default: 30)

    Yields
    -------
    image_tile : np.ndarray
        Array of shape (`tile_size`, `tile_size`).

    bounds : tuple
        Tile boundaries on layer 0: ((x, y), (width, height))
    """
    if tile_size <= overlap:
        raise ValueError("Overlap has to be smaller than the tile size.")
    if overlap < 0:
        raise ValueError("Overlap can not be negative.")
    if otsu_threshold < 0:
        raise ValueError("Otsu threshold can not be negative.")
    if not 0.0 <= fg_perc_thresh <= 1.0:
        raise ValueError("Foreground threshold has to be between 0 and 1")

    width0, height0 = slide.level_dimensions[0]
    downsample = slide.level_downsamples[lvl]

    # Tile size on level 0
    tile_size0 = int(tile_size * downsample + 0.5)
    overlap0 = int(overlap * downsample + 0.5)

    # Minimum number of foreground pixels to be considered as foreground
    min_fg_count = tile_size ** 2 * fg_perc_thresh

    # We take patches into account if they belong to the foreground or to tumor tissue.
    # Once `num_pos_tiles_threshold` tumor patches have been found, do not verify whether patches belong
    # to tumor tissue anymore to save time.
    num_pos_tiles = 0
    num_pos_tiles_threshold = 100
    
    skip_pos_mask_calc = False
    # Loop through WSI rows and columns
    for y in range(0, height0, tile_size0 - overlap0):

        # Compute number of tumor pixels in row
        if skip_pos_mask_calc:
            n_tumor_pixels_row = 0
        else:
            if slide.has_tumor:
                mask_row = create_tumor_mask(slide, lvl, ((0, y), (width0, tile_size)))
                n_tumor_pixels_row = np.sum(mask_row)
            else:
                n_tumor_pixels_row = 0
            
        for x in range(0, width0, tile_size0 - overlap0):

            # Only proceed if tumor exists in row
            if n_tumor_pixels_row > 0:
                if lvl != 0:
                    mask_this = create_tumor_mask(slide, lvl, ((x, y), (tile_size, tile_size)))
                    pos_count = np.sum(mask_this)
                if lvl == 0:
                    pos_count = np.sum(mask_row[:,x:(x+tile_size)])

                if pos_count > 0:
                    num_pos_tiles +=1
                    if num_pos_tiles > num_pos_tiles_threshold:
                        skip_pos_mask_calc = True

            else:
                pos_count = 0

            tile = np.asarray(slide.read_region((x, y), lvl, (tile_size, tile_size)))
            otsu_mask = create_otsu_mask_by_threshold(rgb2gray(tile), otsu_threshold)

            fg_count = np.sum(otsu_mask)
            if fg_count >= min_fg_count or pos_count > 0:
                yield remove_alpha_channel(tile), ((x, y), (tile_size0, tile_size0))

--- Cell 22 ---
import os
import fnmatch
from collections import namedtuple
from typing import Dict

from PIL import Image
from PIL import ImageDraw
from progress.bar import IncrementalBar

Point = namedtuple('Point', 'x y')

def find_files(pattern, path) -> Dict[str, str]:
    """
    Find files in a directory by given file name pattern.

    Parameters
    ----------
    pattern : str
        File pattern allowing wildcards.

    path : str
        Root directory to search in.

    Returns
    -------
    dict(str: str)
        Dictionary of all found files where the file names are keys and the relative paths
        from search root are values.
    """
    result = {}
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result[name] = os.path.join(root, name)
    return result

class ProgressBar(IncrementalBar):
    @property
    def remaining_fmt(self):
        m, s = divmod(self.eta, 60)
        h, m = divmod(m, 60)
        return f'{h:02}:{m:02}:{s:02}'

    @property
    def elapsed_fmt(self):
        m, s = divmod(self.elapsed, 60)
        h, m = divmod(m, 60)
        return f'{h:02}:{m:02}:{s:02}'

def draw_polygon(image: Image.Image, polygon, *, fill, outline) -> Image.Image:
    """
    Draw a filled polygon on to an image.

    Parameters
    ----------
    image : Image.Image
        Background image to be drawn on.

    polygon :
        Polygon to be drawn.

    fill : color str or tuple
        Fill color.

    outline : color str or tuple
        Outline color.

    Returns
    -------
    Image.Image
        A copy of the background image with the polygon drawn onto.
    """
    img_back = image
    img_poly = Image.new('RGBA', img_back.size)
    img_draw = ImageDraw.Draw(img_poly)
    img_draw.polygon(polygon, fill, outline)
    img_back.paste(img_poly, mask=img_poly)
    return img_back

def get_relative_polygon(polygon, origin: Point, downsample=1):
    """
    Translate the polygon to relative to a point.

    Parameters
    ----------
    polygon : Sequence[Point]
        Polygon points.

    origin : Point
        The new origin the polygons points shall be relative to.

    downsample : int, optional
        Layer downsample >= 1 (Default: 1)

    Returns
    -------
    tuple(Point)
        New polygon with points relative to origin.
    """
    rel_polygon = []
    for point in polygon:
        rel_polygon.append(Point((point.x - origin.x) / downsample,
                                 (point.y - origin.y) / downsample))

    return tuple(rel_polygon)

--- Cell 24 ---
#!/usr/bin/env python
import os
import h5py
from pathlib import Path
import argparse
import yaml
import pandas as pd
import torch

from torch.utils.data import DataLoader
from pretraining.model.byol_model import BYOLModel
from data.camelyon.camelyon_dataset import CamelyonImages, PatchSampler

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(
    description="Extract SSL features from foreground patches for each slide"
)

parser.add_argument('--train', dest='is_train', action='store_true')
parser.add_argument('--test', dest='is_train', action='store_false')
parser.set_defaults(is_train=True)

parser.add_argument(
    "--lvl",
    type=int,
    default=0,
    help="Choose the magnification level (0: highest) for feature extraction"
)
parser.add_argument(
    "--otsu_lvl",
    type=int,
    default=0,
    help="Choose the magnification level for otsu threshold"
)
parser.add_argument(
    "--tile_size",
    type=int,
    default=256,
    help="Choose the tile size"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="Choose the batch size"
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=8,
    help="Number of workers for data loading"
)
parser.add_argument(
    "data_dir",
    help="The directory where the CAMELYON16 dataset is located"
)
parser.add_argument(
    "otsu_fname",
    help="The name of the file that holds Otsu thresholds."
)
parser.add_argument(
    "bounds_dir",
    help="The directory where the bounds file is located"
)
parser.add_argument(
    "coords_dir",
    help="The directory where the coords file is located"
)
parser.add_argument(
    "model_dir",
    help="The directory where the model checkpoint is located"
)
parser.add_argument(
    "feat_save_dir",
    help="The directory where the features shall be located"
)

args = parser.parse_args()

# Get arguments
train = args.is_train
lvl = args.lvl
otsu_lvl = args.otsu_lvl
tile_size = args.tile_size
batch_size = args.batch_size
num_workers = args.num_workers
data_dir = args.data_dir
otsu_fname = args.otsu_fname
bounds_dir = args.bounds_dir
coords_dir = args.coords_dir
model_dir = args.model_dir
feat_save_dir = args.feat_save_dir

# Define dataset
bounds_df = pd.read_pickle(bounds_dir)
coords_df = pd.read_pickle(coords_dir)
sampler = PatchSampler(bounds_df, batch_size=batch_size)
dataset = CamelyonImages(data_dir, otsu_fname, coords_df, lvl, tile_size)
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

h5file = h5py.File(feat_save_dir, "w")

# Load pre-trained model
with open(Path('pretraining/config/train_config.yaml'), 'r') as f:
    config = yaml.safe_load(f)
net = BYOLModel(config)
checkpoint = torch.load(model_dir, map_location=device)
loaded_dict = checkpoint['model']
prefix = 'module.'
n_clip = len(prefix)
adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items()
            if k.startswith(prefix)}
net.load_state_dict(adapted_dict, strict=True)
net = net.online_network.encoder
net.to(device)

net.eval()
with torch.no_grad():
    current_slide = None
    feature_list = []
    pos_idx_list = []
    num_processed = 0
    for data in dataloader:
        # Get data
        patches = data['patch'].to(device)
        pos_idx = data['pos_id'].to(device)
        data_idx = data['data_id'].to(device)
        slide_names = [name for name in data['slide_name'] if name]
        if len(slide_names) > 0:
            slide_label = data['label'].max() # either 0 or 1, but not negative
            slide_name = slide_names[0] # slide names are same after filtering

        # Check if new slide
        is_new_slide = slide_name != current_slide
        if is_new_slide:
            feature_list = []
            pos_idx_list = []
            current_slide = slide_name

        # Extract patches from batch (removes empty elements)
        stopper_idx = torch.where(data_idx < 0, 1, 0).nonzero()
        num_neg_idx = stopper_idx.shape[0]
        if num_neg_idx > 0:
            stop_id = stopper_idx[0]
            patches = patches[:stop_id]
            pos_idx = pos_idx[:stop_id]

        # Extract features
        if patches.shape[0] > 0:
            features = net(patches)
            b, n_feat = features.shape[:2]
            features = features.view(b, n_feat)

            feature_list.append(features)
            pos_idx_list.append(pos_idx)

        is_last_patch = data_idx[-1] == PatchSampler.SLIDE_END_TOKEN
        if is_last_patch:
            num_processed += 1
            print("Nr. slides processed: ", num_processed)

            features_np = torch.cat(feature_list, 0).cpu().numpy()
            pos_idx_np = torch.cat(pos_idx_list, 0).cpu().numpy()
            
            # Save as HDF5 file
            slide_grp = h5file.create_group(slide_name)
            slide_grp.create_dataset('img', data=features_np, compression="gzip", compression_opts=9)
            slide_grp.create_dataset('pos', data=pos_idx_np, compression="gzip", compression_opts=9)
            slide_grp.attrs['label'] = slide_label

h5file.close()
print("Stored features successfully!")


--- Cell 26 ---
#!/usr/bin/env python
import os
import argparse
from tqdm import tqdm
import pandas as pd
import pickle
import multiprocessing as mp

from datamodel import SlideManager
from cam_methods import split_slide

parser = argparse.ArgumentParser(
    description="Compute foreground coordinates for each slide"
)

parser.add_argument('--train', dest='is_train', action='store_true')
parser.add_argument('--test', dest='is_train', action='store_false')
parser.set_defaults(is_train=True)

parser.add_argument(
    "--lvl",
    type=int,
    default=0,
    help="Choose the magnification level (0: highest) for foreground computation"
)
parser.add_argument(
    "--otsu_lvl",
    type=int,
    default=0,
    help="Choose the magnification level for otsu threshold"
)
parser.add_argument(
    "--tile_size",
    type=int,
    default=256,
    help="Choose the tile size."
)
parser.add_argument(
    "--fg_perc_thresh",
    type=float,
    default=0.01,
    help="Minimum percentage of foreground pixels so that tile is considered foreground."
)
parser.add_argument(
    "--overlap",
    type=int,
    default=0,
    help="Overlap between tiles."
)
parser.add_argument(
    "--n_worker",
    type=int,
    default=16,
    help="Number of processes to spawn to parallelize computations"
)
parser.add_argument(
    "data_dir",
    help="The directory where the CAMELYON16 dataset is located"
)
parser.add_argument(
    "otsu_fname",
    help="The name of the file that holds Otsu thresholds."
)
parser.add_argument(
    "out_dir",
    help="Directory where foreground coordinates shall be stored. " + \
    "Filenames will be coords_{train/test}.pkl and bounds_{train/test}.pkl"
)

args = parser.parse_args()

# Get arguments
train = args.is_train
lvl = args.lvl
otsu_lvl = args.otsu_lvl
tile_size = args.tile_size
fg_perc_thresh = args.fg_perc_thresh
overlap = args.overlap
n_worker = args.n_worker
data_dir = args.data_dir
otsu_fname = args.otsu_fname
out_dir = args.out_dir

subset = 'train' if train else 'test'
bounds_path = os.path.join(out_dir, 'bounds_' + subset + '.pkl')
coords_path = os.path.join(out_dir, 'coords_' + subset + '.pkl')

def get_foreground_coords(name):
    slide = slide_man.get_slide(name)
    otsu_threshold = slide.get_otsu_threshold(otsu_lvl)
    tile_iter = split_slide(slide, lvl, otsu_threshold, fg_perc_thresh, tile_size, overlap)
    
    x_vals, y_vals = [], []
    for _, bounds in tile_iter:
        x, y = bounds[0]
        x_vals.append(x)
        y_vals.append(y)
    names = [name] * len(x_vals)
    print("Finished slide: ", name)

    return x_vals, y_vals, names

slide_man = SlideManager(data_dir=data_dir, otsu_fname=otsu_fname)
slide_names = slide_man.get_slide_names_subset(train=train)

# Computing of foreground coordinates can take a long time, thus parallelize
pool = mp.Pool(n_worker)
fg_coords = list(tqdm(
    pool.imap(get_foreground_coords, slide_names), total=len(slide_names)
))

# Create lists to be populated
start_idx, end_idx = [], []
all_idx, pos_idx = [], []
x_vals, y_vals = [], []
names = []

start = 0
for slide_id, slide_coords in enumerate(fg_coords):
    for patch_id, (x, y, name) in enumerate(zip(*slide_coords)):
        x_vals.append(x)
        y_vals.append(y)

        all_idx.append(start + patch_id)
        pos_idx.append(patch_id)
        
        names.append(name)
    
    print("Finished slide #", slide_id)

    end = start + patch_id
    
    start_idx.append(start)
    end_idx.append(end)

    start = end + 1

# Create dataframes
lvls = [lvl] * len(start_idx)
bounds_df = pd.DataFrame(
    {
        'level': lvls,
        'names': slide_names,
        'start_id': start_idx,
        'end_id': end_idx
    }
)
coords_df = pd.DataFrame(
    {
        'id': all_idx,
        'pos_id': pos_idx,
        'name': names,
        'x': x_vals,
        'y': y_vals
    }
)

# Save dataframes
bounds_file = open(bounds_path, 'wb')
pickle.dump(bounds_df, bounds_file)
bounds_file.close()

coords_file = open(coords_path, 'wb')
pickle.dump(coords_df, coords_file)
coords_file.close()

print("Done storing foreground coordinates.")

--- Cell 28 ---
import csv
import argparse
import multiprocessing as mp

from datamodel import SlideManager, Slide
from data.camelyon.cam_methods import get_otsu_threshold

parser = argparse.ArgumentParser(
    description="Compute Otsu thresholds from WSIs"
)
parser.add_argument(
    "--lvl",
    type=int,
    default=0,
    help="Choose the magnification level (0: highest) from which to compute thresholds"
)
parser.add_argument(
    "--n_worker",
    type=int,
    default=16,
    help="Number of processes to spawn to parallelize computations"
)
parser.add_argument(
    "data_dir",
    help="The directory where the CAMELYON16 dataset is located"
)
parser.add_argument(
    "otsu_fname",
    help="The directory where to store otsu thresholds."
)

args = parser.parse_args()

# Get arguments
lvl = args.lvl
n_worker = args.n_worker
data_dir = args.data_dir
otsu_fname = args.otsu_fname

# Create slide manager to access all slides
slide_man = SlideManager(data_dir=data_dir, otsu_fname=otsu_fname)

# Multiprocessing function
def get_slide_threshold(name):
    """
    Obtains a slide and computes the otsu threshold
    """
    slide_path = slide_man.slide_paths[name]
    slide = Slide(name, slide_path)

    threshold = get_otsu_threshold(slide, level=lvl, step_size=1000)

    del slide
    return name, lvl, threshold

# Calculating the Otsu threshold can take a long time
# depending on the magnification level, thus parallelize.
pool = mp.Pool(n_worker)
slide_thresholds = list(
    pool.map(get_slide_threshold, slide_man.slide_names)
)

# Write thresholds to file
f = open(out_dir, "w")

writer = csv.writer(f)
header = ['name', 'level', 'threshold']
writer.writerow(header)

for slide_name, level, threshold in slide_thresholds:
    writer.writerow([slide_name, level, threshold])

f.close()
print("Done saving thresholds!")

--- Cell 30 ---
import os
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from .datamodel import SlideManager
from .cam_methods import remove_alpha_channel

class PatchSampler(Sampler):

    FILL_TOKEN = -1
    SLIDE_END_TOKEN = -2

    def __init__(self, bounds, num_samples=None, batch_size=1):
        self.bounds = bounds
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_slides = self.bounds.shape[0]
    
    def __len__(self):
        return self.num_samples

    def __iter__(self):

        slide_idx = list(range(self.num_slides))
        self.all_patch_idx = []
        for slide_id in slide_idx:

            row = self.bounds.iloc[slide_id]
            start_id = row['start_id']
            end_id = row['end_id']

            patch_idx = list(range(start_id, end_id+1))
            num_patches = len(patch_idx)

            # Add tokens to fill up batch
            remainder = (num_patches + 1) % self.batch_size # +1 extra patch
            num_to_add = self.batch_size - remainder# if remainder else 0
            patch_idx = patch_idx + [self.FILL_TOKEN] * num_to_add

            # Add token to identify end of slide
            patch_idx.append(self.SLIDE_END_TOKEN)
            self.all_patch_idx.extend(patch_idx)

        return iter(self.all_patch_idx)


class CamelyonImages(Dataset):

    def __init__(self, data_dir, otsu_fname, coords_df, lvl, tile_size):

        self.slide_man = SlideManager(data_dir=data_dir, otsu_fname=otsu_fname)
        self.coords_df = coords_df

        self.lvl = lvl
        self.tile_size = tile_size

        transform_list = [
            transforms.Lambda(lambda x: remove_alpha_channel(np.asarray(x))),
            transforms.ToPILImage(),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]
        self.transform = transforms.Compose(transform_list)

        self.current_slide_name = None
        self.current_slide = None
    
    def __len__(self):
        return len(self.coords_df)

    def __getitem__(self, i):

        data = {}
        is_empty = i < 0
        if not is_empty:
            row = self.coords_df.iloc[i]
            slide_name, x, y, pos_id = row[['name', 'x', 'y', 'pos_id']]

            if slide_name != self.current_slide_name:
                slide = self.slide_man.get_slide(slide_name)
                
                self.current_slide_name = slide_name
                self.current_slide = slide
            else:
                slide = self.current_slide

            patch = slide.read_region((x, y), self.lvl, (self.tile_size, self.tile_size))
            data['patch'] = self.transform(patch)
            data['label'] = int(slide.has_tumor)
            data['pos_id'] = pos_id
            data['slide_name'] = slide_name
        else:
            # Set dummy data except for label, which is used to identify dummy
            data['patch'] = torch.empty((3, 224, 224))
            data['label'] = -1
            data['pos_id'] = 9999
            data['slide_name'] = ''
        data['data_id'] = i
        return data


class CamelyonFeatures(Dataset):

    def open_hdf5(self):
        self.dataset = h5py.File(self.data_dir, 'r')

    def select_slides(self):
        h5_data = h5py.File(self.data_dir, 'r')
        self.slide_names = list(h5_data.keys())
        self.data_len = len(self.slide_names)
        h5_data.close()

    def __init__(self, conf, train=True):

        self.tasks = conf.tasks

        filename = conf.train_fname if train else conf.test_fname
        self.data_dir = os.path.join(conf.data_dir, filename)

        self.select_slides()
    
    def __len__(self):
        return self.data_len

    def __getitem__(self, i):
        
        if not hasattr(self, 'dataset'):
            self.open_hdf5()

        slide_name = self.slide_names[i]

        slide = self.dataset[slide_name]
        patches = slide['img'][:]
        label = slide.attrs['label']

        data_dict = {'input': patches}
        for task in self.tasks.values():
            data_dict[task['name']] = label

        return data_dict

--- Cell 32 ---
import sys
import numpy as np
import torch

from utils.utils import adjust_learning_rate

def init_batch(device, conf):
    """
    Initialize the memory buffer for the batch consisting of M patches
    """
    if conf.is_image:
        mem_patch = torch.zeros((conf.B, conf.M, conf.n_chan_in, *conf.patch_size)).to(device)
    else:
        mem_patch = torch.zeros((conf.B, conf.M, conf.n_chan_in)).to(device)

    if conf.use_pos:
        mem_pos_enc = torch.zeros((conf.B, conf.M, conf.D)).to(device)
    else:
        mem_pos_enc = None

    # Init the labels for the batch (for multiple tasks in mnist)
    labels = {}
    for task in conf.tasks.values():
        if task['metric'] == 'multilabel_accuracy':
            labels[task['name']] = torch.zeros((conf.B, conf.n_class), dtype=torch.float32).to(device)
        else:
            labels[task['name']] = torch.zeros((conf.B,), dtype=torch.int64).to(device)
    
    return mem_patch, mem_pos_enc, labels

def fill_batch(mem_patch, mem_pos_enc, labels, data, n_prep, n_prep_batch,
            mem_patch_iter, mem_pos_enc_iter, conf):
    """
    Fill the patch, pos enc and label buffers and update helper variables
    """

    n_seq, len_seq = mem_patch_iter.shape[:2]
    mem_patch[n_prep:n_prep+n_seq, :len_seq] = mem_patch_iter
    if conf.use_pos:
        mem_pos_enc[n_prep:n_prep+n_seq, :len_seq] = mem_pos_enc_iter
    
    for task in conf.tasks.values():
        labels[task['name']][n_prep:n_prep+n_seq] = data[task['name']]
    
    n_prep += n_seq
    n_prep_batch += 1

    batch_data = (mem_patch, mem_pos_enc, labels, n_prep, n_prep_batch)

    return batch_data

def shrink_batch(mem_patch, mem_pos_enc, labels, n_prep, conf):
    """
    Adjust batch by removing empty instances (may occur in last batch of an epoch)
    """
    mem_patch = mem_patch[:n_prep]
    if conf.use_pos:
        mem_pos_enc = mem_pos_enc[:n_prep]
    
    for task in conf.tasks.values():
        labels[task['name']] = labels[task['name']][:n_prep]
    
    return mem_patch, mem_pos_enc, labels

def compute_loss(net, mem_patch, mem_pos_enc, criterions, labels, conf):
    """
    Obtain predictions, compute losses for each task and get some logging stats
    """

    # Obtain predictions
    preds = net(mem_patch, mem_pos_enc)

    # Compute losses for each task and sum them up
    loss = 0
    task_losses, task_preds, task_labels = {}, {}, {}
    for task in conf.tasks.values():
        t_name, t_act = task['name'], task['act_fn']

        criterion = criterions[t_name]
        label = labels[t_name]
        pred = preds[t_name].squeeze(-1)

        if t_act == 'softmax':
            pred_loss = torch.log(pred + conf.eps)
            label_loss = label
        else:
            pred_loss = pred.view(-1)
            label_loss = label.view(-1).type(torch.float32)

        task_loss = criterion(pred_loss, label_loss)
        # for logs
        task_losses[t_name] = task_loss.item()
        task_preds[t_name] = pred.detach().cpu().numpy()
        task_labels[t_name] = label.detach().cpu().numpy()

        loss += task_loss
    # Average task losses        
    loss /= len(conf.tasks.values())

    return loss, [task_losses, task_preds, task_labels]


def train_one_epoch(net, criterions, data_loader, optimizer, device, epoch, log_writer, conf):
    """
    Trains the given network for one epoch according to given criterions (loss functions)
    """

    # Set the network to training mode
    net.train()

    # Initialize helper variables
    n_prep, n_prep_batch = 0, 0 # num of prepared images/batches
    mem_pos_enc = None
    start_new_batch = True

    times = [] # only used when tracking efficiency stats
    # Loop through dataloader
    for data_it, data in enumerate(data_loader, start=epoch * len(data_loader)):
        # Move input batch onto GPU if eager execution is enabled (default), else leave it on CPU
        # Data is a dict with keys `input` (patches) and `{task_name}` (labels for given task)
        image_patches = data['input'].to(device) if conf.eager else data['input']

        # If starting a new batch, create placeholders for data which are filled later
        if start_new_batch:
            mem_patch, mem_pos_enc, labels = init_batch(device, conf)
            start_new_batch = False

            # If tracking efficiency, record time from here.
            if conf.track_efficiency:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
        
        # Apply IPS to input patches
        mem_patch_iter, mem_pos_enc_iter = net.ips(image_patches)
        
        # Fill batch placeholders with patches from IPS step
        batch_data = fill_batch(mem_patch, mem_pos_enc, labels, data, n_prep, n_prep_batch,
                                mem_patch_iter, mem_pos_enc_iter, conf)
        mem_patch, mem_pos_enc, labels, n_prep, n_prep_batch = batch_data

        # Check if the current batch is full or if it is the last batch
        batch_full = (n_prep == conf.B)
        is_last_batch = n_prep_batch == len(data_loader)

        # Do training step as soon as batch is full (or last batch)
        if batch_full or is_last_batch:

            if not batch_full:
                # Last batch may not be full, so remove empty instances
                mem_patch, mem_pos_enc, labels = shrink_batch(mem_patch, mem_pos_enc, labels, n_prep, conf)
            
            # Calculate and set new learning rate
            adjust_learning_rate(conf.n_epoch_warmup, conf.n_epoch, conf.lr, optimizer, data_loader, data_it+1)
            optimizer.zero_grad()

            # Compute loss
            loss, task_info = compute_loss(net, mem_patch, mem_pos_enc, criterions, labels, conf)
            task_losses, task_preds, task_labels = task_info

            # Backpropagate error and update parameters
            loss.backward()
            optimizer.step()

            # If tracking efficiency, log the time and memory usage
            if conf.track_efficiency:
                end_event.record()
                torch.cuda.synchronize()
                if epoch == conf.track_epoch and data_it > 0 and not is_last_batch:
                    times.append(start_event.elapsed_time(end_event))
                    print("time: ", times[-1])

            # Update log
            log_writer.update(task_losses, task_preds, task_labels)

            # Reset helper variables
            n_prep = 0
            start_new_batch = True
    
    if conf.track_efficiency:
        if epoch == conf.track_epoch:
            print("avg. time: ", np.mean(times))

            stats = torch.cuda.memory_stats()
            peak_bytes_requirement = stats["allocated_bytes.all.peak"]
            print(f"Peak memory requirement: {peak_bytes_requirement / 1024 ** 3:.4f} GB")

            print("TORCH.CUDA.MEMORY_SUMMARY: ", torch.cuda.memory_summary())
            sys.exit()


# Disable gradient calculation during evaluation
@torch.no_grad()
def evaluate(net, criterions, data_loader, device, log_writer, conf):

    # Set the network to evaluation mode
    net.eval()

    # Remaining parts similar to training loop
    n_prep, n_prep_batch = 0, 0
    mem_pos_enc = None
    start_new_batch = True
    
    for data in data_loader:
        image_patches = data['input'].to(device) if conf.eager else data['input']

        if start_new_batch:
            mem_patch, mem_pos_enc, labels = init_batch(device, conf)
            start_new_batch = False
        
        mem_patch_iter, mem_pos_enc_iter = net.ips(image_patches)
        
        batch_data = fill_batch(mem_patch, mem_pos_enc, labels, data, n_prep, n_prep_batch,
                                mem_patch_iter, mem_pos_enc_iter, conf)
        mem_patch, mem_pos_enc, labels, n_prep, n_prep_batch = batch_data

        batch_full = (n_prep == conf.B)
        is_last_batch = n_prep_batch == len(data_loader)

        if batch_full or is_last_batch:

            if not batch_full:
                mem_patch, mem_pos_enc, labels = shrink_batch(mem_patch, mem_pos_enc, labels, n_prep, conf)
            
            _, task_info = compute_loss(net, mem_patch, mem_pos_enc, criterions, labels, conf)
            task_losses, task_preds, task_labels = task_info

            log_writer.update(task_losses, task_preds, task_labels)

            n_prep = 0
            start_new_batch = True

--- Cell 34 ---
import yaml

camelyon_config_yaml = """
#opt
n_epoch: 50           # number of epochs
B: 16                 # batch size
B_seq: 1              # sequential batch size, set either to
                      # B (eager and lazy loading) or 1 (eager sequential loading)
n_epoch_warmup: 10    # number of warm-up epochs
lr: 0.0003            # learning rate
wd: 0.1               # weight decay

#dset
n_class: 1                            # number of classes
data_dir: 'data/camelyon/dsets'       # directory of dataset
train_fname: 'feat_train_500ep.hdf5'  # filename of extracted features of training set
test_fname: 'feat_test_500ep.hdf5'    # filename of extracted features of test set
n_worker: 64                          # number of workers
pin_memory: False                     # use pin memory in dataloader
eager: True                           # eager or lazy loading

#misc
eps: 0.000001
seed: 0
track_efficiency: False   # for training, needs to be False
track_epoch: 0            # only relevant if efficiency stats are tracked.

#enc
is_image: False         # should a convolutional patch encoder be used?
enc_type: 'resnet50'    # used backbone, set either to 'resnet18' or 'resnet50'
pretrained: False       # should ImageNet weights be used?
n_chan_in: 2048         # number of input channels from resnet 50

#ips
shuffle: True             # should patches be shuffled?
shuffle_style: 'batch'    # 'batch' or 'instance'. 'batch' shuffles each instance of the batch the same way
n_token: 1                # number of learnable query tokens, corresponds to number of tasks
M: 5000                   # memory size
I: 5000                   # iteration size

#aggr
use_pos: False      # should positional encoding be used?
H: 8                # number of transformer layer heads
D: 512              # dimension of features
D_k: 64             # dimension of query/keys per head
D_v: 64             # dimension of values per head
D_inner: 2048       # intermediate layer dimension in MLP
attn_dropout: 0.1   # attention dropout
dropout: 0.1        # standard dropout

# define name, activation function of final layer and metric to be used
tasks:
  task0:
    id: 0
    name: 'metastases'
    act_fn: 'sigmoid'
    metric: 'auc'

"""

camelyon_conf = Struct(**yaml.safe_load(camelyon_config_yaml))

--- Cell 35 ---
import yaml

mnist_config_yaml = """
#opt
n_epoch: 150          # number of epochs
B: 16                 # batch size
B_seq: 16             # sequential batch size, set either to
                      # B (eager and lazy loading) or 1 (eager sequential loading)
n_epoch_warmup: 10    # number of warm-up epochs
lr: 0.001             # learning rate
wd: 0.1               # weight decay

#dset
n_class: 10                                                   # number of classes
data_dir: 'data/megapixel_mnist/dsets/megapixel_mnist_1500'   # directory of dataset
n_worker: 8                                                   # number of workers
pin_memory: True                                              # use pin memory in dataloader
eager: True                                                   # eager or lazy loading

#misc
eps: 0.000001
seed: 0
track_efficiency: False   # for training, needs to be False
track_epoch: 0            # only relevant if efficiency stats are tracked.

#enc
is_image: True          # should a convolutional patch encoder be used?
enc_type: 'resnet18'    # used backbone, set either to 'resnet18' or 'resnet50'
pretrained: False       # should ImageNet weights be used?
n_chan_in: 1            # number of input channels
n_res_blocks: 2         # number of residual ResNet blocks, mnist only uses 2

#ips
shuffle: True               # should patches be shuffled?
shuffle_style: 'batch'      # 'batch' or 'instance'. 'batch' shuffles each instance of the batch the same way
n_token: 4                  # number of learnable query tokens, corresponds to number of tasks (mnist has 4 tasks)
N: 900                      # number of total patches, needs to be consistent with patch size/stride
M: 100                      # memory size
I: 100                      # iteration size
patch_size: [50, 50]        # dims of patch
patch_stride: [50, 50]      # stride of patch, use 25 per side for 50% overlap

#aggr
use_pos: True       # should positional encoding be used?
H: 8                # number of transformer layer heads
D: 128              # dimension of features
D_k: 16             # dimension of query/keys per head
D_v: 16             # dimension of values per head
D_inner: 512        # intermediate layer dimension in MLP
attn_dropout: 0.1   # attention dropout
dropout: 0.1        # standard dropout

# define name, activation fn of final layer and metric to be used
tasks:
  task0:
    id: 0
    name: 'majority'
    act_fn: 'softmax'
    metric: 'accuracy'
  task1:
    id: 1
    name: 'max'
    act_fn: 'softmax'
    metric: 'accuracy'
  task2:
    id: 2
    name: 'top'
    act_fn: 'softmax'
    metric: 'accuracy'
  task3:
    id: 3
    name: 'multi'
    act_fn: 'sigmoid'
    metric: 'multilabel_accuracy'


"""

mnist_conf = Struct(**yaml.safe_load(mnist_config_yaml))

--- Cell 36 ---
import yaml

traffic_config_yaml = """
#opt
n_epoch: 150          # number of epochs
B: 16                 # batch size
B_seq: 16             # sequential batch size, set either to
                      # B (eager and lazy loading) or 1 (eager sequential loading)
n_epoch_warmup: 10    # number of warm-up epochs
lr: 0.0003            # learning rate
wd: 0.1               # weight decay

#dset
n_class: 4                      # number of classes
data_dir: 'data/traffic/dsets'  # directory of dataset
n_worker: 8                     # number of workers
pin_memory: True                # use pin memory in dataloader
eager: True                     # eager or lazy loading

#misc
eps: 0.000001
seed: 0
track_efficiency: False   # for training, needs to be False
track_epoch: 0            # only relevant if efficiency stats are tracked.

#enc
is_image: True          # should a convolutional patch encoder be used?
enc_type: 'resnet18'    # used backbone, set either to 'resnet18' or 'resnet50'
pretrained: True        # should ImageNet weights be used?
n_chan_in: 3            # number of input channels
n_res_blocks: 4         # number of residual ResNet blocks

#ips
shuffle: True                 # should patches be shuffled?
shuffle_style: 'batch'        # shuffle each instance the same way? 'batch' or 'instance'
n_token: 1                    # Number of learnable query tokens
N: 192                        # Number of total patches, needs to be consistent with patch size/stride
M: 10                         # memory size
I: 32                         # iteration size
patch_size: [100, 100]        # dims of patch
patch_stride: [100, 100]      # stride of patch

#aggr
use_pos: False        # should positional encoding be used?
H: 8                  # number of transformer layer heads
D: 512                # dimension of features
D_k: 64               # dimension of query/keys per head
D_v: 64               # dimension of values per head
D_inner: 2048         # hidden dimension of MLP
attn_dropout: 0.1     # attention dropout
dropout: 0.1          # standard dropout

tasks:
  task0:
    id: 0
    name: 'sign'
    act_fn: 'softmax'
    metric: 'accuracy'

"""

traffic_conf = Struct(**yaml.safe_load(traffic_config_yaml))

--- Cell 38 ---
#!/usr/bin/env python

import os
import yaml
from pprint import pprint

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.utils import Logger, Struct
from data.megapixel_mnist.mnist_dataset import MegapixelMNIST
from data.traffic.traffic_dataset import TrafficSigns
from data.camelyon.camelyon_dataset import CamelyonFeatures
from architecture.ips_net import IPSNet
from training.iterative import train_one_epoch, evaluate

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = 'traffic' # either one of {'mnist', 'camelyon', 'traffic'}

conf_map = {'mnist': mnist_conf, 'traffic': traffic_conf, 'camelyon': camelyon_conf}
conf = conf_map[dataset]
pprint(vars(conf))

# fix the seed for reproducibility
torch.manual_seed(conf.seed)
np.random.seed(conf.seed)

# define datasets and dataloaders
if dataset == 'mnist':
    train_data = MegapixelMNIST(conf, train=True)
    test_data = MegapixelMNIST(conf, train=False)
elif dataset == 'traffic':
    train_data = TrafficSigns(conf, train=True)
    test_data = TrafficSigns(conf, train=False)
elif dataset == 'camelyon':
    train_data = CamelyonFeatures(conf, train=True)
    test_data = CamelyonFeatures(conf, train=False)

train_loader = DataLoader(train_data, batch_size=conf.B_seq, shuffle=True,
    num_workers=conf.n_worker, pin_memory=conf.pin_memory, persistent_workers=True)
test_loader = DataLoader(test_data, batch_size=conf.B_seq, shuffle=False,
    num_workers=conf.n_worker, pin_memory=conf.pin_memory, persistent_workers=True)

# define network
net = IPSNet(device, conf).to(device)

loss_nll = nn.NLLLoss()
loss_bce = nn.BCELoss()

# define optimizer, lr not important at this point
optimizer = torch.optim.AdamW(net.parameters(), lr=0, weight_decay=conf.wd)

criterions = {}
for task in conf.tasks.values():
    criterions[task['name']] = loss_nll if task['act_fn'] == 'softmax' else loss_bce

log_writer_train = Logger(conf.tasks)
log_writer_test = Logger(conf.tasks)

for epoch in range(conf.n_epoch):
    
    train_one_epoch(net, criterions, train_loader, optimizer, device, epoch, log_writer_train, conf)

    log_writer_train.compute_metric()

    more_to_print = {'lr': optimizer.param_groups[0]['lr']}
    log_writer_train.print_stats(epoch, train=True, **more_to_print)

    evaluate(net, criterions, test_loader, device, log_writer_test, conf)
    
    log_writer_test.compute_metric()
    log_writer_test.print_stats(epoch, train=False)
