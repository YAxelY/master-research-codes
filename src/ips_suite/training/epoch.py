"""One epoch The learning-rate step counts DataLoader iterations, as in training/iterative.py: max_steps = n_epoch * len(loader), with a loader of batch size B_seq. The optimiser itself steps every B images. Returns a dict: loss, y_true, y_pred, y_prob, ids, coords (if collect), prof (Profiler)."""

import numpy as np
import time
import torch

from ips_suite.models.batching import _embed_from_store, _fill_store, _new_store
from ips_suite.training.profiler import Profiler
from ips_suite.training.schedule import adjust_learning_rate


IMAGE_MODELS = ('baseline', 'mil')    # take whole images, one optimiser step per batch


def run_epoch(net, loader, conf, device, optimizer=None, epoch=0,
              criterion=None, is_baseline=False, collect=False):
    """One pass over a loader. With collect=True (evaluation) also returns,
    per image: selected patches, their attention in the aggregator, the coarse
    zones of a cascade, and for MIL the attention logits of every patch."""
    train = optimizer is not None
    net.train(train)
    prof = Profiler()

    tot_loss, n_seen = 0.0, 0
    y_true, y_pred, y_prob, ids, coords_out = [], [], [], [], []
    attn_out, zones_out, maps_out = [], [], []
    n_step_epoch = max(1, len(loader))
    it = epoch * n_step_epoch
    lazy = (not is_baseline) and net.loading == 'lazy'
    is_mil = bool(getattr(net, 'is_mil', False))
    is_cascade = hasattr(net, 'cascade')
    dev_aug = bool(getattr(loader.dataset, 'aug_on_device', False))

    def stack(xs):
        return np.stack(xs) if xs else None

    t_mark = time.perf_counter()

    if is_baseline:
        for imgs, labels, idx in loader:
            prof.t['data'] += time.perf_counter() - t_mark
            if dev_aug:
                with prof.phase('augment'):
                    imgs = loader.dataset.device_augment(imgs.to(device))
            with prof.phase('step'):
                imgs, labels = imgs.to(device), labels.to(device)
                it += 1
                if train:
                    adjust_learning_rate(conf.n_epoch_warmup, conf.n_epoch,
                                         conf.lr, optimizer, n_step_epoch, it)
                    optimizer.zero_grad()
                with (torch.enable_grad() if train else torch.no_grad()):
                    logits = net(imgs)
                    loss = criterion(logits, labels)
                    if train:
                        loss.backward()
                        optimizer.step()
                bs = labels.shape[0]
                tot_loss += loss.item() * bs
                n_seen += bs
                prob = torch.softmax(logits.detach(), -1)
                y_true += labels.tolist()
                y_pred += prob.argmax(-1).tolist()
                y_prob += prob.cpu().numpy().tolist()
                ids += idx.tolist()
                if collect and is_mil:
                    top = net.last_attn.topk(min(net.M, net.last_attn.shape[1]), dim=1)
                    for b in range(bs):
                        coords_out.append(net.spec.candidates[top.indices[b]].cpu().numpy())
                        attn_out.append(top.values[b].float().cpu().numpy())
                        maps_out.append(net.last_logits[b].float().cpu().numpy())
            prof.n_img += bs
            t_mark = time.perf_counter()
        return dict(loss=tot_loss / max(n_seen, 1), y_true=y_true, y_pred=y_pred,
                    y_prob=y_prob, ids=ids, coords=stack(coords_out),
                    attn=stack(attn_out), zones=None, maps=stack(maps_out), prof=prof)

    store = _new_store(len(net.spec.sizes))
    labels_buf, n_prep = [], 0

    def flush(step_it):
        nonlocal store, labels_buf, n_prep, tot_loss, n_seen
        if n_prep == 0:
            return
        B = n_prep
        lab = torch.stack(labels_buf)
        if train:
            adjust_learning_rate(conf.n_epoch_warmup, conf.n_epoch,
                                 conf.lr, optimizer, n_step_epoch, step_it)
            optimizer.zero_grad()
        with (torch.enable_grad() if train else torch.no_grad()):
            emb = _embed_from_store(net, store, B)
            logits = net.classify(emb)
            loss = criterion(logits, lab)
            if train:
                loss.backward()
                optimizer.step()
            if collect:     # weight of each retained patch in the final decision
                attn_out.extend(net.transf.get_scores(emb).float().cpu().numpy())
        tot_loss += loss.item() * B
        n_seen += B
        prob = torch.softmax(logits.detach(), -1)
        y_true.extend(lab.tolist())
        y_pred.extend(prob.argmax(-1).tolist())
        y_prob.extend(prob.cpu().numpy().tolist())
        store = _new_store(len(net.spec.sizes))
        labels_buf = []
        n_prep = 0

    for imgs, labels, idx in loader:
        prof.t['data'] += time.perf_counter() - t_mark
        labels = labels.to(device)
        if dev_aug:
            with prof.phase('augment'):
                imgs = loader.dataset.device_augment(imgs if lazy else imgs.to(device))
        if not lazy:
            with prof.phase('select'):
                imgs = imgs.to(device)
        it += 1
        for k in range(imgs.shape[0]):
            with prof.phase('select'):
                if collect and is_cascade:
                    coords, stages = net.ips(imgs[k], return_stages=True)
                else:
                    coords = net.ips(imgs[k])
                _fill_store(net, store, imgs[k], coords, n_prep)
            if collect:
                c = coords.cpu().numpy()
                if c.shape[0] < net.M_bag:          # pool smaller than budget
                    pad = -np.ones((net.M_bag - c.shape[0], 3), np.int64)
                    c = np.concatenate([c, pad], 0)
                coords_out.append(c)
                if is_cascade:
                    zones_out.append(stages[0].cpu().numpy())
            ids.append(int(idx[k]))
            labels_buf.append(labels[k])
            n_prep += 1
            prof.n_img += 1
            if n_prep == conf.B:
                with prof.phase('step'):
                    flush(it)
        t_mark = time.perf_counter()
    with prof.phase('step'):
        flush(it)
    return dict(loss=tot_loss / max(n_seen, 1), y_true=y_true, y_pred=y_pred,
                y_prob=y_prob, ids=ids, coords=stack(coords_out),
                attn=stack(attn_out), zones=stack(zones_out), maps=None, prof=prof)
