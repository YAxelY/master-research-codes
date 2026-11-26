"""Exploitation by batch: one encoder call per scale, ONE projector call for the whole batch (the projector holds a BatchNorm1d)."""

import torch


def _embed_from_store(net, store, B):
    """store[sid] = (list of patches, list of batch indices, list of slots)."""
    raws, bidx, sidx, sids = [], [], [], []
    for sid in range(len(net.spec.sizes)):
        pats, bb, ss = store[sid]
        if not pats:
            continue
        x = torch.cat(pats, dim=0)
        raws.append(net.encoder(x))
        bidx.append(torch.cat(bb))
        sidx.append(torch.cat(ss))
        sids.append(torch.full((x.shape[0],), sid, dtype=torch.long,
                               device=x.device))
    if not raws:
        raise RuntimeError("empty batch")
    feats = net.projector(torch.cat(raws, dim=0))
    if net.scale_emb is not None:
        feats = net.scale_emb(feats, torch.cat(sids))
    emb = torch.zeros(B, net.M_bag, net.D, device=feats.device, dtype=feats.dtype)
    return emb.index_put((torch.cat(bidx), torch.cat(sidx)), feats)


def _new_store(n_scales):
    return {sid: ([], [], []) for sid in range(n_scales)}


@torch.no_grad()
def _fill_store(net, store, img, coords, b):
    """Extract the retained patches and file them by scale. No gradient:
    these are data, the graph starts at the encoder."""
    for sid in range(len(net.spec.sizes)):
        mask = coords[:, 0] == sid
        if not bool(mask.any()):
            continue
        idx = torch.nonzero(mask, as_tuple=True)[0]
        sub = coords[idx]
        pats, bb, ss = store[sid]
        pats.append(net.patches(img, sub[:, 1:], sid))
        bb.append(torch.full((idx.numel(),), b, dtype=torch.long,
                             device=idx.device))
        ss.append(idx)


def _embed_single(net, img, coords):
    store = _new_store(len(net.spec.sizes))
    _fill_store(net, store, img, coords, 0)
    return _embed_from_store(net, store, 1)
