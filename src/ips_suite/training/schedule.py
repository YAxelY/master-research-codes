"""Learning-rate schedule, identical to the original IPS code"""

import math


def adjust_learning_rate(n_epoch_warmup, n_epoch, max_lr, optimizer, n_step_epoch, step):
    max_steps    = int(n_epoch * n_step_epoch)
    warmup_steps = int(n_epoch_warmup * n_step_epoch)
    if step < warmup_steps:
        lr = max_lr * step / max(warmup_steps, 1)
    else:
        s = step - warmup_steps
        m = max(max_steps - warmup_steps, 1)
        q = 0.5 * (1 + math.cos(math.pi * s / m))
        end_lr = max_lr * 0.001
        lr = max_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr
    return lr
