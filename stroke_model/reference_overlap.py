"""V2 with per-stroke membership and sparse multi-label supervision."""

import torch
from torch.nn import functional as F

from .reference_model import ReferenceStrokeModel, flow_smoothness

ARCHITECTURE = 'reference_v2_overlap_v1'


class OverlapStrokeModel(ReferenceStrokeModel):
    output_mode = 'overlap'

    def stroke_probabilities(self, logits, active):
        return logits.sigmoid()


def overlap_annotations(image, target_masks, valid, active):
    """Only actual annotated ink contributes; disagreements and blank ink stay unknown."""
    target = (target_masks > .5) & (valid > .5) & (image > .5) & active[:, :, None, None]
    known = target.any(1, keepdim=True)
    return target, known


def overlap_loss(output, target, known):
    active = output['active'][:, :, None, None]
    supervised = known & active
    # Inactive slots carry -inf logits; BCE must only see finite values there.
    logits = output['logits'].masked_fill(~active, 0.)
    target = target.to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    bce = (bce * supervised).sum() / supervised.sum().clamp_min(1)
    pred = output['probabilities'] * supervised
    mass = target.sum((-2, -1))
    present = mass > 0
    dice = (2 * (pred * target).sum((-2, -1)) + 1e-6) / (pred.sum((-2, -1)) + mass + 1e-6)
    dice_loss = ((1 - dice) * present).sum() / present.sum().clamp_min(1)
    return bce + dice_loss + .001 * flow_smoothness(output['flow'])


def overlap_seed_focal_loss(output, target, gamma):
    """Retain the extra supervision of difficult positive seeds, including shared ones."""
    log_true = F.logsigmoid(output['logits'].masked_fill(~target, 0.))
    values = -(1 - log_true.exp()).clamp_min(1e-8).pow(gamma) * log_true * target
    counts = target.sum((-2, -1))
    return (values.sum((-2, -1)) / counts.clamp_min(1)).sum() / (counts > 0).sum().clamp_min(1)
