import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from stroke_model.global_alignment import predict_with_global_alignment
from stroke_model.infer_topology import export_hard_masks, load_topology
from stroke_model.reference_model import ReferenceStrokeModel
from stroke_model.reference_overlap import ARCHITECTURE, OverlapStrokeModel, overlap_annotations, overlap_loss
from stroke_model.train_reference import evaluate, main


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, image):
        return [self.conv(image)]


def tiny_overlap(**kwargs):
    return OverlapStrokeModel(backbone=TinyBackbone(), channels=[8], **kwargs)


class OverlapTests(unittest.TestCase):
    def test_shared_seeds_train_both_channels_and_unannotated_ink_has_no_gradient(self):
        masks = torch.tensor([[[[1., 1., 0., 0.]], [[1., 0., 0., 0.]], [[0., 0., 0., 0.]]]])
        image = torch.ones(1, 1, 1, 4)
        active = torch.tensor([[True, True, False]])
        logits = torch.zeros_like(masks, requires_grad=True)
        scores = logits.masked_fill(~active[:, :, None, None], -torch.inf)
        output = dict(logits=scores, probabilities=scores.sigmoid(), flow=torch.zeros(1, 2, 2, 2), active=active)
        target, known = overlap_annotations(image, masks, image, active)
        loss = overlap_loss(output, target, known)
        self.assertTrue(loss.isfinite())
        loss.backward()
        self.assertTrue((logits.grad[0, :2, 0, 0] < 0).all())
        self.assertGreater(logits.grad[0, 1, 0, 1].item(), 0.)
        self.assertEqual(logits.grad[..., 2:].count_nonzero(), 0)
        self.assertEqual(logits.grad[:, 2].count_nonzero(), 0)

    def test_new_checkpoint_trains_reloads_and_keeps_aligned_shared_masks(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        size = 40
        target = torch.zeros(1, 2, size, size)
        target[:, 0, 20, 12:29] = 1
        target[:, 1, 12:29, 20] = 1
        image = target.amax(1, keepdim=True)
        labels = torch.where(target.sum(1) == 1, target.argmax(1), -1)
        valid = torch.ones_like(image)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for split in ('train', 'val'):
                path = root / f'{split}.npz'
                np.savez_compressed(path, image=image[0].numpy(), reference=image[0].numpy(),
                                    strokes=target[0].numpy(), labels=labels[0].numpy(),
                                    target_masks=target[0].numpy(), valid_pixels=valid[0].numpy())
                rows.append(dict(path=str(path), group_id=split, split=split))
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(rows), encoding='utf-8')
            initializer = root / 'legacy.pt'
            legacy = ReferenceStrokeModel(image_size=size, pretrained=False,
                                          backbone=TinyBackbone(), channels=[8])
            torch.save(dict(architecture='reference_v2', image_size=size, model=legacy.state_dict()), initializer)
            run = root / 'run'
            args = ['train', '--manifest', str(manifest), '--output', str(run), '--epochs', '1',
                    '--batch-size', '1', '--num-workers', '0', '--image-size', str(size),
                    '--allow-overlap', '--init-checkpoint', str(initializer), '--junction-weight', '.05',
                    '--seed-focal-weight', '.1', '--ownership-weight', '.1', '--affinity-weight', '.02',
                    '--direction-affinity-weight', '.03']
            with patch.object(sys, 'argv', args), patch('stroke_model.train_reference.OverlapStrokeModel', tiny_overlap):
                main()
            state = torch.load(run/'best.pt', weights_only=True)
            self.assertEqual(state['architecture'], ARCHITECTURE)
            stats = json.loads((run/'history.json').read_text(encoding='utf-8'))[0]
            self.assertEqual(stats['val_shared_seed_pixels'], 1)
            self.assertEqual(stats['junction_applications'], 1)
            self.assertEqual(stats['direction_affinity_applications'], 1)
            with patch('stroke_model.infer_topology.OverlapStrokeModel', tiny_overlap):
                model, loaded_size, epoch = load_topology(run/'best.pt', 'cpu')
            self.assertEqual((model.output_mode, loaded_size, epoch), ('overlap', size, 1))
            with patch.object(model, 'score_strokes', return_value=torch.full_like(target, 4.)), torch.no_grad():
                aligned = predict_with_global_alignment(model, image, image, target, enabled=True)
                hard = export_hard_masks(image[0].numpy(), aligned['masks'][0].numpy(), output_mode=model.output_mode)
                self.assertEqual(hard[:, 20, 20].sum(), 2)
                self.assertEqual(hard[:, 0, 0].sum(), 0)
                metrics = evaluate(model, [(image, image, target, labels, target, valid)], 'cpu', return_details=True)
                self.assertEqual(metrics['val_shared_seed_exact_recall'], 1.)
                self.assertEqual(metrics['val_false_shared_seed_rate'], 1.)
