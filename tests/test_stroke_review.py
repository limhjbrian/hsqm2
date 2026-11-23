import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from stroke_model.stroke_review import index_training_rows, publish_reviewed_sample
from stroke_model.train_reference import StrokeDataset


class StrokeReviewTests(unittest.TestCase):
    def test_training_index_matches_both_font_and_character(self):
        rows = [{'source_image': name, 'split': split} for name, split in [
            ('I_Font_A_초_1.png', 'train'), ('I_Font_A_초_2.png', 'train'),
            ('I_Font_B_초_1.png', 'train'), ('I_Font_A_오_1.png', 'train'),
            ('I_Font_A_초_3.png', 'val'), ('I_초_1.png', 'train')]]
        grouped = index_training_rows(rows)
        self.assertEqual(grouped, {
            ('Font_A', '초'): rows[:2], ('Font_B', '초'): [rows[2]], ('Font_A', '오'): [rows[3]]})

    def test_confirmed_prediction_loads_in_trainer_and_preserves_shared_masks(self):
        image = np.ones((1, 3, 3), dtype=np.float32)
        masks = np.zeros((2, 3, 3), dtype=bool)
        masks[0, 1, :2] = True
        masks[1, 1:, 1] = True
        prediction = dict(image=image, reference=image, reference_strokes=masks, hard_masks=masks)
        row = dict(source_image='I_Font_A_초_1.png', group_id='reference:demo', char_id='초', path='original.npz')
        with TemporaryDirectory() as directory:
            path = publish_reviewed_sample(directory, prediction=prediction, training_row=row)
            self.assertEqual(path, publish_reviewed_sample(directory, prediction=prediction, training_row=row))
            sample = StrokeDataset([{'path': path}], size=3, include_topology=True, allow_overlap=True)[0]
            self.assertEqual(sample[3][1, 1].item(), -1)
            self.assertEqual(sample[3][1, 0].item(), 0)
            self.assertEqual(sample[4][:, 1, 1].sum().item(), 2)
            with np.load(path, allow_pickle=False) as packed:
                metadata = json.loads(packed['sample_metadata'].item())
            self.assertEqual(metadata['group_id'], 'reference:demo')
            self.assertEqual(metadata['report']['overlap_pixels'], 1)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_fallback_keeps_existing_annotations_byte_for_byte(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'sample.npz'
            np.savez_compressed(source, target_masks=np.ones((2, 2, 2)))
            output = publish_reviewed_sample(root/'reviewed', training_row={'path': source})
            self.assertEqual(output.name, source.name)
            self.assertEqual(output.read_bytes(), source.read_bytes())
