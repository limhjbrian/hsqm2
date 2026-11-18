import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import torch
from PIL import Image

from stroke_model.infer_reference import run_inference_csv, select_input_files


class FakeModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.images = []

    def forward(self, image, reference, strokes):
        self.images.append(image.clone())
        probabilities = torch.ones_like(strokes) / strokes.shape[1]
        return {'probabilities': probabilities, 'masks': image * probabilities,
                'active': strokes.flatten(2).sum(-1) > 0}


class InferenceTests(unittest.TestCase):
    def test_select_by_filename(self):
        frame = pd.DataFrame({'input_path': ['a/first.png', 'b/second.png', 'c/first.png']})
        self.assertEqual(select_input_files(frame, 'input_path', ['second.png', 'a/first.png']), [2, 1])
        self.assertEqual(select_input_files(frame, 'input_path', 'second.png'), [2])
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            select_input_files(frame, 'input_path', ['first.png'])
        with self.assertRaisesRegex(ValueError, 'not found'):
            select_input_files(frame, 'input_path', ['missing.png'])

    def test_missing_input_stroke_does_not_require_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            horizontal = np.zeros((16, 16), dtype=np.uint8)
            horizontal[3:5, 2:14] = 255
            vertical = np.zeros_like(horizontal)
            vertical[2:14, 7:9] = 255
            # Reference has two strokes; input only contains the first.
            for name, ink in [('input', horizontal), ('ref', np.maximum(horizontal, vertical)),
                              ('s1', horizontal), ('s2', vertical)]:
                Image.fromarray(255 - ink).save(root / f'{name}.png')
            row = {'input_path': 'input.png', 'target_path': 'ref.png',
                   'target_stroke_paths': 's1.png;s2.png'}
            torch.save({'model': {}, 'image_size': 16, 'width': 4}, root / 'best.pt')
            # Both absent annotations and invalid annotation paths must be ignored.
            for variant in ('absent', 'invalid'):
                data = dict(row)
                if variant == 'invalid':
                    data.update(input_stroke_paths='does_not_exist.png', total_paths=999)
                pd.DataFrame([data]).to_csv(root / 'paths.csv', index=False)
                model = FakeModel()
                with patch('stroke_model.infer_reference.ReferenceStrokeModel', return_value=model) as load:
                    report = run_inference_csv(root / 'best.pt', root / 'paths.csv',
                                               input_files=['input.png'], save_dir=root / variant, show_soft=True,
                                               compare_alignment=variant == 'invalid')
                load.assert_called_once()
                self.assertFalse(report['uses_input_stroke_annotations'])
                self.assertEqual(report['results'][0]['csv_row'], 1)
                with np.load(report['results'][0]['npz']) as result:
                    self.assertEqual(result['soft_masks'].shape, (2, 16, 16))
                    self.assertEqual(result['labels'][0, 0], -1)
                saved = json.loads((root / variant / 'inference.json').read_text())
                self.assertNotIn('metrics', saved)
                self.assertTrue((root / variant / 'row_0001_soft_1.png').is_file())
                if variant == 'invalid':
                    self.assertEqual(len(model.images), 2)
                    self.assertFalse(torch.equal(*model.images))
                    self.assertTrue(report['compare_alignment'])
                    result = report['results'][0]
                    for key in ('comparison_preview', 'unaligned_npz', 'aligned_npz'):
                        self.assertTrue(Path(result[key]).is_file())
                    self.assertTrue(all(Path(p).is_file() for p in result['comparison_strokes']))
                    with np.load(result['unaligned_npz']) as off, np.load(result['aligned_npz']) as on:
                        self.assertNotIn('global_alignment_theta', off)
                        self.assertIn('global_alignment_theta', on)
                        np.testing.assert_array_equal(off['image'], on['image'])
                        for arrays in (off, on):
                            np.testing.assert_allclose(arrays['soft_masks'].sum(0), arrays['image'][0], atol=1e-6)
                            np.testing.assert_array_equal(arrays['labels'] < 0, arrays['image'][0] <= .5)
                else:
                    self.assertEqual(len(model.images), 1)

    def test_notebook_and_standalone_cell_match_runtime_source(self):
        root = Path(__file__).resolve().parents[1]
        source = (root/'notebooks/cells/infer_reference_v2_cell.py').read_text(encoding='utf-8')
        self.assertTrue(source.startswith((root/'stroke_model/infer_reference.py').read_text(encoding='utf-8')))
        notebook = json.loads((root/'notebooks/inference/infer_reference_v2.ipynb').read_text(encoding='utf-8'))
        self.assertEqual(''.join(notebook['cells'][2]['source']), source)
        self.assertIn('"inference-global-alignment"', ''.join(notebook['cells'][1]['source']))
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), '<colab>', 'exec')


if __name__ == '__main__':
    unittest.main()
