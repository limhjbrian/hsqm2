import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import numpy as np
from PIL import Image
import torch

from stroke_model.npz_dataset import publish_npz_shards
from stroke_model.stroke_demo import run_stroke_demo


class DemoFixture(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.output_mode = mode

    def forward(self, image, reference, strokes):
        scores = torch.ones_like(strokes)
        probabilities = scores * .9 if self.output_mode == 'overlap' else scores / strokes.shape[1]
        return dict(probabilities=probabilities, masks=image * probabilities)


class StrokeDemoTests(unittest.TestCase):
    def test_npz_image_and_canvas_use_selected_checkpoint_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = {'partition': root/'v2_topology_runs'/'run'/'best.pt',
                      'overlap': root/'v2_topology_overlap_runs'/'run'/'best.pt'}
            for path in models.values():
                path.parent.mkdir(parents=True)
                path.touch()
            batch = root/'batch'
            batch.mkdir()
            dataset = root/'dataset'/'test_dataset'
            strokes = np.zeros((2, 16, 16), dtype=np.float32)
            strokes[0, 8, 3:13] = 1
            strokes[1, 3:13, 8] = 1
            image = strokes.max(0, keepdims=True)
            for i in (1, 2):
                name = f'I_Font_가_{i}.png'
                metadata = dict(version=1, name=name, size=16, group_id='reference:font-glyph',
                                char_id='가', image_digest=str(i), report={})
                path = batch/(hashlib.sha256(name.encode()).hexdigest()+'.npz')
                np.savez_compressed(path, image=image, reference=image, strokes=strokes,
                                    sample_metadata=np.asarray(json.dumps(metadata)))
            publish_npz_shards(batch, dataset/'npz')
            stream = io.BytesIO()
            Image.fromarray(np.uint8((1-image[0])*255)).save(stream, format='PNG')
            answer = dict(row=0, png='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode())

            def load(path, _device):
                mode = next(key for key, value in models.items() if value == path)
                return DemoFixture(mode), 16, 3

            with patch('stroke_model.stroke_demo.load_topology', side_effect=load), \
                 patch('stroke_model.stroke_demo.canvas_prompt', return_value=answer) as prompt:
                common = dict(local_root=root/'cache', show=False)
                report = run_stroke_demo(root, 'V2', '테스트 이미지', test_image_name='I_Font_가_2.png',
                                         save_dir=root/'image', **common)
                self.assertEqual(report['source'], 'I_Font_가_2.png')
                self.assertEqual(report['checkpoint'], str(models['partition']))
                self.assertEqual(report['dataset_format'], 'npz')
                with np.load(report['npz']) as result:
                    self.assertEqual(int(result['hard_masks'][:, 8, 8].sum()), 1)
                report = run_stroke_demo(root, '교차점 중복 허용', '직접 쓰기', save_dir=root/'canvas', **common)
                self.assertEqual(report['checkpoint'], str(models['overlap']))
                self.assertEqual(len(prompt.call_args.args[0]), 1)
                self.assertEqual(prompt.call_args.args[0][0]['label'], 'Font_가')
                with np.load(report['npz']) as result:
                    self.assertEqual(int(result['hard_masks'][:, 8, 8].sum()), 2)
                self.assertTrue((root/'canvas'/'prediction_overview.png').is_file())
                with self.assertRaisesRegex(ValueError, '모델 종류'):
                    run_stroke_demo(root, 'V2', '직접 쓰기', checkpoint=models['overlap'], **common)

                # Explicit legacy CSV selection is still available when NPZ TARs exist.
                for filename in ('old.png', 'reference.png'):
                    (dataset/filename).write_bytes(stream.getvalue())
                csv = dataset/'paths.csv'
                csv.write_text('input_path,target_path,target_stroke_paths\n'
                               'old.png,reference.png,reference.png\n', encoding='utf-8')
                report = run_stroke_demo(root, 'V2', '테스트 이미지', csv_path=csv,
                                         save_dir=root/'legacy', **common)
                self.assertEqual((report['dataset_format'], report['source']), ('csv', 'old.png'))
