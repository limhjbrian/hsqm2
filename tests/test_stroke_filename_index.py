import ast
import csv
from io import BytesIO, StringIO
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from stroke_model.prepare_stroke_zips import prepare_stroke_zips
from stroke_model.stroke_filename_index import build_filename_csv


def image_bytes(mask):
    stream = BytesIO()
    Image.fromarray(255 - mask).save(stream, format='PNG')
    return stream.getvalue()


class FilenameIndexTests(unittest.TestCase):
    def test_numeric_stroke_order_and_reject_missing_or_duplicate(self):
        root = Path('raw')
        glyph, sample = 'Font_With_Underscore_가', 'I_Font_With_Underscore_가_12'
        inventory = {
            'input': [root/'i'/f'{sample}.png'],
            'target': [root/'t'/f'{glyph}.png'],
            'input_stroke': [root/'s'/f'{sample}_{n}.png' for n in range(10, 0, -1)] + [root/'s'/f'{sample}_c.png'],
            'target_stroke': [root/'r'/f'{glyph}_{n}.png' for n in range(10, 0, -1)] + [root/'r'/f'{glyph}_c.png'],
        }
        row = next(csv.DictReader(StringIO(build_filename_csv(inventory, root).decode('utf-8'))))
        self.assertEqual(ast.literal_eval(row['input_stroke_paths'])[-2:],
                         [f's/{sample}_9.png', f's/{sample}_10.png'])
        self.assertEqual(ast.literal_eval(row['target_stroke_paths'])[-2:],
                         [f'r/{glyph}_9.png', f'r/{glyph}_10.png'])
        inventory['input_stroke'].pop(1)
        with self.assertRaisesRegex(ValueError, r'누락: \[9\]'):
            build_filename_csv(inventory, root)
        inventory['input_stroke'].append(root/'other_archive'/f'{sample}_10.png')
        with self.assertRaisesRegex(ValueError, '중복'):
            build_filename_csv(inventory, root)
        inventory['input_stroke'].pop()
        inventory['target'].clear()
        with self.assertRaisesRegex(ValueError, '대응하는 전체 이미지가 없는'):
            build_filename_csv(inventory, root)

    def test_prepare_split_archives_without_csv_and_reuse_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, cache = root/'source', root/'cache'
            source.mkdir()
            first = np.zeros((32, 32), dtype=np.uint8)
            first[5:8, 5:23] = 255
            second = np.zeros_like(first)
            second[15:25, 16:19] = 255
            files = {role: {} for role in ('input', 'target', 'input_stroke', 'target_stroke')}
            for char, shift, variants in [('가', 0, [1, 12]), ('나', 3, [1])]:
                glyph = f'Font_{char}'
                masks = [np.roll(m, shift, axis=0) for m in (first, second)]
                files['target'][f'{glyph}.png'] = image_bytes(np.maximum(*masks))
                for index, mask in enumerate(masks, 1):
                    files['target_stroke'][f'{glyph}_{index}.png'] = image_bytes(mask)
                for variant in variants:
                    sample = f'I_{glyph}_{variant}'
                    files['input'][f'{sample}.png'] = image_bytes(np.maximum(*masks))
                    for index, mask in enumerate(masks, 1):
                        files['input_stroke'][f'{sample}_{index}.png'] = image_bytes(mask)
            for role, images in files.items():
                if role == 'target':
                    folder = source/role/'part'
                    folder.mkdir(parents=True)
                    with tarfile.open(folder/'reference_data.tar', 'w') as archive:
                        for name, content in images.items():
                            member = tarfile.TarInfo(name)
                            member.size = len(content)
                            archive.addfile(member, BytesIO(content))
                else:
                    with zipfile.ZipFile(source/f'{role}.zip', 'w') as archive:
                        for name, content in images.items():
                            archive.writestr(f'images/{name}', content)

            manifest = prepare_stroke_zips(source, cache, size=32, independent=True, index_mode='filenames')
            snapshot = manifest.parent/'selected_paths.csv'
            self.assertTrue(snapshot.is_file())
            self.assertFalse(list(source.glob('*.csv')))
            rows = json.loads(manifest.read_text(encoding='utf-8'))
            self.assertEqual(len(rows), 3)
            splits = {Path(row['source_image']).stem: row['split'] for row in rows}
            self.assertEqual(splits['I_Font_가_1'], splits['I_Font_가_12'])
            self.assertNotEqual(splits['I_Font_가_1'], splits['I_Font_나_1'])
            with np.load(rows[0]['path']) as example:
                np.testing.assert_array_equal(example['target_masks'][0] > .5, first > 0)
                np.testing.assert_array_equal(example['target_masks'][1] > .5, second > 0)
            # A later run can read the saved list through the original CSV path.
            restored = prepare_stroke_zips(source, root/'fresh_cache', csv_path=snapshot,
                                           size=32, independent=True)
            restored_rows = json.loads(restored.read_text(encoding='utf-8'))
            self.assertEqual([row['split'] for row in rows], [row['split'] for row in restored_rows])


if __name__ == '__main__':
    unittest.main()
