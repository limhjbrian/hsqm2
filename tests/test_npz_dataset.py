import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np
from PIL import Image

from stroke_model.npz_dataset import (export_inksight_npz, migrate_png_archives,
                                      prepare_npz_dataset, publish_npz_shards)


class NpzDatasetTests(unittest.TestCase):
    def test_migrate_legacy_tars_into_existing_shard_and_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pngs = root / 'png'
            source = root / 'dataset'
            roles = ('input', 'target', 'input_stroke', 'target_stroke')
            for role in roles:
                (pngs / role).mkdir(parents=True)
                (source / role / 'nested').mkdir(parents=True)
            for i in range(3):
                glyph, sample = f'Font_{i}', f'I_Font_{i}_1'
                ink = np.full((24, 24), 255, np.uint8)
                ink[2 + 4*i:4 + 4*i, 4:17] = 0
                for role, name in zip(roles, (sample, glyph, sample + '_1', glyph + '_1')):
                    pixels = np.full_like(ink, 255) if role == 'target_stroke' and i == 2 else ink
                    Image.fromarray(pixels).save(pngs / role / f'{name}.png')
            archives = []
            for role in roles:
                path = source / role / 'nested' / 'legacy.tar'
                with tarfile.open(path, 'w') as archive:
                    for image in (pngs / role).glob('*.png'):
                        archive.add(image, arcname=image.name)
                archives.append((path, path.read_bytes()))

            batch = export_inksight_npz(pngs, root / 'existing', size=24,
                                       expected_samples={'I_Font_0_1'})
            shard, = publish_npz_shards(batch, source / 'npz')
            with self.assertRaisesRegex(ValueError, '빈 기준 획'):
                migrate_png_archives(source, root / 'stage', size=24)
            with tarfile.open(shard) as archive:
                self.assertEqual(len(archive.getnames()), 1)
            result = migrate_png_archives(source, root / 'stage', size=24, skip_invalid=True)
            self.assertEqual(result['samples'], 2)
            self.assertEqual(result['updated_shards'], [shard.name])
            self.assertEqual([item['name'] for item in result['skipped']], ['I_Font_2_1.png'])
            manifest = prepare_npz_dataset(source, root / 'cache', size=24)
            rows = json.loads(manifest.read_text(encoding='utf-8'))
            self.assertEqual({row['source_image'] for row in rows}, {'I_Font_0_1.png', 'I_Font_1_1.png'})
            self.assertEqual(migrate_png_archives(source, root / 'stage', size=24,
                                                 skip_invalid=True)['updated_shards'], [])
            self.assertTrue(all(path.read_bytes() == data for path, data in archives))

    def test_export_merge_rollover_and_restore_old_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pngs = root / 'png'
            for role in ('input', 'target', 'input_stroke', 'target_stroke'):
                (pngs / role).mkdir(parents=True)

            for char, offset in enumerate(range(0, 12, 3)):
                ink = np.full((24, 24), 255, dtype=np.uint8)
                ink[2 + offset:4 + offset, 4:17] = 0
                glyph = f'Font_{char}'
                sample = f'I_{glyph}_1'
                for role, filename in [('input', sample), ('target', glyph),
                                       ('input_stroke', sample + '_1'), ('target_stroke', glyph + '_1')]:
                    Image.fromarray(ink).save(pngs / role / (filename + '.png'))

            source = root / 'dataset'
            destination = source / 'npz'
            batch = export_inksight_npz(pngs, root / 'batch', size=24,
                                       expected_samples={'I_Font_0_1', 'I_Font_1_1'})
            first, = publish_npz_shards(batch, destination, max_bytes=32 * 1024)
            self.assertEqual(publish_npz_shards(batch, destination, max_bytes=32 * 1024), [])
            original = prepare_npz_dataset(source, root / 'cache', size=24)
            identities = original.parent / 'selected_samples.json'
            rows = json.loads(original.read_text(encoding='utf-8'))
            self.assertEqual(len(rows), 2)
            with np.load(rows[0]['path'], allow_pickle=False) as sample:
                self.assertEqual(sample['target_masks'].shape, (1, 24, 24))
                np.testing.assert_array_equal(sample['labels'] == 0, sample['image'][0] > .5)

            addition = export_inksight_npz(pngs, root / 'addition', size=24,
                                          expected_samples={'I_Font_2_1'})
            self.assertEqual(publish_npz_shards(addition, destination, max_bytes=32 * 1024), [first])
            with tarfile.open(first) as archive:
                self.assertEqual(len(archive.getnames()), 3)
            restored = prepare_npz_dataset(source, root / 'cache', size=24, snapshot=identities)
            keys = lambda data: [(r['source_image'], r['split']) for r in data]
            self.assertEqual(keys(rows), keys(json.loads(restored.read_text(encoding='utf-8'))))

            small = root / 'small'
            a, = publish_npz_shards(batch, small, max_bytes=20 * 1024)
            b, = publish_npz_shards(addition, small, max_bytes=20 * 1024)
            self.assertNotEqual(a, b)
            self.assertLessEqual(max(p.stat().st_size for p in (a, b)), 20 * 1024)

            npz, = addition.glob('*.npz')
            npz.write_bytes(npz.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, '내용이 다릅니다'):
                publish_npz_shards(addition, destination)
            self.assertEqual(first.stat().st_size, 10240)
