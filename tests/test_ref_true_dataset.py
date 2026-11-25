import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from stroke_model.npz_dataset import prepare_npz_dataset
from stroke_model.ref_true_dataset import (copy_approved_json, discover_glyphs, generate_approved,
                                           read_coordinates, remove_rejected_json, save_review)
from stroke_model.train_reference import StrokeDataset


def write_glyph(path, shift=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    points = [[(54, 118), (170, 118)], [(110, 50), (110, 124)], [(155, 155)]]
    path.write_text(json.dumps({'strokes': [[dict(x=x+shift, y=y) for x, y in stroke] for stroke in points]}),
                    encoding='utf-8')


class RefTrueDatasetTests(unittest.TestCase):
    def test_approved_copy_is_used_and_rejected_source_is_deleted_without_copy(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            source, destination = base/'jsons', base/'ref is true'
            archive, review = destination/'approved_jsons', destination/'json_review.json'
            accepted, rejected = 'A/초.json', 'A/가.json'
            write_glyph(source/accepted)
            write_glyph(source/rejected)
            _, digest = read_coordinates(source/accepted)
            copy = copy_approved_json(source, archive, accepted, digest)
            self.assertEqual(copy.read_bytes(), (source/accepted).read_bytes())
            write_glyph(source/accepted, 5)
            self.assertEqual(copy_approved_json(source, archive, accepted, digest), copy)
            self.assertEqual(read_coordinates(copy)[1], digest)

            _, rejected_digest = read_coordinates(source/rejected)
            write_glyph(source/rejected, 3)
            with self.assertRaises(ValueError):
                remove_rejected_json(source, rejected, rejected_digest)
            self.assertTrue((source/rejected).exists())
            _, rejected_digest = read_coordinates(source/rejected)
            remove_rejected_json(source, rejected, rejected_digest)
            self.assertFalse((source/rejected).exists())
            self.assertFalse((archive/rejected).exists())

            result = generate_approved(source, review, destination, approved_root=archive, size=64,
                                       scale_range=(1., 1.), aspect_range=(1., 1.), rotation=0.)
            self.assertEqual(result['samples'], 10)
            manifest = prepare_npz_dataset(result['dataset'], local_root=base/'cache', size=64, split='test')
            rows = json.loads(manifest.read_text(encoding='utf-8'))
            for row in rows:
                self.assertEqual(row['char_id'], '초')
                with np.load(row['path'], allow_pickle=False) as sample:
                    metadata = json.loads(sample['sample_metadata'].item())
                    np.testing.assert_array_equal(sample['reference'], sample['image'])
                    # Sample the horizontal stroke away from its ends and intersection.
                    ref_width = int((sample['strokes'][0, :, 23] > .5).sum())
                    input_width = int((sample['target_masks'][0, :, 23] > .5).sum())
                    self.assertLessEqual(abs(ref_width - 2.3), 1)
                    self.assertLessEqual(abs(input_width - 7), 1)
                self.assertEqual(metadata['coordinates_sha256'], digest)

    def test_exact_approvals_produce_ten_samples_per_font_glyph(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            source, destination = base/'jsons', base/'ref is true'
            review = destination/'json_review.json'
            for filename, shift in [('A/초.json', 0), ('A/A_초.json', 4), ('B/초.json', 7),
                                    ('A/가.json', 12), ('A/너.json', 15), ('A/I_초_1.json', 20)]:
                write_glyph(source/filename, shift)
            groups = discover_glyphs(source)
            self.assertEqual(groups['A', '초'], ['A/A_초.json', 'A/초.json'])
            self.assertNotIn(('A', 'I_초_1'), groups)
            self.assertIsNone(generate_approved(source, review, destination)['dataset'])

            for font, glyph, file, approved in [('A', '초', 'A/A_초.json', True), ('B', '초', 'B/초.json', True),
                                                ('A', '가', 'A/가.json', True), ('A', '너', 'A/너.json', False)]:
                _, digest = read_coordinates(source/file)
                save_review(review, font, glyph, file, digest, approved)
            write_glyph(source/'A/가.json', 18)
            result = generate_approved(source, review, destination, size=64)
            self.assertEqual((result['glyphs'], result['samples']), (2, 20))
            self.assertEqual(len(result['stale']), 1)
            self.assertEqual(generate_approved(source, review, destination, size=64), result)

            manifest = prepare_npz_dataset(result['dataset'], local_root=base/'cache', size=64)
            rows = json.loads(manifest.read_text(encoding='utf-8'))
            dataset = StrokeDataset(rows, size=64, include_topology=True, allow_overlap=True)
            self.assertEqual(len(dataset), 20)
            self.assertEqual({row['char_id'] for row in rows}, {'초'})
            by_font = {}
            for index, row in enumerate(rows):
                image, reference, strokes, labels, target, valid = dataset[index]
                self.assertGreater((reference > .5).sum().item(), ((strokes > .5).any(dim=0)).sum().item())
                self.assertTrue(image.equal(target.amax(dim=0, keepdim=True)))
                self.assertTrue(((target > 0) & (target < 1)).any())
                hard_target = target > .5
                self.assertTrue((labels[hard_target.sum(0) > 1] == -1).all())
                self.assertTrue(valid.all())
                with np.load(row['path'], allow_pickle=False) as sample:
                    metadata = json.loads(sample['sample_metadata'].item())
                by_font.setdefault(metadata['font'], []).append(metadata['augmentation'])
            for settings in by_font.values():
                self.assertEqual(len(settings), 10)
                self.assertEqual(len({item['aspect'] for item in settings}), 10)
                self.assertTrue(all(item['scale_x'] != item['scale_y'] for item in settings))

            _, digest = read_coordinates(source/'B/초.json')
            save_review(review, 'B', '초', 'B/초.json', digest, False)
            updated = generate_approved(source, review, destination, size=64)
            self.assertEqual(updated['samples'], 10)
            self.assertNotEqual(updated['dataset'], result['dataset'])
