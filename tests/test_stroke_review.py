import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from stroke_model.stroke_review import SELECTION_PATH, _load_selection, _save_selection, find_predictions


class StrokeReviewTests(unittest.TestCase):
    def test_saved_predictions_keep_source_and_relative_path(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = root/'v2_topology_runs'/'run'
            demo = run/'stroke_demo_20261118_100230'
            batch = run/'topo_test_20261104_120000'
            demo.mkdir(parents=True)
            batch.mkdir()
            (demo/'prediction.npz').touch()
            (batch/'row_00038.npz').touch()
            (batch/'row_manifest.npz').touch()
            (demo/'inference.json').write_text(json.dumps({'source': 'I_초_1.png', 'model': 'V2'}), encoding='utf-8')
            (batch/'inference.json').write_text(json.dumps({'results': [{'csv_row': 38, 'input': '/data/I_가_1.png'}]}),
                                               encoding='utf-8')
            items = list(find_predictions(root))
            self.assertEqual([item['source'] for item in items], ['I_초_1.png', 'I_가_1.png'])
            self.assertEqual(items[0]['prediction'], (demo/'prediction.npz').relative_to(root).as_posix())

    def test_saved_choices_can_be_reopened_and_individually_removed(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/SELECTION_PATH
            sample = {'prediction': 'a.npz', 'source': '초'}
            other = {'prediction': 'b.npz', 'source': '가'}
            self.assertEqual(_load_selection(path), {})
            _save_selection(path, {'a.npz': sample}, set())
            _save_selection(path, {'b.npz': other}, set())
            self.assertEqual(_load_selection(path), {'a.npz': sample, 'b.npz': other})
            _save_selection(path, {}, {'a.npz'})
            self.assertEqual(_load_selection(path), {'b.npz': other})
