"""Build the training index from the dataset's numbered PNG filenames."""

import csv
from io import StringIO
import re
import unicodedata


def _pngs_by_stem(paths, role):
    result = {}
    for path in sorted(paths):
        if path.suffix.lower() != '.png':
            continue
        stem = unicodedata.normalize('NFC', path.stem)
        if stem in result:
            raise ValueError(f'{role}: 중복 파일명: {result[stem]} / {path}')
        result[stem] = path
    return result


def _strokes_by_parent(paths, role):
    grouped = {}
    numbered_paths = (path for path in paths if not path.stem.endswith('_c'))
    for stem, path in _pngs_by_stem(numbered_paths, role).items():
        match = re.fullmatch(r'(.+)_([1-9][0-9]*)', stem)
        if not match:
            raise ValueError(f'{role}: <이미지ID>_<획번호>.png 규칙과 맞지 않습니다: {path.name}')
        parent, index = match.groups()
        grouped.setdefault(parent, {})[int(index)] = path
    return grouped


def build_filename_csv(files_by_role, raw):
    """Return a CSV snapshot with unambiguous paths relative to extracted data."""
    variant = re.compile(r'I_(.+)_[0-9]+')
    inputs = _pngs_by_stem(files_by_role['input'], 'input')
    references = _pngs_by_stem(files_by_role['target'], 'target')
    input_strokes = _strokes_by_parent(files_by_role['input_stroke'], 'input_stroke')
    reference_strokes = _strokes_by_parent(files_by_role['target_stroke'], 'target_stroke')
    if not inputs:
        raise ValueError('input 압축 파일에서 PNG를 찾지 못했습니다.')
    for parents, strokes, role in (
        (inputs, input_strokes, 'input_stroke'),
        (references, reference_strokes, 'target_stroke'),
    ):
        orphans = sorted(strokes.keys() - parents.keys())
        if orphans:
            raise ValueError(f'{role}: 대응하는 전체 이미지가 없는 획 묶음 {len(orphans)}개: {orphans[:5]}')

    output = StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(('input_path', 'target_path', 'total_paths', 'input_stroke_paths', 'target_stroke_paths'))

    def relative(path):
        return path.relative_to(raw).as_posix()

    for sample_id, image in sorted(inputs.items()):
        match = variant.fullmatch(sample_id)
        if not match:
            raise ValueError(f'input: I_<기준ID>_<변형번호>.png 규칙과 맞지 않습니다: {image.name}')
        ref_id = match.group(1)
        if ref_id not in references:
            raise ValueError(f'{image.name}: 기준 이미지 {ref_id}.png가 없습니다.')
        expected = reference_strokes.get(ref_id, {})
        annotated = input_strokes.get(sample_id, {})
        stroke_ids = sorted(expected)
        if not stroke_ids or stroke_ids != list(range(1, len(stroke_ids) + 1)):
            raise ValueError(f'{ref_id}.png: 기준 획 번호가 1부터 연속이어야 합니다: {stroke_ids}')
        if annotated.keys() != expected.keys():
            missing = sorted(expected.keys() - annotated.keys())
            extra = sorted(annotated.keys() - expected.keys())
            raise ValueError(f'{image.name}: 정답 획 번호 불일치 (누락: {missing}, 초과: {extra})')

        writer.writerow((relative(image), relative(references[ref_id]), len(stroke_ids),
                         repr([relative(annotated[k]) for k in stroke_ids]),
                         repr([relative(expected[k]) for k in stroke_ids])))
    return output.getvalue().encode('utf-8')
