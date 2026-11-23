"""Review target-glyph predictions and export manually confirmed training samples."""

from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
import random
import re
import shutil
from tempfile import TemporaryDirectory
import unicodedata
from uuid import uuid4

import numpy as np


OUTPUT_FOLDER = 'reviewed_stroke_npz'
INPUT_NAME = re.compile(r'I_(.+)_([^_]+)_\d+\.png', re.IGNORECASE)


def index_training_rows(rows):
    """Index only the train partition, preserving the exact font and glyph."""
    groups = defaultdict(list)
    for row in rows:
        name = unicodedata.normalize('NFC', row['source_image'])
        match = INPUT_NAME.fullmatch(name)
        if row['split'] == 'train' and match:
            groups[match.groups()].append(row)
    return dict(groups)


def _read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def publish_reviewed_sample(destination, *, prediction=None, training_row):
    """Save approved predictions in the InkSight layout or copy a real training sample."""
    original = Path(training_row['path'])
    if prediction is not None:
        image = prediction['image'].astype(np.float32)
        reference = prediction['reference'].astype(np.float32)
        strokes = prediction['reference_strokes'].astype(np.float32)
        foreground = image[0] > .5
        targets = prediction['hard_masks'].astype(bool) & foreground
        counts = targets.sum(0)
        if not counts.any():
            raise ValueError('정답 획으로 저장할 픽셀이 없습니다.')
        exclusive = counts == 1
        labels = np.full(foreground.shape, -1, dtype=np.int64)
        labels[exclusive] = targets.argmax(0)[exclusive]
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        identity = hashlib.sha256(reference.tobytes() + strokes.tobytes() + targets.tobytes()).hexdigest()
        reference_name = training_row['source_image'].removeprefix('I_').rsplit('_', 1)[0]
        name = f'reviewed_{reference_name}_{identity}.png'
        metadata = dict(version=1, size=image.shape[-1], name=name, char_id=training_row['char_id'],
                        group_id=training_row['group_id'], image_digest=digest,
                        report=dict(foreground_pixels=int(foreground.sum()),
                                    annotation_coverage=float(exclusive.sum() / foreground.sum()),
                                    overlap_pixels=int((counts > 1).sum()),
                                    missing_pixels=int((foreground & (counts == 0)).sum())))
        arrays = dict(image=image, reference=reference, strokes=strokes, labels=labels,
                      target_masks=targets.astype(np.float32), valid_pixels=np.ones_like(image),
                      sample_metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)))
        filename = hashlib.sha256(name.encode('utf-8')).hexdigest() + '.npz'
    else:
        filename = original.name

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / filename
    if path.exists():
        return path
    partial_path = destination / f'.{uuid4().hex}.partial'
    try:
        if prediction is None:
            shutil.copyfile(original, partial_path)
        else:
            with partial_path.open('wb') as stream:
                np.savez_compressed(stream, **arrays)
        partial_path.replace(path)
    finally:
        partial_path.unlink(missing_ok=True)
    return path


def review_training_candidates(hsqm, model_name, *, checkpoint='', training_root='', seed=42,
                               threshold=.5, global_alignment=True):
    import ipywidgets as widgets
    from IPython.display import display
    import torch

    from .convert_reference_csv import existing_path
    from .infer_topology import load_topology, preview_topology, save_topology_prediction
    from .npz_dataset import prepare_npz_dataset
    from .stroke_demo import _select_checkpoint

    root = existing_path(hsqm)
    destination = root / 'dataset' / OUTPUT_FOLDER
    checkpoint, expected_mode = _select_checkpoint(root, model_name, checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, size, epoch = load_topology(checkpoint, device)
    if model.output_mode != expected_mode:
        raise ValueError('선택한 모델과 체크포인트의 종류가 다릅니다.')
    source = training_root or root / 'dataset' / 'automated_train_dataset'
    print(f'{model_name} · epoch {epoch} · {device} | 학습 NPZ 준비 중', flush=True)
    manifest = prepare_npz_dataset(source, size=size, seed=seed)
    groups = index_training_rows(json.loads(manifest.read_text(encoding='utf-8')))
    if not groups:
        raise ValueError('학습 분할에서 I_<폰트>_<글자>_<번호>.png 이름의 샘플을 찾지 못했습니다.')

    glyphs_by_font = defaultdict(list)
    for font_name, character in sorted(groups):
        glyphs_by_font[font_name].append(character)
    font = widgets.Dropdown(options=list(glyphs_by_font), description='폰트')
    glyph = widgets.Dropdown(description='글자')
    run = widgets.Button(description='Target 추출', button_style='info')
    fallback = widgets.Button(description='학습 샘플 보기', disabled=True)
    confirm = widgets.Button(description='확인 · NPZ 저장', button_style='success', disabled=True)
    status = widgets.HTML()
    preview = widgets.VBox()
    displayed = None
    remaining = []

    def message(text):
        status.value = html.escape(str(text))

    def reset(_=None):
        nonlocal displayed
        displayed = None
        fallback.disabled = confirm.disabled = True
        preview.children = ()
        message('폰트와 글자를 고른 뒤 Target 추출을 누르세요.')

    def update_glyphs(_=None):
        glyph.options = glyphs_by_font[font.value]
        reset()

    def show(prefix):
        preview.children = tuple(widgets.Image(
            value=prefix.with_name(f'{prefix.name}_{suffix}.png').read_bytes(), format='png',
            layout=widgets.Layout(width='100%')) for suffix in ('overview', 'strokes'))

    def show_target(_):
        nonlocal displayed, remaining
        reset()
        message('Target에 모델을 실행 중입니다.')
        rows = groups[font.value, glyph.value]
        row = rows[0]
        try:
            baseline = _read_npz(row['path'])
            reference, strokes = (torch.from_numpy(baseline[key]).float() for key in ('reference', 'strokes'))
            with TemporaryDirectory() as temp:
                prefix = Path(temp) / 'target'
                summary = save_topology_prediction(model, device, reference, reference, strokes, prefix,
                                                   'Target prediction', threshold=threshold, show=False,
                                                   global_alignment=global_alignment)
                prediction = _read_npz(summary['npz'])
                show(prefix)
        except (OSError, ValueError, RuntimeError) as error:
            message(f'Target 추출 실패: {error}')
            return
        remaining = random.sample(rows, len(rows))
        displayed = (row, prediction)
        fallback.disabled = False
        confirm.disabled = not prediction['hard_masks'].any()
        message(f'{font.value} · {glyph.value} · Target 모델 결과')

    def show_fallback(_):
        nonlocal displayed
        row = remaining.pop()
        fallback.disabled = not remaining
        try:
            sample = _read_npz(row['path'])
            masks = sample['target_masks']
            with TemporaryDirectory() as temp:
                prefix = Path(temp) / 'training'
                preview_topology(sample['image'], sample['reference'], sample['strokes'], masks, masks,
                                 prefix, 'Training sample annotations', .5, False)
                show(prefix)
        except (OSError, ValueError, KeyError) as error:
            message(f'학습 샘플 읽기 실패: {error}')
            return
        displayed = (row, None)
        confirm.disabled = False
        message(f"학습 샘플의 기존 정답 · {row['source_image']} · 남은 후보 {len(remaining)}개")

    def save_displayed(_):
        row, prediction = displayed
        try:
            path = publish_reviewed_sample(destination, prediction=prediction, training_row=row)
        except (OSError, ValueError) as error:
            message(f'저장 실패: {error}')
            return
        confirm.disabled = True
        message(f"저장 완료 · {path}")

    font.observe(update_glyphs, names='value')
    glyph.observe(reset, names='value')
    run.on_click(show_target)
    fallback.on_click(show_fallback)
    confirm.on_click(save_displayed)
    update_glyphs()
    display(widgets.VBox([widgets.HBox([font, glyph]), widgets.HBox([run, fallback, confirm]), status, preview]))
