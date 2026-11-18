"""Use a V2 or overlap checkpoint on one dataset image or drawing."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch

from .canvas_topology import (canvas_prompt, canvas_reference_option, load_canvas_reference,
                              reference_canvas_options, save_canvas_input)
from .convert_reference_csv import Resolver, existing_path
from .infer_reference import inference_columns, load_inference_row, select_input_files
from .infer_topology import load_topology, save_topology_prediction
from .npz_dataset import prepare_npz_dataset


MODEL_RUNS = {
    'V2': ('partition', ('v2_topology_runs/*/best.pt', 'stroke_zip_runs/*/best.pt',
                         'reference_v2_data/*/reference_v2_run_*/best.pt')),
    '교차점 중복 허용': ('overlap', ('v2_topology_overlap_runs/*/best.pt',)),
}


def _select_checkpoint(hsqm, model_name, checkpoint):
    if model_name not in MODEL_RUNS:
        raise ValueError(f'지원하는 모델: {", ".join(MODEL_RUNS)}')
    expected, patterns = MODEL_RUNS[model_name]
    if checkpoint:
        return existing_path(checkpoint), expected
    candidates = [p for pattern in patterns for p in hsqm.glob(pattern) if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f'{model_name}의 best.pt가 없습니다. CHECKPOINT_PATH를 지정하세요.')
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p))), expected


def _image_index(frame, column, name, number):
    if name:
        return select_input_files(frame, column, name)[0] - 1
    if not 1 <= number <= len(frame):
        raise ValueError(f'테스트 이미지 번호는 1~{len(frame)} 중에서 골라 주세요.')
    return number - 1


def _npz_tensors(path, *keys):
    with np.load(path, allow_pickle=False) as sample:
        return [torch.from_numpy(sample[key]).float() for key in keys]


def _npz_canvas_options(rows):
    options, seen = [], set()
    for index, row in enumerate(rows):
        group = row['group_id']
        if group in seen:
            continue
        reference, = _npz_tensors(row['path'], 'reference')
        # Normal InkSight groups already identify the reference by its image hash.
        # For imported writer groups, use the reference itself to keep glyphs distinct.
        if not group.startswith('reference:'):
            group = 'reference:' + hashlib.sha256(reference.numpy().tobytes()).hexdigest()
            if group in seen:
                continue
        seen.add(group)
        name = row['source_image']
        match = re.fullmatch(r'I_(.+)_\d+', Path(name).stem)
        label = match.group(1) if match else row['char_id'] or name
        options.append(canvas_reference_option(index, label, reference.numpy()))
    return sorted(options, key=lambda option: option['label'])


def run_stroke_demo(hsqm, model_name, input_mode, *, checkpoint=None, dataset_root=None,
                    csv_path=None, test_image_name='', test_image_number=1, threshold=.5,
                    global_alignment=True, save_dir=None, show=True, local_root='/content/hsqm_npz'):
    if input_mode not in ('직접 쓰기', '테스트 이미지'):
        raise ValueError('입력 방식은 직접 쓰기 또는 테스트 이미지여야 합니다.')
    hsqm = existing_path(hsqm)
    checkpoint, expected = _select_checkpoint(hsqm, model_name, checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, size, epoch = load_topology(checkpoint, device)
    if model.output_mode != expected:
        raise ValueError(f'CHECKPOINT_PATH의 모델 종류가 선택한 {model_name}과 다릅니다: {checkpoint}')
    print(f'모델: {model_name} / epoch: {epoch} / {checkpoint}', flush=True)

    dataset = existing_path(dataset_root or hsqm / 'dataset' / 'test_dataset')
    is_canvas = input_mode == '직접 쓰기'
    use_npz = not csv_path and any((dataset / 'npz').glob('samples-*.tar'))
    if use_npz:
        manifest = prepare_npz_dataset(dataset, local_root=local_root, size=size, split='test')
        rows = json.loads(manifest.read_text(encoding='utf-8'))
        rows.sort(key=lambda row: (row['source_image'], row['path']))
        if is_canvas:
            answer = canvas_prompt(_npz_canvas_options(rows), 384, 18)
            row = rows[int(answer['row'])]
            reference, strokes = _npz_tensors(row['path'], 'reference', 'strokes')
            source = row['source_image']
        else:
            frame = pd.DataFrame(rows)
            index = _image_index(frame, 'source_image', test_image_name, test_image_number)
            row = rows[index]
            image, reference, strokes = _npz_tensors(row['path'], 'image', 'reference', 'strokes')
            source = row['source_image']
    else:
        if not csv_path:
            csv_path = next((dataset / name for name in ('path (1).csv', 'path.csv', 'paths.csv')
                             if (dataset / name).is_file()), None)
        if not csv_path:
            raise FileNotFoundError(f'{dataset}: 테스트 NPZ TAR 또는 CSV를 찾지 못했습니다.')
        csv_path = existing_path(csv_path)
        frame = pd.read_csv(csv_path)
        columns = inference_columns(frame)
        resolve = Resolver(csv_path, dataset)
        if is_canvas:
            answer = canvas_prompt(reference_canvas_options(frame, columns, resolve, size), 384, 18)
            row = frame.iloc[int(answer['row'])]
            reference, strokes = load_canvas_reference(row, columns, resolve, size)
            source = str(row[columns['reference']])
        else:
            index = _image_index(frame, columns['image'], test_image_name, test_image_number)
            image, reference, strokes, path, _ = load_inference_row(frame.iloc[index], columns, resolve, size)
            source = path.name

    destination = Path(save_dir) if save_dir else checkpoint.parent / ('stroke_demo_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    destination.mkdir(parents=True, exist_ok=False)
    if is_canvas:
        image = save_canvas_input(answer, destination / 'input.png', size)
    model_title = 'V2' if expected == 'partition' else 'Overlap'
    input_title = 'Canvas' if is_canvas else f'Test image {index + 1}'
    prediction = save_topology_prediction(model, device, image, reference, strokes,
                  destination / 'prediction', f'{model_title} | {input_title}', threshold=threshold,
                  show=show, global_alignment=global_alignment)
    report = dict(model=model_name, input_mode=input_mode, checkpoint=str(checkpoint), checkpoint_epoch=epoch,
                  output_mode=expected, global_alignment=global_alignment, source=source,
                  dataset_format='npz' if use_npz else 'csv', **prediction)
    (destination / 'inference.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('저장:', destination, flush=True)
    return report
