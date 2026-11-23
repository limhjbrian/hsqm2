"""Let the user choose saved stroke predictions for dataset augmentation."""

import html
import json
from pathlib import Path
from tempfile import NamedTemporaryFile


SELECTION_PATH = Path('dataset/selected_stroke_predictions.json')
RUN_PATTERNS = ('*/*/stroke_demo_*/prediction.npz',
                'reference_v2_data/*/*/stroke_demo_*/prediction.npz',
                '*/*/topo_test_*/row_*.npz')


def find_predictions(root):
    paths = {path for pattern in RUN_PATTERNS for path in root.glob(pattern) if path.is_file()}
    metadata = {}
    batch_rows = {}
    for path in sorted(paths, key=lambda p: (p.parent.name.split('_', 2)[-1], p.name), reverse=True):
        folder = path.parent
        if folder not in metadata:
            try:
                metadata[folder] = json.loads((folder/'inference.json').read_text(encoding='utf-8'))
            except (FileNotFoundError, json.JSONDecodeError):
                metadata[folder] = {}
            batch_rows[folder] = {item['csv_row']: item for item in metadata[folder].get('results', [])}
        report = metadata[folder]
        if folder.name.startswith('topo_test_'):
            row_number = path.stem.removeprefix('row_')
            if not row_number.isdecimal():
                continue
            row = int(row_number)
            result = batch_rows[folder].get(row, {})
            source = Path(result.get('input', '')).name or f'CSV 행 {row}'
            mode = '테스트 이미지'
        else:
            source = Path(report.get('source', '')).name or folder.name
            mode = report.get('input_mode', '')
        yield dict(prediction=path.relative_to(root).as_posix(), source=source, input_mode=mode,
                   model=report.get('model', report.get('output_mode', '')), checkpoint=report.get('checkpoint'))


def _load_selection(path):
    if not path.exists():
        return {}
    return {item['prediction']: item for item in json.loads(path.read_text(encoding='utf-8'))['selected']}


def _save_selection(path, added, removed):
    current = _load_selection(path)
    for key in removed:
        current.pop(key, None)
    current.update(added)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as temporary:
        json.dump({'selected': [current[key] for key in sorted(current)]}, temporary, ensure_ascii=False, indent=2)
        temporary.write('\n')
    staged = Path(temporary.name)
    try:
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)
    return current


def review_stroke_predictions(hsqm):
    import ipywidgets as widgets
    from IPython.display import display

    root = Path(hsqm)
    path = root / SELECTION_PATH
    candidates = {item['prediction']: item for item in find_predictions(root)}
    if not candidates:
        print('Drive에 저장된 추출 결과가 없습니다. 먼저 2번 셀에서 결과를 저장하세요.')
        return
    saved = _load_selection(path)
    checked = set(saved)
    dropdown = widgets.Dropdown(options=[
        (' · '.join(text for text in (item['source'], item['model'], Path(key).parent.name) if text), key)
        for key, item in candidates.items()], description='추출 결과', layout=widgets.Layout(width='95%'))
    selected = widgets.Checkbox(description='학습 후보로 선택', indent=False)
    previous = widgets.Button(description='이전')
    following = widgets.Button(description='다음')
    save = widgets.Button(description='선택 목록 저장', button_style='primary')
    status = widgets.HTML()
    preview = widgets.VBox()
    keys = list(candidates)

    def show_candidate(_=None):
        key = dropdown.value
        item = candidates[key]
        index = keys.index(key)
        previous.disabled = index == 0
        following.disabled = index == len(keys) - 1
        selected.value = key in checked
        details = f"{index + 1}/{len(keys)} · {item['source']} · {item['input_mode']}"
        panels = [widgets.HTML(value=html.escape(details))]
        prediction = root / key
        for suffix in ('overview', 'strokes'):
            image = prediction.with_name(f'{prediction.stem}_{suffix}.png')
            if image.is_file():
                panels.append(widgets.Image(value=image.read_bytes(), format='png',
                                            layout=widgets.Layout(width='100%')))
        if len(panels) == 1:
            panels.append(widgets.HTML(value='저장된 미리보기가 없습니다.'))
        preview.children = panels

    def update_checked(change):
        if change['new']:
            checked.add(dropdown.value)
        else:
            checked.discard(dropdown.value)
        status.value = f'선택 {len(checked)}개 · 저장된 선택 {len(saved)}개'

    def save_checked(_):
        nonlocal saved
        added = {key: candidates[key] for key in checked - saved.keys()}
        try:
            saved = _save_selection(path, added, saved.keys() - checked)
        except OSError as error:
            status.value = html.escape(f'저장 실패: {error}')
            return
        checked.clear()
        checked.update(saved)
        selected.value = dropdown.value in checked
        status.value = f'{len(saved)}개 저장 · {html.escape(str(path))}'

    dropdown.observe(show_candidate, names='value')
    selected.observe(update_checked, names='value')
    previous.on_click(lambda _: setattr(dropdown, 'index', dropdown.index - 1))
    following.on_click(lambda _: setattr(dropdown, 'index', dropdown.index + 1))
    save.on_click(save_checked)
    show_candidate()
    status.value = f'선택 {len(checked)}개'
    display(widgets.VBox([dropdown, widgets.HBox([previous, following, selected, save]), status, preview]))
