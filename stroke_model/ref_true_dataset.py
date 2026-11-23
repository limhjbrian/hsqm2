"""Render manually approved InkSight coordinates as portable training NPZ TARs."""

from collections import defaultdict
import hashlib
import html
from io import BytesIO
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unicodedata
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from .npz_dataset import publish_npz_shards

CANVAS = 224
SUPERSAMPLE = 2
VARIANTS = 10
RENDER_VERSION = 1


def discover_glyphs(root):
    """List original and edited JSON aliases, without augmented inputs."""
    root = Path(root)
    groups = defaultdict(list)
    for path in sorted(root.glob('*/*.json')):
        if not path.is_file():
            continue
        font = unicodedata.normalize('NFC', path.parent.name)
        stem = unicodedata.normalize('NFC', path.stem)
        if not re.fullmatch(r'I_.+_\d+', stem):
            glyph = stem.removeprefix(font + '_')
            if glyph:
                groups[font, glyph].append(path.relative_to(root).as_posix())
    for (font, glyph), files in groups.items():
        files.sort(key=lambda file: not unicodedata.normalize('NFC', Path(file).stem).startswith(font + '_'))
    return dict(groups)


def read_coordinates(path):
    raw = path.read_bytes()
    try:
        strokes = tuple(np.array([(p['x'], p['y']) for p in stroke], dtype=np.float64)
                        for stroke in json.loads(raw)['strokes'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('strokes의 x, y 좌표를 읽을 수 없습니다.') from error
    if not strokes or any(points.ndim != 2 or not len(points) or not np.isfinite(points).all()
                          or points.min() < 0 or points.max() > CANVAS for points in strokes):
        raise ValueError('빈 획이 있거나 좌표가 224×224 캔버스 범위를 벗어났습니다.')
    return strokes, hashlib.sha256(raw).hexdigest()


def load_review(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def save_review(path, font, glyph, file, digest, approved):
    path = Path(path)
    review = load_review(path)
    review.setdefault(font, {})[glyph] = dict(file=file, sha256=digest, approved=approved)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{uuid4().hex}.partial')
    try:
        temporary.write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def copy_approved_json(root, approved_root, file, digest):
    """Keep a verified snapshot of the JSON the user saw."""
    source, destination = Path(root)/file, Path(approved_root)/file
    if destination.is_file():
        return destination
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('JSON이 바뀌었습니다. 다시 열어서 확인하세요.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{uuid4().hex}.partial')
    try:
        temporary.write_bytes(raw)
        if not destination.is_file():
            temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def remove_rejected_json(root, file, digest):
    source = Path(root)/file
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError('JSON이 바뀌었습니다. 다시 열어서 확인하세요.')
    source.unlink()


def _rasterize(strokes, width):
    """Draw rounded strokes on a padded, supersampled canvas."""
    radius = width * SUPERSAMPLE / 2
    pad = int(np.ceil(radius)) + 2
    side = CANVAS * SUPERSAMPLE + 2 * pad
    layers = []
    for points in strokes:
        xy = list(map(tuple, np.rint(points * SUPERSAMPLE + pad).astype(int)))
        layer = Image.new('L', (side, side))
        draw = ImageDraw.Draw(layer)
        if len(xy) > 1:
            draw.line(xy, width=max(1, round(2 * radius)), fill=255, joint='curve')
        for x, y in xy:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
        layers.append(layer)
    return layers, pad


def _warp(layers, pad, matrix, size):
    side = CANVAS * SUPERSAMPLE
    inverse = np.linalg.inv(matrix)
    center = np.full(2, side / 2)
    coefficients = np.column_stack((inverse, center + pad - inverse @ center)).ravel()
    out = [layer.transform((side, side), Image.Transform.AFFINE, coefficients, Image.Resampling.BILINEAR)
           .resize((size, size), Image.Resampling.LANCZOS) for layer in layers]
    return np.stack(out).astype(np.float32) / 255


def _variant_matrix(rng, bounds, scale_range, aspect_range, rotation):
    scale = rng.uniform(*scale_range)
    aspect = rng.uniform(*aspect_range)
    degrees = rng.uniform(-rotation, rotation)
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    scale_x, scale_y = scale * np.sqrt(aspect), scale / np.sqrt(aspect)
    matrix = np.array([[c, -s], [s, c]]) @ np.diag([scale_x, scale_y])
    extent = np.abs((bounds - CANVAS / 2) @ matrix.T).max()
    fit = min(1., (CANVAS / 2 - 2) / max(extent, 1.))
    return matrix * fit, dict(scale=float(scale * fit), scale_x=float(scale_x * fit), scale_y=float(scale_y * fit),
                              aspect=float(aspect), rotation=float(degrees))


def _write_glyph(batch, font, glyph, coordinates, digest, *, size, width, scale_range, aspect_range,
                 rotation, seed):
    reference_layers, reference_pad = _rasterize(coordinates, 4.)
    strokes = _warp(reference_layers, reference_pad, np.eye(2), size)
    if not (strokes > .5).any(axis=(1, 2)).all():
        raise ValueError(f'{font}/{glyph}: {size}px 기준 마스크에 사라지는 획이 있습니다.')
    reference = strokes.max(0, keepdims=True)
    group = 'reference:' + hashlib.sha256(reference.tobytes()).hexdigest()
    canvas_width = width * CANVAS / size
    layers, pad = _rasterize(coordinates, canvas_width)
    points = np.concatenate(coordinates)
    minimum, maximum = points.min(0) - canvas_width / 2, points.max(0) + canvas_width / 2
    bounds = np.array([(x, y) for x in (minimum[0], maximum[0]) for y in (minimum[1], maximum[1])])
    key = hashlib.sha256(json.dumps([font, glyph, digest, seed], ensure_ascii=False).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(key[:8], 'big'))
    for index in range(VARIANTS):
        matrix, augmentation = _variant_matrix(rng, bounds, scale_range, aspect_range, rotation)
        thick = _warp(layers, pad, matrix, size)
        image = thick.max(0, keepdims=True)
        target = thick > .5
        if not target.any(axis=(1, 2)).all():
            raise ValueError(f'{font}/{glyph}: 변형 {index + 1}에서 사라진 획이 있습니다. 굵기나 배율을 조정하세요.')
        counts = target.sum(0)
        exclusive = counts == 1
        labels = np.full(counts.shape, -1, dtype=np.int64)
        labels[exclusive] = target.argmax(0)[exclusive]
        ink_count = int((counts > 0).sum())
        name = f'I_{font}_{glyph}_{index + 1}.png'
        metadata = dict(version=1, size=size, name=name, char_id=glyph, group_id=group,
                        image_digest=hashlib.sha256(image.tobytes()).hexdigest(), font=font,
                        coordinates_sha256=digest, augmentation=augmentation,
                        report=dict(foreground_pixels=ink_count, annotation_coverage=float(exclusive.sum()/ink_count),
                                    overlap_pixels=int((counts > 1).sum()), missing_pixels=0))
        filename = hashlib.sha256(name.encode('utf-8')).hexdigest() + '.npz'
        np.savez_compressed(batch/filename, image=image, reference=reference, strokes=strokes,
                            target_masks=target.astype(np.float32), labels=labels, valid_pixels=np.ones_like(image),
                            sample_metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)))


def generate_approved(root, review_path, destination, *, approved_root=None, size=128, width=7., scale_range=(.85, 1.15),
                      aspect_range=(.8, 1.25), rotation=12., seed=42, max_bytes=256 * 2**20):
    """Publish one reusable snapshot containing exactly the current approvals."""
    if size < 16 or width <= 0 or width >= size / 2 or rotation < 0:
        raise ValueError('해상도, 획 굵기 또는 회전 범위를 확인하세요.')
    if any(not 0 < low <= high for low, high in (scale_range, aspect_range)):
        raise ValueError('크기와 가로세로 비율은 양수 범위여야 합니다.')
    root, destination = Path(root), Path(destination)
    groups = discover_glyphs(root)
    approved, stale = {}, []
    if approved_root is not None:
        approved_root = Path(approved_root)
        for (font, glyph), files in sorted(discover_glyphs(approved_root).items()):
            try:
                approved[font, glyph] = read_coordinates(approved_root/files[0])
            except (OSError, ValueError) as error:
                stale.append(f'{font}/{glyph}: {error}')
    for font, review in sorted(load_review(review_path).items()):
        for glyph, record in sorted(review.items()):
            if not record['approved'] or (font, glyph) in approved:
                continue
            file = record['file']
            try:
                if file not in groups.get((font, glyph), ()):
                    raise FileNotFoundError(file)
                points, digest = read_coordinates(root/file)
                if digest != record['sha256']:
                    raise ValueError('검수 이후 JSON 내용 변경')
            except (OSError, ValueError) as error:
                stale.append(f'{font}/{glyph}: {error}')
                continue
            approved[font, glyph] = (points, digest)
    if not approved:
        return dict(dataset=None, glyphs=0, samples=0, stale=stale)
    settings = dict(size=size, width=width, scale_range=scale_range, aspect_range=aspect_range,
                    rotation=rotation, seed=seed)
    identity = [RENDER_VERSION, VARIANTS, settings, max_bytes,
                [(font, glyph, digest) for (font, glyph), (_, digest) in approved.items()]]
    snapshot = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    dataset = destination / f'snapshot-{snapshot[:20]}'
    if not dataset.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='ref-true-npz-') as local, \
                TemporaryDirectory(prefix='.ref-true-', dir=destination) as pending:
            batch, staging = Path(local), Path(pending)/'dataset'
            for (font, glyph), (points, digest) in approved.items():
                _write_glyph(batch, font, glyph, points, digest, **settings)
            publish_npz_shards(batch, staging/'npz', max_bytes=max_bytes)
            try:
                staging.rename(dataset)
            except OSError:
                if not dataset.is_dir():
                    raise
    return dict(dataset=str(dataset), glyphs=len(approved), samples=VARIANTS*len(approved), stale=stale)


def _preview(coordinates, width, size):
    from matplotlib.figure import Figure

    layers, pad = _rasterize(coordinates, width * CANVAS / size)
    masks = _warp(layers, pad, np.eye(2), size)
    rows = (len(masks) + 4) // 5
    figure = Figure(figsize=(10, 3 + 1.75 * rows), layout='constrained')
    grid = figure.add_gridspec(1 + rows, 5)
    paths = figure.add_subplot(grid[0, :3])
    paths.set(xlim=(0, CANVAS), ylim=(CANVAS, 0), aspect='equal', title='Coordinates and stroke order')
    paths.axis('off')
    for index, points in enumerate(coordinates):
        color = f'C{index % 10}'
        paths.plot(*points.T, marker='.', color=color)
        paths.annotate(str(index + 1), points[0], color=color, fontweight='bold')
    merged = figure.add_subplot(grid[0, 3:])
    merged.imshow(masks.max(0), cmap='gray_r', vmin=0, vmax=1)
    merged.set_title('Combined thick strokes')
    merged.axis('off')
    for index, mask in enumerate(masks):
        axis = figure.add_subplot(grid[1 + index//5, index % 5])
        axis.imshow(mask, cmap='gray_r', vmin=0, vmax=1)
        axis.set_title(str(index + 1))
        axis.axis('off')
    stream = BytesIO()
    figure.savefig(stream, format='png', dpi=100)
    return stream.getvalue()


def review_coordinates(root, review_path, *, approved_root, width=7., size=128):
    import ipywidgets as widgets
    from IPython.display import display

    root, approved_root = Path(root), Path(approved_root)
    if root.resolve() == approved_root.resolve():
        raise ValueError('원본과 통과 JSON 폴더는 달라야 합니다.')
    groups = discover_glyphs(root)
    archive = discover_glyphs(approved_root)
    done = {key for key, files in groups.items()
            if {unicodedata.normalize('NFC', file) for file in files}.intersection(
                unicodedata.normalize('NFC', file) for file in archive.get(key, ()))}
    groups = {key: files for key, files in groups.items() if key not in done}
    if not groups:
        print(f'검수할 JSON이 없습니다. 통과 폴더에서 확인된 글자: {len(done)}개')
        return
    by_font = defaultdict(list)
    for font_name, character in sorted(groups):
        by_font[font_name].append(character)
    font = widgets.Dropdown(options=list(by_font), description='폰트')
    glyph = widgets.Dropdown(description='글자')
    files = widgets.Dropdown(description='JSON')
    previous = widgets.Button(description='이전 글자')
    approve = widgets.Button(description='통과 · 다음', button_style='success')
    reject = widgets.Button(description='제외 · 원본 삭제', button_style='danger')
    status = widgets.HTML()
    preview = widgets.Image(format='png', layout=widgets.Layout(width='100%'))
    digest = None

    def show(_=None):
        nonlocal digest
        digest = None
        previous.disabled = glyph.index in (None, 0)
        approve.disabled = reject.disabled = True
        preview.value = b''
        if not files.value:
            return
        source = root/files.value
        try:
            coordinates, digest = read_coordinates(source)
            preview.value = _preview(coordinates, width, size)
        except ValueError as error:
            try:
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
            except OSError:
                pass
            reject.disabled = digest is None
            status.value = html.escape(f'검수 불가: {error}')
            return
        except OSError as error:
            status.value = html.escape(f'검수 불가: {error}')
            return
        approve.disabled = reject.disabled = False
        status.value = html.escape(f'{font.value} / {glyph.value} · {len(coordinates)}획 · 기본 통과')

    def update_files(_=None):
        if glyph.value not in by_font[font.value]:
            return
        record = load_review(review_path).get(font.value, {}).get(glyph.value, {})
        options = groups[font.value, glyph.value]
        files.unobserve(show, names='value')
        try:
            files.options = [(Path(file).name, file) for file in options]
            files.value = record['file'] if record.get('file') in options else options[0]
        finally:
            files.observe(show, names='value')
        show()

    def update_glyphs(_=None):
        glyph.unobserve(update_files, names='value')
        try:
            glyph.options = by_font[font.value]
            if glyph.value is None:
                glyph.index = 0
        finally:
            glyph.observe(update_files, names='value')
        update_files()

    def mark(approved):
        if digest is None or not files.value:
            return
        key = font.value, glyph.value
        file = files.value
        try:
            if approved:
                copy_approved_json(root, approved_root, file, digest)
            else:
                remove_rejected_json(root, file, digest)
        except (OSError, ValueError) as error:
            status.value = html.escape(f'검수 처리 실패: {error}')
            return
        if not approved:
            groups[key] = [alias for alias in groups[key] if alias != file and (root/alias).is_file()]
            if groups[key]:
                update_files()
                return
        index = glyph.index
        by_font[font.value].remove(glyph.value)
        if by_font[font.value]:
            glyph.unobserve(update_files, names='value')
            try:
                glyph.options = by_font[font.value]
                glyph.index = min(index, len(glyph.options) - 1)
            finally:
                glyph.observe(update_files, names='value')
            update_files()
            return
        del by_font[font.value]
        if by_font:
            font.options = list(by_font)
            return
        font.disabled = glyph.disabled = files.disabled = previous.disabled = approve.disabled = reject.disabled = True
        preview.value = b''
        status.value = '검수를 마쳤습니다. 생성 셀을 실행하세요.'

    font.observe(update_glyphs, names='value')
    glyph.observe(update_files, names='value')
    files.observe(show, names='value')
    previous.on_click(lambda _: setattr(glyph, 'index', glyph.index - 1))
    approve.on_click(lambda _: mark(True))
    reject.on_click(lambda _: mark(False))
    update_glyphs()
    display(widgets.VBox([widgets.HBox([font, glyph]), files,
                          widgets.HBox([previous, approve, reject]), status, preview]))
