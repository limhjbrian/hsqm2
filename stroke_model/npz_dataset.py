"""Export ready-to-train InkSight samples and maintain bounded TAR shards."""

from collections import deque
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import tarfile
from tempfile import TemporaryDirectory
import unicodedata
from uuid import uuid4

import numpy as np

from .convert_reference_csv import convert_csv, existing_path, normalized
from .stroke_filename_index import build_filename_csv

ROLES = ('input', 'target', 'input_stroke', 'target_stroke')
MEMBER = re.compile(r'[0-9a-f]{64}\.npz')
SHARD = re.compile(r'samples-\d{6}\.tar')


def _hash(stream):
    digest = hashlib.sha256()
    while chunk := stream.read(8 * 2**20):
        digest.update(chunk)
    return digest.hexdigest()


def _digest(path):
    with path.open('rb') as stream:
        return _hash(stream)


def _metadata(path):
    with np.load(path, allow_pickle=False) as data:
        return json.loads(data['sample_metadata'].item())


def export_inksight_npz(png_root, output, size=128, expected_samples=None):
    """Validate PNG associations once, on the generation side."""
    root, output = Path(png_root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    files = {role: [p for p in (root / role).rglob('*') if p.is_file()] for role in ROLES}
    if expected_samples is not None:
        expected = {unicodedata.normalize('NFC', name) for name in expected_samples}
        references = {name[2:].rsplit('_', 1)[0] for name in expected}
        parents = {'input': expected, 'input_stroke': expected,
                   'target': references, 'target_stroke': references}
        for role in ROLES:
            files[role] = [p for p in files[role] if p.suffix.lower() == '.png' and
                           unicodedata.normalize('NFC', p.stem.rsplit('_', 1)[0]
                                                 if role.endswith('_stroke') else p.stem) in parents[role]]
        missing = expected - {unicodedata.normalize('NFC', p.stem) for p in files['input']}
        if missing:
            raise ValueError(f'이번 작업에서 생성되지 않은 입력 이미지: {sorted(missing)[:5]}')
    _export_npz_index(build_filename_csv(files, root), root, output, size)
    return output


def _export_npz_index(content, root, output, size, *, skip_invalid=False, skip_missing=False):
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='inksight-export-', dir=output.parent) as temp:
        work = Path(temp)
        index = work / 'index.csv'
        index.write_bytes(content)
        manifest = convert_csv(index, work / 'prepared', size=size, dataset_root=root,
                               split='test', local_only=True, independent=True, portable=True,
                               skip_missing=skip_missing)
        report = json.loads((manifest.parent / 'conversion_report.json').read_text(encoding='utf-8'))
        rows = json.loads(manifest.read_text(encoding='utf-8'))
        skipped = [{'name': normalized(row['input_path']).rsplit('/', 1)[-1],
                    'reason': row['reason'], 'detail': row.get('detail', '')}
                   for row in report['skipped_rows']]
        export = {}
        for row, statistics in zip(rows, report['rows'], strict=True):
            path = Path(row['path'])
            name = _metadata(path)['name']
            if statistics['annotation_coverage'] <= 0:
                skipped.append({'name': name, 'reason': 'no_supervised_foreground'})
                continue
            sample_id = hashlib.sha256(name.encode('utf-8')).hexdigest()
            if sample_id in export:
                raise ValueError(f'동일한 입력 이름이 여러 번 나타납니다: {name}')
            export[sample_id] = path
        invalid = [row for row in skipped if row['reason'] != 'missing_file']
        if invalid and not skip_invalid:
            raise ValueError(f'빈 기준 획 또는 소유 주석이 없는 입력 {len(invalid)}개: {invalid[:5]}')
        if not export:
            raise ValueError('변환 가능한 NPZ 샘플이 없습니다.')
        output.mkdir(parents=True)
        for sample_id, path in export.items():
            shutil.move(path, output / f'{sample_id}.npz')
    return skipped


def migrate_png_archives(source, local_root='/content/hsqm_stroke_zip', size=128,
                         csv_path=None, max_bytes=256 * 2**20, skip_invalid=False, skip_missing=True):
    """Convert the selected Drive dataset once, preserving its original archives."""
    from .prepare_stroke_zips import archive_filename_csv, stage_stroke_archives

    source = existing_path(source)
    cache, raw, signature = stage_stroke_archives(source, local_root)
    missing = []
    if csv_path:
        content = existing_path(csv_path).read_bytes()
    else:
        paths = [Path(path).relative_to(source) for path, _, _ in signature]
        content = archive_filename_csv(raw, paths, missing=missing if skip_missing else None)
    with TemporaryDirectory(prefix='png-to-npz-', dir=cache) as temp:
        output = Path(temp) / 'npz'
        skipped = _export_npz_index(content, raw, output, size, skip_invalid=skip_invalid,
                                    skip_missing=skip_missing)
        count = len(list(output.glob('*.npz')))
        shards = publish_npz_shards(output, source / 'npz', max_bytes=max_bytes)
    return {'samples': count, 'skipped': missing + skipped, 'updated_shards': [p.name for p in shards],
            'destination': str(source / 'npz')}


def _shards(directory):
    return sorted(p for p in directory.glob('samples-*.tar') if SHARD.fullmatch(p.name))


def _members(archive):
    for member in archive:
        if not member.isfile() or not MEMBER.fullmatch(member.name):
            raise ValueError(f'NPZ TAR 멤버 형식이 잘못됐습니다: {member.name}')
        yield member


def _signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _member_bytes(path):
    return 512 + ((path.stat().st_size + 511) // 512) * 512


def publish_npz_shards(batch, destination, max_bytes=256 * 2**20):
    """Merge into existing NPZ TARs with room. Retry identical samples safely."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    pending = {p.name: p for p in sorted(Path(batch).glob('*.npz'))}
    if not pending or any(not MEMBER.fullmatch(name) for name in pending):
        raise ValueError('InkSight에서 생성한 NPZ가 필요합니다.')
    costs = {name: _member_bytes(path) for name, path in pending.items()}
    if max(costs.values()) + tarfile.RECORDSIZE > max_bytes:
        raise ValueError('NPZ 하나가 TAR 최대 크기를 초과합니다. 최대 크기를 늘려 주세요.')
    signatures, seen = {}, set()
    for shard in _shards(destination):
        signatures[shard] = _signature(shard)
        with tarfile.open(shard, 'r:') as archive:
            for member in _members(archive):
                if member.name in seen:
                    raise ValueError(f'기존 TAR에 중복 샘플: {member.name}')
                seen.add(member.name)
                if member.name in pending:
                    with archive.extractfile(member) as stream:
                        identical = _hash(stream) == _digest(pending[member.name])
                    if not identical:
                        raise ValueError(f'같은 이름의 기존 샘플 내용이 다릅니다: {_metadata(pending[member.name])["name"]}')
                    del pending[member.name]

    next_id = max((int(p.stem.split('-')[-1]) for p in signatures), default=0) + 1
    written = []
    names = deque(pending)
    with TemporaryDirectory(prefix='inksight-tar-') as temp:
        while names:
            shard = next((p for p in reversed(signatures) if
                          signatures[p][0] + costs[names[0]] + tarfile.RECORDSIZE <= max_bytes), None)
            old = signatures.pop(shard) if shard else None
            if shard is None:
                shard = destination / f'samples-{next_id:06d}.tar'
                next_id += 1
            remaining = max_bytes - (old[0] if old else 0) - tarfile.RECORDSIZE
            local = Path(temp) / shard.name
            if old:
                shutil.copyfile(shard, local)
            with tarfile.open(local, 'a' if old else 'w', format=tarfile.USTAR_FORMAT) as archive:
                while names and costs[names[0]] <= remaining:
                    name = names.popleft()
                    archive.add(pending[name], arcname=name)
                    remaining -= costs[name]
            partial = destination / f'.{shard.name}.{uuid4().hex}.partial'
            try:
                shutil.copyfile(local, partial)
                if partial.stat().st_size != local.stat().st_size or partial.stat().st_size > max_bytes:
                    raise OSError(f'TAR 복사 또는 크기 확인 실패: {shard.name}')
                if (_signature(shard) if shard.exists() else None) != old:
                    raise RuntimeError(f'다른 작업이 TAR를 변경했습니다: {shard.name}. 다시 시도하세요.')
                partial.replace(shard)
            finally:
                partial.unlink(missing_ok=True)
            written.append(shard)
            local.unlink()
    return written


def prepare_npz_dataset(source, local_root='/content/hsqm_npz', size=128, seed=42,
                        independent=True, split=None, snapshot=None):
    """Stage NPZ bytes and write local manifests without decoding PNGs or masks."""
    source = existing_path(source)
    shards = _shards(existing_path(source / 'npz'))
    if not shards:
        raise FileNotFoundError(f'NPZ TAR가 없습니다: {source / "npz"}')
    if split not in (None, 'test'):
        raise ValueError('split은 None 또는 test여야 합니다.')
    selected = json.loads(Path(snapshot).read_text(encoding='utf-8')) if snapshot else None
    files = {}
    for shard in shards:
        signature = _signature(shard)
        key = hashlib.sha256(repr((str(shard.resolve()), signature)).encode()).hexdigest()[:24]
        cache = Path(local_root) / key
        if not cache.is_dir():
            cache.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=cache.parent) as temp:
                temporary = Path(temp)
                local_tar = temporary / shard.name
                shutil.copyfile(shard, local_tar)
                if _signature(shard) != signature or local_tar.stat().st_size != signature[0]:
                    raise RuntimeError(f'TAR 복사 중 변경됐습니다: {shard.name}')
                samples = temporary / 'samples'
                samples.mkdir()
                with tarfile.open(local_tar, 'r:') as archive:
                    for member in _members(archive):
                        with archive.extractfile(member) as stream, (samples / member.name).open('xb') as dst:
                            shutil.copyfileobj(stream, dst)
                samples.rename(cache)
        for path in cache.glob('*.npz'):
            if path.name in files:
                raise ValueError(f'NPZ TAR 사이에 중복 샘플: {path.name}')
            files[path.name] = path

    output = Path(local_root) / f'run-{uuid4().hex}'
    rows, reports, identities, seen_images = [], [], {}, {}
    for name in (selected if selected is not None else sorted(files)):
        if name not in files:
            raise ValueError(f'학습 당시 NPZ가 없습니다: {name}')
        path = files[name]
        digest = _digest(path)
        if selected is not None and digest != selected[name]:
            raise ValueError(f'학습 당시 NPZ와 내용이 다릅니다: {name}')
        metadata = _metadata(path)
        if metadata['version'] != 1 or metadata['size'] != size:
            raise ValueError(f'{metadata["name"]}: NPZ 크기 또는 버전이 학습 설정과 다릅니다.')
        input_digest = metadata['image_digest']
        group = metadata['group_id'] if independent else 'image:' + input_digest
        if seen_images.setdefault(input_digest, group) != group:
            raise ValueError(f'동일 입력 이미지가 다른 그룹에 있습니다: {metadata["name"]}')
        identities[name] = digest
        reports.append(metadata['report'])
        rows.append(dict(path=str(path.resolve()), group_id=group, source='inksight_npz',
                         char_id=metadata['char_id'], source_image=metadata['name']))

    groups = sorted({row['group_id'] for row in rows})
    if not groups or (split is None and len(groups) < 2):
        raise ValueError('학습과 검증 분리를 위해 최소 2개 그룹이 필요합니다.')
    random.Random(seed).shuffle(groups)
    validation = set(groups[:max(1, round(len(groups) * .2))])
    for row in rows:
        row['split'] = split or ('val' if row['group_id'] in validation else 'train')
    output.mkdir(parents=True)
    report = dict(rows=reports, label_policy='생성 시 저장한 획 마스크와 주석 재사용',
                  split_note='test 전용' if split else '기준 이미지 그룹 분할' if independent else '원본 이미지 분할',
                  skipped_count=0)
    for filename, data in {'manifest.json': rows, 'conversion_report.json': report,
                           'selected_samples.json': identities,
                           '.complete.json': dict(index_mode='npz', independent=independent)}.items():
        (output / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return output / 'manifest.json'
