"""Build the runnable Colab topology extractor training notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(overlap=False):
    cells = []

    def add(kind, source):
        if kind == 'code':
            compile(source, '<colab>', 'exec')
        cell = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    legacy_intro = '''# Topo 모델 학습 — 기존 V2 + 정션·방향 관계 보조 손실
GPU 런타임에서 위부터 실행하세요. 입력 글자의 실제 획 정답으로 학습합니다.
기존 V2의 동결 Swin-T/FPN + 국소 flow + 공유 추출 헤드 + softmax를 그대로 사용합니다.
전역 affine/ConvLSTM/독립 sigmoid는 사용하지 않습니다. 모든 soft 획의 합은 원본 잉크와 같습니다.
정션 손실은 예측과 실제 정답의 구조 차이를 비교합니다. 실제 정답에 있는 가지는 제거 대상이 아닙니다.
방향 관계 손실은 AAF(ECCV 2018)의 같은/다른 영역 관계 감독에 착안한 경량 변형입니다.
실제 입력의 주석 구간에서 국소 방향을 구해 같은 획의 연결과 다른 획과의 분리를 함께 감독합니다.
직선 강제나 기준 픽셀 복사는 없으며, 곡선/꺾임은 국소 방향으로 처리합니다.
AAF 전체 재현(적대적 반경 학습)은 아닙니다. 사용자 체크포인트의 실제 개선과 GPU 속도는 검증 전입니다.
기존 After 모델을 보완하려면 아래 INIT_CHECKPOINT에 그 best.pt를 지정하고 EPOCHS=3,
LEARNING_RATE=1e-4로 먼저 비교하세요. 결과는 새 폴더에 저장하고 원본 best.pt는 수정하지 않습니다.

데이터: 공유 Drive의 `2026 자율연구/HSQM/dataset/automated_train_dataset` 안
아래와 같은 폴더별 TAR 묶음을 사용합니다. 각 TAR은 독립 압축 파일입니다.
```
automated_train_dataset/
  input_stroke/  # 폴더와 하위 폴더의 압축 파일 전체
  input/
  target_stroke/
  target/
```
파일명·번호·개수를 제한하지 않고 하위 폴더까지 TAR/ZIP을 전부 찾습니다.
TAR.GZ/TGZ/TAR.BZ2/TAR.XZ도 지원합니다. 기존 네 ZIP 형식도 지원합니다.
기본 `INDEX_MODE='npz'`는 생성 노트북이 올린 `npz/samples-*.tar`를 사용합니다.
기존 PNG 압축 파일은 `INDEX_MODE='filenames'`로 읽을 수 있습니다.
`INDEX_MODE='csv'`는 `path (1).csv`를 우선 사용하고, 없으면 `path.csv`/`paths.csv`를 검색합니다.
입력=input, 기준=target, 정답=input_stroke, 기준 획=target_stroke입니다.
얇은 정답은 기존 V2처럼 주석된 픽셀만 CE/Dice로 감독합니다. 그 밖의 원본 잉크를 배경으로 벌주지 않습니다.
실제 정답 획 마스크는 정션 감독과 선택적 약한 영역 주석 생성에 사용합니다. CSV는 완전한 실제 획 경로를 제공한다고 가정합니다.
선택적으로 실제 입력 정답선에서 보수적으로 확장한 약한 영역 주석과 이웃 일관성 손실을 사용합니다.
이는 완전한 정답이 아니며 교차점/모호한 경계/seed 없는 잉크는 감독하지 않습니다.
기준 획이 비어 있는 샘플은 전체 행을 제외하고 계속 진행하며 `skipped_samples.csv`에 기록합니다.
기준/정답 획 순서와 극성은 2번 셀에서 확인하세요.

기본은 ImageNet Swin 사전학습에서 새 학습입니다. 기존 V2 `best.pt`를 지정하면
전체 V2를 초기화합니다(optimizer/epoch 복원 아님). 정션 성능 개선과 GPU 속도는 아직 검증 전입니다.
'''
    add('markdown', '''# 교차점_중복_허용

기존 Topo의 Swin-T/FPN, 기준 획 힌트, 학습 설정과 정션·방향 보조 손실을 사용합니다.
획별 sigmoid로 바꿔 한 픽셀이 두 획에 동시에 속할 수 있습니다.
InkSight 생성 노트북의 학습용 `automated_train_dataset/npz/samples-*.tar`를 기본으로 읽습니다.
NPZ 정답에 표시된 겹침만 공유 정답으로 쓰며, 주석이 없는 잉크는 기본 손실에서 제외합니다.

GPU 런타임에서 위부터 실행하세요. 기존 Topo와 비교하려면 같은 NPZ, seed, 초기 체크포인트를 사용하세요.
정션·방향·영역 보조 설정은 아래 학습 셀에 있습니다. 출력은 `v2_topology_overlap_runs`에 저장합니다.
마지막 셀은 주석 Dice, 공유 픽셀 복원율, 단일 획에 생긴 불필요한 중복률과 예측 그림을 표시합니다.
''' if overlap else legacy_intro)
    add('code', '''from pathlib import Path
import json, subprocess, sys, tempfile
root = Path(tempfile.mkdtemp(prefix='hsqm_topology_', dir='/content'))
subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'main',
                'https://github.com/limhjbrian/hsqm2.git', str(root)], check=True)
commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(root/'requirements.txt')], check=True)
sys.path.insert(0, str(root))
for key in list(sys.modules):
    if key == 'stroke_model' or key.startswith('stroke_model.'):
        del sys.modules[key]
import torch
if not torch.cuda.is_available():
    raise RuntimeError('런타임 유형을 GPU로 변경한 뒤 다시 실행하세요.')
print('GPU:', torch.cuda.get_device_name(), '\\n코드 버전:', commit)
''')
    data_cell = '''from google.colab import drive
drive.mount('/content/drive')
from stroke_model.prepare_stroke_zips import prepare_stroke_zips
from stroke_model.convert_reference_csv import existing_path
from stroke_model.train_reference import read_manifest
from stroke_model.annotation_data import ActualStrokeDataset
import matplotlib.pyplot as plt
DATA_ROOT = '/content/drive/Shareddrives/2026 자율연구/HSQM/dataset/automated_train_dataset'
INDEX_MODE = 'npz'  # 'npz': InkSight NPZ TAR. 기존 PNG TAR은 'filenames' 또는 'csv'.
CSV_PATH = None  # csv 모드에서만 지정. 기본은 path (1).csv, 그다음 path.csv/paths.csv.
IMAGE_SIZE = 128
SEED = 42
data_root = existing_path(DATA_ROOT)
if INDEX_MODE == 'csv' and CSV_PATH is None:
    try:
        CSV_PATH = existing_path(data_root/'path (1).csv')
    except FileNotFoundError:
        pass
manifest = prepare_stroke_zips(data_root, csv_path=CSV_PATH, size=IMAGE_SIZE, seed=SEED,
                               independent=True, index_mode=INDEX_MODE)
report = json.loads((manifest.parent/'conversion_report.json').read_text(encoding='utf-8'))
print('학습 목록 방식:', INDEX_MODE, '\\n저장된 목록:', manifest.parent / ('selected_samples.json' if INDEX_MODE == 'npz' else 'selected_paths.csv'))
print(report['label_policy'], '\\n검증 분리:', report['split_note'])
if INDEX_MODE != 'npz':
    print('빈 기준 획으로 제외:', report['skipped_count'], '개 / 목록:', manifest.parent/'skipped_samples.csv')
rows = read_manifest(manifest)
print('train:', sum(r['split']=='train' for r in rows), 'val:', sum(r['split']=='val' for r in rows))
dataset = ActualStrokeDataset(rows, IMAGE_SIZE)
for index in range(min(3, len(dataset))):
    image, reference, strokes, target, valid = dataset[index]
    k = len(strokes)
    fig, axes = plt.subplots(2, k+2, figsize=(2*(k+2), 4), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    axes[0,0].imshow(image[0], cmap='gray', vmin=0, vmax=1); axes[0,0].set_title('Input')
    axes[1,0].imshow(reference[0], cmap='gray', vmin=0, vmax=1); axes[1,0].set_title('Reference')
    for s in range(k):
        axes[0,s+1].imshow(target[s], cmap='gray', vmin=0, vmax=1); axes[0,s+1].set_title(f'Actual GT {s+1}')
        axes[1,s+1].imshow(strokes[s], cmap='gray', vmin=0, vmax=1); axes[1,s+1].set_title(f'Ref stroke {s+1}')
    axes[0,-1].imshow((target > .5).sum(0), vmin=0, vmax=max(2,k)); axes[0,-1].set_title('GT overlap count')
    plt.tight_layout(); plt.show()
'''
    if overlap:
        data_cell += '''for split in ('train', 'val'):
    shared = sum(entry['overlap_pixels'] for row, entry in zip(rows, report['rows'], strict=True)
                 if row['split'] == split)
    print(split, '겹침 주석 픽셀:', shared)
    if not shared:
        print('이 분할에는 명시적인 공유 정답이 없습니다.')
'''
    add('code', data_cell)
    training_cell = '''# 2번 셀의 획 순서와 실제 정답을 확인한 뒤 실행하세요.
from datetime import datetime
from zoneinfo import ZoneInfo
import shutil
EPOCHS = 30
BATCH_SIZE = 8  # 기존 V2와 동일. 메모리 부족 시 4 또는 2
LEARNING_RATE = 2e-4
INIT_CHECKPOINT = None  # 선택: 같은 해상도/폭의 V2 또는 현재 V2+정션 best.pt. 비교 실험은 같은 파일로 초기화.
JUNCTION_WEIGHT = 0.05  # 모든 보조 가중치를 0으로 하면 기존 V2 기본 손실
JUNCTION_EVERY = 4  # 4배치마다 정션 항 계산
JUNCTION_MAX_STROKES = 4  # 적용 배치마다 전체 획 중 최대 4개를 순환 선택
JUNCTION_HARD_FRACTION = 0.5  # 계산 획 수는 유지. 절반은 다른 획 정답을 침범한 채널, 절반은 순환
SEED_FOCAL_WEIGHT = 0.1  # 실제 주석이 있는 작은/어려운 획에 집중. 0이면 이전 방식
SEED_FOCAL_GAMMA = 2.0
DIRECTION_AFFINITY_WEIGHT = 0.03  # AAF 착안: 방향을 반영한 같은/다른 획 관계. 0이면 이전 학습
DIRECTION_AFFINITY_EVERY = 2  # 매 2배치, 고정 반경 1/3/5. 추론에는 추가 연산 없음
OWNERSHIP_WEIGHT = None  # None: 초기 모델 설정을 계승. 새 학습은 0. 직접 0.1 등 지정 가능
AFFINITY_WEIGHT = None  # None: 초기 모델 설정을 계승. 새 학습은 0. 직접 0.02 등 지정 가능
EPOCH_WARNING_SECONDS = 33  # 현재 약 30s 대비 비교 예산. 초과 시 경고하며 자동 중단하지 않음.
local_checkpoint = None
inherited = {}
if INIT_CHECKPOINT:
    local_checkpoint = root/'v2_initializer.pt'
    shutil.copyfile(INIT_CHECKPOINT, local_checkpoint)
    inherited = torch.load(local_checkpoint, map_location='cpu', weights_only=True)
if OWNERSHIP_WEIGHT is None:
    OWNERSHIP_WEIGHT = inherited.get('ownership_weight') or 0.0
if AFFINITY_WEIGHT is None:
    AFFINITY_WEIGHT = inherited.get('affinity_weight') or 0.0
del inherited
print('영역 감독 계승:', OWNERSHIP_WEIGHT, AFFINITY_WEIGHT)
if OWNERSHIP_WEIGHT or AFFINITY_WEIGHT:
    from stroke_model.ownership_refinement import preview_ownership_samples
    preview_ownership_samples(manifest, count=3)
project_root = next((parent.parent for parent in data_root.parents if parent.name == 'dataset'), data_root.parents[1])
output = project_root/'v2_topology_runs'/datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')
output.mkdir(parents=True, exist_ok=False)
for name in ('conversion_report.json', 'selected_paths.csv', 'selected_samples.json', 'skipped_samples.csv', '.complete.json'):
    if (manifest.parent / name).is_file():
        shutil.copyfile(manifest.parent/name, output/name.lstrip('.'))
(output/'git_commit.txt').write_text(commit)
cmd = [sys.executable, '-u', '-m', 'stroke_model.train_reference',
       '--manifest', str(manifest), '--output', str(output), '--epochs', str(EPOCHS),
       '--batch-size', str(BATCH_SIZE), '--lr', str(LEARNING_RATE), '--image-size', str(IMAGE_SIZE),
       '--seed', str(SEED), '--num-workers', '2',
       '--junction-weight', str(JUNCTION_WEIGHT), '--junction-every', str(JUNCTION_EVERY),
       '--junction-max-strokes', str(JUNCTION_MAX_STROKES),
       '--junction-hard-fraction', str(JUNCTION_HARD_FRACTION),
       '--seed-focal-weight', str(SEED_FOCAL_WEIGHT), '--seed-focal-gamma', str(SEED_FOCAL_GAMMA),
       '--direction-affinity-weight', str(DIRECTION_AFFINITY_WEIGHT),
       '--direction-affinity-every', str(DIRECTION_AFFINITY_EVERY),
       '--ownership-weight', str(OWNERSHIP_WEIGHT), '--affinity-weight', str(AFFINITY_WEIGHT),
       '--epoch-warning-seconds', str(EPOCH_WARNING_SECONDS)]
if local_checkpoint:
    cmd += ['--init-checkpoint', str(local_checkpoint)]
(output/'initial_checkpoint_source.txt').write_text(str(INIT_CHECKPOINT))
print('모델: 기존 V2 + 정션 보조 손실 / 결과:', output, flush=True)
from IPython.display import display
live = display({'text/plain': '학습 프로세스 시작 중...'}, raw=True, display_id=True)
print('진행률: epoch, 배치 수, 경과/예상 남은 시간, loss, lr (약 5초마다 갱신)')
print('전체 실행 로그:', output/'training.log')
with (output/'training.log').open('w', encoding='utf-8') as log:
    with subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding='utf-8', bufsize=1) as proc:
        try:
            for line in proc.stdout:
                log.write(line); log.flush()
                if line.strip():
                    live.update({'text/plain': line.rstrip()}, raw=True)
                if line.lstrip().startswith('{'):
                    print(line.rstrip(), flush=True)  # epoch별 결과도 출력에 보존
            returncode = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
            raise
if returncode:
    raise subprocess.CalledProcessError(returncode, cmd)
print('학습 완료:', output/'best.pt')
'''
    if overlap:
        training_cell = training_cell.replace("/'v2_topology_runs'/", "/'v2_topology_overlap_runs'/")
        training_cell = training_cell.replace("if local_checkpoint:\n", "cmd += ['--allow-overlap']\nif local_checkpoint:\n")
        training_cell = training_cell.replace('기존 V2 + 정션 보조 손실 / 결과:', 'Topo + 획별 sigmoid / 결과:')
        training_cell = training_cell.replace('모든 보조 가중치를 0으로 하면 기존 V2 기본 손실', '교차점 중복을 허용한 모델에도 정션 손실 적용')
    add('code', training_cell)
    if overlap:
        add('code', '''# 선택된 best.pt의 검증용 주석 점수와 겹침 미리보기
from stroke_model.infer_topology import load_topology, export_hard_masks
from stroke_model.reference_v2_topology import TopologyDataset, collate_topology
from stroke_model.train_reference import evaluate
from torch.utils.data import DataLoader
import numpy as np

model, size, epoch = load_topology(output/'best.pt', 'cuda')
val_items = [(row, entry['overlap_pixels']) for row, entry in zip(rows, report['rows'], strict=True)
             if row['split'] == 'val']
val_items.sort(key=lambda item: item[1], reverse=True)
validation = TopologyDataset([row for row, _ in val_items], size, allow_overlap=True)
loader = DataLoader(validation, batch_size=1, collate_fn=collate_topology)
metrics = evaluate(model, loader, 'cuda', return_details=True)
(output/'validation_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
print('Epoch:', epoch, metrics)
for index in range(min(5, len(validation))):
    image, reference, strokes, _, target, valid = validation[index]
    with torch.inference_mode():
        prediction = model(image[None].cuda(), reference[None].cuda(), strokes[None].cuda())
    soft = prediction['masks'][0].cpu().numpy()
    hard = export_hard_masks(image.numpy(), soft, output_mode=model.output_mode)
    annotated = (target > .5) & (valid > .5) & (image > .5)
    np.savez_compressed(output/f'val_{index:03d}.npz', image=image.numpy(), reference=reference.numpy(),
                        reference_strokes=strokes.numpy(), target_masks=target.numpy(),
                        valid_pixels=valid.numpy(), soft_masks=soft, hard_masks=hard,
                        probabilities=prediction['probabilities'][0].cpu().numpy())
    views = [(image[0], 'Input'), (annotated.sum(0), 'GT stroke count'),
             (hard.sum(0), 'Predicted stroke count')]
    fig, axes = plt.subplots(3, len(strokes)+1, figsize=(2*(len(strokes)+1), 6), squeeze=False)
    for row_index, (view, title) in enumerate(views):
        axes[row_index,0].imshow(view, cmap='gray', vmin=0, vmax=1 if row_index == 0 else max(2, len(strokes)))
        axes[row_index,0].set_title(title)
    for stroke in range(len(strokes)):
        for row_index, (view, title) in enumerate(((annotated[stroke], 'Annotation'),
                                                   (soft[stroke], 'Soft'), (hard[stroke], 'Hard'))):
            axes[row_index,stroke+1].imshow(view, cmap='gray', vmin=0, vmax=1)
            axes[row_index,stroke+1].set_title(f'{title} {stroke+1}')
    for ax in axes.flat:
        ax.axis('off')
    fig.tight_layout(); fig.savefig(output/f'val_{index:03d}.png', dpi=120); plt.show(); plt.close(fig)
''')
    else:
        add('code', '''# 학습에서 선택한 best.pt의 검증 예측. 독립 test 평가가 아닙니다.
from stroke_model.reference_model import ReferenceStrokeModel, exclusive_masks
from stroke_model.train_reference import StrokeDataset, collate, evaluate
from torch.utils.data import DataLoader
checkpoint = torch.load(output/'best.pt', map_location='cpu', weights_only=True)
model = ReferenceStrokeModel(image_size=checkpoint['image_size'], width=checkpoint['width'], pretrained=False)
model.load_state_dict(checkpoint['model'], strict=True); model = model.cuda().eval()
validation = StrokeDataset([r for r in rows if r['split']=='val'], checkpoint['image_size'])
metrics = {'val_macro_dice_annotated': evaluate(model, DataLoader(validation, batch_size=1, collate_fn=collate), 'cuda')}
(output/'validation_metrics.json').write_text(json.dumps(metrics, indent=2))
print(metrics)
for index in range(min(5, len(validation))):
    image, reference, strokes, labels = validation[index]
    with torch.no_grad():
        prediction = model(image[None].cuda(), reference[None].cuda(), strokes[None].cuda())
    masks = prediction['masks'][0].cpu()
    hard = exclusive_masks(prediction, (image[None].cuda() > .5))[0].cpu()
    k = len(strokes)
    fig, axes = plt.subplots(3, k+1, figsize=(2*(k+1), 6), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    axes[0,0].imshow(image[0], cmap='gray', vmin=0, vmax=1); axes[0,0].set_title('Input')
    axes[1,0].imshow(reference[0], cmap='gray', vmin=0, vmax=1); axes[1,0].set_title('Reference')
    for s in range(k):
        for row_index, tensor, title in ((0,labels==s,'Annotation'), (1,masks[s],'Soft'), (2,hard[s],'Exclusive')):
            axes[row_index,s+1].imshow(tensor, cmap='gray', vmin=0, vmax=1)
            axes[row_index,s+1].set_title(f'{title} {s+1}')
    plt.tight_layout(); fig.savefig(output/f'val_{index:03d}.png', dpi=120); plt.show()
''')
    name = '교차점_중복_허용.ipynb' if overlap else 'train_topology_zips.ipynb'
    notebook = dict(cells=cells, metadata=dict(colab=dict(name=name),
        accelerator='GPU', kernelspec=dict(display_name='Python 3', language='python', name='python3')),
        nbformat=4, nbformat_minor=5)
    (ROOT/'notebooks/training'/name).write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n',
                                                encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
    main(overlap=True)
