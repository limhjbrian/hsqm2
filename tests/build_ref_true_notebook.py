"""Build the Colab for reviewing InkSight JSON and rendering thick targets."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAME = 'JSON 검수 및 획 데이터 생성.ipynb'


def main():
    intro = '''# JSON 검수 및 획 데이터 생성

1. Drive를 연결합니다. 원본 좌표는 `HSQM/dataset/automated_train_dataset/jsons/<폰트>/`에서 가져옵니다.
2. 폰트와 글자를 골라 좌표, 획 순서, 굵은 합성 이미지와 개별 획을 확인합니다. **통과**를 누르면 현재 JSON의 검수 기록이 Drive에 바로 저장됩니다. 같은 글자에 원본과 편집본이 함께 있으면 JSON 드롭다운에서 선택할 수 있습니다. **제외**를 누르거나 검수 이후 JSON이 수정되면 다음 생성에 포함되지 않습니다.
3. 생성 셀을 실행합니다. 통과한 폰트·글자마다 가로세로 비율, 전체 크기, 회전을 무작위로 조합해 10장을 만듭니다. 변형 전 얇은 좌표 마스크를 기준 획으로 쓰고, 변형한 굵은 획을 합쳐 입력을 만들며, 합치기 전 획을 정답으로 저장합니다.

출력은 `HSQM/dataset/ref is true/snapshot-…/npz/samples-*.tar`입니다. 학습 노트북의 `DATA_ROOT`에는 생성 셀이 출력한 `snapshot-…` 폴더를 지정하세요. 검수 결과나 설정이 바뀌면 새 폴더에 저장하며, 기존 출력은 보존합니다.
'''
    setup = '''# @title 1. 코드 준비 및 Drive 연결
from pathlib import Path
import subprocess
import sys
import tempfile
from google.colab import drive

root = Path(tempfile.mkdtemp(prefix='hsqm_ref_true_', dir='/content'))
subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'main',
                'https://github.com/limhjbrian/hsqm2.git', str(root)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(root/'requirements.txt'), 'ipywidgets'], check=True)
sys.path.insert(0, str(root))
for key in tuple(sys.modules):
    if key == 'stroke_model' or key.startswith('stroke_model.'):
        del sys.modules[key]
drive.mount('/content/drive')

from stroke_model.convert_reference_csv import existing_path
from stroke_model.ref_true_dataset import generate_approved, review_coordinates

dataset = existing_path('/content/drive/Shareddrives/2026 자율연구/HSQM/dataset')
json_root = existing_path(dataset/'automated_train_dataset'/'jsons')
destination = dataset/'ref is true'
review_path = destination/'json_review.json'
'''
    review = '''# @title 2. JSON 검수
IMAGE_SIZE = 128 # @param {type:"integer"}
THICKNESS_PX = 7.0 # @param {type:"number"}

from google.colab import output
output.enable_custom_widget_manager()
review_coordinates(json_root, review_path, size=IMAGE_SIZE, width=THICKNESS_PX)
'''
    generate = '''# @title 3. 통과한 JSON만 변형하고 NPZ TAR 저장
SCALE_MIN = 0.85 # @param {type:"number"}
SCALE_MAX = 1.15 # @param {type:"number"}
ASPECT_MIN = 0.8 # @param {type:"number"}
ASPECT_MAX = 1.25 # @param {type:"number"}
ROTATION_DEG = 12.0 # @param {type:"number"}
SEED = 42 # @param {type:"integer"}
MAX_TAR_MIB = 256 # @param {type:"integer"}

result = generate_approved(json_root, review_path, destination, size=IMAGE_SIZE, width=THICKNESS_PX,
                           scale_range=(SCALE_MIN, SCALE_MAX), aspect_range=(ASPECT_MIN, ASPECT_MAX),
                           rotation=ROTATION_DEG, seed=SEED, max_bytes=MAX_TAR_MIB * 2**20)
for item in result['stale']:
    print('재검수 필요:', item)
if result['dataset']:
    print(f"통과한 폰트·글자 {result['glyphs']}개, NPZ {result['samples']}장")
    print('학습 DATA_ROOT:', result['dataset'])
else:
    print('현재 통과한 JSON이 없습니다. 2번 셀에서 글자를 검수하세요.')
'''
    cells = [dict(cell_type='markdown', metadata={}, source=intro.splitlines(keepends=True))]
    for source in (setup, review, generate):
        compile(source, '<colab>', 'exec')
        cells.append(dict(cell_type='code', metadata={}, source=source.splitlines(keepends=True),
                          execution_count=None, outputs=[]))
    notebook = dict(cells=cells, metadata=dict(colab=dict(name=NAME),
        kernelspec=dict(display_name='Python 3', language='python', name='python3')),
        nbformat=4, nbformat_minor=5)
    (ROOT/'notebooks/data'/NAME).write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n',
                                            encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
