"""Build the Colab notebook for choosing training NPZ samples."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAME = '학습 데이터 고르기.ipynb'


def main():
    intro = '''# 학습 데이터 고르기

1. 코드 준비 및 Drive 연결을 한 뒤 2번 셀을 실행합니다.
2. 폰트와 글자를 고르고 **Target 추출**을 누릅니다. InkSight NPZ에 들어 있는 원본 폰트 글자(`reference`, 생성 노트북의 `target`)에 모델을 실행합니다.
3. 결과가 괜찮으면 **확인 · NPZ 저장**을 누릅니다. 다른 후보를 보려면 **학습 샘플 보기**를 누르세요. 같은 폰트와 글자의 기존 학습 샘플을 무작위 순서로 하나씩 보여주며, 이때는 기존 정답 획을 표시합니다.

확인한 결과는 `HSQM/dataset/reviewed_stroke_npz/`에 개별 NPZ로 저장됩니다. 후보는 `TRAINING_ROOT`의 학습 분할에서 가져옵니다. `SEED`는 실제 학습에 사용한 분할 값과 맞추세요.
'''
    setup = '''# @title 1. 코드 준비 및 Drive 연결
from pathlib import Path
import subprocess
import sys
import tempfile
from google.colab import drive

root = Path(tempfile.mkdtemp(prefix='hsqm_stroke_review_', dir='/content'))
subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'main',
                'https://github.com/limhjbrian/hsqm2.git', str(root)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(root/'requirements.txt'), 'ipywidgets'], check=True)
sys.path.insert(0, str(root))
for key in tuple(sys.modules):
    if key == 'stroke_model' or key.startswith('stroke_model.'):
        del sys.modules[key]
drive.mount('/content/drive')
print('코드 버전:', subprocess.check_output(['git', '-C', str(root), 'rev-parse', '--short', 'HEAD'], text=True).strip())
'''
    review = '''# @title 2. Target 추출 후 학습용 샘플 선택
MODEL = "교차점 중복 허용" # @param ["V2", "교차점 중복 허용"]
CHECKPOINT_PATH = "" # @param {type:"string"}
TRAINING_ROOT = "" # @param {type:"string"}
SEED = 42 # @param {type:"integer"}
GLOBAL_ALIGNMENT = True # @param {type:"boolean"}
OVERLAP_THRESHOLD = 0.5 # @param {type:"number"}

from google.colab import output
from stroke_model.stroke_review import review_training_candidates

HSQM = '/content/drive/Shareddrives/2026 자율연구/HSQM'
output.enable_custom_widget_manager()
review_training_candidates(HSQM, MODEL, checkpoint=CHECKPOINT_PATH, training_root=TRAINING_ROOT,
                           seed=SEED, threshold=OVERLAP_THRESHOLD, global_alignment=GLOBAL_ALIGNMENT)
'''
    cells = [dict(cell_type='markdown', metadata={}, source=intro.splitlines(keepends=True))]
    for source in (setup, review):
        compile(source, '<colab>', 'exec')
        cells.append(dict(cell_type='code', metadata={}, source=source.splitlines(keepends=True),
                          execution_count=None, outputs=[]))
    notebook = dict(cells=cells, metadata=dict(accelerator='GPU', colab=dict(name=NAME),
        kernelspec=dict(display_name='Python 3', language='python', name='python3')),
        nbformat=4, nbformat_minor=5)
    (ROOT/'notebooks/data'/NAME).write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n',
                                            encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
