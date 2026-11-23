"""Build the unified Colab for trying extracted strokes."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = '획 추출 모델 사용.ipynb'


def main():
    intro = '''# 획 추출 모델 사용

1. 준비 셀에서 코드를 불러오고 Drive에 연결합니다.
2. 실행 셀에서 **MODEL**은 `V2` 또는 `교차점 중복 허용`, **INPUT_MODE**는 `직접 쓰기` 또는 `테스트 이미지`를 선택합니다.
3. **학습 후보 고르기**에서 저장된 결과를 열고 입력과 획별 마스크를 확인합니다. 쓸 결과에 체크한 뒤 **선택 목록 저장**을 누르세요. 기존 결과를 고를 때는 2번 셀을 건너뛰어도 됩니다.

`직접 쓰기`에서는 기준 글자를 고른 뒤 같은 글자를 쓰고 **획 추출**을 누르세요.
`테스트 이미지`에서는 `TEST_IMAGE_NUMBER`로 한 장을 고르거나, `TEST_IMAGE_NAME`에 파일명을 적습니다.
기본 테스트 데이터는 `HSQM/dataset/test_dataset/npz/`입니다. 예전 이미지 목록을 쓰려면 `CSV_PATH`를 지정하세요.

`OVERLAP_THRESHOLD`는 교차점 중복 허용 모델에서 원본 잉크 픽셀을 각 획에 포함시키는 최소 점수입니다. 두 획이 모두 넘으면 자홍색, 아무 획에도 포함되지 않은 입력 잉크는 `Uncovered ink` 패널에서 빨간색으로 표시됩니다.

선택한 모델의 최신 `best.pt`를 사용합니다. 특정 체크포인트를 비교하려면 `CHECKPOINT_PATH`를 입력하세요.
모델이나 입력 방식을 바꿀 때는 두 번째 셀만 다시 실행하면 됩니다.

선택 목록은 Drive의 `HSQM/dataset/selected_stroke_predictions.json`에 저장됩니다. 증강용 원본을 고르는 단계이며, 실제 학습 데이터에 편입할 때는 같은 원본을 검증·테스트에도 쓰지 않도록 분리하세요.
'''
    setup = '''# @title 1. 코드 준비 및 Drive 연결
from pathlib import Path
import subprocess
import sys
import tempfile
from google.colab import drive

root = Path(tempfile.mkdtemp(prefix='hsqm_stroke_demo_', dir='/content'))
subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'main',
                'https://github.com/limhjbrian/hsqm2.git', str(root)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(root/'requirements.txt'), 'ipywidgets'], check=True)
sys.path.insert(0, str(root))
for key in tuple(sys.modules):
    if key == 'stroke_model' or key.startswith('stroke_model.'):
        del sys.modules[key]
drive.mount('/content/drive')
HSQM = '/content/drive/Shareddrives/2026 자율연구/HSQM'
print('코드 버전:', subprocess.check_output(['git', '-C', str(root), 'rev-parse', '--short', 'HEAD'], text=True).strip())
'''
    run = '''# @title 2. 모델·입력 선택 후 획 추출
MODEL = "V2" # @param ["V2", "교차점 중복 허용"]
INPUT_MODE = "직접 쓰기" # @param ["직접 쓰기", "테스트 이미지"]
TEST_IMAGE_NUMBER = 1 # @param {type:"integer"}
TEST_IMAGE_NAME = "" # @param {type:"string"}
CHECKPOINT_PATH = "" # @param {type:"string"}
GLOBAL_ALIGNMENT = True # @param {type:"boolean"}
OVERLAP_THRESHOLD = 0.5 # @param {type:"number"}
DATASET_ROOT = "" # @param {type:"string"}
CSV_PATH = "" # @param {type:"string"}

from stroke_model.stroke_demo import run_stroke_demo

report = run_stroke_demo(HSQM, MODEL, INPUT_MODE, checkpoint=CHECKPOINT_PATH, dataset_root=DATASET_ROOT,
                         csv_path=CSV_PATH, test_image_number=TEST_IMAGE_NUMBER, test_image_name=TEST_IMAGE_NAME,
                         threshold=OVERLAP_THRESHOLD, global_alignment=GLOBAL_ALIGNMENT)
print('선택한 입력/기준:', report['source'])
print(f"입력 잉크 중 여러 획이 공유한 비율: {report['overlapping_ink_fraction']:.1%}")
'''
    review = '''# @title 3. 학습 후보 고르기
from google.colab import output
from stroke_model.stroke_review import review_stroke_predictions

output.enable_custom_widget_manager()
review_stroke_predictions(HSQM)
'''
    cells = [dict(cell_type='markdown', metadata={}, source=intro.splitlines(keepends=True))]
    for source in (setup, run, review):
        compile(source, '<colab>', 'exec')
        cells.append(dict(cell_type='code', metadata={}, source=source.splitlines(keepends=True),
                          execution_count=None, outputs=[]))
    notebook = dict(cells=cells, metadata=dict(accelerator='GPU', colab=dict(name=NAME),
        kernelspec=dict(display_name='Python 3', language='python', name='python3')),
        nbformat=4, nbformat_minor=5)
    (ROOT/'notebooks/inference'/NAME).write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n',
                                                 encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
