"""Synchronize the standalone inference cell and filename-selection Colab."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = (ROOT/'stroke_model/infer_reference.py').read_text(encoding='utf-8')
    source += '''
# Run the setup cell first to load this version of global alignment.
from google.colab import drive
drive.mount("/content/drive")
HSQM = existing_path("/content/drive/Shareddrives/2026 자율연구/HSQM")
CHECKPOINT = globals().get("checkpoint")
if CHECKPOINT is None:
    candidates = sorted((HSQM / "stroke_zip_runs").glob("*/best.pt"))
    if not candidates:
        raise FileNotFoundError("Set CHECKPOINT to your V2 best.pt path")
    CHECKPOINT = candidates[-1]
CSV_PATH = globals().get("test_csv", HSQM / "dataset/test_dataset/paths.csv")
DATASET_ROOT = None
INPUT_FILES = ["I_아_2.png", "I_야_2.png"]  # None = all CSV images.
COMPARE_ALIGNMENT = True  # Same checkpoint/image, OFF and ON side by side.
GLOBAL_ALIGNMENT = True  # Used only when COMPARE_ALIGNMENT=False.
SHOW_SOFT = True
print("Checkpoint:", CHECKPOINT)
print("CSV:", CSV_PATH)
inference_results = run_inference_csv(
    CHECKPOINT, CSV_PATH, input_files=INPUT_FILES, dataset_root=DATASET_ROOT,
    show_soft=SHOW_SOFT, global_alignment=GLOBAL_ALIGNMENT,
    compare_alignment=COMPARE_ALIGNMENT,
)
'''
    compile(source, '<inference-cell>', 'exec')
    (ROOT/'notebooks/cells/infer_reference_v2_cell.py').write_text(source, encoding='utf-8', newline='\n')
    path = ROOT/'notebooks/inference/infer_reference_v2.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    notebook['cells'][0]['source'] = '''# V2 획 추출 — 전체 정렬 OFF / ON 비교
INPUT_FILES에 테스트 CSV의 입력 파일명을 지정하세요. None이면 전체 이미지를 처리합니다.
동일한 학습 체크포인트를 한 번만 로드하고, 같은 이미지에 정렬 OFF/ON을 각각 실행합니다.
원본 / 기준 / 정렬된 모델 입력 / OFF 추출 / ON 추출 / 배정 차이를 나란히 표시합니다.
획별 비교 그림에는 기준 획, OFF/ON soft mask, OFF/ON exclusive mask를 함께 표시합니다.
ON 결과는 원래 입력 캔버스 좌표로 복원합니다. 두 결과 모두 표시 범위는 0~1입니다.
전체 정렬은 가로세로 비율을 유지하는 크기·중심 보정이며 회전/기울기는 보정하지 않습니다.
잡점·누락·잘린 글자는 정렬을 방해할 수 있습니다. 배정 변화율은 정확도 개선 점수가 아닙니다.
재학습하거나 입력 획 정답을 읽지 않습니다. OFF/ON 원시 결과와 비교 PNG를 새 폴더에 저장합니다.
COMPARE_ALIGNMENT=False이면 GLOBAL_ALIGNMENT로 단일 추론의 정렬 여부를 선택합니다.
첫 설정 셀부터 실행하세요. 기존 구버전 런타임에서도 설정 셀을 다시 실행해야 합니다.
'''.splitlines(keepends=True)
    setup = ''.join(notebook['cells'][1]['source']).replace('"main"', '"inference-global-alignment"')
    notebook['cells'][1]['source'] = setup.splitlines(keepends=True)
    notebook['cells'][2]['source'] = source.splitlines(keepends=True)
    notebook['cells'][2]['execution_count'] = None
    notebook['cells'][2]['outputs'] = []
    notebook['metadata']['colab']['name'] = 'infer_reference_v2.ipynb'
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<colab>', 'exec')
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
