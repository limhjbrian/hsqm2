# Drive NPZ TAR 기반 획 학습

[InkSight 생성 노트북](../../notebooks/data/inksight_automation.ipynb)에서 학습용은 `test=False`, 평가용은 `test=True`로 설정한다. 검수와 획 생성을 마치고 **검수한 PNG를 학습용 NPZ로 묶기**, **NPZ TAR를 Drive에 병합 또는 새로 저장**, **된거 안된거 구분하기** 순서로 실행한다. PNG 정규화와 획 주석 계산은 생성할 때 끝난다. 실행한 목록을 손으로 관리할 필요가 없다.

NPZ에는 입력, 기준 이미지, 기준 획, 소유 주석, 공유 교차점을 포함하는 정답 획 마스크가 들어간다. `NPZ_IMAGE_SIZE` 기본값은 128이며 학습의 `IMAGE_SIZE`와 맞춘다. 작은 배치도 NPZ TAR를 재사용한다. `MAX_TAR_MIB` 기본값은 256이다. 상한 이내로 들어갈 기존 `npz/samples-*.tar`에 추가하고 공간이 없으면 새 TAR를 만든다. 같은 이름과 같은 내용은 재시도에서 건너뛰고, 같은 이름의 내용이 다르면 기존 파일을 덮어쓰지 않고 중단한다. 한 데이터 폴더에는 생성 노트북 한 세션에서 업로드한다.

학습 Colab 두 개는 기본 `INDEX_MODE='npz'`로 로컬에 TAR를 풀고, 저장된 NPZ를 배치에서 읽는다. `selected_samples.json`을 기록하므로 이후 TAR에 새 샘플을 추가해도 당시 검증 분할을 복원할 수 있다. 별도 평가 셋은 V2 평가 Colab에서 `MODE='test_npz'`로 읽는다.

기존 네 역할의 PNG TAR은 새 NPZ TAR와 분리된다. 기존 PNG 아카이브만 학습하려면 `INDEX_MODE='filenames'` 또는 `INDEX_MODE='csv'`를 선택한다.

## Drive의 기존 PNG TAR를 한 번만 변환하기

InkSight 생성 노트북 맨 위의 **기존 Drive PNG TAR를 NPZ TAR로 변환** 셀만 실행한다. `RUN_LEGACY_MIGRATION = True`로 바꾸고, 기본 `MIGRATION_DATASET = 'automated_train_dataset'`을 사용한다. 평가용은 `'test_dataset'`으로 지정해 별도로 실행한다. 변환은 원래 네 역할의 압축 파일을 읽어 해당 데이터 폴더의 `npz/`에 저장하며, 학습·평가에서 쓰는 NPZ와 크기 설정(`MIGRATION_IMAGE_SIZE`, 기본 128)을 맞춘다.

기본은 파일명과 획 번호로 연결한다. 기존 CSV의 별도 매핑이나 그룹 ID를 보존해야 하면 `MIGRATION_CSV_PATH`에 파일 경로를 지정한다. 빈 기준 획이나 소유 주석이 전혀 없는 샘플을 확인한 뒤 제외하고 진행하려면 `MIGRATION_SKIP_INVALID = True`로 바꾼다. 제외 항목은 셀 결과에 표시된다. 누락 파일과 모호한 매핑은 계속 오류로 처리한다.

PNG TAR는 읽기만 한다. 변환된 샘플은 여유가 있는 NPZ TAR에 병합하고, 이미 같은 내용으로 변환된 샘플은 건너뛴다. 변환이 끝나면 학습 Colab의 기본 `INDEX_MODE='npz'`로 읽는다.

## 기존 PNG 압축 파일 방식

`prepare_stroke_zips`는 폴더별 TAR 묶음도 지원한다. 데이터 폴더 아래
`input_stroke`, `input`, `target_stroke`, `target` 폴더의 하위 폴더까지 모든 TAR/ZIP을
파일명·번호·개수 제한 없이 자동 검색한다. 각 TAR은 독립 압축 파일이다.
역할별 TAR이 없을 때 기존 `역할.zip`으로 대체한다. Topo 모델 학습은
`notebooks/training/train_topology_zips.ipynb`를 사용하며 실제 입력 획의 공유 교차점을 보존한다.

## CSV 없이 학습 목록 만들기

Topo 또는 V2 ZIP 학습 Colab의 **데이터 준비 셀**에서 `INDEX_MODE = 'filenames'`,
`CSV_PATH = None`으로 설정한다. 이후 네 역할의 압축 파일을 추가하고 준비 셀을 다시 실행하면 된다.
예를 들어 입력 변형이 `I_Font_가_12.png`이면 다음 파일들이 짝을 이룬다.

| 압축 파일 역할 | PNG 이름 예시 |
|---|---|
| `input` | `I_Font_가_12.png` |
| `target` | `Font_가.png` |
| `input_stroke` | `I_Font_가_12_1.png`, `I_Font_가_12_2.png`, … |
| `target_stroke` | `Font_가_1.png`, `Font_가_2.png`, … |

같은 획 번호끼리 연결하며 기준 획 번호는 1부터 연속이어야 한다. 동명 PNG, 기준 이미지 누락,
전체 이미지 없이 남은 획 묶음, 기준과 정답의 획 번호 불일치는 준비 단계에서 오류로 표시한다.
InkSight의 결합 획 미리보기인 `*_c.png`는 획 목록에서 제외한다.
누락 획을 감독하려면 해당 번호의 빈 정답 마스크 파일을 유지한다. 파일의 번호나 픽셀 내용 자체를
잘못 내보낸 경우에는 이름 검사만으로 구분할 수 없으므로 준비 셀의 획 미리보기도 확인한다.

사용한 목록은 `selected_paths.csv`로 자동 저장된다. 별도 이름 규칙이나 `writer_id`/`group_id`가
들어간 기존 목록을 사용하려면 `INDEX_MODE = 'csv'`를 선택한다.

## 기존 평가 노트북 재사용

`notebooks/evaluation/test_reference_v2.ipynb`의 기본 MODE="zip_val"은 최신 stroke_zip_runs/*/best.pt를 선택한다.
checkpoint에 경로를 지정하면 특정 실행을 평가한다. 기존 PNG 방식은 학습 당시 저장한 selected_paths.csv와 run_config의 seed/크기로
로컬 데이터를 복원하고 val만 평가한다. 새 실행에서는 압축 파일의 상대 경로, 크기, 수정 시각을 확인한다.
기존 evaluate/preview_reference를 그대로 사용하여 val Dice, 정답 커버리지, exclusive 예측, 획별 확률/soft mask/entropy를 저장한다.
결과는 checkpoint 옆 val_results_시간에 저장한다. 이는 모델 선택에 사용한 validation이며 독립 test 평가가 아니다.
MODE="test_csv"는 기존 별도 test CSV 전체 평가를 유지한다.
미리보기는 CSV char_id 또는 파일명의 단독 한글 음절로 묶어서 글자별로 골고루 선택한다.
글자 정보가 없으면 무작위 샘플을 보여주며 글자별 샘플 수도 출력한다. 실제 val에 없는 글자는 표시할 수 없다.
마지막 선택 셀의 PREVIEW_COUNT/CHARACTERS/PREVIEW_SEED를 바꾸면 전체 평가 없이 시각화만 다시 한다.

[Colab에서 실행](https://colab.research.google.com/github/limhjbrian/hsqm2/blob/main/notebooks/training/train_stroke_zips.ipynb)하고 GPU 런타임에서 위부터 실행한다.
첫 셀은 GitHub main 최신 코드를 clone하고 전체 커밋 SHA를 출력한다. 실행 커밋을 결과 폴더의 git_commit.txt에도 저장한다.
모델 구현은 기존 ReferenceStrokeModel이며 기본은 Swin 사전학습 + 새 분할 head 학습이다.
호환되는 reference_v2, image_size=128 best.pt가 있으면 INIT_CHECKPOINT로 가중치를 초기화할 수 있다.
optimizer/epoch 복원은 하지 않는다. 자모 모델 가중치가 획 분리 성능을 높인다는 보장은 없다.

기본 폴더는 `/content/drive/Shareddrives/2026 자율연구/HSQM/dataset/automated_train_dataset`.
input_stroke.zip, input.zip, target_stroke.zip, target.zip을 런타임 로컬로 한 번 복사하고 해제한다.
CSV 모드는 데이터 폴더 바로 아래의 path.csv/paths.csv를 우선 검색하고, 없으면 해제된 ZIP 내부를 찾는다.
여러 후보면 중단하며 CSV_PATH로 선택한다. CSV가 나중에 올라오면 준비 셀만 재실행한다.
데이터 폴더 밖의 이전 train/test CSV는 검색하지 않는다.

CSV의 입력은 input, 기준 글자는 target, 기준 획은 target_stroke, 정답 획은 input_stroke에 해당하는 기존 CSV 규약을 사용한다.
실제 컬럼 별칭은 convert_reference_csv.py에 정의되어 있다.
기준/정답 획 목록은 같은 순서여야 하며, preview와 선택된 컬럼을 학습 전에 확인한다.
CSV의 오래된 Drive 절대 경로는 로컬 파일 경로의 접미사로 대응한다. 모호한 동명 파일은 중단한다.
로컬 전용 Resolver를 사용하므로 CSV의 원래 Drive 이미지가 남아 있어도 개별 파일을 Drive에서 읽지 않는다.

변환 NPZ와 manifest는 /content에 보관하고 배치마다 로컬에서 읽는다. 추가 증강은 기본으로 하지 않는다.
Drive에는 HSQM/stroke_zip_runs/실행시간 아래 best.pt, history.json, run_config.json, CSV/변환 보고서와 소스 식별 정보를 저장한다.
변환 보고서는 교차/누락 픽셀을 제외하는 정답 커버리지를 포함한다. val Dice는 주석된 전경 픽셀에 한정된다.
writer/group ID가 있으면 그룹 분할하며 없으면 동일 입력 이미지 해시 단위로 80/20 분리한다.
필자 ID가 없을 때 필자 누수를 방지했다고 주장할 수 없다. 런타임 삭제 후에는 ZIP 복사·변환이 다시 필요하다.
ZIP 캐시는 경로/크기/수정 시각을 사용하므로 원본을 변경한 경우 새 업로드가 완료된 후 실행한다.

`python tests/build_stroke_zip_notebook.py`로 소스 변경 후 노트북을 다시 생성한다.
`python -m unittest discover -s tests -p test_stroke_zips.py -v`로 합성 ZIP→CSV→NPZ와 캐시/경로 보호를 검증한다.
실제 Drive ZIP/CSV와 GPU 학습 성능 검증은 Colab 실행에서 수행해야 한다.
