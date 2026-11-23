# Colab 노트북

| 폴더 | 실행 파일 |
|---|---|
| `data/` 데이터셋 생성 | [InkSight 데이터셋 생성](data/inksight_automation.ipynb), [JSON 검수 및 획 데이터 생성](data/JSON%20검수%20및%20획%20데이터%20생성.ipynb), [학습 데이터 고르기](data/학습%20데이터%20고르기.ipynb) |
| `training/` 학습 | [Topo ZIP/TAR](training/train_topology_zips.ipynb), [교차점 중복 허용](training/교차점_중복_허용.ipynb), [V2 ZIP](training/train_stroke_zips.ipynb), [V2 기준 획](training/train_reference_v2.ipynb), [자모](training/jamo.ipynb) |
| `inference/` 추론 | [획 추출 모델 사용](inference/획%20추출%20모델%20사용.ipynb), [Topo 일괄 추론](inference/test_topology.ipynb), [V2 CSV 추론](inference/infer_reference_v2.ipynb) |
| `evaluation/` 평가·진단 | [Topo 체크포인트 비교](evaluation/compare_topology.ipynb), [폰트 일반화](evaluation/test_topology_fonts.ipynb), [V2 평가](evaluation/test_reference_v2.ipynb), [V2 진단](evaluation/diagnose_reference_v2.ipynb), [자모 평가](evaluation/test_jamo.ipynb) |
| `metrics/` 구조 지표 | [원본 구조 지표](metrics/structural_indicator.ipynb), [획 구조 진단](metrics/stroke_structure_v2.ipynb), [Topo 추론과 점수](metrics/topology_with_scores.ipynb) |

`cells/`에는 기존 Colab에 붙여 넣는 Python 셀이 있습니다. 구조 지표 노트북과 관련 셀은 `builders/`의 스크립트로 재생성합니다. 학습·추론·평가 노트북 생성 스크립트는 [tests/](../tests/)에 있습니다.

**학습 데이터 고르기**에서 Target의 모델 결과나 같은 글자·폰트의 기존 학습 샘플을 검토할 수 있습니다. 확인한 샘플은 `HSQM/dataset/reviewed_stroke_npz/`에 개별 NPZ로 저장됩니다.
