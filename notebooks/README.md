# Colab 노트북

| 폴더 | 실행 파일 |
|---|---|
| `data/` 데이터셋 생성 | [InkSight 데이터셋 생성](data/inksight_automation.ipynb) |
| `training/` 학습 | [Topo ZIP/TAR](training/train_topology_zips.ipynb), [V2 ZIP](training/train_stroke_zips.ipynb), [V2 기준 획](training/train_reference_v2.ipynb), [자모](training/jamo.ipynb) |
| `inference/` 추론 | [Topo 추론·캔버스](inference/test_topology.ipynb), [V2 추론](inference/infer_reference_v2.ipynb) |
| `evaluation/` 평가·진단 | [Topo 체크포인트 비교](evaluation/compare_topology.ipynb), [폰트 일반화](evaluation/test_topology_fonts.ipynb), [V2 평가](evaluation/test_reference_v2.ipynb), [V2 진단](evaluation/diagnose_reference_v2.ipynb), [자모 평가](evaluation/test_jamo.ipynb) |
| `metrics/` 구조 지표 | [원본 구조 지표](metrics/structural_indicator.ipynb), [획 구조 진단](metrics/stroke_structure_v2.ipynb), [Topo 추론과 점수](metrics/topology_with_scores.ipynb) |

`cells/`에는 기존 Colab에 붙여 넣는 Python 셀이 있습니다. 구조 지표 노트북과 관련 셀은 `builders/`의 스크립트로 재생성합니다. 학습·추론·평가 노트북 생성 스크립트는 [tests/](../tests/)에 있습니다.
