# HSQM2

한글 획 추출의 일반화 개선을 위한 연구 저장소입니다. 최신 기준 획 조건부 V2와 V2 + 정션/소유/방향 보조 감독, 학습·추론·평가, 원본 구조 지표와 폰트 일반화 테스트를 유지합니다.

## 실행

- [Topo 학습](https://colab.research.google.com/github/limhjbrian/hsqm2/blob/main/notebooks/train_topology_zips.ipynb)
- [Topo 추론·캔버스](https://colab.research.google.com/github/limhjbrian/hsqm2/blob/main/notebooks/test_topology.ipynb)
- [체크포인트 비교](https://colab.research.google.com/github/limhjbrian/hsqm2/blob/main/notebooks/compare_topology.ipynb)
- [폰트 일반화 테스트](https://colab.research.google.com/github/limhjbrian/hsqm2/blob/main/notebooks/test_topology_fonts.ipynb)
- [추출 및 원본 구조 점수](https://colab.research.google.com/github/limhjbrian/hsqm2/blob/main/notebooks/topology_with_scores.ipynb)
- [원본 구조 지표](https://colab.research.google.com/github/limhjbrian/hsqm2/blob/main/notebooks/structural_indicator.ipynb)
- [V2 학습](https://colab.research.google.com/github/limhjbrian/hsqm2/blob/main/notebooks/train_reference_v2.ipynb) · [V2 평가](https://colab.research.google.com/github/limhjbrian/hsqm2/blob/main/notebooks/test_reference_v2.ipynb)
- [자모 학습](https://colab.research.google.com/github/limhjbrian/hsqm2/blob/main/notebooks/jamo.ipynb) · [자모 일반화 평가](https://colab.research.google.com/github/limhjbrian/hsqm2/blob/main/notebooks/test_jamo.ipynb)

## 폴더

| 경로 | 용도 |
|---|---|
| stroke_model/ | V2/Topo 추출기, 데이터 준비, 학습·추론·평가, 구조 지표 |
| font_test/ | 폰트 렌더링 및 일반화 평가 지원 |
| notebooks/ | Colab 실행 노트북 및 생성기 |
| tests/ | 기능 검증 및 노트북 생성기 |
| docs/stroke_model/ | 유지하는 기능의 설명 |
| configs/ | 자모 CSV 예제 |

```console
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## 정리 범위와 한계

확산 생성·역확산 보정·오류 복원 실험, 구형 BiFPN과 독립/전역정렬 Topo 모델 및 전용 실행 파일을 제거했습니다. 최신 V2/Topo 체크포인트 형식과 학습 손실은 유지합니다. 구형 모델 체크포인트는 이 버전에서 지원하지 않습니다.

실제 획 주석을 검사하는 데이터 로더는 구형 trainer에서 분리해 `stroke_model/annotation_data.py`에 유지합니다. 모든 Colab의 코드 다운로드 주소는 이 포크를 가리킵니다.

정리는 성능 개선이나 재학습을 의미하지 않습니다. 주석된 부분의 Dice, 원본 구조 지표, 폰트별 시각 결과는 서로 다른 측정이며 실제 생성물/손글씨 일반화는 별도 검증이 필요합니다. 데이터·체크포인트는 저장소에 포함하지 않습니다. 정리 전 코드는 Git 이력에서 확인할 수 있습니다.

[모델 및 실행 안내](docs/stroke_model/README.md)
