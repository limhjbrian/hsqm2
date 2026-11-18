# 획 추출·평가 안내

HSQM2는 기준 획 조건부 V2와 V2 구조를 유지한 Topo 보조 학습을 제공합니다.

## 모델과 학습

- [V2 구조·데이터·실행](REFERENCE_V2.md)
- [V2 + 정션 감독](V2_TOPOLOGY.md)
- [영역 소유 감독](OWNERSHIP_REFINEMENT.md)
- [작은 획 보조 학습](HARD_STROKE_TUNING.md)
- [방향 및 획 관계 감독](DIRECTIONAL_AFFINITY.md)
- [ZIP/TAR 데이터 준비](STROKE_ZIPS.md)
- [자모 학습 및 평가](JAMO.md)

## 추론과 검증

- [Topo 추론](TEST_TOPOLOGY.md)
- [체크포인트 비교](COMPARE_TOPOLOGY.md)
- [폰트 일반화 테스트](TOPOLOGY_FONTS.md)
- [추출 및 원본 구조 점수](TOPOLOGY_WITH_SCORES.md)
- [캔버스 구조 평가](CANVAS_CHO.md)

## 구조 지표

- [획 구조 진단](STROKE_STRUCTURE.md)
- [과정과 해석](STROKE_STRUCTURE_PROCESS.md)
- [원본 구조 지표](STRUCTURAL_INDICATOR.md)
- [원본 윤곽 진단](SOURCE_QUALITY_V5.md)
- [정션 회귀 진단](JUNCTION_REGRESSION.md)

추론은 `reference_v2` 및 `reference_v2_topology_loss_v1` 체크포인트를 지원합니다.
구형 BiFPN, 독립 sigmoid/ConvLSTM 및 전역정렬 Topo 모델은 제거했습니다.
실제 입력 획 주석 점검은 `annotation_data.ActualStrokeDataset`을 사용합니다.

기존 문서의 구현 당시 검증 기록은 현재 환경에서 재실행했다는 뜻이 아닙니다.
일반화 성능과 새 체크포인트는 실제 데이터에서 별도 확인해야 합니다.
