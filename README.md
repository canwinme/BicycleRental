# Bicycle Rental Demand Prediction 

날씨, 시간대, 요일 등 자전거 이용 환경 데이터를 분석해 **시간대별 공공자전거 대여량을 예측**하고, 두 회귀 모델의 성능 및 주요 예측 요인을 비교한 머신러닝 프로젝트입니다.

## 프로젝트 개요

- **주제:** 시간대별 공공자전거 대여 수요 예측 (회귀)
- **예측 대상:** `cnt` (해당 시간 총 자전거 대여량)
- **분석 목표:** 날씨와 시간 정보를 이용한 수요 예측, 모델별 오차 분석, 주요 변수 중요도 파악
- **비교 모델:** LinearRegression, RandomForestRegressor

## 개발 환경

- Python
- NumPy, Pandas
- Matplotlib
- scikit-learn
- Jupyter Notebook / Google Colab

## 데이터셋

- **출처:** [UCI Machine Learning Repository — Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
- **사용 파일:** [data/hour.csv](data/hour.csv)
- **데이터 규모:** 총 17,379행, 17개 컬럼
- **기간:** 2011~2012년, 시간대별 자전거 대여 기록

### 주요 컬럼

| 컬럼 | 내용 |
|---|---|
| `hr` | 시간대 (0~23시) |
| `weekday` | 요일 |
| `weathersit` | 날씨 상태 |
| `temp`, `atemp` | 기온 및 체감온도 (정규화 값) |
| `hum`, `windspeed` | 습도 및 풍속 (정규화 값) |
| `workingday` | 근무일 여부 |
| `cnt` | 해당 시간대의 총 대여량 (예측 대상) |

`cnt`는 `casual + registered`이므로 **데이터 누수를 방지하기 위해 두 컬럼을 학습 입력에서 제외**했습니다.

## 분석 및 모델링 과정

1. **EDA:** 데이터 크기·결측치·기초 통계량 확인, 시간대별 평균 대여량과 대여량 분포 시각화
2. **전처리:** `instant`, `dteday`, `casual`, `registered` 제거
3. **원-핫 인코딩:** `season`, `mnth`, `hr`, `weekday`, `weathersit` 변환
4. **데이터 분리:** 원본 시간순을 유지한 채 학습 80%, 테스트 20% 분리 (`shuffle=False`)
5. **모델 학습:** LinearRegression, RandomForestRegressor (`n_estimators=100`)
6. **평가 및 분석:** MAE, RMSE, R² 비교, 실제값·예측값 그래프, RandomForest의 변수 중요도 분석

## 모델 성능 비교

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| LinearRegression | 98.7976 | 133.8354 | 0.6315 |
| **RandomForestRegressor** | **55.0312** | **82.9227** | **0.8586** |

MAE와 RMSE는 낮을수록, R²는 높을수록 좋은 결과입니다.

## 주요 변수 중요도 (RandomForestRegressor)

| 변수 | 중요도 |
|---|---:|
| `atemp` (체감온도) | 0.1896 |
| `hr_17` (17시) | 0.0915 |
| `workingday` (근무일 여부) | 0.0908 |
| `hr_18` (18시) | 0.0801 |
| `hum` (습도) | 0.0748 |

체감온도, 17~18시 시간대, 근무일 여부 등이 모델의 대여량 예측에 중요한 변수로 나타났습니다. 변수 중요도는 **모델 내부의 예측 기여도**이며 인과관계를 뜻하지는 않습니다.

## 결과 및 결론

RandomForestRegressor는 LinearRegression보다 MAE와 RMSE가 낮고 R²가 높아 전반적으로 우수한 예측 성능을 보였습니다.

특히 RandomForestRegressor의 MAE는 약 55대로, LinearRegression의 약 99대보다 예측 오차가 작았습니다. 이번 데이터 및 실험 조건에서는 RandomForestRegressor가 더 적합한 모델이라고 판단했습니다.

다만 두 모델 모두 실제 대여량을 완벽하게 예측하지는 못하므로 시간대별 오차를 추가 분석하고 모델 설정을 조정해 개선할 수 있습니다.

## 프로젝트 구조

```text
BicycleRental/
├── data/
│   └── hour.csv                 # UCI 시간별 공공자전거 대여 데이터
├── BicycleRental.ipynb         # 탐색, 전처리, 모델 학습 및 성능 평가
├── README.md                  # 프로젝트 설명 및 결과
└── requirements.txt          # Python 라이브러리
```

## 실행 방법

1. 저장소를 내려받거나 복제합니다.
2. Jupyter Notebook에서 `BicycleRental.ipynb`를 엽니다.
3. 노트북과 같은 위치에 `data/hour.csv`가 있는지 확인합니다.
4. 필요한 라이브러리를 설치한 뒤 셀을 순서대로 실행합니다.

```bash
pip install -r requirements.txt
```
## 출처

- [UCI Machine Learning Repository — Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
- Fanaee-T, H. & Gama, J. (2014), *Event labeling combining ensemble detectors and background knowledge*, Progress in Artificial Intelligence.
