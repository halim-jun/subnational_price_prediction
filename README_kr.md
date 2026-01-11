# 🌾 동아프리카 식량 가격 및 예측 프로젝트

이 프로젝트는 기후 데이터(SPI), 거시경제 지표, 그리고 과거 가격 데이터를 통합하여 동아프리카(주로 에티오피아)의 식량 가격을 분석하고 예측하는 것을 목표로 합니다. SARIMAX 모델을 사용하여 국가 및 지역(Sub-national) 수준의 가격을 예측합니다.

## 📂 프로젝트 구조

소스 코드는 `src/` 디렉토리 내에 다음과 같이 구성되어 있습니다:

```
src/
├── data_pipeline/      # 데이터 수집 및 처리
│   ├── spi/            # 강수량 데이터 (CHIRPS) & SPI 계산
│   └── macro/          # 거시경제 지표 (World Bank, 환율)
│
├── notebook/           # Jupyter 분석 노트북
│   ├── national_level_analysis.ipynb
│   └── subnational_level_prediction_baseline.ipynb
│
├── utils/              # 유틸리티 스크립트
│   └── fix_file_naming.py
├── data/
│   ├── raw/
│   │   ├── climate/            # CHIRPS 강수량 원본 데이터
│   │   ├── worldbank_commodity/# World Bank Commodity (Pink Sheet) 데이터
│   │   └── wfp/                # WFP 식량 가격 데이터 (수동/사전 다운로드 필요)
│   └── processed/
│       ├── spi/                # 계산된 SPI 지수
│       └── external/           # 처리된 거시경제 지표
```

---

## 🚀 시작하기 (Getting Started)

### 1. 필수 조건 (Prerequisites)

- Python 3.9 이상 권장.
- 필요한 패키지는 `requirements.txt`에 명시되어 있습니다.

### 2. 설치 (Installation)

```bash
# 레포지토리 클론
git clone https://github.com/halim-jun/price_prediction
cd wpf_colla_v2

# 데이터 처리용 디렉토리 생성
mkdir -p data/raw data/processed

# 종속성 패키지 설치
pip install -r requirements.txt
```

---

## 🛠️ 데이터 파이프라인 (Data Pipeline)

분석을 위한 데이터셋을 준비하려면 다음 파이프라인들을 실행하세요:

1.  **거시경제 데이터 (Macro Pipeline)**:
    ```bash
    python src/data_pipeline/macro/process_wb_data.py
    python src/data_pipeline/macro/merge_external_data.py
    ```

2.  **기후 데이터 (SPI Pipeline)**:
    ```bash
    # CHIRPS 데이터 다운로드 및 SPI 생성
    python src/data_pipeline/spi/run_spi_generation.py --download-chirps
    ```
    *(상세한 SPI 관련 설명은 `src/data_pipeline/spi/README.md`를 참고하세요)*

---

## ✅ 구현된 데이터 파이프라인 상세

### 1. 기후 데이터 (SPI) ✅
- **위치**: `src/data_pipeline/spi/`
- **출처**: CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data)
- **주요 스크립트**:
  - `run_spi_generation.py`: 자동 다운로더 및 SPI 계산기.
  - `generate_spi_python.py`: 핵심 SPI 계산 로직 (Gamma 분포 사용).
  - `enrich_all_spi.py`: 행정 구역 정보(국가, 지역 등) 추가.
- **특징**:
  - UCSB 서버에수 CHIRPS 데이터 자동 다운로드.
  - 30년 보정 기간 (1991-2020) 적용으로 신뢰도 높은 가뭄 지수 산출.
  - 모델링을 위해 NetCDF 파일을 CSV로 자동 변환.

### 2. 거시경제 데이터 (Macroeconomic) ✅
- **위치**: `src/data_pipeline/macro/`
- **출처**: World Bank Commodity Markets (Pink Sheet)
- **주요 스크립트**:
  - `process_wb_data.py`: 에너지, 식량, 비료 지수 추출.
  - `merge_external_data.py`: 다양한 경제 지표를 하나의 데이터셋으로 병합.
- **주요 지표**:
  - 에너지 지수 (석유, 가스, 석탄)
  - 식량 가격 지수
  - 비료 지수
- **입력**: 엑셀 파일 (`data/raw/worldbank_commodity/*.xlsx`)

### 3. 식량 가격 데이터 (WFP)
- **출처**: WFP VAM (Vulnerability Analysis and Mapping)
- **상태**: `data/raw/wfp/`에 원본 데이터가 존재해야 함.
- **참고**: 현재 예측 모델의 타겟 변수(정답 데이터)로 사용됨.

## 📊 데이터 파이프라인 사용법

### SPI 파이프라인 실행
```bash
# CHIRPS 다운로드부터 SPI 생성까지 전체 과정 실행
python src/data_pipeline/spi/run_spi_generation.py --download-chirps
```

### Macro 파이프라인 실행
```bash
# World Bank 경제 데이터 처리
python src/data_pipeline/macro/process_wb_data.py
python src/data_pipeline/macro/merge_external_data.py
```

## 📓 분석 노트북 (Notebooks)

`src/notebook/` 디렉토리에서 인터랙티브한 분석을 확인할 수 있습니다.

*   `national_level_analysis.ipynb`: 국가 수준의 거시적 트렌드 및 집계 분석.
*   `subnational_level_prediction_baseline.ipynb`: 세부 지역별 가격 예측 베이스라인 모델.

---
