# 데이터 파이프라인 문서

이 디렉토리는 데이터 소스 및 도메인별로 구성된 프로젝트의 데이터 처리 파이프라인을 포함합니다.

## 📂 디렉토리 구조

```
src/data_pipeline/
├── macro/          # 거시경제 및 외부 데이터 처리
├── spi/            # 강수량 (CHIRPS) 및 SPI 계산
├── fews_net/       # 식량 안보 데이터 (IPC)
├── crop_mask/      # 작물 마스크 (ASAP)
├── night_lights/   # 야간 조명 (Black Marble)
├── spatial_weighting/ # 작물 생산량 가중치 적용 (SPAM)
└── utils/          # 유틸리티 및 진단 스크립트
```

---

## 1. 🌦️ SPI 파이프라인 (`src/data_pipeline/spi/`)

CHIRPS 강수량 데이터를 다운로드, 처리, 가공(Enrichment)하는 과정을 다룹니다.

| 스크립트 | 입력 데이터 | 출력 데이터 | 설명 |
|--------|------------|-------------|-------------|
| **`run_spi_generation.py`** | `data/raw/climate/chirps/*.nc` | `data/processed/spi/05_spi_final/*.nc` | **메인 관리자(Orchestrator)**. 전체 파이프라인 실행: 클리핑, 결측치 채우기, SPI 계산 (`generate_spi_python.py` 사용). |
| `generate_spi_python.py` | NetCDF (Raw) | NetCDF (Processed) | `climate-indices` 라이브러리를 사용한 SPI 계산의 핵심 로직. |
| `convert_nc_to_csv.py` | `data/processed/spi/05_spi_final/*.nc` | `data/processed/spi/06_spi_csv/*.csv` | 계산된 SPI NetCDF 파일을 사용하기 쉬운 CSV 형식으로 변환. |
| `enrich_all_spi.py` | `data/processed/spi/06_spi_csv/*.csv` | `data/processed/spi/07_enriched/*.csv` | GeoBoundaries 데이터를 사용하여 SPI CSV에 행정 구역(국가, 지역, 존) 정보 추가. |
| **`generate_weighted_indices.py`** | SPI CSV + SPAM Data | CSV (Admin2 Weighted) | 지역별 작물 생산량을 가중치로 반영한 SPI 지수를 생성합니다. |
| `climate_aggregator.py` | - | - | 가중 평균 계산을 위한 핵심 로직 모듈. |

### 🚀 사용법
```bash
# 전체 파이프라인 실행 (생성 -> 변환)
python src/data_pipeline/spi/run_spi_generation.py
```

---

## 2. 📈 거시경제(Macro) 파이프라인 (`src/data_pipeline/macro/`)

World Bank 상품 가격 및 환율과 같은 거시경제 지표를 처리합니다.

| 스크립트 | 입력 데이터 | 출력 데이터 | 설명 |
|--------|------------|-------------|-------------|
| **`process_wb_data.py`** | `data/raw/worldbank_commodity/*.xlsx` | `data/processed/external/worldbank_indices.csv` | World Bank "Pink Sheet" 데이터에서 에너지, 식량, 비료 지수 추출. |
| `merge_external_data.py` | `worldbank_indices.csv`, etc. | `data/processed/external/external_variables_merged.csv` | 다양한 외부 지표(WB 데이터, 환율, FAO 지수)를 하나의 마스터 시계열 파일로 병합. |

### 🚀 사용법
```bash
# World Bank 데이터 처리
python src/data_pipeline/macro/process_wb_data.py

# 모든 외부 데이터 병합
python src/data_pipeline/macro/merge_external_data.py
```

---

## 3. 🌽 식량 안보 파이프라인 (`src/data_pipeline/fews_net/`)

FEWS NET에서 식량 안보 데이터(IPC Phase)를 다운로드하고 처리합니다.

| 스크립트 | 설명 |
|--------|------|
| **`download.py`** | FEWS NET API를 통해 케냐, 에티오피아, 소말리아의 IPC 데이터를 다운로드합니다. |
| `merge.py` | 분할된 CSV 파일들을 하나로 병합합니다. |

### 🚀 사용법
```bash
python src/data_pipeline/fews_net/download.py
```

---

## 4. 🌾 작물 마스크 파이프라인 (`src/data_pipeline/crop_mask/`)

ASAP 작물 마스크(TIFF)를 처리하고 행정 구역과 매핑합니다.

| 스크립트 | 설명 |
|--------|------|
| **`tiff_to_parquet.py`** | 대용량 TIFF 파일에서 작물 픽셀만 추출하여 Parquet로 변환합니다. |
| `map_to_admin.py` | Parquet 점 데이터를 행정 구역(Admin2)에 매핑합니다. |
| `visualize.py` | TIFF 파일을 시각화하여 PNG로 저장합니다. |

### 🚀 사용법
```bash
python src/data_pipeline/crop_mask/tiff_to_parquet.py
python src/data_pipeline/crop_mask/map_to_admin.py
```

---

## 5. 🌃 야간 조명 파이프라인 (`src/data_pipeline/night_lights/`)

NASA Black Marble(VNP46A4) 데이터를 다운로드하고 집계합니다.

| 스크립트 | 설명 |
|--------|------|
| **`download.py`** | 연간 야간 조명 데이터를 다운로드합니다. (`.env`에 `BEARER_TOKEN` 필요) |
| `process.py` | 다운로드한 데이터를 행정 구역별로 평균 내어 집계합니다. |

### 🚀 사용법
```bash
python src/data_pipeline/night_lights/download.py
python src/data_pipeline/night_lights/process.py
```

---

## 6. ⚖️ 공간 가중치 파이프라인 (`src/data_pipeline/spatial_weighting/`)

단순 평균이 아닌, 작물 생산 지역에 가중치를 둔 기후 지수를 계산합니다. (예: 옥수수 재배지의 가뭄 지수)

| 스크립트 | 설명 |
|--------|------|
| **`spam_loader.py`** | SPAM 2020 데이터(작물 분포)를 로드합니다. |

**참고**: `climate_aggregator.py`와 `generate_weighted_indices.py`는 `src/data_pipeline/spi/`로 이동되었습니다.

