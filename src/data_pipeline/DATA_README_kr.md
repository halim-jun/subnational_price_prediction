# 데이터 파이프라인 문서

이 디렉토리는 데이터 소스 및 도메인별로 구성된 프로젝트의 데이터 처리 파이프라인을 포함합니다.

## 📂 디렉토리 구조

```
src/data_pipeline/
├── macro/          # 거시경제 및 외부 데이터 처리
└── spi/            # 강수량 (CHIRPS) 및 SPI 계산
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
