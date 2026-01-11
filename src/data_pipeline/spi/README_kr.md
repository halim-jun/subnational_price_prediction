# SPI (표준강수지수) 생성 가이드

## 📌 개요

이 디렉토리는 CHIRPS 강수량 데이터로부터 **SPI (Standardized Precipitation Index)**를 생성하는 도구들을 포함합니다.
SPI는 특정 기간 동안의 강수량이 장기 평균과 비교하여 얼마나 건조하거나 습윤한지를 정규 분포로 표준화하여 나타낸 기상학적 가뭄 지표입니다.

---

## 🚀 빠른 시작 (Quick Start)

### 1. 필수 패키지 설치

```bash
pip install climate-indices scipy
```

### 2. 파이프라인 실행

`run_spi_generation.py` 스크립트가 모든 과정(데이터 다운로드, 동아프리카 지역 추출, SPI 계산)을 처리합니다.

```bash
# 일반적인 사용법
python run_spi_generation.py --download-chirps

# 권장: 30년 보정 기간 사용 (WMO 표준)
python run_spi_generation.py \
  --download-chirps \
  --year-start 2016 --year-end 2024 \
  --calibration-start 1991 --calibration-end 2020
```

---

## 📂 파일 구조

```
src/data_pipeline/spi/
├── README.md                     # 이 파일 (영문)
├── README_kr.md                  # 이 파일 (국문)
├── run_spi_generation.py         # 메인 실행 스크립트
├── generate_spi_python.py        # 핵심 SPI 계산 로직 (Pure Python)
├── convert_nc_to_csv.py          # NetCDF 출력을 CSV로 변환하는 도구
└── enrich_all_spi.py             # 행정 구역 정보를 추가하는 도구
```

---

## 🔄 파이프라인 처리 단계

1.  **Clip (자르기)**: 전 세계 CHIRPS 데이터에서 동아프리카 지역만 추출합니다.
2.  **Fill (채우기)**: 해안선 근처의 결측값을 보간합니다.
3.  **Metadata (메타데이터)**: 단위를 `mm`로 표준화하고 시간 속성을 수정합니다.
4.  **Reorder (재정렬)**: 계산을 위해 차원을 `(lat, lon, time)` 순서로 변경합니다.
5.  **Calculate (계산)**: Gamma 분포를 사용하여 SPI를 계산합니다 (`climate-indices` 라이브러리 활용).
6.  **Finalize (마무리)**: 차원을 다시 `(time, lat, lon)`으로 복원하고 CF-compliant NetCDF로 저장합니다.

---

## 📊 출력 결과

결과 파일은 `data/processed/spi/05_spi_final/` 디렉토리에 저장됩니다:

-   `east_africa_spi_gamma_01_month.nc`
-   `east_africa_spi_gamma_03_month.nc`
-   ...및 기타 시간 스케일 파일들.

---

## 🔧 문제 해결 (Troubleshooting)

### 1. "SPI calculation failed" (보정 기간 불일치)
**오류:** `Command '['spi', ...]' returned non-zero exit status 1.`
**원인:** 지정된 보정 기간(예: 1991-2020)이 사용 가능한 데이터 범위 밖에 있습니다.
**해결:** 스크립트가 자동으로 조정을 시도합니다. 실패할 경우, `--year-start`가 보정 기간을 포함하는지 확인하거나 `--calibration-start`를 조정하세요.

### 2. "climate-indices not found"
**오류:** `FileNotFoundError: 'spi' command not found`
**해결:** 패키지를 설치하세요:
```bash
pip install climate-indices
```

### 3. 메모리 오류 (Memory Errors)
**원인:** 너무 넓은 지역이나 긴 시간 범위를 처리하고 있습니다.
**해결:** `--lon-min/max` 또는 `--year-start/end`를 사용하여 범위를 줄이거나, 한 번에 계산할 스케일 수를 줄이세요:
```bash
python run_spi_generation.py --scales 3 6 12
```

### 4. 결측값 경고 (Missing Data Warning)
**경고:** `High percentage of missing values`
**해결:** 파이프라인에는 자동 채우기(filling) 단계가 포함되어 있습니다. 이 단계가 정상적으로 실행되었는지 확인하세요. 커스텀 데이터를 사용하는 경우, 육지 위의 결측값(NaN)을 미리 전처리해야 할 수 있습니다.
