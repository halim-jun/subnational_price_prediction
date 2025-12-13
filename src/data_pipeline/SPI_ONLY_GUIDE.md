# 🎯 SPI만 빠르게 계산하기

이미 **전처리된 강수량 데이터**가 있을 때, SPI만 빠르게 계산하는 간단한 스크립트입니다.

---

## ⚡ 빠른 시작

### 가장 간단한 사용법

```bash
python calculate_spi_only.py --input your_precipitation.nc
```

끝! 🎉

---

## 📋 사용 시나리오

### 시나리오 1: 이미 전처리된 CHIRPS 데이터가 있음

```bash
# SPI 1, 3, 6, 12개월 계산
python calculate_spi_only.py \
  --input ../../data/processed/chirps/east_africa_precip_clean.nc \
  --output ../../data/processed/spi
```

### 시나리오 2: 특정 SPI 스케일만 필요

```bash
# SPI-12만 계산 (장기 가뭄)
python calculate_spi_only.py \
  --input precip.nc \
  --scales 12

# SPI-3, SPI-6만 계산 (계절 가뭄)
python calculate_spi_only.py \
  --input precip.nc \
  --scales 3 6
```

### 시나리오 3: 다른 보정 기간 사용

```bash
# 최근 10년을 보정 기간으로
python calculate_spi_only.py \
  --input precip.nc \
  --cal-start 2010 \
  --cal-end 2020
```

### 시나리오 4: 변수명이 다를 때

```bash
# 변수명이 'rainfall'인 경우
python calculate_spi_only.py \
  --input precip.nc \
  --var-name rainfall
```

---

## 📝 입력 파일 요구사항

### 필수 조건:

1. **파일 형식**: NetCDF (.nc)
2. **변수명**: `precip`, `precipitation`, `prcp`, `rain`, 또는 `rainfall` 중 하나
3. **단위**: `mm`, `millimeters`, `inches` 중 하나
4. **차원**: `time`, `lat`/`latitude`, `lon`/`longitude` 포함

### 권장 사항:

- 월별 데이터 (monthly)
- 최소 30년 이상의 데이터 (통계적 안정성)
- 결측값이 50% 미만

---

## 🎛️ 전체 옵션

```bash
python calculate_spi_only.py --help
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--input`, `-i` | 입력 강수량 NetCDF 파일 (필수) | - |
| `--output`, `-o` | 출력 디렉토리 | `./spi_output` |
| `--var-name` | 강수량 변수명 | 자동 감지 |
| `--scales` | SPI 시간 스케일 (월) | `1 3 6 12` |
| `--cal-start` | 보정 시작 연도 | `1991` |
| `--cal-end` | 보정 종료 연도 | `2020` |
| `--keep-intermediate` | 중간 파일 유지 | False |
| `--skip-validation` | 검증 건너뛰기 | False |

---

## 📤 출력 결과

### 디렉토리 구조

```
spi_output/
├── intermediate/              # 중간 파일 (--keep-intermediate 시)
│   └── input_spi_reordered.nc
└── final/                     # ⭐ 최종 SPI 파일
    ├── spi_gamma_01_month.nc
    ├── spi_gamma_03_month.nc
    ├── spi_gamma_06_month.nc
    └── spi_gamma_12_month.nc
```

### 출력 파일

- **파일 형식**: NetCDF (압축)
- **차원 순서**: `(time, lat, lon)` - CF-compliant
- **변수명**: `spi_gamma_XX_month`
- **SPI 값 범위**: 보통 -3 ~ +3

---

## 🔍 입력 파일 검증

스크립트는 자동으로 입력 파일을 검증합니다:

```
============================================================
Validating input file...
============================================================
✓ File loaded successfully
  Dimensions: {'time': 108, 'lat': 740, 'lon': 540}
  Variables: ['precip']
✓ Precipitation variable: 'precip'
  Units: mm
  Dimension order: ['time', 'lat', 'lon']
✓ All required dimensions present
  Time range: 2016-01-01 to 2024-12-01
  Number of timesteps: 108
  Missing values: 1,234 / 43,243,200 (0.003%)
```

---

## 💡 전체 파이프라인 vs SPI만 계산

### 🔄 전체 파이프라인 (`run_spi_generation.py`)

**언제 사용?**
- CHIRPS 원본 데이터부터 시작
- 데이터 클리핑, 결측값 보간 등 전처리 필요
- 동아프리카 전용

**장점:**
- 모든 것을 자동으로 처리
- 시각화 포함
- 품질 보고서 생성

```bash
python run_spi_generation.py --visualize
```

### ⚡ SPI만 계산 (`calculate_spi_only.py`)

**언제 사용?**
- **이미 전처리된 강수량 데이터 있음** ⭐
- 빠르게 SPI만 계산하고 싶음
- 다양한 지역/데이터 소스 사용

**장점:**
- 매우 빠름 (전처리 생략)
- 간단함
- 어떤 강수량 데이터도 사용 가능

```bash
python calculate_spi_only.py --input my_precip.nc
```

---

## 📊 실행 예시

### 터미널 출력

```bash
$ python calculate_spi_only.py --input precip.nc --scales 3 6 12

**********************************************************************
  SPI CALCULATOR
  Standardized Precipitation Index from NetCDF
**********************************************************************

Started: 2024-12-13 14:30:00

============================================================
Validating input file...
============================================================
✓ File loaded successfully
  Dimensions: {'time': 120, 'lat': 100, 'lon': 150}
  Variables: ['precip']
✓ Precipitation variable: 'precip'
  Units: mm
✓ All required dimensions present
  Time range: 2010-01-01 to 2019-12-31
  Missing values: 234 / 1,800,000 (0.01%)

============================================================
Checking dimension order...
============================================================
Current order: ['time', 'lat', 'lon']
Reordering to: [lat, lon, time]
✓ Dimensions reordered successfully

============================================================
Calculating SPI...
============================================================
Input file: spi_output/intermediate/input_spi_reordered.nc
Scales: [3, 6, 12]
Calibration period: 1991-2020

2024-12-13 14:31:15 INFO Computing 3-month SPI (Gamma)
2024-12-13 14:31:45 INFO Computing 6-month SPI (Gamma)
2024-12-13 14:32:10 INFO Computing 12-month SPI (Gamma)
✓ SPI calculation completed!

============================================================
Reordering SPI outputs to CF-compliant format...
============================================================
Processing: spi_gamma_03_month.nc
  ✓ Saved to: spi_output/final/spi_gamma_03_month.nc
Processing: spi_gamma_06_month.nc
  ✓ Saved to: spi_output/final/spi_gamma_06_month.nc
Processing: spi_gamma_12_month.nc
  ✓ Saved to: spi_output/final/spi_gamma_12_month.nc

**********************************************************************
  COMPLETED SUCCESSFULLY!
**********************************************************************

Finished: 2024-12-13 14:32:30

Output files saved to: spi_output/final

Generated SPI files:
  - spi_gamma_03_month.nc (12.34 MB)
  - spi_gamma_06_month.nc (12.34 MB)
  - spi_gamma_12_month.nc (12.34 MB)
```

---

## 🔧 문제 해결

### ❌ "No precipitation variable found"

**원인**: 변수명이 예상과 다름

**해결**:
```bash
# NetCDF 파일의 변수 확인
ncdump -h your_file.nc

# 변수명 지정
python calculate_spi_only.py --input your_file.nc --var-name rainfall
```

### ❌ "Units 'kg/m^2' may not be recognized"

**원인**: 단위가 비표준

**해결**: 입력 파일의 단위를 `mm`로 변경
```python
import xarray as xr
ds = xr.open_dataset('precip.nc')
ds['precip'].attrs['units'] = 'mm'
ds.to_netcdf('precip_fixed.nc')
```

### ❌ "'spi' command not found"

**원인**: `climate-indices` 패키지 미설치

**해결**:
```bash
pip install climate-indices
```

### ⚠️ "High percentage of missing values: 65%"

**원인**: 결측값이 너무 많음

**해결**: 전처리 필요
```bash
# 전체 파이프라인 사용 (결측값 보간 포함)
python run_spi_generation.py
```

---

## 🆚 비교표

| 특성 | 전체 파이프라인 | SPI만 계산 |
|------|----------------|-----------|
| 입력 | CHIRPS 원본 | 전처리된 강수량 데이터 |
| 전처리 | ✅ 자동 (클리핑, 보간 등) | ❌ 없음 |
| 속도 | 🐢 느림 (전체 처리) | ⚡ 빠름 (SPI만) |
| 시각화 | ✅ 포함 | ❌ 없음 |
| 유연성 | 동아프리카 전용 | 🌍 모든 지역 |
| 스크립트 | `run_spi_generation.py` | `calculate_spi_only.py` |

---

## 📚 관련 문서

- **전체 파이프라인**: `QUICK_START.md`
- **상세 가이드**: `SPI_GENERATION_README.md`
- **튜토리얼**: `Steps_to_Generate_SPI_Using_CHIRPS_Data.ipynb`

---

## 🎯 추천 워크플로우

### 처음 사용자:
1. 전체 파이프라인으로 시작 → `run_spi_generation.py`
2. 프로세스 이해
3. 필요시 SPI만 재계산 → `calculate_spi_only.py`

### 고급 사용자:
1. 자신만의 전처리 수행
2. SPI만 빠르게 계산 → `calculate_spi_only.py` ⭐
3. 결과 통합

---

**마지막 업데이트:** 2024년 12월

