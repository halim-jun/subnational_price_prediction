# SPI (Standardized Precipitation Index) Generation Guide

## 📌 Overview

이 디렉토리는 CHIRPS 강수량 데이터로부터 **SPI (Standardized Precipitation Index)**를 생성하는 도구들을 포함합니다.

SPI는 특정 기간 동안의 강수량이 장기 평균과 비교하여 얼마나 건조하거나 습윤한지를 표준화된 지수로 나타낸 기상학적 가뭄 지표입니다.

---

## 🚀 Quick Start

### 1. 필수 패키지 설치

```bash
pip install climate-indices scipy
```

### 2. Python 스크립트 실행

```python
from generate_spi_python import CHIRPStoSPI

# CHIRPS 데이터 경로
chirps_file = '../../data/raw/chirps/chirps-v2.0.monthly.nc'

# SPI 생성기 초기화
processor = CHIRPStoSPI(
    input_file=chirps_file,
    output_dir='../../data/processed/spi'
)

# 전체 파이프라인 실행
final_dir = processor.run_full_pipeline(
    lon_min=25, lon_max=52,      # 동아프리카 경도 범위
    lat_min=-15, lat_max=22,     # 동아프리카 위도 범위
    year_start=2016,             # 시작 연도
    year_end=2024,               # 종료 연도
    spi_scales=[1, 2, 3, 6, 9, 12],  # SPI 시간 스케일 (개월)
    calibration_start=1991,      # 보정 기간 시작
    calibration_end=2020         # 보정 기간 종료
)

print(f"✅ SPI 파일 생성 완료: {final_dir}")
```

---

## 📂 파일 구조

```
src/data_pipeline/
├── generate_spi_python.py                    # 순수 Python SPI 생성 스크립트 (CDO/NCO 불필요)
├── Steps_to_Generate_SPI_Using_CHIRPS_Data.ipynb  # 상세 튜토리얼 노트북
└── SPI_GENERATION_README.md                  # 이 파일
```

---

## 🔄 처리 파이프라인

### Step 1: 데이터 클리핑
- CHIRPS 전역 데이터에서 동아프리카 지역만 추출
- 시간 범위 선택 (예: 2016-2024)

### Step 2: 결측값 보간
- 해안선 근처 결측값을 nearest neighbor 또는 linear 방법으로 보간

### Step 3: 메타데이터 수정
- 강수량 단위를 `mm`로 통일 (climate-indices 패키지 요구사항)
- 시간 속성 표준화

### Step 4: 차원 재정렬
- SPI 계산을 위해 차원을 `(lat, lon, time)` 순서로 변경

### Step 5: SPI 계산
- `climate-indices` 패키지를 사용하여 다양한 시간 스케일의 SPI 계산
- Gamma 분포 기반 계산 사용

### Step 6: 후처리
- CF-Convention을 따르기 위해 차원을 `(time, lat, lon)` 순서로 복원
- 압축된 NetCDF 형식으로 저장

---

## 📊 출력 구조

```
data/processed/spi/
├── 01_clipped/                           # 클리핑된 데이터
│   └── east_africa_chirps_clipped.nc
├── 02_filled/                            # 결측값 보간 완료
│   └── east_africa_chirps_filled.nc
├── 03_metadata_revision/                 # 메타데이터 수정 완료
│   ├── east_africa_chirps_metadata_fixed.nc
│   └── input_spi.nc                      # SPI 계산용 입력
├── 04_spi_intermediate/                  # SPI 계산 중간 결과
│   ├── east_africa_spi_gamma_01_month.nc
│   ├── east_africa_spi_gamma_02_month.nc
│   ├── east_africa_spi_gamma_03_month.nc
│   ├── east_africa_spi_gamma_06_month.nc
│   ├── east_africa_spi_gamma_09_month.nc
│   └── east_africa_spi_gamma_12_month.nc
└── 05_spi_final/                         # 최종 CF-compliant SPI 파일 ⭐
    ├── east_africa_spi_gamma_01_month.nc
    ├── east_africa_spi_gamma_02_month.nc
    ├── east_africa_spi_gamma_03_month.nc
    ├── east_africa_spi_gamma_06_month.nc
    ├── east_africa_spi_gamma_09_month.nc
    └── east_africa_spi_gamma_12_month.nc
```

최종 SPI 파일은 **`05_spi_final/`** 디렉토리에 저장됩니다.

---

## 📈 SPI 해석

### SPI 값의 의미

| SPI 값 | 상태 | 설명 |
|--------|------|------|
| ≥ 2.0 | 극단적으로 습윤 | Extremely wet |
| 1.5 ~ 2.0 | 심각하게 습윤 | Severely wet |
| 1.0 ~ 1.5 | 중간 정도 습윤 | Moderately wet |
| -1.0 ~ 1.0 | 정상 범위 | Near normal |
| -1.5 ~ -1.0 | 중간 정도 건조 | Moderate drought |
| -2.0 ~ -1.5 | 심각한 가뭄 | Severe drought |
| ≤ -2.0 | 극심한 가뭄 | Extreme drought |

### 시간 스케일별 의미

| 시간 스케일 | 의미 | 용도 |
|------------|------|------|
| **SPI-1, SPI-3** | 단기 가뭄 | 농업 생산성, 작물 생육 영향 |
| **SPI-6** | 계절 가뭄 | 계절별 강수 패턴 분석 |
| **SPI-9, SPI-12** | 장기 기상학적 가뭄 | 기후 변화 추세 분석 |
| **SPI-18, SPI-24+** | 수문학적 가뭄 | 수자원 관리, 저수지 수위 |

---

## 🔧 고급 사용법

### 단계별 실행 (더 세밀한 제어)

```python
from generate_spi_python import CHIRPStoSPI

processor = CHIRPStoSPI('chirps-v2.0.monthly.nc')

# 각 단계를 개별적으로 실행
clipped = processor.step1_clip_region(25, 52, -15, 22, 2016, 2024)
filled = processor.step2_fill_missing(clipped)
metadata_fixed = processor.step3_fix_metadata(filled)
reordered = processor.step4_reorder_for_spi(metadata_fixed)
spi_dir = processor.step5_calculate_spi(reordered, scales=[1, 3, 6, 12])
final = processor.step6_reorder_output(spi_dir)
```

### 결측값 보간 방법 변경

```python
# 'nearest' (기본값) 또는 'linear' 보간
filled = processor.step2_fill_missing(
    clipped,
    method='linear',      # 선형 보간 사용
    distance_limit=10     # 탐색 거리 증가
)
```

### 다른 보정 기간 사용

```python
# 최근 10년을 보정 기간으로 사용
spi_dir = processor.step5_calculate_spi(
    reordered,
    scales=[1, 3, 6, 12],
    calibration_start=2010,  # 더 최근 기간
    calibration_end=2020
)
```

---

## 🆚 CDO/NCO vs Python 비교

### 기존 방법 (CDO/NCO 사용)

**장점:**
- ✅ 매우 빠른 처리 속도
- ✅ 대용량 데이터 처리에 최적화

**단점:**
- ❌ 별도 설치 필요 (Linux/Mac 중심, Windows에서 어려움)
- ❌ 복잡한 명령어 체인
- ❌ 디버깅 어려움
- ❌ 진행 상황 추적 어려움

### 새로운 방법 (순수 Python)

**장점:**
- ✅ 플랫폼 독립적 (Windows, Mac, Linux 모두 동일)
- ✅ 설치 간편 (`pip install`)
- ✅ 읽기 쉬운 코드
- ✅ 상세한 진행 상황 표시
- ✅ 쉬운 디버깅 및 커스터마이징
- ✅ Python 생태계와 통합 (numpy, pandas, xarray)

**단점:**
- ⚠️ 큰 데이터셋에서는 CDO보다 느릴 수 있음

### 명령어 대응표

| CDO/NCO 명령어 | Python 대체 |
|----------------|-------------|
| `cdo sellonlatbox,25,52,-15,22` | `ds.sel(lon=slice(25, 52), lat=slice(-15, 22))` |
| `cdo selyear,2016/2024` | `ds.sel(time=slice('2016', '2024'))` |
| `cdo -fillmiss` | `scipy.interpolate.griddata()` |
| `cdo -remapbil` | `xarray` + `scipy` 보간 |
| `cdo -setattribute,precip@units="mm"` | `ds['precip'].attrs['units'] = 'mm'` |
| `ncpdq -a lat,lon,time` | `ds.transpose('lat', 'lon', 'time')` |
| `ncks --fix_rec_dmn` | xarray에서 자동 처리 |

---

## 🐛 문제 해결

### 1. `spi` 명령어를 찾을 수 없음

```bash
pip install climate-indices
```

### 2. 메모리 부족 오류

큰 데이터셋의 경우 청크(chunk) 단위 처리:

```python
ds = xr.open_dataset(file, chunks={'time': 12})
```

### 3. `climate-indices` 설치 오류

의존성 패키지 먼저 설치:

```bash
pip install numpy scipy xarray netCDF4
pip install climate-indices
```

### 4. 결측값이 너무 많음

`step2_fill_missing`에서 `distance_limit` 파라미터 증가:

```python
filled = processor.step2_fill_missing(clipped, distance_limit=20)
```

---

## 📚 참고 자료

### SPI 관련
- [WMO SPI Guide](https://library.wmo.int/viewer/39629/)
- [Climate Indices Python Package](https://climate-indices.readthedocs.io/)
- [NCAR Climate Data Guide - SPI](https://climatedataguide.ucar.edu/climate-data/standardized-precipitation-index-spi)

### CHIRPS 데이터
- [CHIRPS Official Website](https://www.chc.ucsb.edu/data/chirps)
- [CHIRPS Paper (Nature, 2015)](https://doi.org/10.1038/sdata.2015.66)

### Python 라이브러리
- [xarray Documentation](https://docs.xarray.dev/)
- [netCDF4-python](https://unidata.github.io/netcdf4-python/)
- [scipy.interpolate](https://docs.scipy.org/doc/scipy/reference/interpolate.html)

---

## 📞 도움말

문제가 발생하거나 질문이 있으시면:
1. `Steps_to_Generate_SPI_Using_CHIRPS_Data.ipynb` 노트북의 상세 설명 참고
2. 코드 주석 확인
3. GitHub Issues 또는 프로젝트 관리자에게 문의

---

**Last Updated:** December 2024

