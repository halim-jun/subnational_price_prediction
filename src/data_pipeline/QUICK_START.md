# 🚀 SPI 생성 빠른 시작 가이드

## 📝 요약

CHIRPS 강수량 데이터로부터 SPI (Standardized Precipitation Index)를 생성하는 **실행 가능한 Python 스크립트**입니다.

---

## ⚡ 빠른 실행

### 1. 패키지 설치

```bash
pip install climate-indices scipy
```

### 2. CHIRPS 데이터 다운로드 + SPI 생성 (한 번에!) ⭐

```bash
cd src/data_pipeline
python run_spi_generation.py --download-chirps
```

**끝!** 🎉

스크립트가 자동으로:
- 📥 CHIRPS 전체 데이터 다운로드 (~7 GB, 1981-현재)
- 🔄 동아프리카 지역 추출
- 📊 SPI 계산 (30년 보정!)

---

### 또는 이미 CHIRPS 파일이 있다면:

```bash
python run_spi_generation.py
```

---

## 📋 상세 사용법

### 기본 실행 (기본 설정 사용)

```bash
python run_spi_generation.py
```

**기본 설정:**
- 지역: 동아프리카 (lon: 25-52°E, lat: 15°S-22°N)
- 기간: 2016-2024
- SPI 스케일: 1, 2, 3, 6, 9, 12개월
- 보정 기간: 1991-2020

### 시각화 포함 실행

```bash
python run_spi_generation.py --visualize
```

### CHIRPS 다운로드 + 30년 보정 (권장!) ⭐⭐⭐

```bash
python run_spi_generation.py \
  --download-chirps \
  --year-start 1991 \
  --year-end 2024 \
  --calibration-start 1991 \
  --calibration-end 2020 \
  --visualize
```

**장점:**
- ✅ 30년 보정 기간 (WMO 표준)
- ✅ 통계적으로 안정적인 SPI
- ✅ 신뢰할 수 있는 가뭄 지표

### 맞춤 설정 실행

```bash
python run_spi_generation.py \
  --download-chirps \
  --year-start 2010 --year-end 2024 \
  --scales 3 6 12 \
  --visualize \
  --viz-year 2024
```

### 기존 SPI 데이터 시각화만 하기

```bash
python run_spi_generation.py --skip-spi --visualize --viz-year 2024
```

---

## 🎛️ 옵션 설명

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--input`, `-i` | 입력 CHIRPS NetCDF 파일 경로 | `../../data/raw/chirps/chirps-v2.0.monthly.nc` |
| `--output`, `-o` | 출력 디렉토리 | `../../data/processed/spi` |
| `--lon-min` | 최소 경도 | `25` (동아프리카) |
| `--lon-max` | 최대 경도 | `52` (동아프리카) |
| `--lat-min` | 최소 위도 | `-15` (동아프리카) |
| `--lat-max` | 최대 위도 | `22` (동아프리카) |
| `--year-start` | 시작 연도 | `2016` |
| `--year-end` | 종료 연도 | `2024` |
| `--scales` | SPI 시간 스케일 (월) | `1 2 3 6 9 12` |
| `--calibration-start` | 보정 시작 연도 | `1991` |
| `--calibration-end` | 보정 종료 연도 | `2020` |
| `--visualize` | 시각화 생성 | `False` |
| `--viz-year` | 시각화할 연도 | `2024` |
| `--skip-spi` | SPI 생성 건너뛰기 | `False` |
| `--download-chirps` | CHIRPS 데이터 다운로드 | `False` |
| `--force-download` | 강제 재다운로드 | `False` |
| `--chirps-dir` | CHIRPS 저장 디렉토리 | `../../data/raw/chirps` |

---

## 📤 출력 결과

### 디렉토리 구조

```
data/processed/spi/
├── 01_clipped/                           # 클리핑된 원본 데이터
├── 02_filled/                            # 결측값 보간 완료
├── 03_metadata_revision/                 # 메타데이터 수정
├── 04_spi_intermediate/                  # SPI 중간 결과
├── 05_spi_final/                         # ⭐ 최종 SPI 파일
│   ├── east_africa_spi_gamma_01_month.nc
│   ├── east_africa_spi_gamma_02_month.nc
│   ├── east_africa_spi_gamma_03_month.nc
│   ├── east_africa_spi_gamma_06_month.nc
│   ├── east_africa_spi_gamma_09_month.nc
│   └── east_africa_spi_gamma_12_month.nc
├── visualizations/                       # 시각화 (--visualize 옵션 시)
│   ├── east_africa_spi3_2024.png
│   ├── east_africa_spi6_2024.png
│   └── east_africa_spi12_2024.png
└── spi_generation_report.txt             # 요약 보고서
```

### 생성되는 파일들

1. **SPI NetCDF 파일** (`05_spi_final/`)
   - 각 시간 스케일별 SPI 값
   - CF-compliant 형식
   - 압축 적용

2. **시각화 PNG** (`visualizations/`)
   - 월별 SPI 지도 (12개월 그리드)
   - 가뭄/습윤 상태 색상 표시
   - 고해상도 (300 DPI)

3. **요약 보고서** (`spi_generation_report.txt`)
   - 생성된 파일 목록
   - 기본 통계량
   - 가뭄 발생 빈도

---

## 📊 실행 예시

### 터미널 출력

```
**********************************************************************
  SPI GENERATION FOR EAST AFRICA
  Standardized Precipitation Index from CHIRPS Data
**********************************************************************

Started: 2024-12-13 12:30:00

Configuration:
  Input file: ../../data/raw/chirps/chirps-v2.0.monthly.nc
  Output directory: ../../data/processed/spi
  Region: Lon [25, 52], Lat [-15, 22]
  Time period: 2016-2024
  SPI scales: [1, 2, 3, 6, 9, 12]
  Calibration period: 1991-2020

============================================================
STEP 1: Clipping to East Africa region (2016-2024)
============================================================
Loading CHIRPS data from: ../../data/raw/chirps/chirps-v2.0.monthly.nc
Original shape: {'latitude': 2000, 'longitude': 7200, 'time': 504}
Selecting years: 2016-2024
Selecting lon: [25, 52], lat: [-15, 22]
Saving to: ../../data/processed/spi/01_clipped/east_africa_chirps_clipped.nc
✓ Clipped shape: {'latitude': 740, 'longitude': 540, 'time': 108}
...

**********************************************************************
  SPI GENERATION COMPLETED SUCCESSFULLY!
**********************************************************************

Final SPI files saved to: ../../data/processed/spi/05_spi_final/
```

---

## 💡 사용 팁

### 1. 첫 실행시 시간 단축

처음 실행할 때는 결측값 보간이 시간이 걸릴 수 있습니다. 필요한 SPI 스케일만 선택하면 시간을 절약할 수 있습니다:

```bash
# 주요 스케일만 계산 (3, 6, 12개월)
python run_spi_generation.py --scales 3 6 12
```

### 2. 다른 지역에 적용

```bash
# 예: 서아프리카
python run_spi_generation.py \
  --lon-min -20 --lon-max 20 \
  --lat-min 0 --lat-max 20
```

### 3. 최신 데이터만 처리

```bash
# 최근 5년만
python run_spi_generation.py --year-start 2019 --year-end 2024
```

### 4. 도움말 보기

```bash
python run_spi_generation.py --help
```

---

## 🔧 문제 해결

### 문제: `climate-indices` 설치 오류

**해결:**
```bash
pip install --upgrade pip
pip install numpy scipy xarray netCDF4
pip install climate-indices
```

### 문제: 메모리 부족

큰 지역이나 긴 시간 범위 처리 시 메모리가 부족할 수 있습니다.

**해결:**
- 더 작은 지역으로 나누기
- 시간 범위 줄이기
- SPI 스케일 수 줄이기

### 문제: CHIRPS 데이터 없음

**해결:**
```bash
cd ../../data/raw/chirps
wget https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/chirps-v2.0.monthly.nc
```

---

## 📚 관련 파일

- `generate_spi_python.py` - 핵심 SPI 생성 클래스
- `run_spi_generation.py` - 실행 스크립트 (이 파일)
- `Steps_to_Generate_SPI_Using_CHIRPS_Data.ipynb` - 상세 튜토리얼 노트북
- `SPI_GENERATION_README.md` - 전체 문서

---

## 📞 추가 도움말

더 자세한 정보는 다음을 참고하세요:
- 📖 상세 문서: `SPI_GENERATION_README.md`
- 📓 노트북: `Steps_to_Generate_SPI_Using_CHIRPS_Data.ipynb`
- 💻 소스 코드: `generate_spi_python.py`

---

**마지막 업데이트:** 2024년 12월

