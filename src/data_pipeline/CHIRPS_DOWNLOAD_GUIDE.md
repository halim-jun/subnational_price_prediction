# 📥 CHIRPS 데이터 다운로드 가이드

## 🎯 왜 더 긴 기간이 필요한가?

### SPI 보정에는 충분한 데이터가 필요!

```
❌ 짧은 기간 (5-10년)
   → 통계적으로 불안정
   → SPI 값의 신뢰도 낮음

✅ 긴 기간 (30년+)
   → 통계적으로 안정
   → "정상" 강수량을 정확히 정의
   → WMO 권장: 최소 30년!
```

---

## 🚀 빠른 시작

### 방법 1: 스크립트로 자동 다운로드 (권장) ⭐

```bash
cd /Users/halimjun/Coding_local/wpf_colla_v2
source venv/bin/activate
cd src/data_pipeline

# CHIRPS 다운로드 + SPI 생성 (한 번에!)
python run_spi_generation.py --download-chirps
```

**끝!** 스크립트가 알아서:
1. 📥 CHIRPS 전체 데이터 다운로드 (~7 GB, 1981-현재)
2. 🔄 동아프리카 지역 추출
3. 📊 SPI 계산 (30년 이상 보정 가능!)

---

### 방법 2: 수동 다운로드

```bash
# 1. 디렉토리 생성
mkdir -p ../../data/raw/chirps
cd ../../data/raw/chirps

# 2. wget으로 다운로드
wget https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/chirps-v2.0.monthly.nc

# 또는 curl
curl -o chirps-v2.0.monthly.nc https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/chirps-v2.0.monthly.nc

# 3. 다운로드 확인
ls -lh chirps-v2.0.monthly.nc
```

---

## 📋 자동 다운로드 상세 사용법

### 기본 다운로드 + SPI 생성

```bash
python run_spi_generation.py --download-chirps
```

**자동 처리:**
- ✅ 파일 존재 확인
- ✅ 이미 있으면 다운로드 건너뜀
- ✅ 데이터 범위 자동 확인
- ✅ 보정 기간 자동 조정

**출력 예시:**
```
============================================================
CHIRPS DATA DOWNLOAD
============================================================

✓ CHIRPS file already exists!
  Location: ../../data/raw/chirps/chirps-v2.0.monthly.nc
  Size: 7068.5 MB
  Time range: 1981-01-01 to 2024-11-01

Use --force-download to re-download
```

### 강제 재다운로드

```bash
# 파일이 손상되었거나 업데이트된 경우
python run_spi_generation.py --download-chirps --force-download
```

### 다른 디렉토리에 다운로드

```bash
python run_spi_generation.py \
  --download-chirps \
  --chirps-dir /path/to/my/data
```

### 다운로드만 하고 SPI는 나중에

```bash
python run_spi_generation.py \
  --download-chirps \
  --skip-spi
```

---

## 📊 CHIRPS 데이터 정보

### 기본 정보

| 항목 | 내용 |
|------|------|
| **시간 범위** | 1981년 1월 ~ 현재 (44년+) |
| **시간 해상도** | 월별 (monthly) |
| **공간 범위** | 전 지구 (50°S - 50°N) |
| **공간 해상도** | 0.05° (~5.5 km) |
| **파일 크기** | ~7 GB |
| **변수** | precipitation (mm/month) |

### 다운로드 시간 예상

| 인터넷 속도 | 예상 시간 |
|------------|----------|
| 100 Mbps | ~10분 |
| 50 Mbps | ~20분 |
| 10 Mbps | ~1.5시간 |

---

## 🎯 권장 사용 시나리오

### 시나리오 1: 처음 시작 (권장)

```bash
# 한 번에 모두 처리
python run_spi_generation.py \
  --download-chirps \
  --year-start 1991 \
  --year-end 2024 \
  --calibration-start 1991 \
  --calibration-end 2020 \
  --visualize
```

**장점:**
- ✅ 30년 보정 기간 (1991-2020)
- ✅ WMO 표준 준수
- ✅ 통계적으로 안정적

### 시나리오 2: 최대 기간 사용

```bash
# 전체 CHIRPS 기간 활용 (1981-현재)
python run_spi_generation.py \
  --download-chirps \
  --year-start 1981 \
  --year-end 2024 \
  --calibration-start 1981 \
  --calibration-end 2010 \
  --visualize
```

**장점:**
- ✅ 44년의 역사적 맥락
- ✅ 극단적 이벤트 포함
- ✅ 가장 신뢰할 수 있는 SPI

### 시나리오 3: 최신 데이터만 (빠른 테스트)

```bash
# 최근 10년만 (테스트용)
python run_spi_generation.py \
  --download-chirps \
  --year-start 2014 \
  --year-end 2024 \
  --calibration-start 2014 \
  --calibration-end 2019 \
  --scales 6 12
```

---

## 🔍 다운로드 후 데이터 확인

### Python으로 확인

```python
import xarray as xr

# 파일 열기
ds = xr.open_dataset('../../data/raw/chirps/chirps-v2.0.monthly.nc')

# 기본 정보
print("Dimensions:", ds.dims)
print("Variables:", list(ds.data_vars))
print("Time range:", ds.time.min().values, "to", ds.time.max().values)

# 시간 범위 확인
years = int(ds.time.dt.year.max().values) - int(ds.time.dt.year.min().values) + 1
print(f"Total years: {years}")

# 닫기
ds.close()
```

### 명령줄로 확인

```bash
# NetCDF 헤더 보기
ncdump -h chirps-v2.0.monthly.nc | head -50

# 또는 Python one-liner
python -c "import xarray as xr; ds=xr.open_dataset('chirps-v2.0.monthly.nc'); print(ds)"
```

---

## ⚡ 다운로드 속도 향상

### 옵션 1: aria2 사용 (멀티 커넥션)

```bash
# aria2 설치
brew install aria2  # Mac
sudo apt install aria2  # Ubuntu

# 다운로드 (16개 연결)
aria2c -x 16 -s 16 \
  https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/chirps-v2.0.monthly.nc
```

### 옵션 2: 백그라운드 다운로드

```bash
# 백그라운드에서 실행
nohup python run_spi_generation.py --download-chirps > download.log 2>&1 &

# 진행 상황 확인
tail -f download.log
```

---

## 🔧 문제 해결

### 문제 1: 다운로드 중단됨

```bash
# wget으로 이어받기
cd ../../data/raw/chirps
wget -c https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/chirps-v2.0.monthly.nc
```

### 문제 2: 디스크 공간 부족

```bash
# 필요한 공간: ~10 GB (원본 + 처리된 파일)
df -h .

# 공간 확보 후 다시 시도
```

### 문제 3: 느린 다운로드 속도

```bash
# 미러 사이트 사용 (있는 경우)
# 또는 aria2로 멀티 커넥션 다운로드
```

### 문제 4: 연결 시간 초과

```bash
# 타임아웃 늘리기
wget --timeout=300 --tries=5 \
  https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/chirps-v2.0.monthly.nc
```

---

## 📚 CHIRPS 데이터 출처

**CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data)**

- **제공**: Climate Hazards Center, UC Santa Barbara
- **웹사이트**: https://www.chc.ucsb.edu/data/chirps
- **논문**: Funk et al. (2015), Scientific Data
  - DOI: 10.1038/sdata.2015.66
- **라이선스**: Public domain (자유롭게 사용 가능)

---

## 💡 팁

### 1. 첫 실행 시

```bash
# 다운로드 + 전체 파이프라인 (한 번에)
python run_spi_generation.py \
  --download-chirps \
  --year-start 1991 \
  --year-end 2024 \
  --visualize
```

### 2. 업데이트된 데이터 받기

```bash
# 매달 새 데이터가 추가됨
python run_spi_generation.py --download-chirps --force-download
```

### 3. 데이터 재사용

```bash
# 다운로드는 한 번만, SPI는 여러 번
python run_spi_generation.py --scales 3 6 12
python run_spi_generation.py --scales 1 2 3 --year-start 2010
```

---

## 🎯 완전 자동화 예시

```bash
#!/bin/bash
# complete_spi_generation.sh

cd /Users/halimjun/Coding_local/wpf_colla_v2
source venv/bin/activate
cd src/data_pipeline

echo "Starting complete SPI generation pipeline..."

# 1. CHIRPS 다운로드
python run_spi_generation.py --download-chirps --skip-spi

# 2. SPI 생성 (30년 보정)
python run_spi_generation.py \
  --year-start 1991 \
  --year-end 2024 \
  --calibration-start 1991 \
  --calibration-end 2020 \
  --scales 1 3 6 12 \
  --visualize

echo "Complete! Check results in ../../data/processed/spi/"
```

실행:
```bash
chmod +x complete_spi_generation.sh
./complete_spi_generation.sh
```

---

**마지막 업데이트**: 2024년 12월

