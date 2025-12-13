# 🔧 SPI 생성 문제 해결 가이드

## ❌ 일반적인 오류와 해결 방법

---

## 1. "SPI calculation failed" - 보정 기간 불일치

### 오류 메시지
```
Error: Command '['spi', '--periodicity', 'monthly', ...]' returned non-zero exit status 1.
```

### 원인
**보정 기간(calibration period)**이 **데이터 범위**와 맞지 않음

예시:
- 데이터: 2016-2024 (9년)
- 보정 기간: 1991-2020 (30년)
- ❌ 1991-2015 데이터가 없음!

### 해결 방법

#### ✅ 자동 수정 (권장)
스크립트가 이제 자동으로 조정합니다:

```bash
python run_spi_generation.py
```

출력 예시:
```
⚠️  WARNING: Calibration period (1991-2020) is outside data range (2016-2024)
Adjusting calibration period to match data range...
✓ Adjusted calibration period: 2016-2020
```

#### ✅ 수동 지정

```bash
# 데이터 범위에 맞게 보정 기간 지정
python run_spi_generation.py \
  --year-start 2016 \
  --year-end 2024 \
  --calibration-start 2016 \
  --calibration-end 2020
```

**중요**: 최소 5년의 보정 기간이 필요합니다!

---

## 2. "calendar attribute" 오류

### 오류 메시지
```
Error: failed to prevent overwriting existing key calendar in attrs on variable 'time'
```

### 원인
xarray가 NetCDF 저장 시 `calendar` 속성 충돌

### 해결 방법
✅ **이미 수정됨!** 최신 버전의 스크립트 사용

업데이트하려면:
```bash
git pull  # 또는 최신 버전 다운로드
```

---

## 3. "No matching distribution found for matplotlib.pyplot"

### 오류 메시지
```
ERROR: No matching distribution found for matplotlib.pyplot
```

### 원인
`matplotlib.pyplot`은 패키지가 아니라 모듈입니다

### 해결 방법

```bash
# ❌ 잘못된 방법
pip install matplotlib.pyplot

# ✅ 올바른 방법
pip install matplotlib
```

Python 코드:
```python
import matplotlib.pyplot as plt  # pyplot은 자동 포함
```

---

## 4. venv 손상 오류

### 오류 메시지
```
OSError: [Errno 2] No such file or directory: '.../METADATA'
```

### 원인
가상 환경이 손상됨

### 해결 방법

```bash
# 1. 손상된 venv 삭제
cd /Users/halimjun/Coding_local/wpf_colla_v2
rm -rf venv

# 2. 새로 생성
python3 -m venv venv

# 3. 활성화
source venv/bin/activate

# 4. 패키지 재설치
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. "climate-indices not found"

### 오류 메시지
```
FileNotFoundError: 'spi' command not found
```

### 원인
`climate-indices` 패키지가 설치되지 않음

### 해결 방법

```bash
source venv/bin/activate
pip install climate-indices
```

---

## 6. 데이터가 너무 적음 오류

### 오류 메시지
```
ERROR: Need at least 5 years of data for SPI calibration
You have: 3 years (2022-2024)
```

### 원인
SPI 계산에는 최소 5년의 데이터 필요 (통계적 안정성)

### 해결 방법

#### 옵션 1: 더 많은 데이터 다운로드
```bash
# 더 긴 기간의 CHIRPS 데이터 다운로드
python run_spi_generation.py --year-start 2010 --year-end 2024
```

#### 옵션 2: 다른 SPI 스케일 사용
```bash
# 짧은 스케일만 사용 (1, 3개월)
python run_spi_generation.py --scales 1 3
```

**권장**: 최소 10-30년의 데이터 사용

---

## 7. 메모리 부족 오류

### 오류 메시지
```
MemoryError: Unable to allocate array...
```

### 원인
데이터가 너무 큼

### 해결 방법

#### 옵션 1: 더 작은 지역
```bash
python run_spi_generation.py \
  --lon-min 30 --lon-max 40 \
  --lat-min 0 --lat-max 10
```

#### 옵션 2: 더 짧은 기간
```bash
python run_spi_generation.py \
  --year-start 2020 --year-end 2024
```

#### 옵션 3: 더 적은 SPI 스케일
```bash
python run_spi_generation.py \
  --scales 6 12  # 6, 12개월만
```

---

## 8. 결측값이 너무 많음

### 경고 메시지
```
⚠️  High percentage of missing values: 65%
```

### 원인
입력 데이터에 결측값이 너무 많음

### 해결 방법

#### 전체 파이프라인 사용 (자동 보간)
```bash
# 결측값 자동 보간 포함
python run_spi_generation.py
```

#### SPI만 계산 시
```bash
# 먼저 데이터 전처리 필요
# 1. 전체 파이프라인으로 전처리
# 2. 그 다음 SPI만 계산
```

---

## 🔍 디버깅 체크리스트

### 1. venv가 활성화되었는지 확인
```bash
which python
# 출력: /path/to/wpf_colla_v2/venv/bin/python (venv 경로여야 함)
```

### 2. 필수 패키지 설치 확인
```bash
python -c "import climate_indices; print('✓ climate-indices')"
python -c "import xarray; print('✓ xarray')"
python -c "import matplotlib; print('✓ matplotlib')"
```

### 3. 데이터 파일 존재 확인
```bash
ls -lh data/raw/chirps/chirps-v2.0.monthly.nc
```

### 4. 데이터 범위 확인
```bash
python -c "
import xarray as xr
ds = xr.open_dataset('data/raw/chirps/chirps-v2.0.monthly.nc')
print(f'Time range: {ds.time.min().values} to {ds.time.max().values}')
"
```

---

## 📞 추가 도움말

### 로그 확인
```bash
# 자세한 오류 정보 보기
python run_spi_generation.py 2>&1 | tee spi_generation.log
```

### 테스트 실행
```bash
# 작은 데이터로 테스트
python run_spi_generation.py \
  --year-start 2020 \
  --year-end 2024 \
  --scales 6 12 \
  --calibration-start 2020 \
  --calibration-end 2024
```

### 도움말 보기
```bash
python run_spi_generation.py --help
python calculate_spi_only.py --help
```

---

## 📚 관련 문서

- **빠른 시작**: `QUICK_START.md`
- **SPI만 계산**: `SPI_ONLY_GUIDE.md`
- **상세 가이드**: `SPI_GENERATION_README.md`
- **튜토리얼**: `Steps_to_Generate_SPI_Using_CHIRPS_Data.ipynb`

---

**마지막 업데이트**: 2024년 12월

