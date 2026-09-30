# Expected-charge image generation

이 문서는 `scripts/`에서 SMILES를 expected-charge 이미지로 변환하는 방식과 대량
사전학습 데이터 생성 경로를 설명한다. 현재 production 경로는
`generate_expected_charge_dataset.py`이며, 기본 출력은 128×128 단일 채널
`electrostatic_potential` 텐서다.

## 어떤 스크립트를 사용하는가

| 스크립트 | 용도 | 기본 출력 |
| --- | --- | --- |
| `generate_expected_charge_dataset.py` | 대량 사전학습 데이터 생성 | 128×128 float16 단일 채널 Torch shard |
| `generate_expected_charge_samples.py` | 한 분자의 내부 성분 진단 | 2행 4열 컬러 PNG와 JSON manifest |
| `preview_layout_variants.py` | layout/H 영향 비교 | 네 가지 조건의 비교 PNG |
| `preview_electron_cloud.py` | 단일 분자의 전체 진단 | 성분별 컬러 PNG와 JSON |

대량 학습에는 첫 번째 스크립트를 사용한다. 나머지는 표현을 분석하는 도구이며,
학습 shard의 데이터 형식이 아니다.

## 생성 과정

### 1. 분자 전처리와 layout

1. RDKit으로 입력 SMILES를 파싱하고 canonical isomeric SMILES와 molecular key를
   만든다.
2. 수소를 명시적으로 포함해 전자 수와 수소의 영향력을 보존한다. production
   설정의 `hydrogen_influence`는 `1.0`이다.
3. 결정적인 원본 2D layout을 먼저 생성한다.
4. 서로 원자를 공유하지 않는 결합선의 proper crossing만 센다.
5. crossing이 0이면 원본 좌표를 그대로 사용한다. crossing이 있을 때만 후보
   layout을 생성하며, crossing 수가 실제로 감소하는 후보만 채택한다.
6. 좌표를 중심에 맞추고 median bond length와 canvas margin을 기준으로 정규화한다.

따라서 crossing 제거 때문에 이미 정상인 분자의 구조나 방향을 불필요하게 바꾸지
않는다. 원자 간 거리, 원자-결합 겹침, 종횡비, 고리 왜곡은 현재 production 선택
점수에 포함하지 않는다.

### 2. 전하 밀도 rasterization

원자핵은 정규화된 Gaussian primitive로, 결합·비공유 전자쌍·비편재화 전자는
방향성을 가진 elliptical Gaussian primitive로 표현한다. 각 primitive는 유한한
raster 위에서 먼저 정규화되므로 계수는 전자 수 단위를 유지한다.

핵 밀도와 전자 밀도는 다음 조건을 만족하도록 구성된다.

```text
Q(x, y) = rho_nuclei(x, y) - rho_electrons(x, y)
integral Q(x, y) dx dy ~= formal molecular charge
```

전자 밀도는 기대 전자 수에 맞게 다시 정규화한다. rasterization은 여러 분자의
primitive를 묶고 제한된 크기의 chunk로 처리해 메모리 사용량을 제어한다.

### 3. 선택 가능한 단일 채널

`--channel`은 다음 세 값 중 하나다.

| 채널 | 정의 | 용도 |
| --- | --- | --- |
| `electrostatic_potential` | softened Coulomb kernel과 `Q`의 선형 합성곱 | 기본 사전학습 target |
| `signed_charge` | 원시 `Q = rho_nuclei - rho_electrons` | 전하 밀도 분석 |
| `field` | `Q / (softsign_scale + abs(Q))` | bounded target 실험 |

기본 electrostatic potential은 다음과 같이 계산한다.

```text
G(r) = 1 / sqrt(r^2 + epsilon^2)
V = G * Q
V <- V - mean(V)
```

구현은 128×128 `Q`를 각 축 방향으로 2배 크기에 zero-padding한 뒤 FFT로 선형
합성곱한다. 이 padding은 원치 않는 circular wrap-around를 막는다. 마지막 평균값
제거는 전위의 임의 상수, 즉 gauge offset을 고정한다.

이 값은 코드가 정의한 `G * Q`가 맞지만, 3차원 전자 파동함수로 계산한 ab-initio
molecular electrostatic potential은 아니다. 전하 보존형 2D valence-charge 모델에
대한 deterministic potential surrogate다.

## 대량 데이터 생성

PowerShell 예시:

```powershell
uv run python scripts/generate_expected_charge_dataset.py `
  --input data/chembl.smi `
  --output data/expected-charge-128 `
  --channel electrostatic_potential `
  --resolution 128 `
  --batch-size 128 `
  --shard-size 4096 `
  --workers 4 `
  --validation-previews 8 `
  --device cuda
```

`--batch-size`를 생략하면 CUDA는 128, CPU는 32를 사용한다. `--workers`의 기본값은
CPU 수에 따라 정해지며 최대 4다. 실제 최적값은 분자 크기, 저장 장치, CPU와 GPU에
따라 달라질 수 있으므로 대표 데이터로 측정해야 한다.

지원 입력:

- Parquet
- CSV, CSV.GZ
- SMI, SMILES, TXT 및 각각의 GZip 파일
- SDF, SDF.GZ

Parquet/CSV에서 열 이름이 다르면 `--smiles-column`과 `--id-column`을 지정한다.
SMI 계열은 첫 token을 SMILES, 두 번째 token을 ID로 읽으며 ID가 없으면 행 번호를
사용한다.

## I/O와 계산 pipeline

현재 경로는 대량 변환에서 다음 병목을 줄인다.

- 입력은 4 MiB buffer로 순차 읽기하며 GZip에도 별도 buffered reader를 둔다.
- Parquet는 batch 전체를 Python dictionary로 바꾸지 않고 필요한 Arrow column만
  list로 변환한다.
- SMI는 binary line을 나눈 후 SMILES와 ID token만 UTF-8로 decode한다.
- RDKit compile 단계는 CPU thread pool에서 수행한다.
- CUDA에서는 현재 batch를 rasterize하는 동안 다음 batch의 RDKit compile을
  진행한다.
- GPU 결과는 pinned host memory로 non-blocking 복사하며 shard 경계의 CUDA event로
  완료를 확인한다.
- 직전 shard는 단일 background writer가 저장한다. 동시에 한 shard만 in-flight로
  유지해 RAM 증가를 제한한다.
- `torch.save`에는 16 MiB output buffer를 사용하고 molecular key는 한 번에 묶어
  기록한다.
- shard와 key sidecar 기록이 끝난 뒤에만 manifest를 commit한다.

CPU 렌더링에서는 RDKit와 rasterization이 같은 CPU 자원을 경쟁하므로 CUDA 경로와
같은 batch overlap을 강제로 적용하지 않는다.

## 출력 형식

각 `fields-NNNNNN.pt` shard에는 다음 값이 저장된다.

| key | 내용 |
| --- | --- |
| 선택한 채널 이름 | `[N, 1, 128, 128]` float16 텐서(기본 설정) |
| `formal_charge` | 정수 형식 전하 |
| `electron_count` | 기대 전자 수 |
| `integrated_charge` | raster에서 적분한 전하 |
| `canonical_smiles` | canonical isomeric SMILES |
| `molecule_key` | 중복 제거용 molecular key |
| `source_id` | 입력 데이터의 ID |

`manifest.json`에는 source 절대 경로, renderer/config hash, source offset, 성공·실패·
중복 수, 전역 통계와 shard 목록이 저장된다. 같은 source와 설정으로 다시 실행하면
마지막으로 commit된 offset부터 이어서 처리한다. 다른 source나 설정으로 같은 출력
디렉터리를 재사용하면 오류를 발생시켜 서로 다른 데이터가 섞이지 않게 한다.

기본적으로 exact molecular key 중복을 건너뛴다. 이미 unique가 보장된 대규모
입력이라면 `--no-deduplicate`로 sidecar와 in-memory key set 비용을 없앨 수 있다.

## validation preview

`--validation-previews N`은 처음 N개 결과를 `validation_previews/`에 PNG로 저장한다.
미리보기는 `coolwarm` 색상과 절댓값 99.5 percentile 기준의 대칭 색상 범위를
사용한다.

학습 텐서에는 색상, 제목, colorbar, 원자 표식이 들어가지 않는다. 현재 대량 생성
미리보기에도 검은 점이나 `x` 형태의 원자 좌표 overlay를 추가하지 않는다. PNG의
전체 canvas는 제목과 colorbar 때문에 128×128보다 클 수 있지만, 그 안에 표시되는
원시 field 데이터는 정확히 128×128이다.

## 검증

빠른 검증:

```powershell
uv run ruff check scripts/generate_expected_charge_dataset.py src/molai/dft/dataset.py
uv run pytest tests/test_structure_dataset.py tests/test_expected_charge.py -q
```

생성 후에는 다음 항목을 확인한다.

1. 채널 shape가 `[N, 1, 128, 128]`이고 dtype이 `torch.float16`인지 확인한다.
2. `integrated_charge`가 `formal_charge`와 수치 오차 안에서 일치하는지 확인한다.
3. 같은 입력과 설정으로 다시 실행했을 때 새 shard 없이 resume되는지 확인한다.
4. 코드 최적화 전후의 텐서를 `torch.equal`로 비교한다.
5. validation PNG에서 crossing과 field 형태를 보고, 좌표 표식이 추가되지 않았는지
   확인한다.

카페인과 ATP 예시 입력:

```text
CN1C=NC2=C1C(=O)N(C(=O)N2C)C caffeine
C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])O)O)N atp-4
```

ATP 예시는 형식 전하 `-4`, 카페인은 `0`이어야 한다.
