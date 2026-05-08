# DESCRIPTION
## 1. 프로젝트 전체 목적

이 프로젝트는 MNIST 데이터셋을 이용해서 손글씨 숫자 이미지를 `0~9` 중 하나로 분류하는 Python 프로그램입니다.

MNIST는 다음 형태의 데이터셋입니다.

- 입력: 28x28 픽셀 grayscale 이미지
- 출력: 숫자 label, 즉 `0`, `1`, `2`, ..., `9`
- train set: 60,000개
- test set: 10,000개

이 프로젝트는 PyTorch나 TensorFlow 같은 deep learning framework를 쓰지 않고, `NumPy`만으로 MLP 모델을 직접 구현합니다. 그래서 “모델이 내부에서 어떻게 계산되는지”를 보기 좋은 교육용 구조입니다.

전체 흐름은 다음과 같습니다.

```text
_data/*.gz
   ↓
src/mnist_loader.py
   ↓
NumPy 배열 x_train, y_train, x_test, y_test
   ↓
src/model.py 의 MLP
   ↓
src/train.py 로 학습
   ↓
models/mnist_mlp.npz 저장
   ↓
src/evaluate.py 로 정확도 평가
src/predict.py 로 개별 예측
```

## 2. 폴더 구조

현재 핵심 구조는 이렇습니다.

```text
260410_mnist/
  _data/
    train-images-idx3-ubyte.gz
    train-labels-idx1-ubyte.gz
    t10k-images-idx3-ubyte.gz
    t10k-labels-idx1-ubyte.gz

  src/
    __init__.py
    mnist_loader.py
    model.py
    train.py
    evaluate.py
    predict.py

  tests/
    test_mnist_loader.py
    test_model.py

  models/
    mnist_mlp.npz

  README.md
  .gitignore
  .venv/
```

각 폴더의 책임은 다음과 같습니다.

- `_data/`: 원본 MNIST 데이터 파일 저장소입니다. 학습 코드가 이 파일을 읽습니다.
- `src/`: 실제 프로그램 코드입니다.
- `tests/`: 데이터 로더와 모델의 최소 동작을 검증합니다.
- `models/`: 학습 후 저장된 모델 파일이 들어갑니다.
- `.venv/`: 프로젝트 전용 Python 가상환경입니다.
- `README.md`: 실행 방법 문서입니다.
- `.gitignore`: git에 올리지 않을 파일 목록입니다.

`_data/`, `models/`, `.venv/`는 일반적으로 git에 올리지 않습니다. 데이터와 학습 결과물, 가상환경은 크거나 환경별로 달라지기 때문입니다.

## 3. 실행 흐름

가장 기본적인 실행 순서는 다음입니다.

```bash
.venv/bin/python -m src.train
```

이 명령은 다음 일을 합니다.

1. `_data`에서 MNIST gzip 파일을 읽습니다.
2. 이미지를 `(60000, 784)` 형태의 NumPy 배열로 바꿉니다.
3. MLP 모델을 생성합니다.
4. mini-batch 단위로 학습합니다.
5. epoch마다 test accuracy를 출력합니다.
6. 학습된 가중치를 `models/mnist_mlp.npz`에 저장합니다.

그다음 평가:

```bash
.venv/bin/python -m src.evaluate
```

이 명령은 저장된 모델을 불러와 test set 전체 정확도와 confusion matrix를 출력합니다.

개별 예측:

```bash
.venv/bin/python -m src.predict --index 0
```

이 명령은 MNIST test set의 0번째 이미지를 모델에 넣고 예측값과 실제값을 출력합니다.

이미지 파일 예측:

```bash
.venv/bin/python -m src.predict --image sample.png
```

이 명령은 외부 이미지 파일을 28x28 grayscale로 바꿔 모델에 넣습니다.

## 4. 기본 개념: 이미지가 모델에 들어가는 방식

MNIST 이미지는 28x28입니다.

```text
28 rows × 28 columns = 784 pixels
```

컴퓨터 입장에서는 이미지를 2차원 그림으로 처리할 수도 있지만, 이 프로젝트의 MLP는 1차원 벡터를 입력으로 받습니다.

그래서 이미지를 이렇게 펼칩니다.

```text
28x28 이미지
↓
길이 784짜리 벡터
```

예를 들면 원래 shape이 다음이라면:

```text
(60000, 28, 28)
```

모델 입력용으로는 다음이 됩니다.

```text
(60000, 784)
```

각 픽셀 값은 원래 `0~255` 정수입니다.

- `0`: 검정
- `255`: 흰색

모델 학습에는 보통 값을 작게 정규화해서 씁니다.

```python
pixel / 255.0
```

그래서 최종 값 범위는 `0.0~1.0`입니다.

## 5. [src/mnist_loader.py](/Users/choies/Code_School/260410_mnist/src/mnist_loader.py:1)

이 파일의 책임은 “MNIST 원본 gzip IDX 파일을 NumPy 배열로 바꾸는 것”입니다.

파일 상단:

```python
import gzip
import struct
from pathlib import Path

import numpy as np
```

각 라이브러리의 의미는 다음과 같습니다.

- `gzip`: `.gz` 압축 파일을 Python에서 직접 열기 위해 필요합니다.
- `struct`: binary file header를 정해진 byte 구조로 해석하기 위해 필요합니다.
- `Path`: 문자열 경로보다 안전하고 읽기 쉬운 파일 경로 처리를 위해 사용합니다.
- `numpy`: 대량 숫자 배열을 빠르게 처리하기 위해 사용합니다.

MNIST 원본 파일은 텍스트 파일이 아닙니다. 사람이 읽는 CSV가 아니라 binary IDX format입니다. 그래서 `open().readlines()` 같은 방식으로 읽을 수 없습니다.

파일명 상수:

```python
TRAIN_IMAGES = "train-images-idx3-ubyte.gz"
TRAIN_LABELS = "train-labels-idx1-ubyte.gz"
TEST_IMAGES = "t10k-images-idx3-ubyte.gz"
TEST_LABELS = "t10k-labels-idx1-ubyte.gz"
```

이 상수들은 `_data/` 안에 있어야 하는 표준 MNIST 파일 이름입니다.

### 5.1 `_read_images`

위치: [src/mnist_loader.py](/Users/choies/Code_School/260410_mnist/src/mnist_loader.py:18)

```python
def _read_images(path: Path) -> np.ndarray:
```

앞의 `_`는 “이 함수는 이 파일 내부에서만 쓰는 private helper 성격”이라는 관례입니다.

핵심 코드:

```python
with gzip.open(path, "rb") as file:
```

- `gzip.open`: gzip 압축 파일을 엽니다.
- `"rb"`: read binary mode입니다. MNIST는 binary이므로 반드시 binary로 읽습니다.
- `with`: 파일을 다 쓴 후 자동으로 닫아줍니다.

```python
magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
```

MNIST image file의 첫 16 bytes는 header입니다.

- `magic`: 파일 종류 식별자
- `count`: 이미지 개수
- `rows`: 행 수, MNIST는 28
- `cols`: 열 수, MNIST는 28

`">IIII"`의 의미:

- `>`: big-endian byte order
- `I`: unsigned int 4 bytes
- `IIII`: 4-byte integer 4개를 읽겠다는 뜻

즉 16 bytes를 4개의 정수로 해석합니다.

```python
if magic != 2051:
    raise ValueError(...)
```

MNIST image file의 magic number는 `2051`입니다. 이 값이 아니면 image 파일이 아니거나 손상된 파일입니다.

```python
data = np.frombuffer(file.read(), dtype=np.uint8)
```

나머지 모든 byte를 NumPy 배열로 바꿉니다.

- `np.frombuffer`: byte 데이터를 복사 없이 배열처럼 해석합니다.
- `dtype=np.uint8`: 픽셀은 `0~255`이므로 unsigned 8-bit integer입니다.

```python
expected = count * rows * cols
```

예상 픽셀 개수입니다.

MNIST train image라면:

```text
60000 * 28 * 28 = 47,040,000
```

```python
if data.size != expected:
    raise ValueError(...)
```

파일 크기가 header 정보와 맞지 않으면 데이터 손상을 의심해야 하므로 예외를 냅니다.

```python
return data.reshape(count, rows * cols).astype(np.float32) / 255.0
```

여기가 전처리 핵심입니다.

- `reshape(count, rows * cols)`: `(60000, 28, 28)`이 아니라 `(60000, 784)`로 펼칩니다.
- `astype(np.float32)`: 모델 계산을 위해 실수형으로 바꿉니다.
- `/ 255.0`: 픽셀 값을 `0.0~1.0`으로 정규화합니다.

### 5.2 `_read_labels`

위치: [src/mnist_loader.py](/Users/choies/Code_School/260410_mnist/src/mnist_loader.py:32)

label 파일은 image 파일보다 단순합니다.

```python
magic, count = struct.unpack(">II", file.read(8))
```

label file header는 8 bytes입니다.

- `magic`: label 파일 식별자
- `count`: label 개수

```python
if magic != 2049:
```

MNIST label file의 magic number는 `2049`입니다.

```python
labels = np.frombuffer(file.read(), dtype=np.uint8)
```

label은 `0~9` 숫자이므로 `uint8`로 읽습니다.

```python
return labels.astype(np.int64)
```

최종적으로 label은 `int64`로 바꿉니다. 이유는 NumPy에서 indexing할 때 integer type이 안정적이고, loss 계산에서 class index로 사용하기 때문입니다.

이 프로젝트는 label을 one-hot으로 바꾸지 않습니다.

즉 label이 다음처럼 유지됩니다.

```python
[7, 2, 1, 0, 4, ...]
```

one-hot을 쓰면 다음처럼 됩니다.

```python
7 -> [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
```

하지만 이 프로젝트의 cross entropy 계산은 정수 label을 바로 사용하므로 one-hot이 필요 없습니다.

### 5.3 `load_mnist`

위치: [src/mnist_loader.py](/Users/choies/Code_School/260410_mnist/src/mnist_loader.py:45)

```python
def load_mnist(data_dir: str | Path = "_data") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
```

이 함수가 외부에서 쓰는 메인 API입니다.

반환값은 항상 이 순서입니다.

```python
x_train, y_train, x_test, y_test
```

각 shape은 다음입니다.

```text
x_train: (60000, 784)
y_train: (60000,)
x_test:  (10000, 784)
y_test:  (10000,)
```

`train.py`, `evaluate.py`, `predict.py` 모두 이 함수를 통해 데이터를 읽습니다.

## 6. [src/model.py](/Users/choies/Code_School/260410_mnist/src/model.py:1)

이 파일은 실제 neural network 모델을 담당합니다.

모델 이름은 `MLP`입니다.

MLP는 Multi-Layer Perceptron의 약자입니다. 쉽게 말하면 “fully connected layer를 여러 층 쌓은 신경망”입니다.

이 프로젝트의 구조는 다음입니다.

```text
입력층: 784
은닉층: 128
출력층: 10
```

숫자로 보면:

```text
x:       (batch_size, 784)
w1:      (784, 128)
b1:      (128,)
hidden:  (batch_size, 128)
w2:      (128, 10)
b2:      (10,)
output:  (batch_size, 10)
```

출력 10개는 각각 숫자 `0~9`일 확률입니다.

### 6.1 `__init__`

위치: [src/model.py](/Users/choies/Code_School/260410_mnist/src/model.py:10)

```python
class MLP:
    def __init__(...)
```

모델 객체를 만들 때 가중치와 bias를 초기화합니다.

```python
input_size: int = 784
hidden_size: int = 128
output_size: int = 10
seed: int = 42
```

- `input_size=784`: 28x28 이미지이기 때문입니다.
- `hidden_size=128`: 은닉층 neuron 개수입니다.
- `output_size=10`: 숫자 class가 10개이기 때문입니다.
- `seed=42`: 난수 초기화를 재현 가능하게 하기 위해 사용합니다.

```python
rng = np.random.default_rng(seed)
```

NumPy의 modern random generator입니다. 같은 seed를 쓰면 같은 초기 가중치가 만들어집니다. 실험 재현성을 위해 중요합니다.

```python
self.w1 = rng.standard_normal((input_size, hidden_size)) * np.sqrt(2.0 / input_size)
```

`w1`은 입력층에서 은닉층으로 가는 가중치입니다.

shape:

```text
(784, 128)
```

왜 `np.sqrt(2.0 / input_size)`를 곱하나요?

이것은 ReLU 계열 activation에 자주 쓰는 He initialization에 가까운 방식입니다. 가중치를 너무 크게 시작하면 값이 폭발하고, 너무 작게 시작하면 gradient가 약해질 수 있습니다. 적절한 scale로 시작하기 위해 사용합니다.

```python
self.b1 = np.zeros(hidden_size, dtype=np.float32)
```

bias는 0으로 시작합니다. bias는 각 neuron에 더해지는 보정값입니다.

```python
self.w2
self.b2
```

두 번째 layer의 가중치와 bias입니다.

- `w2`: hidden에서 output으로 가는 가중치, shape `(128, 10)`
- `b2`: output bias, shape `(10,)`

```python
self._cache
self._grads
```

- `_cache`: forward 때 계산한 중간값을 저장합니다. backward에서 필요합니다.
- `_grads`: backward 때 계산한 gradient를 저장합니다. step에서 필요합니다.

### 6.2 Softmax

위치: [src/model.py](/Users/choies/Code_School/260410_mnist/src/model.py:35)

```python
@staticmethod
def _softmax(logits: np.ndarray) -> np.ndarray:
```

softmax는 raw score를 확률처럼 바꾸는 함수입니다.

모델의 마지막 layer는 숫자 10개에 대한 점수, 즉 `logits`를 만듭니다.

예:

```text
[1.2, -0.3, 2.1, ...]
```

이 값은 확률이 아닙니다. 합이 1도 아니고 음수도 있을 수 있습니다.

softmax를 적용하면:

```text
[0.12, 0.03, 0.25, ...]
```

처럼 모든 값이 양수이고, row별 합이 1이 됩니다.

```python
shifted = logits - np.max(logits, axis=1, keepdims=True)
```

이 코드는 numerical stability를 위한 것입니다.

softmax는 내부적으로 `exp()`를 씁니다. 큰 값에 `exp()`를 적용하면 overflow가 날 수 있습니다. 예를 들어 `exp(1000)`은 너무 큽니다.

그래서 row별 최대값을 빼서 가장 큰 logit이 0이 되게 합니다. softmax 결과는 상수 하나를 전체에서 빼도 변하지 않습니다.

```python
exp = np.exp(shifted)
return exp / np.sum(exp, axis=1, keepdims=True)
```

각 값을 exponentiate하고, row sum으로 나눠 확률 분포로 만듭니다.

### 6.3 `forward`

위치: [src/model.py](/Users/choies/Code_School/260410_mnist/src/model.py:41)

forward는 입력을 모델에 통과시켜 예측 확률을 만드는 과정입니다.

```python
x = x.astype(np.float32, copy=False)
```

입력을 float32로 맞춥니다. `copy=False`는 가능하면 복사하지 말라는 뜻입니다.

```python
hidden_linear = x @ self.w1 + self.b1
```

첫 번째 선형 변환입니다.

수식:

```text
hidden_linear = xW1 + b1
```

`@`는 Python의 matrix multiplication operator입니다.

shape:

```text
x:             (batch_size, 784)
w1:            (784, 128)
x @ w1:        (batch_size, 128)
b1:            (128,)
hidden_linear: (batch_size, 128)
```

NumPy는 `b1`을 batch 전체에 자동으로 broadcast합니다.

```python
hidden = np.maximum(hidden_linear, 0.0)
```

ReLU activation입니다.

ReLU는 다음 함수입니다.

```text
ReLU(z) = max(z, 0)
```

음수는 0으로 만들고, 양수는 그대로 둡니다.

왜 ReLU가 필요한가요?

선형 변환만 여러 번 쌓으면 결국 하나의 선형 변환과 같습니다. 신경망이 복잡한 패턴을 학습하려면 비선형 activation이 필요합니다.

```python
logits = hidden @ self.w2 + self.b2
```

두 번째 선형 변환입니다.

shape:

```text
hidden: (batch_size, 128)
w2:     (128, 10)
logits: (batch_size, 10)
```

```python
probabilities = self._softmax(logits)
```

숫자 10개에 대한 확률로 바꿉니다.

```python
self._cache = (x, hidden_linear, hidden)
```

backward 때 필요한 중간값을 저장합니다.

왜 저장하나요?

gradient 계산에는 forward 때의 입력값과 activation 결과가 필요합니다. 예를 들어 `dw2`를 계산하려면 `hidden`이 필요하고, ReLU gradient를 계산하려면 `hidden_linear`가 필요합니다.

### 6.4 `predict`

위치: [src/model.py](/Users/choies/Code_School/260410_mnist/src/model.py:50)

```python
return np.argmax(self.forward(x), axis=1)
```

`forward(x)`는 각 class 확률을 반환합니다.

예:

```text
[0.01, 0.02, 0.03, 0.90, ...]
```

`np.argmax(..., axis=1)`은 row별로 가장 큰 값의 index를 반환합니다.

가장 큰 확률이 index 3이면 예측 숫자는 `3`입니다.

### 6.5 `loss`

위치: [src/model.py](/Users/choies/Code_School/260410_mnist/src/model.py:53)

loss는 모델이 얼마나 틀렸는지를 수치로 나타냅니다. 학습은 이 loss를 줄이는 방향으로 진행됩니다.

```python
clipped = np.clip(probabilities[np.arange(labels.size), labels], 1e-12, 1.0)
```

여기서 중요한 부분은 advanced indexing입니다.

예를 들어 batch label이:

```python
labels = [7, 2, 1]
```

이고 probabilities shape이 `(3, 10)`이면:

```python
probabilities[np.arange(3), labels]
```

는 다음을 뽑습니다.

```text
0번째 샘플의 7번 확률
1번째 샘플의 2번 확률
2번째 샘플의 1번 확률
```

즉 “정답 class에 모델이 부여한 확률”만 뽑습니다.

`np.clip(..., 1e-12, 1.0)`은 log(0)을 막기 위한 장치입니다. 확률이 0이면 `log(0)`은 정의되지 않아 `-inf`가 됩니다.

```python
return float(-np.mean(np.log(clipped)))
```

cross entropy loss입니다.

정답 확률이 높을수록 loss는 작아집니다.

예:

```text
정답 확률 0.9 -> -log(0.9) 작음
정답 확률 0.1 -> -log(0.1) 큼
```

### 6.6 `backward`

위치: [src/model.py](/Users/choies/Code_School/260410_mnist/src/model.py:58)

backward는 backpropagation, 즉 역전파입니다. 모델의 가중치를 어떻게 바꿔야 loss가 줄어드는지 gradient를 계산합니다.

```python
if self._cache is None:
    raise RuntimeError("forward must be called before backward")
```

backward는 forward 결과가 있어야 계산할 수 있습니다.

```python
x, hidden_linear, hidden = self._cache
```

forward에서 저장한 중간값을 꺼냅니다.

```python
dlogits = probabilities.copy()
dlogits[np.arange(batch_size), labels] -= 1.0
dlogits /= batch_size
```

softmax + cross entropy 조합에서 gradient는 간단해집니다.

```text
dlogits = probabilities - one_hot(labels)
```

정답 class 위치에서 1을 빼는 것이 one-hot label을 빼는 효과입니다.

`/ batch_size`는 batch 평균 loss에 대한 gradient로 맞추기 위한 것입니다.

```python
dw2 = hidden.T @ dlogits
db2 = np.sum(dlogits, axis=0)
```

두 번째 layer의 gradient입니다.

- `dw2`: `w2`를 얼마나 바꿀지
- `db2`: `b2`를 얼마나 바꿀지

```python
dhidden = dlogits @ self.w2.T
```

출력층 gradient를 은닉층 방향으로 되돌립니다.

```python
dhidden[hidden_linear <= 0.0] = 0.0
```

ReLU의 derivative입니다.

ReLU는 입력이 0 이하였던 곳에서는 gradient가 0입니다. 왜냐하면 그 구간에서는 출력이 계속 0이기 때문입니다.

```python
dw1 = x.T @ dhidden
db1 = np.sum(dhidden, axis=0)
```

첫 번째 layer의 gradient입니다.

```python
self._grads = (...)
```

계산한 gradient를 저장합니다. 실제 가중치 업데이트는 `step()`에서 합니다.

### 6.7 `step`

위치: [src/model.py](/Users/choies/Code_School/260410_mnist/src/model.py:84)

```python
self.w1 -= learning_rate * dw1
```

이것이 gradient descent입니다.

개념적으로는 다음입니다.

```text
새 가중치 = 기존 가중치 - learning_rate × gradient
```

gradient는 loss가 증가하는 방향입니다. 그래서 빼면 loss가 줄어드는 방향으로 이동합니다.

`learning_rate`는 한 번에 얼마나 움직일지 정하는 값입니다.

- 너무 크면 학습이 불안정합니다.
- 너무 작으면 학습이 느립니다.

### 6.8 `save`와 `load`

위치:
- [src/model.py](/Users/choies/Code_School/260410_mnist/src/model.py:94)
- [src/model.py](/Users/choies/Code_School/260410_mnist/src/model.py:109)

```python
np.savez(...)
```

`np.savez`는 여러 NumPy 배열을 하나의 `.npz` 파일에 저장합니다.

여기서는 다음을 저장합니다.

- 모델 구조 정보: `input_size`, `hidden_size`, `output_size`, `seed`
- 학습된 parameter: `w1`, `b1`, `w2`, `b2`

왜 저장해야 하나요?

학습에는 시간이 걸립니다. 매번 예측할 때마다 다시 학습하면 비효율적입니다. 한 번 학습한 가중치를 저장해두고 평가/예측 때 불러옵니다.

## 7. [src/train.py](/Users/choies/Code_School/260410_mnist/src/train.py:1)

이 파일은 학습 실행용입니다.

```bash
.venv/bin/python -m src.train
```

명령으로 실행합니다.

`-m src.train`은 Python에게 “`src` 패키지 안의 `train` 모듈을 실행하라”는 뜻입니다. 그래서 파일 내부의 relative import가 안정적으로 동작합니다.

### 7.1 `accuracy`

위치: [src/train.py](/Users/choies/Code_School/260410_mnist/src/train.py:14)

```python
def accuracy(model, x, y, batch_size=1024):
```

모델 정확도를 계산합니다.

```python
for start in range(0, x.shape[0], batch_size):
```

한 번에 전체 test set을 예측하지 않고 batch로 나눕니다.

왜 나누나요?

10,000개 정도는 한 번에 해도 되지만, 일반적으로 큰 데이터에서는 메모리 사용량을 줄이기 위해 batch 평가를 합니다.

```python
correct += int(np.sum(model.predict(x[start:end]) == y[start:end]))
```

예측값과 정답이 같은 개수를 셉니다.

```python
return correct / y.size
```

정확도는 다음입니다.

```text
맞춘 개수 / 전체 개수
```

### 7.2 `train`

위치: [src/train.py](/Users/choies/Code_School/260410_mnist/src/train.py:22)

기본값:

```python
epochs=5
batch_size=128
learning_rate=0.1
hidden_size=128
seed=42
```

용어 정리:

- `epoch`: 전체 train data를 한 번 다 보는 단위입니다.
- `batch`: 데이터를 작게 나눈 묶음입니다.
- `mini-batch SGD`: 전체 데이터를 한 번에 쓰지 않고 batch마다 gradient를 계산해 업데이트하는 방식입니다.
- `learning_rate`: 업데이트 크기입니다.

```python
x_train, y_train, x_test, y_test = load_mnist(data_dir)
```

데이터를 로드합니다.

```python
model = MLP(hidden_size=hidden_size, seed=seed)
```

모델을 생성합니다.

```python
indices = rng.permutation(x_train.shape[0])
```

매 epoch마다 train data 순서를 섞습니다.

왜 섞나요?

항상 같은 순서로 학습하면 데이터 순서에 모델이 영향을 받을 수 있습니다. mini-batch 학습에서는 shuffle이 일반적입니다.

```python
batch_indices = indices[start : start + batch_size]
x_batch = x_train[batch_indices]
y_batch = y_train[batch_indices]
```

이번 batch에 해당하는 입력과 label을 가져옵니다.

학습의 핵심 loop:

```python
probabilities = model.forward(x_batch)
batch_loss = model.loss(probabilities, y_batch)
model.backward(probabilities, y_batch)
model.step(learning_rate)
```

순서가 중요합니다.

1. `forward`: 예측 확률 계산
2. `loss`: 얼마나 틀렸는지 계산
3. `backward`: gradient 계산
4. `step`: 가중치 업데이트

```python
test_accuracy = accuracy(model, x_test, y_test)
```

epoch마다 test set 정확도를 계산합니다.

주의할 점: 실제 연구/실무에서는 test set을 자주 보면서 튜닝하면 안 됩니다. validation set을 따로 두는 것이 더 엄밀합니다. 하지만 이 프로젝트는 교육용이고 단순 구조라 epoch별 test accuracy를 출력합니다.

```python
model.save(model_path)
```

학습이 끝난 모델을 저장합니다.

### 7.3 `argparse`

위치: [src/train.py](/Users/choies/Code_School/260410_mnist/src/train.py:65)

```python
parser = argparse.ArgumentParser(...)
```

`argparse`는 command line argument를 처리하는 표준 라이브러리입니다.

예를 들어 다음처럼 실행할 수 있게 해줍니다.

```bash
.venv/bin/python -m src.train --epochs 10 --learning-rate 0.05
```

이렇게 하면 코드 수정 없이 학습 설정을 바꿀 수 있습니다.

```python
if __name__ == "__main__":
    main()
```

이 패턴은 이 파일이 직접 실행될 때만 `main()`을 호출하게 합니다. 다른 파일에서 import할 때는 자동 실행되지 않습니다.

## 8. [src/evaluate.py](/Users/choies/Code_School/260410_mnist/src/evaluate.py:1)

이 파일은 저장된 모델을 test set으로 평가합니다.

```bash
.venv/bin/python -m src.evaluate
```

### 8.1 `confusion_matrix`

위치: [src/evaluate.py](/Users/choies/Code_School/260410_mnist/src/evaluate.py:14)

confusion matrix는 어떤 숫자를 어떤 숫자로 헷갈렸는지 보여주는 표입니다.

```python
matrix = np.zeros((classes, classes), dtype=np.int64)
```

10x10 행렬을 만듭니다.

- row: 실제 label
- column: 예측 label

예를 들어 `matrix[7, 9] = 12`라면 실제 7인 이미지를 9로 예측한 경우가 12개라는 뜻입니다.

```python
for actual, predicted in zip(y_true, y_pred, strict=True):
```

`zip(..., strict=True)`는 두 배열 길이가 다르면 예외를 냅니다. Python 3.10+ 기능입니다. 평가에서는 실제 label과 예측 label 길이가 반드시 같아야 하므로 좋은 안전장치입니다.

```python
matrix[int(actual), int(predicted)] += 1
```

실제 class와 예측 class 위치의 count를 1 증가시킵니다.

### 8.2 `evaluate`

위치: [src/evaluate.py](/Users/choies/Code_School/260410_mnist/src/evaluate.py:21)

```python
_, _, x_test, y_test = load_mnist(data_dir)
```

train set은 평가에 필요 없으므로 `_`로 받습니다. Python에서 `_`는 “이 값은 쓰지 않는다”는 관례입니다.

```python
model = MLP.load(model_path)
```

저장된 모델을 불러옵니다.

```python
predictions = model.predict(x_test)
```

test set 전체에 대한 예측을 만듭니다.

```python
test_accuracy = accuracy(model, x_test, y_test)
matrix = confusion_matrix(y_test, predictions)
```

정확도와 confusion matrix를 계산합니다.

## 9. [src/predict.py](/Users/choies/Code_School/260410_mnist/src/predict.py:1)

이 파일은 개별 예측용입니다.

두 가지 입력 방식을 지원합니다.

```bash
.venv/bin/python -m src.predict --index 0
```

또는:

```bash
.venv/bin/python -m src.predict --image sample.png
```

#### 9.1 `prepare_image`

위치: [src/predict.py](/Users/choies/Code_School/260410_mnist/src/predict.py:15)

외부 이미지 파일을 모델 입력 형태로 바꿉니다.

```python
image = Image.open(path).convert("L").resize((28, 28))
```

- `Image.open`: 이미지 파일을 엽니다.
- `.convert("L")`: grayscale로 바꿉니다. `"L"`은 Pillow에서 8-bit grayscale mode입니다.
- `.resize((28, 28))`: MNIST와 같은 크기로 바꿉니다.

왜 Pillow가 필요한가요?

NumPy는 배열 계산에는 좋지만 PNG/JPG 같은 이미지 파일을 직접 읽고 resize하는 기능은 Pillow가 담당하는 것이 일반적입니다.

```python
pixels = np.asarray(image, dtype=np.float32) / 255.0
```

Pillow image를 NumPy 배열로 바꾸고 `0.0~1.0`으로 정규화합니다.

```python
if float(np.mean(pixels)) > 0.5:
    pixels = 1.0 - pixels
```

MNIST는 보통 검은 배경에 밝은 숫자 형태입니다. 그런데 사용자가 만든 이미지는 흰 배경에 검은 숫자일 수 있습니다.

평균 픽셀값이 0.5보다 크면 전체적으로 밝은 이미지라고 보고 색을 반전합니다.

```python
return pixels.reshape(1, 784)
```

모델은 batch 입력을 기대합니다. 이미지 하나라도 shape은 `(784,)`가 아니라 `(1, 784)`로 맞춥니다.

### 9.2 `predict_index`

위치: [src/predict.py](/Users/choies/Code_School/260410_mnist/src/predict.py:23)

MNIST test set 안의 특정 index를 예측합니다.

```python
if index < 0 or index >= x_test.shape[0]:
    raise ValueError(...)
```

index 범위를 검증합니다.

```python
probabilities = model.forward(x_test[index : index + 1])[0]
```

`x_test[index:index+1]`를 쓰는 이유는 shape을 `(1, 784)`로 유지하기 위해서입니다.

만약 `x_test[index]`를 쓰면 shape이 `(784,)`가 됩니다. 모델은 batch dimension이 있는 2차원 입력을 기대하므로 slicing을 사용합니다.

```python
predicted = int(np.argmax(probabilities))
actual = int(y_test[index])
confidence = float(probabilities[predicted])
```

- `predicted`: 가장 확률이 높은 class
- `actual`: 실제 정답
- `confidence`: 예측 class에 부여한 확률

### 9.3 `predict_image`

위치: [src/predict.py](/Users/choies/Code_School/260410_mnist/src/predict.py:35)

외부 이미지 파일을 예측합니다.

```python
x = prepare_image(image_path)
probabilities = model.forward(x)[0]
```

전처리 후 모델에 넣습니다.

### 9.4 mutually exclusive argument

위치: [src/predict.py](/Users/choies/Code_School/260410_mnist/src/predict.py:47)

```python
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--index", type=int)
group.add_argument("--image")
```

이 코드는 `--index`와 `--image` 중 정확히 하나만 받도록 합니다.

즉 다음은 허용됩니다.

```bash
.venv/bin/python -m src.predict --index 0
```

다음도 허용됩니다.

```bash
.venv/bin/python -m src.predict --image sample.png
```

하지만 둘 다 넣으면 안 됩니다.

```bash
.venv/bin/python -m src.predict --index 0 --image sample.png
```

입력 방식이 모호해지기 때문입니다.

## 10. 테스트 구조

테스트는 두 파일입니다.

- [tests/test_mnist_loader.py](/Users/choies/Code_School/260410_mnist/tests/test_mnist_loader.py:1)
- [tests/test_model.py](/Users/choies/Code_School/260410_mnist/tests/test_model.py:1)

실행:

```bash
.venv/bin/python -m unittest discover -s tests
```

`unittest`는 Python 표준 테스트 프레임워크입니다. 별도 설치가 필요 없습니다.

`test_mnist_loader.py`는 다음을 검증합니다.

- train image shape이 `(60000, 784)`인지
- train label shape이 `(60000,)`인지
- test image shape이 `(10000, 784)`인지
- test label shape이 `(10000,)`인지
- 이미지 dtype이 `float32`인지
- 이미지 값이 `0.0~1.0`인지
- label이 integer인지

`test_model.py`는 다음을 검증합니다.

- `forward` 출력 shape이 맞는지
- softmax 결과 row sum이 1인지
- 확률값이 `0.0~1.0`인지
- `save/load` 후 예측이 유지되는지

테스트의 역할은 “나중에 코드를 수정했을 때 핵심 동작이 깨졌는지 빠르게 확인하는 것”입니다.

## 11. 전체 학습 계산을 한 번에 연결해서 보면

한 batch가 들어왔을 때 실제 계산은 다음 순서입니다.

```text
x_batch
  shape: (128, 784)
  ↓
x @ w1 + b1
  shape: (128, 128)
  ↓
ReLU
  shape: (128, 128)
  ↓
hidden @ w2 + b2
  shape: (128, 10)
  ↓
softmax
  shape: (128, 10)
  ↓
cross entropy loss
  scalar
  ↓
backward
  dw1, db1, dw2, db2 계산
  ↓
step
  w1, b1, w2, b2 업데이트
```

모델이 학습한다는 것은 결국 `w1`, `b1`, `w2`, `b2` 값을 조금씩 바꿔서 정답 class 확률을 높이는 과정입니다.


## 12. 왜 이 파일 구조가 좋은가

이 프로젝트는 책임별로 나눠져 있습니다.

- `mnist_loader.py`: 데이터 파일 읽기만 담당
- `model.py`: 모델 계산만 담당
- `train.py`: 학습 loop만 담당
- `evaluate.py`: 저장된 모델 평가만 담당
- `predict.py`: 개별 입력 예측만 담당
- `tests/`: 동작 검증 담당

이렇게 나누면 장점이 있습니다.

- 데이터 로딩 문제가 생기면 `mnist_loader.py`만 보면 됩니다.
- 모델 수식을 바꾸고 싶으면 `model.py`만 보면 됩니다.
- epoch, batch size 같은 학습 방식을 바꾸고 싶으면 `train.py`를 보면 됩니다.
- 예측 입력 형식을 확장하고 싶으면 `predict.py`를 보면 됩니다.

즉 한 파일이 모든 일을 하지 않게 해서 유지보수가 쉬워집니다.

## 13. 현재 성능과 의미

검증 시 학습 결과는 다음이었습니다.

```text
epoch 5/5 loss=0.1526 test_accuracy=0.9574
```

정확도 `0.9574`는 test set 10,000개 중 약 9,574개를 맞췄다는 뜻입니다.

이 정도면 NumPy MLP 교육용 구현으로 충분히 정상적인 결과입니다. CNN을 쓰면 더 높은 정확도를 기대할 수 있지만, 이 프로젝트의 목표는 framework 없이 기본 원리를 이해하는 것이므로 현재 구조가 적절합니다.

## 14. 노트

- `_data/` 파일명은 현재 코드에서 표준 이름으로 고정되어 있습니다.
- 입력 이미지는 반드시 28x28로 변환되어야 합니다.
- 모델 입력 shape은 항상 `(batch_size, 784)`여야 합니다.
- `backward()`는 반드시 `forward()` 뒤에 호출해야 합니다.
- `step()`은 반드시 `backward()` 뒤에 호출해야 합니다.
- 학습된 모델 파일이 없으면 `evaluate.py`, `predict.py`는 실행할 수 없습니다. 먼저 `train.py`를 실행해야 합니다.
- 외부 이미지 예측은 MNIST와 스타일이 다르면 성능이 떨어질 수 있습니다. MNIST는 정중앙에 놓인 28x28 손글씨 숫자라서, 일반 사진이나 크게 치우친 숫자에는 약합니다.