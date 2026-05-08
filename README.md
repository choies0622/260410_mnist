# 260410_mnist

This is handwritten digit recognition model based on MNIST dataset.  
I implemented a NumPy-based MLP from scratch so that it can run in the environment containing `NumPy`, `Pillow`, and `matplotlib` installed in the `.venv`, without relying on any external frameworks.

MNIST 데이터셋 기반 손글씨 숫자 인식 모델.  
외부 프레임워크 없이 `.venv`에 설치된 `numpy`, `Pillow`, `matplotlib` 환경에서 실행되도록 NumPy 기반 MLP를 구현했습니다.

## Project Structure

```text
_data/                  # MNIST IDX gzip files, git not tracted
src/
  mnist_loader.py       # MNIST IDX gzip loader
  model.py              # NumPy MLP model
  train.py              # training
  evaluate.py           # evaluation
  predict.py            # prediction
tests/
  test_mnist_loader.py
  test_model.py
models/                 # generated model files, git not tracted
```

## Data Files

`_data` must contain the following four files. ([MNIST dataset source](https://ossci-datasets.s3.amazonaws.com/mnist/))  
`_data` 폴더에 아래 네 파일이 있어야 합니다.

```text
train-images-idx3-ubyte.gz
train-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz
```

## Commands

Unit test / 단위 테스트:

```bash
.venv/bin/python -m unittest discover -s tests
```

Train / 학습:

```bash
.venv/bin/python -m src.train
```

Once training is complete, the model will be saved to `models/mnist_mlp.npz`.  
학습이 끝나면 `models/mnist_mlp.npz`에 모델이 저장됩니다.

Evaluate / 평가:

```bash
.venv/bin/python -m src.evaluate
```

MNIST test sample predict / MNIST test sample 예측:

```bash
.venv/bin/python -m src.predict --index 0
```

Image file predict / 이미지 파일 예측:

```bash
.venv/bin/python -m src.predict --image sample.png
```

## Model

- Input: 28x28 image flattened to 784 values
- Hidden layer: 128 units with ReLU
- Output: 10 classes with softmax
- Loss: cross entropy
- Optimizer: mini-batch stochastic gradient descent

Default training setting is `epochs=5`, `batch_size=128`, `learning_rate=0.1`,
`hidden_size=128`, `seed=42`. If nessecery, it can be adjusted using CLI options.  
기본 학습 설정은 `epochs=5`, `batch_size=128`, `learning_rate=0.1`,
`hidden_size=128`, `seed=42`이다. 필요하면 CLI 옵션으로 조정할 수 있다.

## Detail
For more detail, see `DOCS.md`.