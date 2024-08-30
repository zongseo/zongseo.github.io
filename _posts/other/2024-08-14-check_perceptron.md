---
layout: single
title:  "퍼셉트론, 다층 퍼셉트론 살펴보기"
categories: OTHER
tag: [python, ai, bigdata]
toc: true
author_profile: false
---


# 퍼셉트론 모델 이해하기: OR 및 AND 게이트



퍼셉트론은 인공지능 모델의 기본 단위이며, 인간의 뇌가 작동하는 방식을 모방합니다.

퍼셉트론은 여러 입력 신호를 받아 이를 가중치와 바이어스를 사용하여 처리한 후 단일 이진 결과를 출력합니다.



## 논리 연산으로서의 퍼셉트론



### OR 게이트

OR 게이트는 입력 중 하나라도 참이면 참을 출력합니다. 모든 입력이 거짓이면 거짓을 출력합니다.



### AND 게이트

AND 게이트는 모든 입력이 참일 때만 참을 출력합니다. 그 외의 경우에는 거짓을 출력합니다.



퍼셉트론 모델에서:

- **가중치**(w1, w2, ...)는 각 입력의 중요도를 결정합니다.

- **바이어스**(b)는 가중합과 함께 출력을 조정하여 기대 출력에 더 잘 맞도록 합니다.

- **활성화 함수**는 가중합과 바이어스를 기반으로 뉴런이 활성화될지를 결정합니다.



간단한 결정 규칙 사용: 입력의 가중합과 바이어스가 작은 임곗값(엡실론)보다 크면 퍼셉트론은 1(참)을 출력하고, 그렇지 않으면 0(거짓)을 출력합니다.



## 예제 코드

아래는 OR 게이트를 행동하는 퍼셉트론의 구현입니다.

가중치와 바이어스를 OR 연산의 참 테이블에 맞춰 조정합니다. 이 매개변수들을 조정하여 AND 게이트로 만들 수도 있습니다.



```python
import numpy as np

# 부동소수점 정밀도 문제를 피하기 위한 작은 양수
epsilon = 0.0000001

def perceptron_or(x1, x2):
    """
    OR 게이트를 퍼셉트론 모델을 사용하여 구현합니다.

    인자:
    x1 (int): 첫 번째 입력 (0 또는 1)
    x2 (int): 두 번째 입력 (0 또는 1)

    반환:
    int: 입력 중 하나라도 1이면 1, 그 외에는 0

    부동소수점 연산에서 발생할 수 있는 아주 작은 오차를 방지하기 위해,
    가중합 결과가 epsilon보다 크면 1을 반환하고, 그렇지 않으면 0을 반환합니다.
    이는 0을 직접 비교하는 대신 머신 엡실론을 사용하여 정밀도 문제를 관리하는 방법입니다.
    """
    X = np.array([x1, x2])
    W = np.array([1.0, 1.0])  # 각 입력에 대한 가중치
    B = -0.5                  # 바이어스: OR 참 테이블에 맞게 조정됨
    sum = np.dot(W, X) + B
    return 1 if sum > epsilon else 0
```


```python
def perceptron_and(x1, x2):
    """
    AND 게이트를 퍼셉트론 모델을 사용하여 구현합니다.

    인자:
    x1 (int): 첫 번째 입력 (0 또는 1)
    x2 (int): 두 번째 입력 (0 또는 1)

    반환:
    int: 모든 입력이 1이면 1, 그 외에는 0

    부동소수점 연산에서 발생할 수 있는 아주 작은 오차를 방지하기 위해,
    가중합 결과가 epsilon보다 크면 1을 반환하고, 그렇지 않으면 0을 반환합니다.
    이는 정밀도 문제를 관리하기 위해 0과의 직접적인 비교 대신 머신 엡실론을 사용합니다.
    """
    X = np.array([x1, x2])
    W = np.array([1.0, 1.0])  # 가중치는 OR 게이트와 동일
    B = -1.5                  # 바이어스: AND 참 테이블에 맞게 조정됨
    sum = np.dot(W, X) + B
    return 1 if sum > epsilon else 0
```


```python
# 퍼셉트론 기반 OR 게이트와 AND 게이트의 출력을 테스트합니다.
print("OR Gate Outputs:")
print(perceptron_or(0, 0))  # 예상 출력: 0
print(perceptron_or(1, 0))  # 예상 출력: 1
print(perceptron_or(0, 1))  # 예상 출력: 1
print(perceptron_or(1, 1))  # 예상 출력: 1

print("AND Gate Outputs:")
print(perceptron_and(0, 0))  # 예상 출력: 0
print(perceptron_and(1, 0))  # 예상 출력: 0
print(perceptron_and(0, 1))  # 예상 출력: 0
print(perceptron_and(1, 1))  # 예상 출력: 1
```

<pre>
OR Gate Outputs:
0
1
1
1
AND Gate Outputs:
0
0
0
1
</pre>
## 학습 과정

퍼셉트론의 학습 과정은 오차를 기반으로 가중치를 조절하며 이루어집니다. 각 입력에 대해 예측값과 실제값의 차이를 계산하고, 이 오차를 사용해 가중치를 업데이트합니다. 학습률(`eta`)은 이 업데이트의 크기를 결정합니다.



## 예시 코드 설명

제공된 코드는 AND 논리 연산을 수행하는 퍼셉트론의 학습과 예측 과정을 구현합니다. 초기 가중치는 모두 0이며, 입력 데이터에는 각 입력 조합과 함께 항상 1인 바이어스 입력이 포함됩니다. 코드는 학습을 반복하며 가중치를 조정하고, 학습이 끝난 후 입력 데이터에 대한 예측을 수행합니다.



```python
import numpy as np

epsilon = 0.0000001  # 부동소수점 오차를 방지하기 위한 매우 작은 값

def step_func(t):
    """
    활성화 함수로 단계 함수(step function)를 사용합니다.
    epsilon보다 크면 1을 반환하고, 그렇지 않으면 0을 반환합니다.
    """
    return 1 if t > epsilon else 0

# 입력 데이터 세트와 항상 1인 바이어스 입력
X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

y = np.array([0, 0, 0, 1])  # AND 연산의 정답 데이터
```


```python
# 가중치 초기화
W = np.zeros(len(X[0]))

def perceptron_fit(X, Y, epochs=10):
    """
    퍼셉트론 학습 함수입니다.
    epochs: 전체 데이터 세트에 대한 학습 반복 횟수
    """
    global W
    eta = 0.2  # 학습률
    for t in range(epochs):
        print("epoch=", t, "======================")
        for i in range(len(X)):
            predict = step_func(np.dot(X[i], W))
            error = Y[i] - predict
            W += eta * error * X[i]  # 가중치 업데이트
            print("현재 처리 입력=", X[i], "정답=", Y[i], "출력=", predict, "변경된 가중치=", W)
        print("================================")

def perceptron_predict(X, Y):
    """
    퍼셉트론을 사용한 예측 함수입니다.
    각 입력 데이터에 대해 예측 결과를 출력합니다.
    """
    global W
    for x in X:
         print(x[0], x[1], "->", step_func(np.dot(x, W)))
```


```python
# 학습 및 예측 함수 호출
perceptron_fit(X, y, 6)
perceptron_predict(X, y)
```

<pre>
epoch= 0 ======================
현재 처리 입력= [0 0 1] 정답= 0 출력= 0 변경된 가중치= [0. 0. 0.]
현재 처리 입력= [0 1 1] 정답= 0 출력= 0 변경된 가중치= [0. 0. 0.]
현재 처리 입력= [1 0 1] 정답= 0 출력= 0 변경된 가중치= [0. 0. 0.]
현재 처리 입력= [1 1 1] 정답= 1 출력= 0 변경된 가중치= [0.2 0.2 0.2]
================================
epoch= 1 ======================
현재 처리 입력= [0 0 1] 정답= 0 출력= 1 변경된 가중치= [0.2 0.2 0. ]
현재 처리 입력= [0 1 1] 정답= 0 출력= 1 변경된 가중치= [ 0.2  0.  -0.2]
현재 처리 입력= [1 0 1] 정답= 0 출력= 0 변경된 가중치= [ 0.2  0.  -0.2]
현재 처리 입력= [1 1 1] 정답= 1 출력= 0 변경된 가중치= [0.4 0.2 0. ]
================================
epoch= 2 ======================
현재 처리 입력= [0 0 1] 정답= 0 출력= 0 변경된 가중치= [0.4 0.2 0. ]
현재 처리 입력= [0 1 1] 정답= 0 출력= 1 변경된 가중치= [ 0.4  0.  -0.2]
현재 처리 입력= [1 0 1] 정답= 0 출력= 1 변경된 가중치= [ 0.2  0.  -0.4]
현재 처리 입력= [1 1 1] 정답= 1 출력= 0 변경된 가중치= [ 0.4  0.2 -0.2]
================================
epoch= 3 ======================
현재 처리 입력= [0 0 1] 정답= 0 출력= 0 변경된 가중치= [ 0.4  0.2 -0.2]
현재 처리 입력= [0 1 1] 정답= 0 출력= 0 변경된 가중치= [ 0.4  0.2 -0.2]
현재 처리 입력= [1 0 1] 정답= 0 출력= 1 변경된 가중치= [ 0.2  0.2 -0.4]
현재 처리 입력= [1 1 1] 정답= 1 출력= 0 변경된 가중치= [ 0.4  0.4 -0.2]
================================
epoch= 4 ======================
현재 처리 입력= [0 0 1] 정답= 0 출력= 0 변경된 가중치= [ 0.4  0.4 -0.2]
현재 처리 입력= [0 1 1] 정답= 0 출력= 1 변경된 가중치= [ 0.4  0.2 -0.4]
현재 처리 입력= [1 0 1] 정답= 0 출력= 0 변경된 가중치= [ 0.4  0.2 -0.4]
현재 처리 입력= [1 1 1] 정답= 1 출력= 1 변경된 가중치= [ 0.4  0.2 -0.4]
================================
epoch= 5 ======================
현재 처리 입력= [0 0 1] 정답= 0 출력= 0 변경된 가중치= [ 0.4  0.2 -0.4]
현재 처리 입력= [0 1 1] 정답= 0 출력= 0 변경된 가중치= [ 0.4  0.2 -0.4]
현재 처리 입력= [1 0 1] 정답= 0 출력= 0 변경된 가중치= [ 0.4  0.2 -0.4]
현재 처리 입력= [1 1 1] 정답= 1 출력= 1 변경된 가중치= [ 0.4  0.2 -0.4]
================================
0 0 -> 0
0 1 -> 0
1 0 -> 0
1 1 -> 1
</pre>
## 퍼셉트론 모델 사용 예제



### 코드 설명

이 예제에서는 `scikit-learn` 라이브러리의 `Perceptron` 클래스를 사용하여 간단한 논리 연산을 학습하는 방법을 보여줍니다. 여기서 사용된 데이터 세트는 간단한 AND 연산을 나타내며, 퍼셉트론은 이를 모델링하기 위해 학습됩니다.



### 바이어스에 대한 설명

퍼셉트론에서 바이어스는 모델의 결정 경계를 조정하는 역할을 합니다. `scikit-learn`의 `Perceptron` 클래스는 자동으로 바이어스를 처리하며, 사용자가 명시적으로 값을 설정할 필요가 없습니다. 이는 내부적으로 각 샘플의 특성 벡터에 자동으로 바이어스 항을 추가하는 방식으로 구현되어 있습니다.



```python
from sklearn.linear_model import Perceptron

# 샘플 데이터 정의
X = [[0, 0], [0, 1], [1, 0], [1, 1]]  # 입력 샘플
y = [0, 0, 0, 1]                      # AND 연산 결과

# Perceptron 모델 초기화
clf = Perceptron(tol=1e-3, random_state=0)  # 학습 종료 조건 설정, 난수 시드 설정

# 모델 학습
clf.fit(X, y)

# 학습된 모델로 예측 수행
print(clf.predict(X))  # 출력 예측
```

<pre>
[0 0 0 1]
</pre>
## 퍼셉트론 가중치 업데이트 메커니즘 상세 설명



퍼셉트론은 인공 신경망의 가장 기본적인 형태로, 간단한 입력과 출력 사이의 관계를 학습할 수 있습니다. 가중치 업데이트 과정은 퍼셉트론이 데이터로부터 학습하고, 예측 오차를 최소화하는 방법을 보여줍니다.



### 오차 계산



퍼셉트론은 현재 가중치를 사용하여 각 입력 데이터에 대한 예측값을 계산합니다. 이 예측값은 활성화 함수를 통과한 결과입니다. 예측값과 실제 타깃 값(`Y[i]`)의 차이를 오차(`error`)로 계산합니다. 오차는 다음과 같이 정의됩니다:



오차 계산 공식:

```plaintext

error = Y[i] - predict

```

이 공식은 현재 가중치를 사용하여 계산된 예측값(predict)과 실제 타깃 값(Y[i]) 간의 차이를 나타냅니다. 이 오차는 가중치를 조정하는 데 사용됩니다.



### 가중치 업데이트 규칙



퍼셉트론은 이 오차를 사용하여 각 가중치를 업데이트합니다. 가중치 업데이트는 학습률(eta)과 오차, 그리고 현재 입력값(X[i])의 곱을 통해 이루어집니다. 학습률은 이 업데이트의 크기를 조절하는 중요한 파라미터입니다.



가중치 업데이트 공식:

```plaintext

W = W + eta * error * X[i]

```

이 공식에 따라, 각 입력값에 대응하는 가중치는 오차와 학습률, 해당 입력값의 곱만큼 조정됩니다. 오차가 크면 큰 조정이 이루어져 빠르게 학습할 수 있으며, 오차가 작으면 작은 조정을 통해 점차 최적의 가중치에 도달합니다.

이러한 가중치 업데이트 메커니즘은 퍼셉트론이 학습 데이터에 대해 점차 더 정확한 예측을 할 수 있도록 도와줍니다. 오차를 줄이는 방향으로 가중치를 지속적으로 조정함으로써, 모델은 데이터의 패턴을 효과적으로 학습하게 됩니다.


## 퍼셉트론과 배타적 논리합(XOR)



### 배타적 논리합(XOR)의 문제점

퍼셉트론은 간단한 선형 분류 문제를 해결할 수 있는 강력한 도구입니다. 그러나 퍼셉트론이 해결할 수 없는 특정 유형의 문제들도 있으며, 그 중 대표적인 예가 배타적 논리합(XOR) 문제입니다.



XOR 문제는 두 입력값이 서로 다를 때 참(1)을 반환하고, 같을 때 거짓(0)을 반환합니다. 이 문제는 단층 퍼셉트론으로는 해결할 수 없는데, 그 이유는 XOR 연산이 선형적으로 분리할 수 없기 때문입니다. 즉, 단일 선형 결정 경계로는 두 클래스를 구분할 수 없습니다.



### 마빈 민스키와 시모어 페퍼트의 연구

1969년, 마빈 민스키와 시모어 페퍼트는 퍼셉트론의 한계를 공식적으로 증명하여, 단층 퍼셉트론이 XOR 문제를 해결할 수 없음을 보여주었습니다. 이 발견은 인공지능 연구에 큰 실망을 가져다주었고, 이는 1970년대와 1980년대에 걸쳐 인공지능 연구의 첫 번째 겨울로 이어졌습니다.



### XOR 문제 시각화 코드 예제

아래 파이썬 코드는 XOR 문제를 시각화하고, 단층 퍼셉트론으로는 이를 해결할 수 없음을 보여줍니다.



```python
import numpy as np
import matplotlib.pyplot as plt

# XOR 데이터 세트 정의
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
targets = np.array([0, 1, 1, 0])

# 데이터 시각화
plt.scatter(inputs[:, 0], inputs[:, 1], c=targets, cmap='viridis')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.title('XOR Problem Visualization')
plt.colorbar(label='Output')
plt.grid(True)
plt.show()
```

<pre>
<Figure size 640x480 with 2 Axes>
</pre>

```python
import numpy as np

epsilon = 0.0000001  # 부동소수점 오차를 방지하기 위한 매우 작은 값

def step_func(t):
    """
    활성화 함수로 단계 함수(step function)를 사용합니다.
    epsilon보다 크면 1을 반환하고, 그렇지 않으면 0을 반환합니다.
    """
    return 1 if t > epsilon else 0

# XOR 입력 데이터 세트와 항상 1인 바이어스 입력
X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

y = np.array([0, 1, 1, 0])  # XOR 연산의 정답 데이터
```


```python
# 가중치 초기화
W = np.zeros(len(X[0]))

def perceptron_fit(X, Y, epochs=10):
    """
    퍼셉트론 학습 함수입니다.
    epochs: 전체 데이터 세트에 대한 학습 반복 횟수
    """
    global W
    eta = 0.1  # 학습률
    for t in range(epochs):
        print("epoch=", t, "======================")
        for i in range(len(X)):
            predict = step_func(np.dot(X[i], W))
            error = Y[i] - predict
            W += eta * error * X[i]  # 가중치 업데이트
            print("현재 처리 입력=", X[i], "정답=", Y[i], "출력=", predict, "변경된 가중치=", W)
        print("================================")

def perceptron_predict(X, Y):
    """
    퍼셉트론을 사용한 예측 함수입니다.
    각 입력 데이터에 대해 예측 결과를 출력합니다.
    """
    global W
    for x in X:
         print(x[0], x[1], "->", step_func(np.dot(x, W)))
```


```python
# 학습 및 예측 함수 호출
perceptron_fit(X, y, 6)
perceptron_predict(X, y)
```

<pre>
epoch= 0 ======================
현재 처리 입력= [0 0 1] 정답= 0 출력= 0 변경된 가중치= [0. 0. 0.]
현재 처리 입력= [0 1 1] 정답= 1 출력= 0 변경된 가중치= [0.  0.1 0.1]
현재 처리 입력= [1 0 1] 정답= 1 출력= 1 변경된 가중치= [0.  0.1 0.1]
현재 처리 입력= [1 1 1] 정답= 0 출력= 1 변경된 가중치= [-0.1  0.   0. ]
================================
epoch= 1 ======================
현재 처리 입력= [0 0 1] 정답= 0 출력= 0 변경된 가중치= [-0.1  0.   0. ]
현재 처리 입력= [0 1 1] 정답= 1 출력= 0 변경된 가중치= [-0.1  0.1  0.1]
현재 처리 입력= [1 0 1] 정답= 1 출력= 0 변경된 가중치= [0.  0.1 0.2]
현재 처리 입력= [1 1 1] 정답= 0 출력= 1 변경된 가중치= [-0.1  0.   0.1]
================================
epoch= 2 ======================
현재 처리 입력= [0 0 1] 정답= 0 출력= 1 변경된 가중치= [-0.1  0.   0. ]
현재 처리 입력= [0 1 1] 정답= 1 출력= 0 변경된 가중치= [-0.1  0.1  0.1]
현재 처리 입력= [1 0 1] 정답= 1 출력= 0 변경된 가중치= [0.  0.1 0.2]
현재 처리 입력= [1 1 1] 정답= 0 출력= 1 변경된 가중치= [-0.1  0.   0.1]
================================
epoch= 3 ======================
현재 처리 입력= [0 0 1] 정답= 0 출력= 1 변경된 가중치= [-0.1  0.   0. ]
현재 처리 입력= [0 1 1] 정답= 1 출력= 0 변경된 가중치= [-0.1  0.1  0.1]
현재 처리 입력= [1 0 1] 정답= 1 출력= 0 변경된 가중치= [0.  0.1 0.2]
현재 처리 입력= [1 1 1] 정답= 0 출력= 1 변경된 가중치= [-0.1  0.   0.1]
================================
epoch= 4 ======================
현재 처리 입력= [0 0 1] 정답= 0 출력= 1 변경된 가중치= [-0.1  0.   0. ]
현재 처리 입력= [0 1 1] 정답= 1 출력= 0 변경된 가중치= [-0.1  0.1  0.1]
현재 처리 입력= [1 0 1] 정답= 1 출력= 0 변경된 가중치= [0.  0.1 0.2]
현재 처리 입력= [1 1 1] 정답= 0 출력= 1 변경된 가중치= [-0.1  0.   0.1]
================================
epoch= 5 ======================
현재 처리 입력= [0 0 1] 정답= 0 출력= 1 변경된 가중치= [-0.1  0.   0. ]
현재 처리 입력= [0 1 1] 정답= 1 출력= 0 변경된 가중치= [-0.1  0.1  0.1]
현재 처리 입력= [1 0 1] 정답= 1 출력= 0 변경된 가중치= [0.  0.1 0.2]
현재 처리 입력= [1 1 1] 정답= 0 출력= 1 변경된 가중치= [-0.1  0.   0.1]
================================
0 0 -> 1
0 1 -> 1
1 0 -> 0
1 1 -> 0
</pre>
## 다층 퍼셉트론을 이용한 XOR 문제 해결

다층 퍼셉트론(Multilayer Perceptron, MLP)은 단층 퍼셉트론의 한계를 극복하고, XOR과 같은 비선형 문제를 해결할 수 있는 강력한 구조입니다. XOR 문제는 두 입력이 서로 다를 때 1을 반환하고, 같을 때 0을 반환하는 논리 연산입니다. 단층 퍼셉트론으로는 이를 구현할 수 없지만, 다층 구조를 통해 가능합니다.



### XOR 문제와 다층 퍼셉트론

XOR 연산은 선형으로 분리할 수 없는 데이터를 처리해야 하므로, 하나 이상의 숨겨진 계층(hidden layers)을 필요로 합니다. 다층 퍼셉트론에서는 NAND, OR, AND와 같은 기본 논리 게이트를 조합하여 XOR 연산을 구현합니다.



- **NAND 게이트**: 두 입력이 모두 1일 때 0을 출력하고, 그 외에는 1을 출력합니다.

- **OR 게이트**: 두 입력 중 하나라도 1이면 1을 출력합니다.

- **AND 게이트**: 두 입력이 모두 1일 때만 1을 출력합니다.



### XOR 구현 코드와 설명

아래 Python 코드는 XOR 연산을 다층 퍼셉트론을 사용하여 구현한 예제입니다. 각 게이트는 퍼셉트론으로 모델링되어 서로 다른 가중치와 바이어스를 사용합니다.



```python
import numpy as np

# 가중치와 바이어스 초기화
w11 = np.array([-2, -2])  # NAND 게이트 가중치
w12 = np.array([2, 2])    # OR 게이트 가중치
w2 = np.array([1, 1])     # AND 게이트 가중치
b1 = 3  # NAND 게이트 바이어스
b2 = -1 # OR 게이트 바이어스
b3 = -1 # AND 게이트 바이어스

# 다층 퍼셉트론 함수
def MLP(x, w, b):
    y = np.sum(w * x) + b
    return 1 if y > 0 else 0

# 논리 게이트 정의
def NAND(x1, x2):
    return MLP(np.array([x1, x2]), w11, b1)

def OR(x1, x2):
    return MLP(np.array([x1, x2]), w12, b2)

def AND(x1, x2):
    return MLP(np.array([x1, x2]), w2, b3)

# XOR 게이트 구현
def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))

# 메인 실행 부분
if __name__ == '__main__':
    for x in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = XOR(x[0], x[1])
        print("입력값: " + str(x) + ", 출력값: " + str(y))
```

<pre>
입력값: (0, 0), 출력값: 0
입력값: (1, 0), 출력값: 1
입력값: (0, 1), 출력값: 1
입력값: (1, 1), 출력값: 0
</pre>
## 활성화 함수의 중요성과 다양한 활성화 함수 소개

활성화 함수는 인공 신경망의 중추적인 요소로, 신경망이 선형 문제뿐만 아니라 복잡한 비선형 문제도 해결할 수 있게 합니다. 이는 인공 신경망이 단순한 분류 문제에서 더 복잡한 문제들까지 다룰 수 있게 만드는 핵심적인 특성입니다.



### 선형 함수의 한계

선형 활성화 함수는 출력이 입력의 선형 조합으로 표현되기 때문에, 네트워크의 층을 아무리 많이 추가해도 결국 하나의 선형 함수로 표현될 수 있습니다. 이는 모델이 복잡한 문제, 특히 XOR 같은 비선형 문제를 학습할 수 없다는 것을 의미합니다.



### 비선형 활성화 함수

비선형 활성화 함수를 사용하면 이러한 한계를 극복할 수 있습니다. 비선형 함수는 신경망에 비선형성을 도입하여 각 층이 이전 층의 출력을 비선형적으로 변형할 수 있게 합니다. 이를 통해 신경망은 더 복잡한 패턴을 학습하고 다양한 문제를 해결할 수 있습니다.



- 계단 함수 (Step Function): 최초의 활성화 함수 중 하나로, 특정 임곗값을 기준으로 출력이 바뀝니다. 이는 이진 분류 문제에 적합하지만, 미분 불가능한 점과 출력값의 변화가 갑작스러워 신경망이 점진적으로 학습하기 어렵다는 단점이 있습니다.



- 시그모이드 함수 (Sigmoid Function); 시그모이드 함수는 출력값을 0과 1 사이로 압축하여, 확률과 같은 형태를 모델링하기에 적합합니다. 하지만 깊은 네트워크에서는 그라디언트 소실 문제를 일으킬 수 있습니다.



- 하이퍼볼릭 탄젠트 함수 (Hyperbolic Tangent, tanh): 시그모이드와 유사하지만, 출력 범위가 -1에서 1로 더 넓습니다. 이는 데이터가 중심에 정규화되는 효과를 가져와 학습 초기 단계에서 유리합니다.



- 렐루 함수 (ReLU - Rectified Linear Unit): 음수를 입력받았을 때 0을 반환하고, 양수는 그대로 반환하는 함수입니다. 계산 효율성과 그라디언트 소실 문제가 덜하므로 현대 신경망에서 가장 널리 사용됩니다.



### Python 코드 예제 및 설명

아래는 각 활성화 함수를 Python으로 구현하고, 그 결과를 시각화하는 코드입니다.



```python
import numpy as np
import matplotlib.pyplot as plt

# 계단 함수
def step_function(x):
    return np.array(x > 0, dtype=int)

x = np.arange(-10.0, 10.0, 0.1)
y = step_function(x)
plt.plot(x, y, label='Step Function')
plt.ylim(-0.1, 1.1)  # y축의 범위 지정
```

<pre>
(-0.1, 1.1)
</pre>
<pre>
<Figure size 640x480 with 1 Axes>
</pre>

```python
# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

y = sigmoid(x)
plt.plot(x, y, label='Sigmoid')

# 하이퍼볼릭 탄젠트 함수
y = np.tanh(x)
plt.plot(x, y, label='Tanh')

# 렐루 함수
def relu(x):
    return np.maximum(0, x)

y = relu(x)
plt.plot(x, y, label='ReLU')

plt.title('Activation Functions')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()
```

<pre>
<Figure size 640x480 with 1 Axes>
</pre>
## 다층 퍼셉트론의 순방향 패스 이해

다층 퍼셉트론(MLP)은 신경망의 기본 구성 요소로, 여러 계층의 뉴런으로 이루어져 있습니다. 이러한 구조는 복잡한 비선형 문제를 해결할 수 있는 능력을 제공합니다. 순방향 패스는 입력에서부터 출력까지의 데이터 흐름 과정을 말하며, 이 과정에서 각 뉴런은 가중치와 활성화 함수를 통해 신호를 처리합니다.



### 활성화 함수의 역할

활성화 함수는 신경망이 비선형 문제를 해결할 수 있도록 돕습니다. 대표적으로 사용되는 시그모이드 함수는 입력받아 0과 1 사이의 출력을 반환하며, 이는 뉴런의 활성화 정도를 나타냅니다. 시그모이드 함수의 미분은 그라디언트 기반 학습에 사용되며, 이는 가중치를 조정하는 데 중요한 역할을 합니다.



### 다층 퍼셉트론 순방향 패스 코드 설명

아래 코드는 XOR 문제를 해결하기 위한 다층 퍼셉트론의 구현 예제입니다. 코드는 각 레이어의 가중치 계산과 활성화 함수를 적용하는 과정을 보여줍니다.



```python
import numpy as np

# 활성화 함수 및 미분
def actf(x):
    return 1 / (1 + np.exp(-x))  # 시그모이드 함수

def actf_deriv(x):
    return x * (1 - x)  # 시그모이드 미분

# 네트워크 구조 정의
inputs, hiddens, outputs = 2, 2, 1
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 훈련 데이터
T = np.array([[0], [1], [1], [0]])  # 목표 출력

# 가중치와 바이어스 초기화
W1 = np.array([[0.10, 0.20], [0.30, 0.40]])
W2 = np.array([[0.50], [0.60]])
B1 = np.array([0.1, 0.2])
B2 = np.array([0.3])

# 순방향 패스 함수
def predict(x):
    layer0 = x
    Z1 = np.dot(layer0, W1) + B1
    layer1 = actf(Z1)  # 첫 번째 은닉층 활성화
    Z2 = np.dot(layer1, W2) + B2
    layer2 = actf(Z2)  # 출력층 활성화
    return layer0, layer1, layer2

# 테스트 함수
def test():
    for x, y in zip(X, T):
        x = np.reshape(x, (1, -1))  # 입력을 2차원 배열로 변환
        layer0, layer1, layer2 = predict(x)
        print(f"입력값: {x}, 목푯값: {y}, 예측값: {layer2}")

test()
```

<pre>
입력값: [[0 0]], 목푯값: [0], 예측값: [[0.70938314]]
입력값: [[0 1]], 목푯값: [1], 예측값: [[0.72844306]]
입력값: [[1 0]], 목푯값: [1], 예측값: [[0.71791234]]
입력값: [[1 1]], 목푯값: [0], 예측값: [[0.73598705]]
</pre>
## 오차 역전파와 그 등장 배경

오차 역전파(backpropagation)는 신경망 훈련에서 중요한 알고리즘으로, 네트워크의 오차를 감소시키기 위해 출력층에서 입력층으로 가중치를 조정합니다. 이 알고리즘은 1980년대에 큰 주목을 받기 시작했으며, 신경망이 복잡한 문제를 효과적으로 학습할 수 있게 하는 데 중요한 역할을 합니다.



### 오차 역전파의 등장 배경

신경망의 초기 연구에서는 가중치를 무작위로 조정하거나 간단한 규칙에 따라 업데이트했습니다. 그러나 이러한 방법은 복잡한 문제에서 신경망의 성능을 제한했습니다. 오차 역전파는 네트워크 전체에 걸쳐 가중치를 시스템적으로 조정함으로써, 네트워크가 학습 과정에서 발생하는 오류를 최소화할 수 있도록 도와줍니다.



### 오차 계산: 평균 제곱 오차 (MSE)

오차 역전파에서 중요한 단계 중 하나는 네트워크의 출력과 실제 목푯값 사이의 오차를 계산하는 것입니다. 평균 제곱 오차(Mean Squared Error, MSE)는 이 오차를 측정하는 흔히 사용되는 방법입니다. MSE는 예측값과 목푯값 간의 차이의 제곱을 평균 내어 계산합니다.



```python
# Python 코드 예제: MSE 계산과 경사도 소실 문제 설명
import numpy as np

# 평균 제곱 오차 함수
def MSE(target, y):
    """
    Mean Squared Error를 계산하는 함수
    :param target: 실제 목푯값 배열
    :param y: 예측값 배열
    :return: 계산된 MSE 값
    """
    return 0.5 * np.sum((y - target)**2)

# 예측값과 목푯값
y = np.array([0.0, 0.0, 0.8, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
target = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# MSE 계산
mse_value = MSE(target, y)
print(f"Calculated MSE: {mse_value}")
```

<pre>
Calculated MSE: 0.029999999999999992
</pre>
## 경사 하강법 (Gradient Descent) 이해 및 실습

경사 하강법은 최적화 알고리즘의 일종으로, 주로 기계학습과 딥러닝에서 비용 함수를 최소화하기 위해 사용됩니다. 이 방법은 파라미터를 반복적으로 조정하면서 손실 함수의 최솟값을 찾는 과정입니다.



### 경사 하강법의 원리

- **손실 함수 선택**: 모델의 예측값과 실젯값 사이의 오차를 나타내는 함수를 정의합니다. 이 예에서는 평균 제곱 오차(Mean Squared Error, MSE)를 사용합니다.

- **그래디언트 계산**: 손실 함수의 파라미터에 대한 기울기(미분)를 계산합니다. 이 그래디언트는 손실을 가장 빠르게 증가시키는 방향을 나타냅니다.

- **파라미터 업데이트**: 현재 파라미터에서 그래디언트에 학습률(learning rate)을 곱한 값을 빼서, 손실을 줄이는 방향으로 파라미터를 조정합니다.



### 경사 하강법의 한계

- **학습률 선택**: 학습률이 너무 높으면 손실 함수가 발산할 수 있고, 너무 작으면 학습이 매우 느려질 수 있습니다.

- **지역 최솟값(Local Minima)**: 경사 하강법은 지역 최솟값에 빠질 위험이 있으며, 이는 전역 최솟값(Global Minima)이 아닐 수 있습니다.

- **플래토 현상(Plateaus)**: 그래디언트가 0에 가까운 평평한 지역에서는 학습이 정체될 수 있습니다.



```python
import numpy as np
import matplotlib.pyplot as plt

# 경사 하강법을 이용하여 최소값 찾기
x = 10  # 초깃값
learning_rate = 0.2  # 학습률
precision = 0.00001  # 정밀도
max_iterations = 100  # 최대 반복 횟수

# 손실 함수: (x-3)^2 + 10
loss_func = lambda x: (x-3)**2 + 10
# 그래디언트 함수: 손실 함수의 도함수
gradient = lambda x: 2 * (x - 3)

# 경사 하강법 실행
for i in range(max_iterations):
    grad = gradient(x)
    x = x - learning_rate * grad  # 파라미터 업데이트
    loss = loss_func(x)
    print(f"Iteration {i+1}: x = {x:.5f}, Loss = {loss:.5f}")

    if abs(grad) < precision:
        print("Gradient close to zero; stopping.")
        break
```

<pre>
Iteration 1: x = 7.20000, Loss = 27.64000
Iteration 2: x = 5.52000, Loss = 16.35040
Iteration 3: x = 4.51200, Loss = 12.28614
Iteration 4: x = 3.90720, Loss = 10.82301
Iteration 5: x = 3.54432, Loss = 10.29628
Iteration 6: x = 3.32659, Loss = 10.10666
Iteration 7: x = 3.19596, Loss = 10.03840
Iteration 8: x = 3.11757, Loss = 10.01382
Iteration 9: x = 3.07054, Loss = 10.00498
Iteration 10: x = 3.04233, Loss = 10.00179
Iteration 11: x = 3.02540, Loss = 10.00064
Iteration 12: x = 3.01524, Loss = 10.00023
Iteration 13: x = 3.00914, Loss = 10.00008
Iteration 14: x = 3.00549, Loss = 10.00003
Iteration 15: x = 3.00329, Loss = 10.00001
Iteration 16: x = 3.00197, Loss = 10.00000
Iteration 17: x = 3.00118, Loss = 10.00000
Iteration 18: x = 3.00071, Loss = 10.00000
Iteration 19: x = 3.00043, Loss = 10.00000
Iteration 20: x = 3.00026, Loss = 10.00000
Iteration 21: x = 3.00015, Loss = 10.00000
Iteration 22: x = 3.00009, Loss = 10.00000
Iteration 23: x = 3.00006, Loss = 10.00000
Iteration 24: x = 3.00003, Loss = 10.00000
Iteration 25: x = 3.00002, Loss = 10.00000
Iteration 26: x = 3.00001, Loss = 10.00000
Iteration 27: x = 3.00001, Loss = 10.00000
Iteration 28: x = 3.00000, Loss = 10.00000
Iteration 29: x = 3.00000, Loss = 10.00000
Gradient close to zero; stopping.
</pre>
### 코드 설명:

- **손실 함수와 그래디언트**: (x-3)^2 + 10은 간단한 2차 함수로, x=3에서 최소값을 갖습니다. 그래디언트는 2*(x-3)입니다.

- **경사 하강 실행**: 시작점 x=10에서 시작하여, 각 단계마다 그래디언트에 따라 x 값을 조정하고 손실을 계산하여 출력합니다.

- **시각화**: 2차원 벡터 필드에서 x와 y에 대한 그래디언트를 시각화하여 경사 하강법의 방향을 보여줍니다.



```python
# XOR 문제 해결을 위한 다층 퍼셉트론
import numpy as np

# 시그모이드 활성화 함수와 그 미분
def actf(x):
    """활성화 함수로 시그모이드 함수를 사용"""
    return 1 / (1 + np.exp(-x))

def actf_deriv(x):
    """시그모이드 함수의 도함수"""
    return x * (1 - x)

# 네트워크 구조 설정
inputs, hiddens, outputs = 2, 2, 1
learning_rate = 0.2

# 훈련 데이터와 목표 출력 (XOR 문제)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
T = np.array([[0], [1], [1], [0]])

# 가중치와 바이어스 초기화
W1 = np.array([[0.10, 0.20], [0.30, 0.40]])
W2 = np.array([[0.50], [0.60]])
B1 = np.array([0.1, 0.2])
B2 = np.array([0.3])
```


```python
# 순방향 전파 계산
def predict(x):
    """네트워크의 순방향 계산을 수행"""
    layer0 = x
    Z1 = np.dot(layer0, W1) + B1
    layer1 = actf(Z1)
    Z2 = np.dot(layer1, W2) + B2
    layer2 = actf(Z2)
    return layer0, layer1, layer2

# 역전파 및 가중치 업데이트
def fit():
    global W1, W2, B1, B2
    for i in range(90000):
        for x, y in zip(X, T):
            x = np.reshape(x, (1, -1))
            y = np.reshape(y, (1, -1))

            layer0, layer1, layer2 = predict(x)

            # 출력층의 오차와 델타 계산
            layer2_error = layer2 - y
            layer2_delta = layer2_error * actf_deriv(layer2)

            # 은닉층의 오차와 델타 계산
            layer1_error = layer2_delta.dot(W2.T)
            layer1_delta = layer1_error * actf_deriv(layer1)

            # 가중치와 바이어스 업데이트
            W2 -= learning_rate * np.dot(layer1.T, layer2_delta)
            W1 -= learning_rate * np.dot(layer0.T, layer1_delta)
            B2 -= learning_rate * np.sum(layer2_delta, axis=0)
            B1 -= learning_rate * np.sum(layer1_delta, axis=0)

        # 파라미터 값 변화를 3000번 반복마다 출력
        if (i + 1) % 3000 == 0:
            print(f"After {i + 1} iterations:")
            print(f"W1: {W1}\nW2: {W2}\nB1: {B1}\nB2: {B2}\n")

# 테스트 함수
def test():
    """학습된 네트워크로 XOR 문제 테스트"""
    for x, y in zip(X, T):
        x = np.reshape(x, (1, -1))
        layer0, layer1, layer2 = predict(x)
        print(f"Input: {x}, Target: {y}, Predicted: {layer2}")

# 모델 학습 실행
fit()
# 학습 결과 테스트
test()
```

<pre>
After 3000 iterations:
W1: [[-0.15887832 -0.02249762]
 [-0.13077359 -0.05368119]]
W2: [[0.14800796]
 [0.19945944]]
B1: [-0.35521223 -0.25351778]
B2: [-0.14179549]

After 6000 iterations:
W1: [[-0.24061267 -0.072481  ]
 [-0.33310181 -0.29946148]]
W2: [[-0.06520414]
 [ 0.10975503]]
B1: [-0.63183937 -0.57181538]
B2: [-0.01698159]

After 9000 iterations:
W1: [[-4.16896225 -0.41789535]
 [-4.29802654 -1.27714375]]
W2: [[-4.33660137]
 [ 1.67518112]]
B1: [0.5570265  0.00142186]
B2: [0.20168304]

After 12000 iterations:
W1: [[-5.75110507 -3.79650076]
 [-5.84602427 -3.80915635]]
W2: [[-7.80871476]
 [ 7.481593  ]]
B1: [2.15160484 5.58144586]
B2: [-3.40041243]

After 15000 iterations:
W1: [[-5.9914569  -4.18947206]
 [-6.06589364 -4.20141118]]
W2: [[-8.5908069]
 [ 8.3608668]]
B1: [2.34570496 6.20562871]
B2: [-3.88147665]

After 18000 iterations:
W1: [[-6.11857319 -4.38446387]
 [-6.18390176 -4.39569285]]
W2: [[-9.02027759]
 [ 8.82151791]]
B1: [2.43789266 6.51078363]
B2: [-4.12753921]

After 21000 iterations:
W1: [[-6.20451434 -4.5122721 ]
 [-6.26428488 -4.52295216]]
W2: [[-9.31674406]
 [ 9.13333249]]
B1: [2.4970472  6.70934332]
B2: [-4.29224311]

After 24000 iterations:
W1: [[-6.26917896 -4.60656174]
 [-6.325063   -4.61680692]]
W2: [[-9.54308114]
 [ 9.36872527]]
B1: [2.54012553 6.85515486]
B2: [-4.41573296]

After 27000 iterations:
W1: [[-6.32088001 -4.68089095]
 [-6.37382909 -4.69078032]]
W2: [[-9.72607929]
 [ 9.55763731]]
B1: [2.57378481 6.96972428]
B2: [-4.51437251]

After 30000 iterations:
W1: [[-6.36387172 -4.74203166]
 [-6.41449169 -4.7516221 ]]
W2: [[-9.87964002]
 [ 9.71531926]]
B1: [2.60129286 7.06373251]
B2: [-4.59641851]

After 33000 iterations:
W1: [[-6.40061852 -4.79383718]
 [-6.44932488 -4.80317118]]
W2: [[-10.01190329]
 [  9.85058645]]
B1: [2.62448542 7.14323154]
B2: [-4.66661085]

After 36000 iterations:
W1: [[-6.43267372 -4.83870245]
 [-6.479767   -4.84781275]]
W2: [[-10.12804497]
 [  9.96899043]]
B1: [2.64449208 7.21197033]
B2: [-4.72791865]

After 39000 iterations:
W1: [[-6.46107838 -4.87821438]
 [-6.50678482 -4.8871269 ]]
W2: [[-10.23155997]
 [ 10.07425166]]
B1: [2.66205534 7.27242628]
B2: [-4.78232321]

After 42000 iterations:
W1: [[-6.48656369 -4.91347694]
 [-6.53105895 -4.92221265]]
W2: [[-10.32491883]
 [ 10.16898422]]
B1: [2.6776883  7.32631889]
B2: [-4.8312117]

After 45000 iterations:
W1: [[-6.5096625  -4.9452881 ]
 [-6.55308646 -4.95386426]]
W2: [[-10.40993229]
 [ 10.25509441]]
B1: [2.69175975 7.37488863]
B2: [-4.87559268]

After 48000 iterations:
W1: [[-6.53077489 -4.97424276]
 [-6.57324124 -4.98267379]]
W2: [[-10.4879669 ]
 [ 10.33401485]]
B1: [2.7045433  7.41905876]
B2: [-4.91622217]

After 51000 iterations:
W1: [[-6.55020879 -5.00079589]
 [-6.5918115  -5.00909405]]
W2: [[-10.56007929]
 [ 10.40684923]]
B1: [2.71624735 7.4595343 ]
B2: [-4.95368127]

After 54000 iterations:
W1: [[-6.56820616 -5.02530301]
 [-6.60902406 -5.03347877]]
W2: [[-10.62710324]
 [ 10.47446571]]
B1: [2.72703414 7.49686545]
B2: [-4.98842621]

After 57000 iterations:
W1: [[-6.58496051 -5.04804702]
 [-6.62506053 -5.05610946]]
W2: [[-10.6897082 ]
 [ 10.53755942]]
B1: [2.73703233 7.53148947]
B2: [-5.02082161]

After 60000 iterations:
W1: [[-6.60062898 -5.0692566 ]
 [-6.64006859 -5.07721365]]
W2: [[-10.74843978]
 [ 10.59669552]]
B1: [2.74634566 7.5637595 ]
B2: [-5.05116348]

After 63000 iterations:
W1: [[-6.6153409  -5.08911917]
 [-6.65416991 -5.09697782]]
W2: [[-10.80374852]
 [ 10.65233965]]
B1: [2.7550589  7.59396464]
B2: [-5.07969536]

After 66000 iterations:
W1: [[-6.62920396 -5.10779019]
 [-6.66746589 -5.11555661]]
W2: [[-10.85601074]
 [ 10.70488002]]
B1: [2.7632422  7.62234446]
B2: [-5.10662]

After 69000 iterations:
W1: [[-6.64230876 -5.12539998]
 [-6.68004193 -5.13307969]]
W2: [[-10.905544  ]
 [ 10.75464359]]
B1: [2.77095425 7.64909968]
B2: [-5.132108]

After 72000 iterations:
W1: [[-6.65473223 -5.14205884]
 [-6.69197062 -5.14965679]]
W2: [[-10.95261871]
 [ 10.80190835]]
B1: [2.77824462 7.67440001]
B2: [-5.15630421]

After 75000 iterations:
W1: [[-6.66654025 -5.15786094]
 [-6.70331413 -5.16538158]]
W2: [[-10.99746701]
 [ 10.84691259]]
B1: [2.78515557 7.69839025]
B2: [-5.17933272]

After 78000 iterations:
W1: [[-6.67778965 -5.17288727]
 [-6.71412618 -5.18033463]]
W2: [[-11.04028968]
 [ 10.88986214]]
B1: [2.79172335 7.72119486]
B2: [-5.20130056]

After 81000 iterations:
W1: [[-6.6885298  -5.18720799]
 [-6.72445344 -5.19458576]]
W2: [[-11.08126154]
 [ 10.93093598]]
B1: [2.79797935 7.74292159]
B2: [-5.22230075]

After 84000 iterations:
W1: [[-6.69880384 -5.20088428]
 [-6.73433674 -5.2081958 ]]
W2: [[-11.12053572]
 [ 10.9702907 ]]
B1: [2.80395089 7.76366435]
B2: [-5.24241463]

After 87000 iterations:
W1: [[-6.70864971 -5.21396978]
 [-6.74381197 -5.22121811]]
W2: [[-11.15824714]
 [ 11.00806411]]
B1: [2.80966189 7.78350542]
B2: [-5.26171369]

After 90000 iterations:
W1: [[-6.7181009  -5.22651178]
 [-6.75291089 -5.23369975]]
W2: [[-11.19451524]
 [ 11.04437812]]
B1: [2.8151334  7.80251731]
B2: [-5.28026114]

Input: [[0 0]], Target: [0], Predicted: [[0.00814407]]
Input: [[0 1]], Target: [1], Predicted: [[0.99154105]]
Input: [[1 0]], Target: [1], Predicted: [[0.99152258]]
Input: [[1 1]], Target: [0], Predicted: [[0.01038517]]
</pre>
### 코드 주석 보완 설명

- **actf 및 actf_deriv 함수**: 활성화 함수로 시그모이드 함수를 사용하며, 해당 함수의 도함수를 계산하는 함수를 정의합니다. 이는 역전파에서 그라디언트 계산에 필요합니다.

- **fit 함수**: 모델을 학습하는 함수로, 네트워크의 파라미터를 업데이트합니다. 각 반복마다 손실을 계산하고, 가중치를 그라디언트 방향으로 조정합니다. 3000번 반복마다 중간 결과를 출력하여 파라미터가 어떻게 변화하는지 관찰할 수 있습니다.

- **predict 함수**: 주어진 입력에 대해 네트워크를 순방향으로 실행하여 출력을 계산합니다.

- **test 함수**: 학습된 모델을 테스트 데이터에 적용하여 결과를 확인합니다.



### 경사도 소실 문제

깊은 네트워크에서 경사도(가중치에 대한 오차의 변화율)를 역전파할 때, 입력층에 가까워질수록 경사도가 점점 작아지는 현상이 발생할 수 있습니다. 이는 신경망의 초기 층이 충분히 학습되지 않게 만들어, 전체 학습 과정의 효율성을 크게 저하할 수 있습니다.



```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# 데이터 파일 경로
file_path = 'dataset_KU_dormitory_v1.csv'

# 데이터 로드
data = pd.read_csv(file_path)

# 독립변수와 종속변수 분리
X = data.drop(['Consumption'], axis=1)
y = data['Consumption']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 특성 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```


```python
# 시그모이드 활성화 함수를 사용하는 MLP 모델 학습
mlp_sigmoid = MLPRegressor(hidden_layer_sizes=(100, 50, 25), activation='logistic', max_iter=1000, random_state=42)
mlp_sigmoid.fit(X_train_scaled, y_train)
y_pred_sigmoid = mlp_sigmoid.predict(X_test_scaled)
mse_sigmoid = mean_squared_error(y_test, y_pred_sigmoid)

# ReLU 활성화 함수를 사용하는 MLP 모델 학습
mlp_relu = MLPRegressor(hidden_layer_sizes=(100, 50, 25), activation='relu', max_iter=1000, random_state=42)
mlp_relu.fit(X_train_scaled, y_train)
y_pred_relu = mlp_relu.predict(X_test_scaled)
mse_relu = mean_squared_error(y_test, y_pred_relu)

# 결과 출력
print(f"Mean Squared Error with Sigmoid Activation: {mse_sigmoid}")
print(f"Mean Squared Error with ReLU Activation: {mse_relu}")
```

<pre>
Mean Squared Error with Sigmoid Activation: 80478.60531328227
Mean Squared Error with ReLU Activation: 3245.732186362485
</pre>

```python
# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_sigmoid, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Consumption')
plt.ylabel('Predicted Consumption')
plt.title('MLP with Sigmoid Activation')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_relu, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Consumption')
plt.ylabel('Predicted Consumption')
plt.title('MLP with ReLU Activation')

plt.show()
```

<pre>
<Figure size 1400x600 with 2 Axes>
</pre>
## ReLU와 Sigmoid 활성화 함수의 역전파 차이



다층 퍼셉트론(MLP)에서 활성화 함수는 뉴런의 출력을 결정하며, 역전파 과정에서 그라디언트를 계산하는 데 중요한 역할을 합니다. ReLU와 Sigmoid는 가장 널리 사용되는 활성화 함수 중 두 가지로, 이들은 역전파 과정에서 각기 다른 특성을 보입니다.



### ReLU (Rectified Linear Unit)

- **정의:**

  \[

  \text{ReLU}(x) = \max(0, x)

  \]

- **미분:**

  - \( x > 0 \): 그라디언트는 1

  - \( x \leq 0 \): 그라디언트는 0

- **특성:**

  - **비선형성:** ReLU는 비선형성을 제공하여 모델이 다양한 데이터 분포를 학습할 수 있게 합니다.

  - **경사도 소실 문제 완화:** ReLU는 음수 값을 0으로 변환하여 특정 뉴런이 죽는 문제(Dying ReLU)를 일으킬 수 있지만, 양수 값의 그라디언트는 항상 1로, 깊은 신경망에서도 그라디언트가 소실되지 않습니다.



### Sigmoid

- **정의:**

  \[

  \sigma(x) = \frac{1}{1 + e^{-x}}

  \]

- **미분:**

  \[

  \sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))

  \]

- **특성:**

  - **비선형성:** Sigmoid 함수는 출력값을 [0, 1] 사이로 압축합니다.

  - **경사도 소실 문제:** Sigmoid 함수는 입력값이 매우 크거나 매우 작을 때 그라디언트가 0에 가까워져, 깊은 신경망에서 경사도가 소실되는 문제를 일으킵니다. 이는 학습이 매우 느려지거나 거의 불가능하게 만듭니다.



```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """시그모이드 활성화 함수"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """시그모이드 함수의 미분"""
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    """ReLU 활성화 함수"""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU 함수의 미분"""
    return np.where(x <= 0, 0, 1)

# 임의의 깊은 네트워크를 시뮬레이션하기 위해 층의 수 설정
layers = 10  # 네트워크의 층 수
neuron_output = 0.1  # 초기 뉴런 출력 (임의로 낮은 값으로 설정하여 그라디언트 소실 확인)
gradients_sigmoid = []  # 각 층의 시그모이드 그라디언트 저장
gradients_relu = []  # 각 층의 ReLU 그라디언트 저장

# 네트워크를 통해 역전파 시뮬레이션 (Sigmoid)
output = neuron_output
for _ in range(layers):
    grad = sigmoid_derivative(output)
    gradients_sigmoid.append(grad)
    output *= grad  # 다음 뉴런 출력을 현재 그라디언트로 업데이트

# 네트워크를 통해 역전파 시뮬레이션 (ReLU)
output = neuron_output
for _ in range(layers):
    grad = relu_derivative(output)
    gradients_relu.append(grad)
    output *= grad  # 다음 뉴런 출력을 현재 그라디언트로 업데이트

# 그라디언트 출력
print("Gradients at each layer (Sigmoid):")
for i, grad in enumerate(gradients_sigmoid):
    print(f"Layer {i+1}: Gradient = {grad}")

print("\nGradients at each layer (ReLU):")
for i, grad in enumerate(gradients_relu):
    print(f"Layer {i+1}: Gradient = {grad}")

# 그라디언트 시각화
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, layers+1), gradients_sigmoid, marker='o')
plt.xlabel('Layer')
plt.ylabel('Gradient')
plt.title('Gradient Vanishing with Depth in Sigmoid Activation')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, layers+1), gradients_relu, marker='o')
plt.xlabel('Layer')
plt.ylabel('Gradient')
plt.title('Gradient Vanishing with Depth in ReLU Activation')
plt.grid(True)

plt.show()
```

<pre>
Gradients at each layer (Sigmoid):
Layer 1: Gradient = 0.24937604019289197
Layer 2: Gradient = 0.24996113627229605
Layer 3: Gradient = 0.24999757153619778
Layer 4: Gradient = 0.2499998482230396
Layer 5: Gradient = 0.2499999905139479
Layer 6: Gradient = 0.24999999940712178
Layer 7: Gradient = 0.2499999999629451
Layer 8: Gradient = 0.24999999999768407
Layer 9: Gradient = 0.24999999999985525
Layer 10: Gradient = 0.24999999999999095

Gradients at each layer (ReLU):
Layer 1: Gradient = 1
Layer 2: Gradient = 1
Layer 3: Gradient = 1
Layer 4: Gradient = 1
Layer 5: Gradient = 1
Layer 6: Gradient = 1
Layer 7: Gradient = 1
Layer 8: Gradient = 1
Layer 9: Gradient = 1
Layer 10: Gradient = 1
</pre>
<pre>
<Figure size 1400x600 with 2 Axes>
</pre>

```python
```


```python
!jupyter nbconvert --to markdown "/content/drive/MyDrive/Colab Notebooks/TEST/notebook_test.ipynb"
```
