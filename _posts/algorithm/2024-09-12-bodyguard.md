---
layout: single
title:  "Bodyguard Algorithm"
categories: ALGORITHM
tag: [sch, algorithm, coding]
toc: true
author_profile: false
---


> bodyguard 문제
> 
> n x m 행렬에서 각 행과 열에 한명의 보디가드라도 존재해야하며 그렇지 못한 곳에 최소한의 인력을 투입해서 매꾸는 문제이다.


![2024_Algo_Assignment01](https://github.com/user-attachments/assets/28de3531-b940-45a2-bc5c-fa7a54206918)



## 문제 해결


1. 각 행과 열의 빈 공간이 없어야 하기 때문에 입력된 OX문자열을 row, col 리스트에 1과 0으로 입력하여 각 행, 열에 단 하나라도 존재하는지 여부를 나타내는 변수를 생성한다.



2. 이렇게 저장된 각 자리의 정보들을 이용하여 최소한의 인력으로 조건을 만족 시킬 수 있는 방법은 더욱 많은 빈자리 개수가 최솟값으로 따라간다.



    예를 들어. row = [1, 0, 0, 0], col = [0, 1, 1, 0]이라고 할 때 가장 최소한의 인력을 위해서는 서로 교차하는 지점에 한 명 씩을 넣어야 한다. 2행 1열[2, 1], 3행 4열[3, 4]를 한 명 씩 배치한다면 마지막으로 4행에 한 명이라도 존재해야 한다.

    

    결국 가장 빈 공간이 많은 행 또는 열의 빈자리 값이 결국 최소 인원 배치가 된다.


## 완성 코드



```python
# 입력을 받는 부분
rows, cols = map(int, input("행과 열의 수를 입력하세요 (예: 4 4): ").split())
matrix = [input() for _ in range(rows)]  # 각 행을 입력받는다

gen_row = []
gen_col = []

# 행별로 'O'의 존재 여부 확인
for row in matrix:
    gen_row.append([1] if 'O' in row else [])

# 열별로 'O'의 존재 여부 확인
for i in range(cols):
    gen_col.append([1] if any('O' in row[i] for row in matrix) else [])

# 비어있는 행과 열의 개수를 계산
empty_rows = sum(1 for r in gen_row if not r)
empty_cols = sum(1 for c in gen_col if not c)

# 결과 출력
result = max(empty_rows, empty_cols)
print("결과:", result)
```

<pre>
결과: 3
</pre>
