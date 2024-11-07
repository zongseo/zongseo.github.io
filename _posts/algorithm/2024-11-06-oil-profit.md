---
layout: single
title:  "Oil Profit problem"
categories: ALGORITHM
tag: [sch, algorithm, coding]
toc: true
author_profile: false
---


> Oil Profit Maximizing 문제
> 
> 모의 투자 프로그램을 통해 최대의 이득을 얻을 수 있는 프로그램을 만들고자한다. 
> 예를 들어, 첫 번쨰 입력 예시에서 19, 15, 11, 10, 20, 13, 30, 11, 34, 15가 주어진 경우, 네번째 날(10)에 인수하여, 아홉번째 날에 매도(종가 34)하는 경우에 최대 이익 24를 얻을 수 있다.
> 이때 출력값은 매수시점 B, 매도시점 S와 최대이익 M을 구하고 이들의 합 B+S+M의 값 4+9+24 = 37을 출력한다. **(Divide-and-Conquer활용)**


![2024_Algo_Assignment03](https://github.com/user-attachments/assets/f017b79f-b9da-485d-b809-3ed2e7a1c4ca)


## 문제 해결


1. 주어진 숫자 배열을 각각 전날과의 시세차익을 포함하는 새로운 배열을 만든다. 즉 9 11 13 5 인경우 2 2 -8을 포함하는 배열을 만든다.

2. **Divide**

    재귀함수를 통해 시세차익 리스트를 최소단위(한개)까지 나눈다.

3. **Conquer**

    최소단위로 쪼개진 숫자들을 각각 왼쪽, 오른쪽, 중간(왼쪽+오른쪽)범위에서 순서대로 최고의 이익(각 범위에서 계속 더해나가면서 최고 값이 나올때까지 비교)이 나오면 반환하여 다시 비교한다.

    각 전날 시세차익을 담는 리스트기 때문에 계속 더해가는 것이 날이 지날수록 변화하는 시세차익 정보를 얻을 수 있다. 즉 최고 값이 되는 값을 찾는다면 날짜 인덱스와 그 값을 반환하여 문제를 해결할 수 있다.


## 완성 코드

```python
def max_crossing_subarray(prices_diff, low, mid, high):
    # 왼쪽 배열과 오른쪽 배열을 중간을 기준으로 연속합의 최대 지점 구하기
    left_sum = float('-inf')
    total = 0
    max_left = mid
    for i in range(mid, low - 1, -1):
        total += prices_diff[i]
        if total > left_sum:
            left_sum = total
            max_left = i
    
    right_sum = float('-inf')
    total = 0
    max_right = mid + 1
    for j in range(mid + 1, high + 1):
        total += prices_diff[j]
        if total > right_sum:
            right_sum = total
            max_right = j
    
    return max_left, max_right, left_sum + right_sum

def max_subarray(prices_diff, low, high):
    # 최소 단위(원소 1개)까지 쪼개지면 본인 인덱스, 인덱스에 해당하는 값(가격차이) 반환
    if low == high:
        return low, high, prices_diff[low]
    
    mid = (low + high) // 2
    left_low, left_high, left_sum = max_subarray(prices_diff, low, mid)
    right_low, right_high, right_sum = max_subarray(prices_diff, mid + 1, high)
    cross_low, cross_high, cross_sum = max_crossing_subarray(prices_diff, low, mid, high)
    
    # 왼쪽 배열, 오른쪽 배열, 크로스 배열 우선순위
    if left_sum >= right_sum and left_sum >= cross_sum:
        return left_low, left_high, left_sum
    elif right_sum >= left_sum and right_sum >= cross_sum:
        return right_low, right_high, right_sum
    else:
        return cross_low, cross_high, cross_sum

def find_max_profit(prices):
    n = len(prices)
    if n < 2:
        return 0  # 가격이 하루치 밖에 없으면 0 반환

    # 1일 간격으로 가격 변화 배열 생성
    prices_diff = [prices[i] - prices[i - 1] for i in range(1, n)]
    
    # 최대 이익을 갖는 날짜 인덱스와 이익 서칭
    buy_day, sell_day, max_profit = max_subarray(prices_diff, 0, len(prices_diff) - 1)
    
    # 1일 간격 가격차이로 최대 이익을 구했기 때문에 반환받은 인덱스를 날짜 개념으로 +1, +2
    buy_day += 1
    sell_day += 2
    
    return buy_day + sell_day + max_profit


prices = [19, 15, 11, 10, 20, 13, 30, 11, 34, 15]

result = find_max_profit(prices)
print(result)
```

<pre>
37
</pre>
