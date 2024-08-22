---
layout: single
title:  "Langchain으로 엑셀 데이터 자동 분석 및 요약하기"
categories: AI
tag: [langchain, llm, ai]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


# PandasDataFrameOutputParser



**Pandas DataFrame**은 Python 프로그래밍 언어에서 널리 사용되는 데이터 구조로, 데이터 조작 및 분석을 위한 강력한 도구입니다. DataFrame은 구조화된 데이터를 효과적으로 다루기 위한 포괄적인 도구 세트를 제공하며, 이를 통해 데이터 정제, 변환 및 분석과 같은 다양한 작업을 수행할 수 있습니다.



이 **출력 파서**는 사용자가 임의의 Pandas DataFrame을 지정하여 해당 DataFrame에서 데이터를 추출하고, 이를 형식화된 사전(dictionary) 형태로 조회할 수 있게 해주는 LLM(Large Language Model) 기반 도구입니다.




```python
from dotenv import load_dotenv

load_dotenv()
```

<pre>
True
</pre>

```python
# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH03-OutputParser")
```

<pre>
LangSmith 추적을 시작합니다.
[프로젝트명]
CH03-OutputParser
</pre>

```python
import pprint
from typing import Any, Dict

import pandas as pd
from langchain.output_parsers import PandasDataFrameOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
```


```python
# ChatOpenAI 모델 초기화 (gpt-3.5-turbo 모델 사용을 권장합니다)
model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
```

`format_parser_output` 함수는 파서 출력을 사전 형식으로 변환하고 출력 형식을 지정하는 데 사용됩니다. 



```python
# 출력 목적으로만 사용됩니다.
def format_parser_output(parser_output: Dict[str, Any]) -> None:
    # 파서 출력의 키들을 순회합니다.
    for key in parser_output.keys():
        # 각 키의 값을 딕셔너리로 변환합니다.
        parser_output[key] = parser_output[key].to_dict()
    # 예쁘게 출력합니다.
    return pprint.PrettyPrinter(width=4, compact=True).pprint(parser_output)
```

- `titanic.csv` 데이터를 읽어온 뒤 DataFrame 을 로드하여 `df` 변수에 할당합니다.

- PandasDataFrameOutputParser를 사용하여 DataFrame을 파싱합니다.




```python
# 원하는 Pandas DataFrame을 정의합니다.
df = pd.read_csv("./data/titanic.csv")
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



```python
# 파서를 설정하고 프롬프트 템플릿에 지시사항을 주입합니다.
parser = PandasDataFrameOutputParser(dataframe=df)

# 파서의 지시사항을 출력합니다.
print(parser.get_format_instructions())
```

<pre>
The output should be formatted as a string as the operation, followed by a colon, followed by the column or row to be queried on, followed by optional array parameters.
1. The column names are limited to the possible columns below.
2. Arrays must either be a comma-separated list of numbers formatted as [1,3,5], or it must be in range of numbers formatted as [0..4].
3. Remember that arrays are optional and not necessarily required.
4. If the column is not in the possible columns or the operation is not a valid Pandas DataFrame operation, return why it is invalid as a sentence starting with either "Invalid column" or "Invalid operation".

As an example, for the formats:
1. String "column:num_legs" is a well-formatted instance which gets the column num_legs, where num_legs is a possible column.
2. String "row:1" is a well-formatted instance which gets row 1.
3. String "column:num_legs[1,2]" is a well-formatted instance which gets the column num_legs for rows 1 and 2, where num_legs is a possible column.
4. String "row:1[num_legs]" is a well-formatted instance which gets row 1, but for just column num_legs, where num_legs is a possible column.
5. String "mean:num_legs[1..3]" is a well-formatted instance which takes the mean of num_legs from rows 1 to 3, where num_legs is a possible column and mean is a valid Pandas DataFrame operation.
6. String "do_something:num_legs" is a badly-formatted instance, where do_something is not a valid Pandas DataFrame operation.
7. String "mean:invalid_col" is a badly-formatted instance, where invalid_col is not a possible column.

Here are the possible columns:
```
PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
```

</pre>
컬럼에 대한 값을 조회하는 예제입니다.



```python
# 열 작업 예시입니다.
df_query = "Age column 을 조회해 주세요."


# 프롬프트 템플릿을 설정합니다.
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{question}\n",
    input_variables=["question"],  # 입력 변수 설정
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },  # 부분 변수 설정
)

# 체인 생성
chain = prompt | model | parser

# 체인 실행
parser_output = chain.invoke({"question": df_query})

# 출력
format_parser_output(parser_output)
```

<pre>
{'Age': {0: 22.0,
         1: 38.0,
         2: 26.0,
         3: 35.0,
         4: 35.0,
         5: nan,
         6: 54.0,
         7: 2.0,
         8: 27.0,
         9: 14.0,
         10: 4.0,
         11: 58.0,
         12: 20.0,
         13: 39.0,
         14: 14.0,
         15: 55.0,
         16: 2.0,
         17: nan,
         18: 31.0,
         19: nan}}
</pre>
첫 번째 행을 검색하는 예시입니다.



```python
# 행 조회 예시입니다.
df_query = "Retrieve the first row."

# 체인 실행
parser_output = chain.invoke({"question": df_query})

# 결과 출력
format_parser_output(parser_output)
```

<pre>
{'0': {'Age': 22.0,
       'Cabin': nan,
       'Embarked': 'S',
       'Fare': 7.25,
       'Name': 'Braund, '
               'Mr. '
               'Owen '
               'Harris',
       'Parch': 0,
       'PassengerId': 1,
       'Pclass': 3,
       'Sex': 'male',
       'SibSp': 1,
       'Survived': 0,
       'Ticket': 'A/5 '
                 '21171'}}
</pre>
특정 열에서 일부 행의 평균을 검색하는 작업 예제입니다.



```python
# row 0 ~ 4의 평균 나이를 구합니다.
df["Age"].head().mean()
```

<pre>
31.2
</pre>

```python
# 임의의 Pandas DataFrame 작업 예시, 행의 수를 제한합니다.
df_query = "Retrieve the average of the Ages from row 0 to 4."

# 체인 실행
parser_output = chain.invoke({"question": df_query})

# 결과 출력
print(parser_output)
```

<pre>
{'mean': 31.2}
</pre>
다음은 요금(Fare) 에 대한 평균 가격을 산정하는 예시입니다.



```python
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



```python
# 잘못 형식화된 쿼리의 예시입니다.
df_query = "Calculate average `Fare` rate."

# 체인 실행
parser_output = chain.invoke({"question": df_query})

# 결과 출력
print(parser_output)
```

<pre>
{'mean': 22.19937}
</pre>

```python
# 결과 검증
df["Fare"].mean()
```

<pre>
22.19937
</pre>