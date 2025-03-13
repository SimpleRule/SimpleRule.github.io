---
title:  "EDA project : selling lectures"
header:
   teaser: "/assets/images/EDA project-lecture sales images/output_73_0.png" 
excerpt: "Analysis of lecture sales data"
categories: 
- EDA
tags:
- project
- team
- analysis
toc_label: Contents
toc: true
toc_sticky: True
toc_h_min: 1
toc_h_max: 3
date: 2025-03-12
last_modified_at: 2025-03-12
---


# EDA project explanation 
## About Project
이어드림스쿨에서 진행한 EDA 팀 프로젝트입니다.\
이어드림스쿨은 중소기업진흥공단에서 기획한 데이터 사이언티스트를 양성하는 부트캠프입니다.\
EDA 팀 프로젝트의 팀원은 총 4명이었으며, 저는 팀장을 맡았습니다.\
프로젝트는 약 2주간 진행되었습니다.

## About FAST CAMPUS
![image](/assets/images/EDA project-lecture sales images/fastcampus_logo.png){: width="60%" height="60%"}{: .center}

Link : [fast campus](https://fastcampus.co.kr/, "fastcampus")

2014년 패스트 캠퍼스라는 사명을 처음으로 사용하였고 2021년 데이원 컴퍼니로 법인명이 변경되었습니다.\
**데이원컴퍼니**는 현재 CIC(Company In Company) 구조로 운영이 되고 있고 4개의 CIC가 만들어져 운영이 되고 있습니다.\
현재 그 중 하나의 회사가 **패스트 캠퍼스(FASTCAMPUS)**입니다.\
패스트 캠퍼스는 25~50세의 성인을 타겟으로 삼아서 **온/오프라인 교육(강의)**을 통해서 수익을 창출하는 회사입니다.

## About DATA
EDA로 분석할 데이터는 **패스트 캠퍼스에서 판매된 강의 정보**입니다.\
이는 **정형 데이터**로, **2022년 1월 1일**부터 **2022년 12월 31**일까지의 판매를 담고 있습니다.

***

# 라이브러리 설치

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from datetime import datetime, timedelta
import os
import re
import nltk
from pprint import pprint
```


```python
plt.rcParams['font.family'] = 'gulim'
 
df = pd.read_csv('data/실습데이터.csv')
```

***

# 데이터 확인
\
**데이터 개수, 컬럼, null, 타입을 확인해 봅니다**

기업 보안상 데이터를 직접 출력하진 않겠습니다.


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 159328 entries, 0 to 159327
    Data columns (total 16 columns):
     #   Column     Non-Null Count   Dtype 
    ---  ------     --------------   ----- 
     0   거래id       159328 non-null  int64 
     1   유형         159328 non-null  object
     2   고객id       159328 non-null  int64 
     3   코스ID       159328 non-null  int64 
     4   사이트        159328 non-null  object
     5   포맷         159328 non-null  object
     6   카테고리       159328 non-null  object
     7   코스(상품) 이름  159328 non-null  object
     8   거래일자       159328 non-null  object
     9   쿠폰이름       159328 non-null  object
     10  판매가격       159328 non-null  object
     11  결제수단       159328 non-null  object
     12  실거래금액      159328 non-null  int64 
     13  쿠폰할인액      159328 non-null  object
     14  거래금액       159328 non-null  object
     15  환불금액       159328 non-null  object
    dtypes: int64(4), object(12)
    memory usage: 19.4+ MB
    


```python
df.isnull().sum() #결측치 없음
```




    거래id         0
    유형           0
    고객id         0
    코스ID         0
    사이트          0
    포맷           0
    카테고리         0
    코스(상품) 이름    0
    거래일자         0
    쿠폰이름         0
    판매가격         0
    결제수단         0
    실거래금액        0
    쿠폰할인액        0
    거래금액         0
    환불금액         0
    dtype: int64



***

# 전처리

1. **'사이트'**컬럼 제거
    - 모든 데이터가 같은 값을 가지고 있음으로 무의미한 컬럼이다.
2. **'-'** 값들을 **0**으로 변경
    - '-'는 공백에 해당하는 값으로 분석하기 위해 0값으로 바꿔준다.
3. 날짜 데이터 변경 **2022. 9. 20. 오후 4:09:23** → **거래월: 9 / 거래일: 20 / 거래 시간: 16**
    - 월별, 일별, 시간별 분석을 위해 시간 컬럼을 분해한다. '분' 정보는 불필요하다고 판단하여 제거한다.
4. **'쿠폰할인액', '거래금액', '환불금액', '거래월', '거래일'** 데이터를 문자열에서 수치형으로 변환
    - 문자열로 되어 있는 숫자를 분석하기 위해서는 integer나 floating point같은 수치형으로 바꿔준다.


```python
df = df.drop('사이트', axis=1)  #사이트 컬럼 제거
df = df.replace('-',0) # - 를 0으로 변경 ##수치화하기 위함
```


```python
# 날짜 전처리
df_date_time = df['거래일자'].str.split('오') #'오'를 기준으로 분리
df_time = df_date_time.str.get(1)
df_time = df_time.str.split(" ") # "전","후"와 시간데이터 나누기
```


```python
new_time = []  #24시간으로 변경 ##오후 시간 +12, 오전 시간 그대로 반영
for i, j in df_time:
    
    if i == '후':
        time_with_date = datetime.strptime(j,'%H:%M:%S') + timedelta(hours=12)
        new_time.append(datetime.strftime(time_with_date,'%H'))
        
    elif i == '전':
        time_with_date = datetime.strptime(j,'%H:%M:%S')
        new_time.append(datetime.strftime(time_with_date,'%H'))


new_time_Se = pd.Series(new_time)
```


```python
df_date_time.dtype
```




    dtype('O')




```python
df_date = df_date_time.str.get(0) #년월일 데이터만 가져오기
YMD = df_date.str.split(". ") #년월일 나누기
month = YMD.str.get(1)  #월 데이터만 저장
day = YMD.str.get(2)  #일 데이터만 저장

df['거래월'] = month  # 월 추가
df['거래일'] = day   # 일 추가
df['거래시간'] = new_time_Se  #시간 추가
```


```python
df = df.drop('거래일자', axis=1) #거래일자 제거
```


```python
# 수치화
df = df.astype({'판매가격':int,
                '쿠폰할인액':int,
                '거래금액':int,
                '환불금액':int,
                '거래월':'int8',
                '거래일': 'int8',
                '거래시간': 'int8'
                })
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 159328 entries, 0 to 159327
    Data columns (total 17 columns):
     #   Column     Non-Null Count   Dtype 
    ---  ------     --------------   ----- 
     0   거래id       159328 non-null  int64 
     1   유형         159328 non-null  object
     2   고객id       159328 non-null  int64 
     3   코스ID       159328 non-null  int64 
     4   포맷         159328 non-null  object
     5   카테고리       159328 non-null  object
     6   코스(상품) 이름  159328 non-null  object
     7   쿠폰이름       159328 non-null  object
     8   판매가격       159328 non-null  int32 
     9   결제수단       159328 non-null  object
     10  실거래금액      159328 non-null  int64 
     11  쿠폰할인액      159328 non-null  int32 
     12  거래금액       159328 non-null  int32 
     13  환불금액       159328 non-null  int32 
     14  거래월        159328 non-null  int8  
     15  거래일        159328 non-null  int8  
     16  거래시간       159328 non-null  int8  
    dtypes: int32(4), int64(4), int8(3), object(6)
    memory usage: 15.0+ MB
    

***

# EDA

## 1. Basic analysis


```python
print("전체 구매 수: ", len(df[df['유형']=="PAYMENT"])) #전체 구매 수 
print("전체 환불한 수: ", len(df[df['유형']=="REFUND"])) #전체 환불한 수
print("고객의 수: ", df['고객id'][df['유형']=="PAYMENT"].nunique()) #고객의 수
print("코스(상품)의의 종류: ", df['코스ID'][df['유형']=="PAYMENT"].nunique()) #코스(상품)의 종류
```

    전체 구매 수:  148010
    전체 환불한 수:  11318
    고객의 수:  77210
    코스(상품)의의 종류:  406
    
\
한 명당 강의를 대략 1.92개 구매했다. 재구매율이 그리 높지 않다고 볼 수 있다.


```python
print("한 명당 구매한 개수: ", round(len(df[df['유형']=="PAYMENT"]) / df['고객id'][df['유형']=="PAYMENT"].nunique(), 2))
# 구매 수 / 고객 수
```

    한 명당 구매한 개수:  1.92
    
\
환불률이 0.0765 % 정도로 그리 높진 않다.


```python
print("구매 대비 환불률: ", round(len(df[df['유형']=="REFUND"]) / len(df[df['유형']=="PAYMENT"]), 4), "%")
# 환불 수 / 구매 수
```

    구매 대비 환불률:  0.0765 %
    
\
**heatmap으로 각 컬럼 간의 상관관계를 시각화해봅니다**

거래 금액이 높으면 실거래 금액이 높고, 할인액이 높으면 거래금액이 낮은 등 상식적으로 당연한 상관관계가 대부분이다.\
즉, 특별히 주목할만한 상관관계가 보이지는 않는다.


```python
# 컬럼 간 상관관계 
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='YlGnBu')
```




    <Axes: >




    
![png](/assets/images/EDA project-lecture sales images/output_31_1.png)
    



```python
#기술 통계
df.describe() 
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
      <th>거래id</th>
      <th>고객id</th>
      <th>코스ID</th>
      <th>판매가격</th>
      <th>실거래금액</th>
      <th>쿠폰할인액</th>
      <th>거래금액</th>
      <th>환불금액</th>
      <th>거래월</th>
      <th>거래일</th>
      <th>거래시간</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.593280e+05</td>
      <td>159328.000000</td>
      <td>159328.000000</td>
      <td>1.593280e+05</td>
      <td>1.593280e+05</td>
      <td>1.593280e+05</td>
      <td>1.593280e+05</td>
      <td>1.593280e+05</td>
      <td>159328.000000</td>
      <td>159328.000000</td>
      <td>159328.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.968703e+06</td>
      <td>50061.549903</td>
      <td>207679.355123</td>
      <td>2.183140e+05</td>
      <td>1.418337e+05</td>
      <td>4.928870e+04</td>
      <td>1.559904e+05</td>
      <td>-1.415675e+04</td>
      <td>6.522294</td>
      <td>15.650331</td>
      <td>14.534646</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.516892e+05</td>
      <td>28850.738273</td>
      <td>3541.408820</td>
      <td>7.867327e+04</td>
      <td>1.397796e+05</td>
      <td>8.447793e+04</td>
      <td>1.093086e+05</td>
      <td>5.633219e+04</td>
      <td>3.618602</td>
      <td>9.129716</td>
      <td>6.090027</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.486194e+06</td>
      <td>0.000000</td>
      <td>2204.000000</td>
      <td>0.000000e+00</td>
      <td>-1.054400e+06</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>-1.054400e+06</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.743640e+06</td>
      <td>25081.500000</td>
      <td>204373.000000</td>
      <td>1.700000e+05</td>
      <td>7.800000e+04</td>
      <td>0.000000e+00</td>
      <td>7.800000e+04</td>
      <td>0.000000e+00</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>11.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.971748e+06</td>
      <td>50082.500000</td>
      <td>207161.000000</td>
      <td>1.990000e+05</td>
      <td>1.620000e+05</td>
      <td>0.000000e+00</td>
      <td>1.620000e+05</td>
      <td>0.000000e+00</td>
      <td>7.000000</td>
      <td>15.000000</td>
      <td>15.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.188854e+06</td>
      <td>75071.000000</td>
      <td>210796.000000</td>
      <td>2.500000e+05</td>
      <td>2.176000e+05</td>
      <td>4.660000e+04</td>
      <td>2.176000e+05</td>
      <td>0.000000e+00</td>
      <td>10.000000</td>
      <td>24.000000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.417084e+06</td>
      <td>99999.000000</td>
      <td>214834.000000</td>
      <td>2.000000e+06</td>
      <td>1.339200e+06</td>
      <td>2.000000e+06</td>
      <td>1.339200e+06</td>
      <td>0.000000e+00</td>
      <td>12.000000</td>
      <td>31.000000</td>
      <td>23.000000</td>
    </tr>
  </tbody>
</table>
</div>


\
**유일 값(unique)과 유일 값의 개수(nunique)를 출력해 봅니다**


```python
print("전체 데이터 수:", len(df[df['유형']==1]), "\n")
for i in df.columns:
    print(f'{i}:', "값:", df[i].unique(), "개수:",df[i].nunique(),"\n")
```

    전체 데이터 수: 0 
    
    거래id: 값: [2417084 2415408 2413897 ... 1551361 1545070 1486194] 개수: 159328 
    
    유형: 값: ['PAYMENT' 'REFUND'] 개수: 2 
    
    고객id: 값: [20053 58309 18075 ... 27994 78593 39658] 개수: 79615 
    
    코스ID: 값: [209016 210664 06067 ... 202072 203013 209132] 개수: 407 
    
    포맷: 값: ['올인원' 'RED'] 개수: 2 
    
    카테고리: 값: ['업무 생산성' '부동산/금융' '영상/3D' '프로그래밍' '마케팅' '일러스트' '디자인' '데이터 사이언스' '부업/창업'
     '투자/재테크' '크리에이티브'] 개수: 11 
    
    코스(상품) 이름: 값: ['올인원 패키지 : 김왼손의 파이썬 업무자동화 유치원' '초격차 패키지 : 한 번에 끝내는 부동산 금융(PF) 실무'
     '편집하는여자의 영상편집 마스터클래스 - 제 6강 다양한 효과를 응용하여 애프터이펙트 마스터' ... 
     '올인원 패키지 : 코딩 첫 걸음 프로젝트' '글로벌 엑스퍼트 : 바이오 데이터사이언스(Bioinformatics)'
     'AI/Data Science Conference : AI Explorer 22'] 개수: 407 
    
    쿠폰이름: 값: [0 '[WELCOME] 프로그래밍 3만원할인'
     '[20% 할인] 입문자를 위한 풀스택 웹 개발 Kit : 기획부터 프로젝트까지 기수강자 대상' ...
     '[단체구매: 엠티오메가] 초격차 패키지 : 한 번에 끝내는 딥러닝/인공지능 무료수강권'
     '[무료수강권] 강의 참고용 발행_지인할인쿠폰' '[WELCOME] 패캠은 처음이지? 3만원 할인쿠폰'] 개수: 1227 
    
    판매가격: 값: [ 189000  501000  549000 ... 2000000  245500  521000] 개수: 625 
    
    결제수단: 값: ['TRANSFER' 'CARD' 'POINT' 'PROMOTION' 'TRANS' 'VBANK'] 개수: 6 
    
    실거래금액: 값: [ 159000  501000  549000 ...    7850   30600 -345240] 개수: 3366 
    
    쿠폰할인액: 값: [     0  30000  39000 ... 200800  60800 129800] 개수: 1200 
    
    거래금액: 값: [159000 501000 549000 ...  39200   7850  30600] 개수: 1605 
    
    환불금액: 값: [      0 -158000  -65000 ... -430000 -709000 -345240] 개수: 1762 
    
    거래월: 값: [12 11 10  9  8  7  6  5  4  3  2  1] 개수: 12 
    
    거래일: 값: [31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10  9  8
      7  6  5  4  3  2  1] 개수: 31 
    
    거래시간: 값: [22 18 23 21 20 19 17 16 15 14 13  0 11 10  9  8  7  6  5  4  3  2  1 12] 개수: 24 
    
    
\
**금액에 관련된 컬럼들을 히스토그램으로 시각화해봅니다**

- **"실거래금액"**은 **"거래금액"**과 **"환불금액"**을 모두 나타낸다.
- 보통 **"판매금액"**보다 더 싼 (실거래)금액으로 거래를 하고 있는 것을 볼 수 있다. 쿠폰에 의한 할인이 원인인 것으로 보인다.
- **150,000 ~ 200,000**원대 강의가 가장 많이 팔린다.


```python
hist_df = df[['판매가격','실거래금액','거래금액','환불금액',"쿠폰할인액"]]
hist_df = hist_df.replace(0, np.nan) # 0값은 사실상 없는 값이니 null로 교체

plt.figure(figsize=(20,6))
sns.histplot(data = hist_df[(hist_df < 7*1e5) & (hist_df > -4*1e5)], # 보이지 않는 히스토그램의 범위는 제외
                  bins=60, 
                  kde=True, 
                  stat= 'percent', 
                  cumulative=False, 
                )

plt.xticks(np.arange(-4*1e5, 7*1e5, 0.5*1e5));
```


    
![png](/assets/images/EDA project-lecture sales images/output_36_0.png)
    

\
**카테고리를 히스토그램으로 시각화해봅니다**
- **"프로그래밍"** 카테고리가 가장 많이 판매되었고, **"데이터 사이언스"**가 두 번째로 많이 판매되었다. 이는 두가지 이유를 생각해 볼 수 있다.
    - 이유 1) **"프로그래밍"**과 **"데이터 사이언스"** 강의의 인기가 많다.
    - 이유 2) 단순히 **"프로그래밍"**과 **"데이터 사이언스"** 강의의 수가 많기 때문에 많이 팔리는 것이다. 


```python
plt.figure(figsize=(15,6))
plt.rcParams['font.family'] = 'gulim'
sns.histplot(data = df[df["유형"]=="PAYMENT"]["카테고리"], # 보이지 않는 히스토그램의 범위는 제외
                  bins=60, 
                  stat= 'percent', 
                  palette='cool'
                );
```

    C:\Users\a0107\AppData\Local\Temp\ipykernel_3612\2265951337.py:3: UserWarning: Ignoring `palette` because no `hue` variable has been assigned.
      sns.histplot(data = df[df["유형"]=="PAYMENT"]["카테고리"], # 보이지 않는 히스토그램의 범위는 제외
    


    
![png](/assets/images/EDA project-lecture sales images/output_38_1.png)
    

\
**카테고리를 원차트(Pie)로 시각화해봅니다**
- **"프로그래밍"**과 **"데이터 사이언스"**가 매출의 절반을 차지하는 것을 알 수 있다.


```python
category_sales = pd.DataFrame()
category_sales['sales'] = df.groupby(['카테고리'])['실거래금액'].sum() #매출
category_sales = category_sales.reset_index() 
category_sales = category_sales.drop(8) #크리에이티브 행 제거 (0이기 때문)
category_sales.sort_values(by=('sales'),ascending=False)

# 카테고리 별 매출을 차지하는 비율
values = category_sales['sales']
objects = category_sales['카테고리']
colors = sns.color_palette('pastel')[0:10]
plt.figure(figsize=(6,6))
plt.pie(values,
        labels=objects,
        autopct='%.1f',
        colors=colors
        ) 
plt.legend(loc=(1.2,0.3), title='카테고리별 거래금액')
plt.show()
```


    
![png](/assets/images/EDA project-lecture sales images/output_40_0.png)
    


***

## 2. 고객군 분석
\
**고객군을 차원 축소해서 살펴보고 클러스터링 합니다.**

**고객id**를 인덱스로 하는 DataFrame을 새로 정의합니다.

아래의 컬럼을 괄호 안에 있는 방식으로 변환해서 DataFrame에 추가합니다.
- **고객id, 유형('PAYMENT' 1, 'REFUND' 0으로 변환), 포맷('올인원' 1, 'RED' 0으로 변환), 카테고리(One-hot encoding), 쿠폰이름(쿠폰 사용 1, 미사용 0으로 변환), 판매가격(MinMax scaling), 결제수단(One-hot encoding), 실거래금액(MinMax scaling), 쿠폰할인액(MinMax scaling), 거래금액(MinMax scaling), 환불금액(MinMax scaling), 거래월(One-hot encoding), 거래일(One-hot encoding), 거래시간(One-hot encoding)**

**'코스(상품) 이름'**이 407개로 많지 않아 One-hot encoding하여 컬럼으로 만들까 생각하였지만, **차원의 저주**를 피하기 위해 **'코스(상품) 이름'**은 제외하기로 결정했습니다.\
**거래월, 거래일, 거래시간**을 One-hot encoding할 때에도 **차원의 저주** 문제가 발생할 수 있음으로 거래월을 **분기**로 변환하고(e.g.1월부터 3월을 0으로 변경), 거래일과 거래시간도 마찬가지로 **4등분**으로 변환했습니다.


```python
df_cluster = pd.DataFrame()
df_cluster[["고객id", "유형", "포맷", "카테고리", "쿠폰이름", "결제수단", "거래월", "거래일", "거래시간"]] = df[["고객id", "유형", "포맷", "카테고리", "쿠폰이름", "결제수단", "거래월", "거래일", "거래시간"]]
```


```python
# 거래월, 일, 시간 분기로 변경경
df_cluster.loc[df["거래월"].between(0, 3), "거래월"] = 0
df_cluster.loc[df["거래월"].between(4, 6), "거래월"] = 1
df_cluster.loc[df["거래월"].between(7, 9), "거래월"] = 2
df_cluster.loc[df["거래월"].between(10, 12), "거래월"] = 3

df_cluster.loc[df["거래일"].between(0, 7), "거래일"] = 0
df_cluster.loc[df["거래일"].between(8, 14), "거래일"] = 1
df_cluster.loc[df["거래일"].between(15, 21), "거래일"] = 2
df_cluster.loc[df["거래일"].between(22, 31), "거래일"] = 3

df_cluster.loc[df["거래시간"].between(0, 6), "거래시간"] = 0
df_cluster.loc[df["거래시간"].between(7, 12), "거래시간"] = 1
df_cluster.loc[df["거래시간"].between(13, 18), "거래시간"] = 2
df_cluster.loc[df["거래시간"].between(19, 24), "거래시간"] = 3
```


```python
from sklearn.preprocessing import MinMaxScaler

# 금액에 관련된 컬럼에 MinMaxScaling을 적용 -> 0~1 범위로 변경
scaler = MinMaxScaler()
df_cluster[['판매가격',"실거래금액","쿠폰할인액", "거래금액", "환불금액"]] = scaler.fit_transform(df[['판매가격',"실거래금액","쿠폰할인액", "거래금액", "환불금액"]])
```


```python
# 'PAYMENT', 'REFUND'을 1, 0으로 변경
_type = df['유형']=='PAYMENT'
df_cluster['유형'] = _type

# '올인원', 'RED'을 1, 0으로 변경
_formats = df['포맷']=='올인원'
df_cluster['포맷'] = _formats

# "카테고리" One-hot encoding
df_cluster = pd.get_dummies(df_cluster, columns=["카테고리"])

# 종류 상관없이 쿠폰을 사용했으면 1, 안했으면 0으로 변경
_coupon = df_cluster['쿠폰이름'] != 0
df_cluster['쿠폰이름'] = _coupon

# "결제수단" One-hot encoding
df_cluster = pd.get_dummies(df_cluster, columns=["결제수단"])

# 거래월, 일, 시간 One-hot encoding
df_cluster = pd.get_dummies(df_cluster, columns=["거래월"])
df_cluster = pd.get_dummies(df_cluster, columns=["거래일"])
df_cluster = pd.get_dummies(df_cluster, columns=["거래시간"])
```


```python
# boolean을 1과 0 값으로 변경
df_cluster = df_cluster.applymap(lambda x: int(x) if isinstance(x, bool) else x)
```

    C:\Users\a0107\AppData\Local\Temp\ipykernel_3612\3487119256.py:2: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
      df_cluster = df_cluster.applymap(lambda x: int(x) if isinstance(x, bool) else x)
    
\
**groupby**로 각 고객이 인덱스가 되도록 만듭니다. 한 고객으로부터 나온 데이터들은 그 값들을 모두 더하여 하나의 데이터로 만듭니다.\
평균이 아닌 합으로 처리하는 이유는, 고객이 같은 선택(같은 유형, 카테고리, 금액, 환불 등을 선택)을 했을 때 해당 방향으로 더 이동하도록 만들기 위함입니다. 즉, 같은 선택을 많이한 고객군이 공간상에서 비슷한 좌표에 위치하도록 합니다.\
이렇게 했을 때 전체 148010개의 데이터가 고객id 개수 79615만큼으로 줄어들게 됩니다. 


```python
df_cluster = df_cluster.groupby('고객id').sum()
```
\
**UMAP과 PCA를 활용해 데이터를 차원축소해서 시각화합니다**

UMAP으로 차원축소한 결과에서는 일부 클러스터가 보입니다. 그런데 형태가 마치 무작위하게 넓게 퍼져있는 데이터의 일부를 억지로 끌어다 모아놓은 것처럼 보입니다. UMAP이 비선형적인 모델이기에 생기는 문제인 것 같아 선형적으로 차원축소를 하는 PCA로도 시각화를 해봅니다. 


```python
import umap

# UMAP 적용 (2차원 축소)
umap_model = umap.UMAP(n_components=2, random_state=42)
embedding_umap = umap_model.fit_transform(df_cluster.values)  # DataFrame을 numpy 배열로 변환 후 적용

# 결과를 DataFrame으로 변환
df_embedded_umap = pd.DataFrame(embedding_umap, columns=["umap1", "umap2"])

# 시각화
plt.figure(figsize=(8, 8))
sns.scatterplot(x=df_embedded_umap["umap1"], y=df_embedded_umap["umap2"],color="g")
plt.xlabel("umap1")
plt.ylabel("umap2")
plt.title("UMAP Dimensionality Reduction")
plt.show()
```

    c:\Users\a0107\anaconda3\Lib\site-packages\umap\umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(
    


    
![png](/assets/images/EDA project-lecture sales images/output_51_1.png)
    


PCA로 차원축소한 데이터는 딱히 군집을 이루고 있지 않아 보입니다.


```python
from sklearn.decomposition import PCA

pca_model = PCA(n_components=2)  # 2개의 주성분만을 유지
embedding_pca = pca_model.fit_transform(df_cluster.values)

# 결과를 DataFrame으로 변환
df_embedded_pca = pd.DataFrame(embedding_pca, columns=["pca1", "pca2"])

# 시각화
plt.figure(figsize=(8, 8))
sns.scatterplot(x=df_embedded_pca["pca1"], y=df_embedded_pca["pca2"],color="g")
plt.xlabel("pca1")
plt.ylabel("pca2")
plt.title("PCA Dimensionality Reduction")
plt.show()
```


    
![png](/assets/images/EDA project-lecture sales images/output_53_0.png)
    

\
**데이터를 DBSCAN으로 클러스터링합니다**

이때, 클러스터링은 차원축소하지 않은 데이터로 합니다.

클러스터한 결과를 UMAP와 PCA로 시각화해보면 클러스터링이 의미있게 이뤄졌다고 보긴 힘듭니다.\
(클러스터별로 추가적인 분석을 해봤을 때 유의미한 차이를 찾지 못했습니다. 이에 대한 코드 및 설명은 생략합니다.)


```python
from sklearn.cluster import DBSCAN

# DBSCAN 모델 생성 및 적용
dbscan = DBSCAN(eps=0.1, min_samples=3)  
dbscan_cluster = dbscan.fit_predict(df_cluster.values)
df_embedded_umap["Cluster"] = dbscan_cluster
df_embedded_pca["Cluster"] = dbscan_cluster

#  시각화
plt.figure(figsize=(8, 8))
sns.scatterplot(data=df_embedded_umap, x="umap1", y="umap2", hue="Cluster", palette="deep", legend=False)
plt.figure(figsize=(8, 8))
sns.scatterplot(data=df_embedded_pca, x="pca1", y="pca2", hue="Cluster", palette="deep", legend=False)

plt.show()        
        
```


    
![png](/assets/images/EDA project-lecture sales images/output_55_0.png)
    



    
![png](/assets/images/EDA project-lecture sales images/output_55_1.png)
    



```python
print("전체 클러스터 개수: ", len(df_embedded_umap.value_counts("Cluster")), "개")
print("클러스터별 포함된 데이터 개수: \n", df_embedded_umap.value_counts("Cluster"))
```

    전체 클러스터 개수:  2580 개
    클러스터별 포함된 데이터 개수: 
     Cluster
    -1       52302
     9         124
     27        111
     196       110
     250       104
             ...  
     1868        3
     2164        3
     997         3
     2166        3
     2578        3
    Name: count, Length: 2580, dtype: int64
    

***

## 3. 가설에 의한 분석과 Insight
\
**가설을 세우고 가설이 맞는지 분석을 통해 알아봅니다. 가설은 아래와 같습니다.**

1. 구매를 많이 한 고객군과 금액을 많이 지불한 고객군이 같을 것이다
2. 잘 팔리는 강의의 제목에 특정 단어가 자주 등장할 것이다
3. 시간은 판매량에 영향을 끼칠 것이다
4. 쿠폰은 매출에 영향을 줄 것이다


### 가설1) 구매를 많이 한 고객군과 금액을 많이 지불한 고객군이 같을 것이다


```python
# 매출을 가장 많이 올린 고객
sum_vip_mask = df.groupby('고객id')['실거래금액'].sum().sort_values(ascending=False) > 0 # 실거래가 마이너스인 고객 제거
sum_vip = df.groupby('고객id')['실거래금액'].sum().sort_values(ascending=False)[sum_vip_mask]
top_rate = round(len(sum_vip) *(10/100)) # 상위 10% 고객 수
print("상위 10% 소비자 수 :", top_rate, "명")
sum_vip_idx = sum_vip.iloc[:top_rate].index # 가장 많은 금액을 지불한 고객id

# 고객별로 "PAYMENT" 횟수에서 "REFUND" 횟수를 제거
real_pay_count = df[(df["유형"]=="PAYMENT")].value_counts("고객id") - df[(df["유형"]=="REFUND")].value_counts("고객id")
real_pay_count.dropna(how='any', inplace=True) # "REFUND"하지 않은 고객은 NULL로 나타남. NULL 제거

# 가장 많은 횟수를 구매한 고객
num_vip = df[(df["유형"]=="PAYMENT")].value_counts("고객id") 
num_vip.update(real_pay_count) # "REFUND" 횟수를 제거한 구매 횟수로 업데이트
num_vip = num_vip.sort_values(ascending=False)
num_vip_idx = num_vip.iloc[:top_rate].index # 가장 많은 횟수를 구매한 고객id

# 두 고객군의 교집합
intersection = len(set(num_vip_idx) & set(sum_vip_idx)) 
print("구매를 많이 한 고객군과 금액을 많이 지불한 고객군의 교집합 수 :", intersection, "명")

print("구매를 많이 한 고객군과 금액을 많이 지불한 고객군이 일치하는 비율: ", round(intersection / top_rate * 100, 2), "%")
```

    상위 10% 소비자 수 : 7094 명
    구매를 많이 한 고객군과 금액을 많이 지불한 고객군의 교집합 수 : 4026 명
    구매를 많이 한 고객군과 금액을 많이 지불한 고객군이 일치하는 비율:  56.75 %
    
\
**lineplot으로 많은 금액을 지불한 고객이 얼마나 많은 제품을 구매했는지 시각화해봅니다**


```python
num_vip_perchase_count = num_vip.iloc[:top_rate]
num_vip_perchase_count = num_vip_perchase_count.values

sum_vip_perchase_count = df[df["고객id"].isin(sum_vip_idx)].value_counts("고객id")
sum_vip_perchase_count.update(real_pay_count) # 환불한 횟수 제거
sum_vip_perchase_count = sum_vip_perchase_count.sort_values(ascending=False)
sum_vip_perchase_count = sum_vip_perchase_count.values

plt.figure(figsize=(15, 6))
sns.lineplot(x=np.arange(0,top_rate), y=sum_vip_perchase_count, label='가장 많은 금액을 지불한 고객이 구매한 횟수', color="g")
sns.lineplot(x=np.arange(0,top_rate), y=num_vip_perchase_count, label='가장 많은 횟수를 구매한 고객이 구매한 횟수', color="r")
plt.xlabel('0에 가까울 수록 많은 횟수를 구매한 고객')
plt.ylabel('구매 횟수')
plt.legend()
plt.title("VIP perchase count")
plt.show()
```


    
![png](/assets/images/EDA project-lecture sales images/output_62_0.png)
    


#### 가설1) 결론(Insight)

- 실질적으로 금액을 지불한 전체 고객이 70944명일 때, 상위 10%는 7094명이 된다. 
- 금액을 가장 많이 지불한 7094명과 구매를 가장 많이 한 7094명이 서로 얼마나 일치하는지 확인해본 결과 4026명이 일치하는 것으로 나타났다.
- 7094명 중 4026명이 일치함으로 약 56.75%가 일치한다.

**결론** : 구매를 많이 한 고객군과 금액을 많이 지불한 고객군은 비슷한 경향성을 따르며, 과반수 이상 일치합니다. 그러나 약 43% 정도는 금액을 많이 지불한 고객이라고 해서 다회 구매한 고객은 아니라는 것을 알 수 있습니다. 이는 한 사람당 구매하는 횟수가 충분히 많지 않기 때문으로 해석할 수 있습니다. 한 사람당 충분히 많은 강의를 구매하면 많은 금액을 지불한 고객군과 그렇지 않은 고객군의 격차가 크게 벌어져 많은 금액을 지불한 고객군과 많은 강의 구매한 고객군이 같아질 것입니다. 그러나 강의 판매 사업의 특성상 한 사람이 많은 강의를 구매하는 것에는 한계가 있습니다. 

이에 사업적 전략으로 두가지를 생각해 볼 수 있습니다. 
1. 첫 번째, 같은 사람이 많은 강의를 구매하는 것을 유도하 되, 어차피 1인당 구매에 한계가 있다면 최대한 많은 사람에게 판매하는 것이 매출에 유리할 수 있습니다.
2. 두 번째, 낮은 퀄리티로 싼 강의를 많이 판매하기 보단 높은 퀄리티의 비싼 강의를 판매하는 것이 전략상 좋을 수 있습니다.


### 가설2) 잘 팔리는 강의의 제목에 특정 단어가 자주 등장할 것이다
\
**강의 제목을 시각화할 수 있도록 처리합니다**


```python
#tokenizing 함수 선언

def text_cleaning(doc):
    #특수문자를 제거
    doc = re.sub("[\{\}\[\]\/?.,;:|\)*~`!^\-_+<>@\#$%&\\\=\(\'\"]", " ", doc)
    return doc

def define_stopwords(path):
    
    SW = set()
    SW.add("있다")
    SW.add("있어요")
    SW.add("대한")
    SW.add("합니다")
    SW.add("하는")
    
    with open(path, encoding="utf-8") as f:
        for word in f:
            SW.add(word.strip())
            
    return SW

SW = define_stopwords("data/stopwords-ko.txt")

def text_tokenizing(doc):

    return [word for word in doc.split() if word not in SW ]
```


```python
# 가장 많이 팔린 강의 top 100과 팔린 횟수 
print(df['코스(상품) 이름'].value_counts().head(100)) 

print("\n", "전체 강의 수: ", len( df['코스(상품) 이름'].value_counts()), "개")
```

    코스(상품) 이름
    초격차 패키지 : 일잘러 필수 스킬 모음.zip                                            2890
    초격차 패키지 : 10개 프로젝트로 완성하는 백엔드 웹개발(Java/Spring)                         2788
    초격차 패키지 : 한 번에 끝내는 프론트엔드 개발                                           2679
    초격차 패키지 : 한 번에 끝내는 Java/Spring 웹 개발 마스터                               2330
    올인원 패키지 : 세계 3등에게 배우는 실무 밀착 데이터 시각화                                   1941
                                                                          ... 
    모두를 위한 앱 개발 : Android 앱 개발의 정석 with Kotlin                             539
    초격차 패키지 : 확실하게 끝내는 포토샵&일러스트레이터                                         533
    이필성의 포토클래스 시선을 사로잡는 특별한 사진을 만드는 법 Online.                              531
    올인원 패키지 : 김민태의 프론트엔드 아카데미 : 제 1강 JavaScript & TypeScript Essential     525
    올인원 패키지 : 실시간 빅데이터 처리를 위한 Spark & Flink                                519
    Name: count, Length: 100, dtype: int64
    
     전체 강의 수:  407 개
    


```python
#가장 잘 팔린, 가장 덜 팔린 100개 강의 제목 
top_product_name = pd.DataFrame()
top_product_name['name'] = df['코스(상품) 이름'].value_counts().head(100).index #top 100(제목100개)
bottom_product_name = pd.DataFrame()
bottom_product_name['name'] = df['코스(상품) 이름'].value_counts(ascending=True).head(100).index #dottom 100

#tokenizing
tokenized_word_t = top_product_name['name'].apply(text_cleaning).apply(text_tokenizing)
tokenized_word_b = bottom_product_name['name'].apply(text_cleaning).apply(text_tokenizing)

#모든 토큰 리스트로 저장
total_tokens_t = [token for doc in tokenized_word_t for token in doc] 
total_tokens_b = [token for doc in tokenized_word_b for token in doc]
text_t = nltk.Text(total_tokens_t) #텍스트 나열
text_b = nltk.Text(total_tokens_b)

#단어 갯수 ##가장 많이 언급된 단어 100개
top100 = text_t.vocab().most_common(100) #가장 잘 팔린 강의의 가장 많이 언급된 단어
bottom100 = text_b.vocab().most_common(100) #가장 안 팔린 강의의 가장 많이 언급된 단어

#top100 딕셔너리에 저장
x_t = []
y_t = []
cloud_t = dict()

for word, count in top100:
    x_t.append(word)
    y_t.append(count)
    cloud_t[word] = count

name_count_data_t = {"words":x_t, "count":y_t}

#bottom100 딕셔너리에 저장
x_b = []
y_b = []
cloud_b = dict()

for word, count in bottom100:
    x_b.append(word)
    y_b.append(count)
    cloud_b[word] = count

name_count_data_b = {"words":x_b, "count":y_b}

```
\
**barplot으로 상위 100개와 하위 100개의 단어 출연 빈도를 시각화해봅니다**
- 상위 데이터들과 달리 하위 데이터들의 제목에는 다양한 단어가 들어가 barplot의 분포가 더 완만하다. 
- 많이 팔리는 강의에 자주 등장하는 단어가 어느정도 정해져 있다.


```python
plt.figure(figsize=(16, 24))
plt.title("상위 100 강의에 등장하는 단어 빈도" , fontsize=18)
plt.xlabel("Samples")
plt.ylabel("Counts")
sns.barplot(x="count", y="words", data=name_count_data_t, palette="rainbow")
plt.show()
```

    C:\Users\a0107\AppData\Local\Temp\ipykernel_3612\2306427469.py:5: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(x="count", y="words", data=name_count_data_t, palette="rainbow")
    


    
![png](/assets/images/EDA project-lecture sales images/output_70_1.png)
    



```python
plt.figure(figsize=(16, 24))
plt.title("하위 100 강의 등장하는 단어", fontsize=18)
plt.xlabel("Samples")
plt.ylabel("Counts")
sns.barplot(x="count", y="words", data=name_count_data_b, palette="rainbow")
plt.show()
```

    C:\Users\a0107\AppData\Local\Temp\ipykernel_3612\4279318511.py:5: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(x="count", y="words", data=name_count_data_b, palette="rainbow")
    


    
![png](/assets/images/EDA project-lecture sales images/output_71_1.png)
    

\
**WordCloud로 상위 100개와 하위 100개의 단어 출연 빈도를 시각화해봅니다**
- 상위 제목들에는 한글이 대부분인데 비해, 하위 제목들에는 영어가 더 많다. 


```python
# top100제목 분석
wc = WordCloud(font_path = "gulim", width=1500, height=800, background_color="white", random_state=0)
plt.imshow(wc.generate_from_frequencies(cloud_t))
plt.axis("off")
plt.show()
```


    
![png](/assets/images/EDA project-lecture sales images/output_73_0.png)
    



```python
# bottom 100 제목 분석
wc = WordCloud(font_path = "gulim", width=1500, height=800, background_color="white", random_state=0)
plt.imshow(wc.generate_from_frequencies(cloud_b))
plt.axis("off")
plt.show()
```


    
![png](/assets/images/EDA project-lecture sales images/output_74_0.png)
    


#### 가설2) 결론(Insight)

**결론** : 잘 팔리는 강의는 특정 단어가 많이 등장한다고 볼 수 있습니다. ['패키지', '초격차', '끝내는', '한번에', '위한', '개발', '실무']와 같은 단어가 포함된 강의가 잘 팔립니다. 그리고 강의의 판매율을 높이려면 제목을 영어보단 한글로 짓는 것이 더 유리할 수 있습니다.


### 가설3) 시간은 판매량에 영향을 끼칠 것이다
\
**lineplot으로 월별 판매 횟수와 판매 금액을 시각화해봅니다**
- 연초에 판매량이 가장 높고 5월에 가장 낮다. 7~9월에 판매량이 증가하다가 10~11월에 하락하고 12월에 다시 증가한다.


```python
monthly_deal_sum = pd.DataFrame()
monthly_deal_sum['판매 횟수'] = df[df['유형']=='PAYMENT']['거래월'].value_counts()
monthly_deal_sum = monthly_deal_sum.sort_index()
monthly_deal_sum['실거래금액 합계'] = df.groupby('거래월')['실거래금액'].sum() / 150000 # 판매 횟수와 금액의 그래프가 비슷한 위치에 있도록 정규화


plt.figure(figsize=(10,6))
sns.set_theme(style='whitegrid')
plt.rcParams['font.family'] = 'gulim'
plt.title("월별 판매 횟수, 금액" , fontsize=18)
plt.xticks(np.arange(0,12,1))
sns.lineplot(data=monthly_deal_sum, x='거래월', y='판매 횟수',color="C3", label='판매 횟수')
sns.lineplot(data=monthly_deal_sum, x='거래월', y='실거래금액 합계',color="C2", label='실거래금액 합계/15*1e4')
```




    <Axes: title={'center': '월별 판매 횟수, 금액'}, xlabel='거래월', ylabel='판매 횟수'>




    
![png](/assets/images/EDA project-lecture sales images/output_78_1.png)
    

\
**lineplot으로 일별 판매 횟수와 판매 금액을 시각화해봅니다**
- 월초에 판매량이 가장 낮다. 3~14일에 판매량이 증가하다가 15~26일에 하락하고 월말에 증가한다.


```python
daily_deal_sum = pd.DataFrame()
daily_deal_sum['판매 횟수'] = df[df['유형']=='PAYMENT']['거래일'].value_counts()
daily_deal_sum = daily_deal_sum.sort_index()
daily_deal_sum['실거래금액 합계'] = df.groupby('거래일')['실거래금액'].sum() / 150000 # 판매 횟수와 금액의 그래프가 비슷한 위치에 있도록 정규화

plt.figure(figsize=(10,6))
sns.set_theme(style='whitegrid')
plt.rcParams['font.family'] = 'gulim'
plt.title("일별 판매 횟수, 금액" , fontsize=18)
plt.xticks(np.arange(0,32,1))
sns.lineplot(data=daily_deal_sum, x='거래일', y='판매 횟수',color="C3", label='판매 횟수')
sns.lineplot(data=daily_deal_sum, x='거래일', y='실거래금액 합계',color="C2", label='실거래금액 합계/15*1e4')
```




    <Axes: title={'center': '일별 판매 횟수, 금액'}, xlabel='거래일', ylabel='판매 횟수'>




    
![png](/assets/images/EDA project-lecture sales images/output_80_1.png)
    

\
**lineplot으로 시간별 판매 횟수와 판매 금액을 시각화해봅니다**
- 새벽 시간대에는 판매량이 낮고, 아침부터 저녁까지 판매량이 증가한다.
- 12시 점심 시간에 판매량이 줄어드는 것도 보인다.


```python
hourly_deal_sum = pd.DataFrame()
hourly_deal_sum['판매 횟수'] = df[df['유형']=='PAYMENT']['거래시간'].value_counts()
hourly_deal_sum = hourly_deal_sum.sort_index()
hourly_deal_sum['실거래금액 합계'] = df.groupby('거래시간')['실거래금액'].sum() / 150000 # 판매 횟수와 금액의 그래프가 비슷한 위치에 있도록 정규화

plt.figure(figsize=(10,6))
sns.set_theme(style='whitegrid')
plt.rcParams['font.family'] = 'gulim'
plt.title("일별 판매 횟수, 금액" , fontsize=18)
plt.xticks(np.arange(0,32,1))
sns.lineplot(data=hourly_deal_sum, x='거래시간', y='판매 횟수',color="C3", label='판매 횟수')
sns.lineplot(data=hourly_deal_sum, x='거래시간', y='실거래금액 합계',color="C2", label='실거래금액 합계/15*1e4')
```




    <Axes: title={'center': '일별 판매 횟수, 금액'}, xlabel='거래시간', ylabel='판매 횟수'>




    
![png](/assets/images/EDA project-lecture sales images/output_82_1.png)
    


#### 가설3) 결론(Insight)

**결론** : 연초에 새해 다짐을 하듯이 월별, 일별, 시간별로 판매량에 "시작"과 "끝"에 의한 심리적 영향이 있는 것으로 보입니다. 월, 일, 시간 모든 기준에서 "끝"에 다가갈수록 판매량이 증가하는 경향을 보입니다. 연도가 끝나갈 때, 달이 끝나갈 때, 하루가 끝나갈 때 사람들은 자기계발에 대한 조바심을 보입니다. 이로 미루어 보아 마케팅을 할 때는 연말, 월말, 일말에 하는 것이 좋을 것입니다.


### 가설4) 쿠폰은 매출에 영향을 줄 것이다
- 쿠폰이 판매가격을 낮추는 데에도 불구하고 정말 매출상으로 도움이 되는지에 대한 의문을 해결한다.
- 쿠폰 시스템이 존재하지 않았을 때의 데이터가 없기 때문에 쿠폰 시스템이 존재하는 데이터와의 AB 테스트가 불가능하다. 
- AB 테스트 대신 쿠폰에 의한 구매 촉진율을 간접적으로 계산하여 매출에 대한 영향을 확인한다.
\
**아래 과정을 거쳐서 쿠폰에 의한 매출의 변화를 간접적으로 확인합니다**
1. 쿠폰에 의한 구매 촉진율을 구한다.
    - 촉진율 = 쿠폰할인액 / 판매가격 
2. 쿠폰이 없을 때의 고객별 매출을 구한다.
    - 쿠폰이 없을 때 고객별 매출 = 고격별 구매한 강의의 판매가격 합 * (1-촉진율)
        - 쿠폰이 없을 때 구매를 촉진하는 정도 = (1-촉진율)
        - 해석: 고격별 구매한 강의의 판매가격 합에서 구매를 촉진한 영향을 제거
3. "쿠폰이 없을 때 매출 총합"과 "쿠폰이 있을 때 매출 총합"을 비교하여 쿠폰이 매출을 높이는데 도움이 되는지 확인한다.


```python
df["촉진율"] = df["쿠폰할인액"] / df["판매가격"]
promotion_rate = df[df["유형"]=="PAYMENT"].groupby("고객id")["촉진율"].mean() # 고객별 촉진율 평균
by_customer_sell_sum = df[df["유형"]=="PAYMENT"].groupby("고객id")["판매가격"].sum() # 고객별 구매한 강으의 판매가격 합
coupon_unaffected_sell= by_customer_sell_sum * (1-promotion_rate) # 쿠폰이 없을 때 고객별 매출
coupon_unaffected_sell_sum = coupon_unaffected_sell[coupon_unaffected_sell > 0].sum() # 쿠폰이 없을 때 매출 총합 # 쿠폰할인액이 판매가격보다 큰 경우 0보다 작다
coupon_affected_sell_sum = df[df["유형"]=="PAYMENT"]["거래금액"].sum() # 쿠폰이 있을 때 매출 총합
```


```python
values = [coupon_affected_sell_sum, coupon_unaffected_sell_sum]
objects = ["쿠폰 사용시 매출","쿠폰 미사용시 매출"]
colors = sns.color_palette('pastel')[2:4]
plt.figure(figsize=(5,5))
plt.pie(values, labels = objects, colors = colors, autopct='%.2f%%', startangle=90)
plt.legend(loc=(1,0.8))
plt.show()
```


    
![png](/assets/images/EDA project-lecture sales images/output_87_0.png)
    


#### 가설4) 결론(Insight)

**결론** : 미묘한 차이이긴 하지만 쿠폰이 없을 때보다 쿠폰이 있을 때 총 매출이 상승하는 것을 볼 수 있습니다. 결과적으로 쿠폰은 매출에 도움이 됩니다.


