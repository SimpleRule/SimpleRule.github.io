# Project Explanation

## About Project
이어드림스쿨에서 진행한 중소기업 협업 팀 프로젝트입니다.\
이어드림스쿨은 중소기업진흥공단에서 기획한 데이터 사이언티스트를 양성하는 부트캠프입니다.\
중소기업이 실제 겪고 있는 문제를 팀으로써 함께 해결하였습니다. 합께 현업한 기업은 **(주)예스스탁**으로, 자동매매 시스템 제공, 투자 솔루션 등 주식 혹은 암호화폐 관련 서비스를 제공하는 기업입니다.

**(주)예스스탁**에서 파견된 멘토는 기업이 겪고 있는 문제와 데이터에 대해서 설명과 조언을 해주었고, 부트캠프 팀 인원은 AI를 통해 문제를 해결하였습니다.\
부트캠프의 팀원은 총 6명이었으며, 3명의 Data scientist, 3명의 Data engineer로 구성되었습니다. 저는 Data scientist이자 팀장 역할을 하였습니다.\
프로젝트는 약 2달간 진행되었습니다.

## About YesStock
![image](/assets/images/1D CNN stock forcasting images/yesstock-01.png){: width="80%" height="80%"}
![image](/assets/images/1D CNN stock forcasting images/yesstock-02.png){: width="80%" height="80%"}
![image](/assets/images/1D CNN stock forcasting images/yesstock-03.png){: width="80%" height="80%"}

## 문제 도출
![image](/assets/images/1D CNN stock forcasting images/apple_samsung.png){: width="60%" height="60%"}

국내 주식 시장에서 외국인 투자자의 매매 현황은 주가에 중요한 영향을 미칩니다. 대한민국의 대표 대기업인 삼성의 시가총액은 약 5백조, 외국의 대기업인 애플의 시가총액은 약 4천조로 규모에서 큰 차이가 납니다. 수급에 의해 결정되는 주가는 내국인보다 큰 규모를 가진 외국인 투자자에 의해 좌우됩니다. 그렇기에 외국인 투자자의 통향을 파악하면 주가의 흐름을 예측하는데 도움이 됩니다.

![image](/assets/images/1D CNN stock forcasting images/foreigner_result.png){: width="60%" height="60%"}

그러나 외국인의 수급은 장이 종료(18:00)된 후에만 확인이 가능합니다. 그럼으로 KOSPI 증권 시장의 (시계열) 데이터를 기반으로 장이 종료되기 전에 외국인의 수급을 추정합니다. 그리고 이를 통해 보다 효율적인 투자 전략을 투자자에게 제공하는 것이 목표입니다.

# Data Explanation

## Type of Data 
**데이터의 종류는 크게 4가지가 주어집니다.**
- 데이터 종류 : **"실시간 채결", "프로그램 매매 실시간 채결", "상위거래원", "장마감 데이터"**
- 각 종류의 데이터는 증권 프로그램에서 하나의 인터페이스를 이루는 데이터입니다. 예로, "실시간 체결" 데이터는 하나의 인터페이스를 구성합니다.
- 각 종류의 데이터는 모두 주식 채결에 대한 시계열 데이터인데, 각자 다른 형태로 구성됩니다.

## Data engineering
![image](/assets/images/1D CNN stock forcasting images/Tick.png){: width="80%" height="80%"}
- 데이터들은 2023년 2월 ~ 10월까지의 KOSPI 종목의 한국거래소 RawData입니다.
- Data scientist들이 필요한 데이터(종목, 컬럼 등)를 Data engineer들에게 요구하면 Data engineer들이 RawData를 정제하여 Data scientist들에게 전달해주었습니다.
- 유저가 활용하는 UI에서는 tick 단위의 RawData를 압축해서 보여줍니다. 현 프로젝트에서 활용하는 RawData는 모든 tick 데이터를 보여줍니다. 압축되지 않은 데이터이기에 시계열로써 더 활용도가 높습니다.

## Data 1) 실시간 채결
![image](/assets/images/1D CNN stock forcasting images/real_time.png){: width="60%" height="60%"}
- 모든 자잘한 채결(tick) 정보를 나타냅니다. 
- 개인, 기관(증권사, 법인), 외국인의 채결 정보가 담겨있습니다 (개인과 기관은 국내).
- 한번에 많은 거래량이 "실시간 채결"에서 나타나는데 "프로그램 매매"에서 나타나지 않으면 해당 채결 주체는 기관이라고 가늠할 수 있습니다. 기관은 프로그램 매매를 잘 사용하지 않기 때문입니다. 그러나 어떤 기준으로 많은 거래량이라고 판가름할지는 명확하지 않습니다.
- 기관계는 똑같은 물량을 비슷한 텀으로 꾸준히 매매하는 "단주매매"를 많이 합니다. 외국인은 많이 하지 않는 방법입니다. 

## Data 2) 프로그램 매매 실시간 채결
![image](/assets/images/1D CNN stock forcasting images/program.png){: width="60%" height="60%"}
- 기관과 외국인이 사용하는데 국내 증권사보다 외국 증권사가 더 많이 사용합니다.
- "실시간 채결"에서 나타나는 거래량 중 프로그램으로 매매된 거래량이 얼마지 보여줍니다.
    - 1000개의 거래량이 있을 때, "실시간 채결"에서는 이를 [300, 250, 200, 250]과 같이 분할해서 1초에서 3초 사이에 빠르게 올라오는 시계열로 나타냅니다. 반면, "프로그램 매매"에서는 거의 같은 시점에 1000개가 그대로 나타납니다.

## Data 3) 상위거래원
![image](/assets/images/1D CNN stock forcasting images/top5.png){: width="60%" height="60%"}
- 가장 높은 거래량을 가진 top5 증권사를 표시합니다. 외국계 증권사가 곧 외국인 수급을 나타낸다고 할 수 있습니다.
- 새로 들어오고 이탈한 증권사의 기록을 모두 확인할 수 있으며, 외국계의 매도와 매수의 차액을 구해 결과적으로 팔고 있는지 사고있는지 확인할 수 있습니다.
- 상위거래원 이하에서 이뤄지는 증권사들의 자잘한 채결은 확인할 수 없습니다.
- "상위 거래원"에 올라오는 정보는 "프로그램 매매", "실시간 채결"에도 나타납니다.
    - "상위 거래원"의 데이터는 "실시간 채결"에 채결 데이터가 올라오고 약간의 딜레이(약 30초~1분) 후 올라옵니다.
- 거래량을 종목별로 볼 수 있습니다.

## Data 4) 장마감 데이터
![image](/assets/images/1D CNN stock forcasting images/gt.png){: width="60%" height="60%"}
- 장마감(18:00) 후 나오는 데이터로, "금융투자", "보험", "투신", "사모", "은행", "개인", "외국인" 등 투자자 종류별로 총합 채결 수량(혹은 금액)을 나타냅니다. 
- 장마감 후 데이터는 세 차례에 걸쳐 올라오는데 가장 마지막에 올라오는 데이터가 확실한 데이터임으로 가장 마지막에 나오는 데이터만을 활용합니다.

# Workflow

## 머신러닝 문제 정의
![image](/assets/images/1D CNN stock forcasting images/Regression_define.png){: width="60%" height="60%"}

DS팀은 현재 프로젝트의 목표를 **외국인 수급을 예측하는 Regression** 문제로 정의하였습니다.\
정확한 외국인 수급은 장이 마감되고 하루에 한 번씩만 나옴으로 장이 열려있는 하루치 데이터 전부를 하나의 시계열 데이터로 취급합니다. 그리고 **장이 마감되고 나오는 외국인 수급을 정답 데이터(Ground Truth)로 사용하여 학습**합니다.

실시간으로 올라오는 채결(Tick)이 외국인인지 내국인인지 실시간으로 분류(classification)하는 방식도 고려해보았지만 오랜 시간동안 손수 라벨링해야 한다는 한계로 인해 장마감 데이터로 Regression하는 방식을 채택했습니다.

## 모델 선정
![image](/assets/images/1D CNN stock forcasting images/model_define.png){: width="80%" height="80%"}

외국인 수급을 예측하기 위해 저희가 선정한 모델은 **1D CNN**입니다. 1D CNN은 기존에도 심장박동처럼 단기적 속성이 강한 시계열 데이터를 효과적으로 학습한다고 알려져 있었습니다. 거기다 채널을 여러개 넣을 수 있어 여러 시계열 데이터끼리의 상호작용을 잘 학습할 수 있을 것이라 기대하였습니다.\
그리고 1D CNN과 성능을 비교하기 위한 베이스라인으로 시계열 데이터에 널리 사용되는 **LSTM**을 선정했습니다.

**input은 여러 종류의 하루치 시계열 데이터**이고, **output(Ground Truth)은 장마감 후 나오는 외국인 수급 스칼라**입니다.\
여러 종류의 하루치 시계열을 각 채널로 만들어 input 데이터를 구성합니다. 정답 데이터(Ground Truth)가 하루치 시계열 데이터마다 하나씩 존재함으로, input을 더 작은 widow size로 자르지 않고 하루치 데이터를 그대로 input으로 사용합니다.\
학습이 완료되고 Inference할 때는 장 마감이 되지 않은 시계열 데이터로 외국인 수급을 예측하게 됩니다. 

**Transformer**나 **MLP**처럼 더 무거운 모델을 사용하지 않는 이유는 **컴퓨팅 리소스의 한계** 때문입니다. 하루치 시계열 데이터의 길이가 매우 길어서 $O(l^2)$의 연산을 하는 Transformer나, 그 이상의 연산을 하는 MLP를 사용하기에는 무리가 있었습니다. 또한, Inductive Bias가 없는 Transformer나 MLP 대신 **Inductive Bias가 있는 CNN을 사용하는 것이 적은 데이터로 더 빠르게 학습할 수 있는 방법이라 판단**하였습니다.

## 전처리 
- **"실시간 채결", "프로그램 매매 실시간 채결", "상위거래원", "장마감 데이터"**에서 필요한 컬럼들만 각각 뽑아서 하나의 시계열 데이터로 만들어줍니다.

### "실시간 채결" 전처리
- "실시간 채결"의 컬럼들은 다음과 같습니다.
    - **['시간', '보드ID', '세션ID', '종목코드', '매매처리시각', '전일대비구분코드', '전일대비가격', '체결가격', '거래량', '시가', '고가', '저가', '누적거래량', '누적거래대금', '최종매도매수구분코드', '매도최우선호가가격', '매수최우선호가가격']**
- 실시간 체결 데이터에는 여러 컬럼이 존재하지만 외국인 수급을 예측하는데에 가장 큰 정보가 될 수 있다고 생각되는 세 가지 컬럼 **"시간", "거래량", "체결가격"**만 사용하였습니다. 
    - 보드ID, 세션 ID : 장개시전, 정규장 등을 의미합니다. 외국인은 정규장 여부와 상관없이 매매함으로 외국인 수급 예측과 무관하다고 판단하여 제거했습니다.
    - 종목코드 : 채결(tick) 데이터의 종목(e.g.삼성전자, LG)을 의미합니다. Data engineer에게 종목별로 데이터를 요구했기 때문에 하루치 데이터 안에서 모두 같은 값을 가짐으로 제거했습니다.
    - 매매처리시각 : 유저의 UI에서 나타나는 시간입니다. "시간"컬럼이 더 정확하기 때문에 "매매처리시각"는 제거했습니다.
    - 전일대비가격 : 전일 장마감 가격 대비 현재 가격이 얼마나 변동했는지 보여줍니다. 사실상 "체결가격"에 크게 의존(dependency)하는 값임으로 불필요합니다.
    - 시가 : "체결가격"과 거의 동일한 값임으로 불필요합니다.
    - 고가, 저가 : 현재까지 가장 높았던 "시가"와 가장 낮았던 "시가"를 보여줍니다. 역시 "체결가격"에 크게 의존(dependency)하는 값임으로 불필요합니다.
    - 누적거래량 : 지금까지의 "거래량"을 모두 누적해서 더한 값입니다. "거래량"에 크게 의존(dependency)하는 값임으로 불필요합니다.
    - 누적거래대금 : 현재까지 거래된 거래대금을 모두 누적해서 더한 것입니다 (거래대금 = "거래량"$\times$"체결가격"). "거래량"과 "체결가격"에 크게 의존(dependency)하는 값임으로 불필요합니다.
    - 최종매도매수구분코드 : 체결가격이 오르면 2로 유지되고 하락하면 1로 유지됩니다. 역시 외국인 수급 예측과 무관하다고 판단하여 제거했습니다.
    - 매도(매수)최우선호가가격 : 현재 가장 낮은 호가(혹은 높은 호가)를 나타냄으로 외국인 수급 예측과 무관하다고 판단하여 제거했습니다.
    - 결론적으로, 시계열 데이터에서 중요한 "시간" 정보와 대부분의 컬럼과 의존성(dependency)을 지닌 "거래량", "체결가격"만 사용하기로 결정했습니다.

![image](/assets/images/1D CNN stock forcasting images/relation.png){: width="60%" height="60%"}

체결가격을 뽑은 이유는 그래프에서도 보실 수 있 듯이 **체결가격에 변동이 생기면 거래량에도 높은 상관성으로 변화**가 생기는 것을 알 수 있습니다. 이에 대한 가설은 두 가지를 세워 볼 수 있습니다.
- 첫번째, 프로그램 매매가 특정 가격 범위에 들어오면 자동으로 체결 하는 방식을 사용함으로 주가 변동에 반응한 **프로그램 매매의 거래일 가능성**이 있습니다. 
- 두번째, 단순히 거래량이 많기 때문에 **수요와 공급의 원리에 따라 주가가 변화**하는 것일 수 있습니다.
첫 번째 가설이 성립한다면 프로그램 매매를 많이 사용하는 외국인의 수급을 예측하는 데에 도움이 될 수 있는 정보이기 때문에 체결가격 정보를 추가로 사용하였습니다.

### "프로그램 매매 실시간 채결" 전처리
- "프로그램 매매 실시간 채결"의 컬럼들은 다음과 같습니다.
    - **['시간', '매도차익거래잔량', '매수차익거래잔량', '매도비차익잔량', '매수비차익잔량', '매도차익수량', '매수차익수량', '매도비차익수량', '매수비차익수량', '위탁매도차익체결수량', '자기매도차익체결수량', '위탁매수차익체결수량', '자기매수차익체결수량', '위탁매도비차익체결수량', '자기매도비차익체결수량', '위탁매수비차익체결수량', '자기매수비차익체결수량', '위탁매도차익거래대금', '자기매도차익거래대금', '위탁매수차익거래대금', '자기매수차익거래대금', '위탁매도비차익거래대금', '자기매도비차익거래대금', '위탁매수비차익거래대금', '자기매수비차익거래대금']**
- "프로그램 매매 실시간 채결"에도 여러가지 컬럼이 존재하지만 일반 서비스 유저가 보는 프로그램 매매 값은 **"순매수 수량"**이며, 이는 아래의 수식을 통해서 산출됩니다. **"프로그램 매매 실시간 채결"**에서는 **"시간"**과 **순매수 수량**을 산출해 사용했습니다.
    - **순매수 수량** = **매수 수량** - **매도 수량**
        - **매수수량** = 위탁매수차익체결수량 + 자기매수차익체결수량 + 위탁매수비차익체결수량 + 자기매수비차익체결수량
        - **매도수량** = 위탁매도차익체결수량 + 자기매도차익체결수량 + 위탁매도비차익체결수량 + 자기매도비차익체결수량
    - 차익거래 : 현물과 선물의 시세 차이를 이용해서 수익을 내는 방식입니다.
    - 비차익거래 : 15개 이상의 주식 종목을 하나의 묶음으로 정하고 특정 조건이 성립하면 동시에 매매하는 방식을 의미합니다. 원칙적으론 현물과 선물을 동시에 다루는 차익거래와는 달리 현물만을 취급합니다.
    - 위탁매매 : 중개상인 또는 증권업자가 고객의 의뢰를 받고 상품 또는 증권을 매매하는 것입니다.
    - 자기매매 : 증권사가 보유한 고유자금으로 유가증권을 매매하는 것을 말하며 시장의 투자동향 지표중 하나입니다. 증권사가 시장에서 투자자 역할을 하는 활동으로, 증권사들의 매매는 증시 수급에 큰 영향을 미치는 단기 기대치의 변화를 반영하기 때문에 시장에서 주목합니다.
    - 매수 잔량, 매도 잔량 : 호가창에서 매수와 매도를 위해 신청한 종목의 개수로 아직 거래가 성립되지 않은 것들을 말합니다. 매수 잔량은 현재 호가보다 낮은 가격에 걸어둔 매수 주문의 총개수입니다. 반대로 매도 잔량은 현재 호가보다 높은 가격에 걸어둔 매도 주문의 총개수를 말합니다.

### "상위거래원" 전처리
- **"상위거래원"** 데이터는 장이 열려있는 동안 유일하게 확신할 수 있는 외국계 거래이기 때문에 중요하게 생각하고 전처리를 하였습니다.
- 가장 많은 거래량을 보인 증권사를 1위부터 5위까지 보여줍니다.
-  "상위거래원"의 컬럼들은 다음과 같습니다.
    - **['시간', '1단계매도회원번호', '1단계매도체결수량', '1단계매도거래대금', '1단계매수회원번호', '1단계매수체결수량', '1단계매수거래대금', '2단계매도회원번호', '2단계매도체결수량', '2단계매도거래대금', '2단계매수회원번호', '2단계매수체결수량', '2단계매수거래대금', '3단계매도회원번호', '3단계매도체결수량', '3단계매도거래대금', '3단계매수회원번호', '3단계매수체결수량', '3단계매수거래대금', '4단계매도회원번호', '4단계매도체결수량', '4단계매도거래대금', '4단계매수회원번호', '4단계매수체결수량', '4단계매수거래대금', '5단계매도회원번호', '5단계매도체결수량', '5단계매도거래대금', '5단계매수회원번호', '5단계매수체결수량', '5단계매수거래대금']**
- **"상위거래원"** 데이터에서는 1부터 5단계까지의 매도, 매수수량과 회원번호를 데이터로 사용하였습니다. 
    - 상위 거래원의 거래량과 함께 거래원의 회원번호가 들어오는데 이 회원번호를 확인하여 외국계와 국내(기관)를 구분할 수 있습니다. 저희는 이 번호를 외국계와 국내로만 구분하기 위해 **외국계를 1**로, **국내(기관)를 0**으로 바꾸어 바이너리로 만들었습니다.

### 데이터 통합
![image](/assets/images/1D CNN stock forcasting images/combine.png){: width="80%" height="80%"}
- **"실시간 채결", "프로그램 매매 실시간 채결", "상위거래원"**에서 추출한 컬럼들을 시간을 기준으로 합쳐서(Concat) **7개의 컬럼을 가진 하나의 데이터 프레임**으로 만듭니다.
- 각 컬럼이 1D CNN의 input으로 들어갈 채널이 됩니다.

### 시계열 길이 고정
- 하루치 시계열의 길이를 **100,000**으로 고정합니다.
- 원본 시계열은 (행 기준) 최소 80,000 ~ 최대 300,000 길이를 갖습니다. 100,000보다 짧은 시계열에는 뒷부분(미래)에 0벡터를 붙여주어 100,000으로 맞춰주었고, 100,000보다 긴 시계열은 행을 2~4개씩 더함으로써 압축하였습니다. 
    - 10만이상 20만 이하인 데이터는 행을 2개씩 더하여 길이를 반으로 만들고, 20만이상 30만 이하인 데이터는 행을 3개씩 더하여 3분의 1로 만들고, 30만 이상 40만 이하인 데이터는 행을 4개씩 더하여 4분의 1로 만들었습니다.

### Normalization
![image](/assets/images/1D CNN stock forcasting images/scaling.png){: width="80%" height="80%"}
- Normalization으로 적용한 함수는 위와 같습니다. 데이터가 있을 때 그 데이터의 절대값의 최대값으로 데이터를 나누어 주었습니다. 
- 현 프로젝트가 외국인이 현재 매도를 하고 있는지 매수를 하고 있는지 파악하는 것이 중요하기 때문에 음수와 양수가 유지될 수 있는 Normalization을 했습니다. 
- 이와 같은 Normalization을 적용하면 값은 **-1 ~ 1** 범위를 갖게 됩니다.

## 전처리 코드

### Step 1
- A301, B901, C301, C101로 나뉘어 있는 데이터를 통합해서 각 종목별로 나눈다.
- 또한, 각 종목별로 모든 날짜가 통합되어있는 데이터를 일별로 나눈다.
- 결과적으로 데이터는 종목별로 파일을 생성해 나뉘고 각 종목 파일에는 날짜별로 파싱되어 데이터가 저장된다.
- 저장 경로: 코도가 있는 폴더에서 DATA 폴더를 생성해 두면 그 안에 각 종목 파일이 생긴다.


```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import glob
import os
```


```python
# 각 종목 파일 생성
file_paths = glob.glob('./A301/*') 
file_name = []
for file_path in file_paths:
    file_name.append(file_path.split('\\')[-1].split('_')[0])

path = './DATA/'

# 해당 경로에 폴더가 없다면 폴더 생성
for i in file_name:
    if not os.path.exists(f'{path}{i}'):
        os.makedirs(f'{path}{i}')


# 날짜별로 분류, 필요한 컬럼만 뽑아서 저장
instances = ['A301','B901','C301', 'C101']
for instance in instances:
    file_paths = glob.glob(f'./{instance}/' + "*") 
    file_name = []
    for file_path in file_paths:
        file_name.append(file_path.split('\\')[-1])

    for i in file_name: # 종목 이름 순환
        isin = i.split('_')[0]
        if instance == 'A301':
            required_columns = ['server_time','Trading Price','Trading volume','day']
        if instance == 'B901':
            required_columns = ['server_time','Member Number 1 for Ask','Ask_Trading Volume 1','Member Number 1 for Bid','Bid_Trading Volume 1',
            'Member Number 2 for Ask','Ask_Trading Volume 2','Member Number 2 for Bid','Bid_Trading Volume 2','Member Number 3 for Ask',
            'Ask_Trading Volume 3','Member Number 3 for Bid','Bid_Trading Volume 3','Member Number 4 for Ask','Ask_Trading Volume 4','Member Number 4 for Bid',
            'Bid_Trading Volume 4','Member Number 5 for Ask','Ask_Trading Volume 5','Member Number 5 for Bid','Bid_Trading Volume 5','day']
        if instance == 'C301':
            required_columns = ['server_time','Arbitrage Ask Trust Trading Volume','Arbitrage Ask Principal Trading Volume',
            'Arbitrage Bid Trust Trading Volume','Arbitrage Bid Principal Trading Volume','Non-Arbitrage Ask Trust Trading Volume',
            'Non-Arbitrage Ask Principal Trading Volume','Non-Arbitrage Bid Trust Trading Volume','Non-Arbitrage Bid Principal Trading Volume','day']
        if instance == 'C101':
            required_columns = ['Investor Code','Accumulated Ask Trading Volume','Accumulated Bid Trading Volume','day']
        
        DF = pd.read_parquet(f'./{instance}/' + i, columns=required_columns) #실시간 체결
        day = DF['day'].unique()
            
        for j in day: #날짜 순환
            df = DF[DF['day']==j]
            df.drop('day',axis=1, inplace=True)

            df.to_parquet(f'./DATA/{isin}/{instance}_{j}.parquet', index=False, engine='pyarrow')

```

### Step 2
- 각 파일에 있는 데이터를 가져와서 전처리를 진행한다.
- 정답 데이터는 학습데이터 가장 오른쪽 컬럼에 생성된다. 
- 정답 데이터 값은 스칼라이고 같은 스칼라 값이 모든 행에 똑같이 들어간다.
- 모두 돌리는데 대략 89분이 걸린다.


```python
# 각 종목명(폴더명 가져오기)
above_file_paths = glob.glob('./DATA/*')
above_file_paths
above_file_names = []
for file_path in above_file_paths:
    above_file_names.append(file_path.split('\\')[-1])

remove = ['KR7022100002','KR7086520004','KR7091990002','KR7247540008'] #비어있음
above_file_names = [x for x in above_file_names if x not in remove]

# 특정 경로에 있는 모든 A301 이름 모두 가져오기
file_pattern = 'A301_*.parquet'
file_paths = glob.glob(f'./DATA/KR7000270009/' + file_pattern) 
# 파일 이름만 추출, 날짜만 추출
A301S_file_name = []
dates = []
for file_path in file_paths:
    A301S_file_name.append(file_path.split('\\')[-1])  # 경로에서 파일 이름만 추출
    _file_name = file_path.split('\\')[-1]
    cleaned_name = _file_name.replace('A301_', '').replace('.parquet', '')
    dates.append(cleaned_name)

# 특정 경로에 있는 모든 B901S 이름 모두 가져오기
file_pattern = 'B901_*.parquet'
file_paths = glob.glob(f'./DATA/KR7000270009/' + file_pattern) 
B901S_file_name = []
for file_path in file_paths:
    B901S_file_name.append(file_path.split('\\')[-1])  # 경로에서 파일 이름만 추출

# 특정 경로에 있는 모든 C301S 이름 모두 가져오기
file_pattern = 'C301_*.parquet'
file_paths = glob.glob(f'./DATA/KR7000270009/' + file_pattern) 
C301S_file_name = []
for file_path in file_paths:
    C301S_file_name.append(file_path.split('\\')[-1])  # 경로에서 파일 이름만 추출

# 특정 경로에 있는 모든 C101S 이름 모두 가져오기
file_pattern = 'C101_*.parquet'
file_paths = glob.glob(f'./DATA/KR7000270009/' + file_pattern) 
C101S_file_name = []
for file_path in file_paths:
    C101S_file_name.append(file_path.split('\\')[-1])  # 경로에서 파일 이름만 추출

# 스케일러 정의
# 시간 컬럼 제외
# 절댓값의 최댓값으로 나눠주어 절댓값 중 가장 큰 값이 1 혹은 -1이 되고, 0값과 음수, 양수가 유지된다.
def scaler(df):
    df[df.columns[1:]] = df[df.columns[1:]]/abs(df[df.columns[1:]]).max()
    df = df.fillna(value=0)
    return df
```

|column|max| 
|------|---|
|거래량  |       6827315.00|
|체결가격 |            30.56|
|순매수수량|       1373542.00|
|외국인 매도수량 |   1469274.00|
|외국인 매수수량|    1946016.00|
|기관 매도수량   |  8381906.00|
|기관 매수수량  |   8446100.00|

|column|min| 
|------|---|
|거래량      |         0.0|
|체결가격     |         0.0|
|순매수수량  |    -1910596.0|
|외국인 매도수량   |       0.0|
|외국인 매수수량  |        0.0|
|기관 매도수량     |      0.0|
|기관 매수수량    |       0.0|



```python
for above_file_name in above_file_names: #모든 종목 순환
    for A,B,C,Y,date in zip(A301S_file_name, B901S_file_name, C301S_file_name, C101S_file_name, dates): #모든 날짜 순환

        dfa = pd.read_parquet(f'./DATA/{above_file_name}/' + A) #실시간 체결
        dfb = pd.read_parquet(f'./DATA/{above_file_name}/' + B) #상위거래원
        dfc = pd.read_parquet(f'./DATA/{above_file_name}/' + C) #프로그램매매
        dfy = pd.read_parquet(f'./DATA/{above_file_name}/' + Y) #장마감

        # 이름 변경
        korcol={
        'server_time':'시간',
        'Trading Price':'체결가격',
        'Trading volume':'거래량',}
        dfa = dfa.rename(columns=korcol)

        korcol={'server_time': '시간',
        'Member Number 1 for Ask': '1단계매도회원번호',
        'Ask_Trading Volume 1': '1단계매도체결수량',
        'Member Number 1 for Bid': '1단계매수회원번호',
        'Bid_Trading Volume 1': '1단계매수체결수량',
        'Member Number 2 for Ask': '2단계매도회원번호',
        'Ask_Trading Volume 2': '2단계매도체결수량',
        'Member Number 2 for Bid': '2단계매수회원번호',
        'Bid_Trading Volume 2': '2단계매수체결수량',
        'Member Number 3 for Ask': '3단계매도회원번호',
        'Ask_Trading Volume 3': '3단계매도체결수량',
        'Member Number 3 for Bid': '3단계매수회원번호',
        'Bid_Trading Volume 3': '3단계매수체결수량',
        'Member Number 4 for Ask': '4단계매도회원번호',
        'Ask_Trading Volume 4': '4단계매도체결수량',
        'Member Number 4 for Bid': '4단계매수회원번호',
        'Bid_Trading Volume 4': '4단계매수체결수량',
        'Member Number 5 for Ask': '5단계매도회원번호',
        'Ask_Trading Volume 5': '5단계매도체결수량',
        'Member Number 5 for Bid': '5단계매수회원번호',
        'Bid_Trading Volume 5': '5단계매수체결수량',}
        dfb = dfb.rename(columns=korcol)

        korcol={'server_time': '시간'}
        dfc = dfc.rename(columns=korcol)

        # 공백이 있으면 타입변화가 되지 않음으로 공백을 문자열 0으로 변경
        dfa.replace('     ', '0', inplace=True)
        dfb.replace('     ', '0', inplace=True)
        dfc.replace('     ', '0', inplace=True)
        dfy.replace('     ', '0', inplace=True)

        # 데이터 타입 변경
        # 시간 제외 int로 변환
        columns = dfa.columns.difference(['시간'])
        dfa[columns] = dfa[columns].astype(int)

        columns = dfb.columns.difference(['시간'])
        dfb[columns] = dfb[columns].astype(int)

        columns = dfc.columns.difference(['시간'])
        dfc[columns] = dfc[columns].astype(int)


        # 프로그램 매매 순매수 수량 구하기
        buy = dfc['Arbitrage Bid Trust Trading Volume']+dfc['Arbitrage Bid Principal Trading Volume']+dfc['Non-Arbitrage Bid Trust Trading Volume']+dfc['Non-Arbitrage Bid Principal Trading Volume']
        sell = dfc['Arbitrage Ask Trust Trading Volume']+dfc['Arbitrage Ask Principal Trading Volume']+dfc['Non-Arbitrage Ask Trust Trading Volume']+dfc['Non-Arbitrage Ask Principal Trading Volume']
        accrued_amount = buy - sell

        occurred_amount = []

        for i in range(len(accrued_amount)):
            if i == 0:    
                occurred_amount.append(accrued_amount[0])
            if i > 0:
                occurred_amount.append(accrued_amount[i] - accrued_amount[i-1])
        dfc['순매수수량'] = occurred_amount

        drop_columns = ['Arbitrage Ask Trust Trading Volume','Arbitrage Ask Principal Trading Volume','Arbitrage Bid Trust Trading Volume','Arbitrage Bid Principal Trading Volume',
            'Non-Arbitrage Ask Trust Trading Volume','Non-Arbitrage Ask Principal Trading Volume','Non-Arbitrage Bid Trust Trading Volume','Non-Arbitrage Bid Principal Trading Volume',]
        dfc = dfc.drop(drop_columns,axis=1)


        # 상위거래원 데이터는 대략 1분마다 한 번씩 올라온다
        # 하나의 증권사가 상위거래원에 뜰 만큼 거래를 하고 추가적으로 거래를 하면 추가 거래한 양이 해당 증권사의 거래수량에 누적되어 나타남
        # 그걸 그 순간의 거래량으로 바꾸는 코드
        sellbuy = ['매도', '매수']
        for z in sellbuy:
            for j in range(1,6):
                member = {}
                current_volume = []
                for i in range(len(dfb)):
                    # 첫 번째 거래는 그냥 추가
                    if i == 0: 
                        _x = dfb[f'{j}단계{z}체결수량'][i]
                    
                    # 회원번호가 이전과 같으면 현재에서 이전 거래량을 뺀다.
                    if i>0 and dfb[f'{j}단계{z}회원번호'][i] == dfb[f'{j}단계{z}회원번호'][i-1]: #&과 and는 다르다. &로 하면 안됨
                        _x = dfb[f'{j}단계{z}체결수량'][i] - dfb[f'{j}단계{z}체결수량'][i-1]
                    
                    # 회원번호가 달라졌는데 이전에 등장하지 않은 회원번호일 때.
                    if i>0 and dfb[f'{j}단계{z}회원번호'][i] != dfb[f'{j}단계{z}회원번호'][i-1] and dfb[f'{j}단계{z}회원번호'][i] not in member.keys():
                        _x = dfb[f'{j}단계{z}체결수량'][i] # 그냥 추가
                        member[dfb[f'{j}단계{z}회원번호'][i-1]] = dfb[f'{j}단계{z}체결수량'][i-1] #딕셔너리에 회원번호, 거래량 메모
                    
                    # 회원번호가 달라졌는데 이전에 등장한 회원번호일 때.
                    if i>0 and dfb[f'{j}단계{z}회원번호'][i] != dfb[f'{j}단계{z}회원번호'][i-1] and dfb[f'{j}단계{z}회원번호'][i] in member.keys():
                        _x = dfb[f'{j}단계{z}체결수량'][i] - member[dfb[f'{j}단계{z}회원번호'][i]] #같은 회원번호가 마지막으로 가졌던 거래량 빼기
                        member[dfb[f'{j}단계{z}회원번호'][i-1]] = dfb[f'{j}단계{z}체결수량'][i-1] #딕셔너리에 회원번호, 거래량 새롭게 갱신

                    current_volume.append(_x)
                dfb[f'{j}단계{z}체결수량'] = current_volume


        # 외국인 회원을 전부 1로 그외는 0으로 바꾼다.
        foreign = [
            29,33,35,36,37,38,40,41,42,43,44,45,54,58,60,61,62,67,74,75,506,513,
            516,519,520,521,523,537,538,539,611,907,908,939,942]

        for i in range(1, 6):
            _foreign_label = []
            for j in dfb[f'{i}단계매도회원번호']:
                if j in foreign:
                    _foreign_label.append(1)
                else:
                    _foreign_label.append(0)
            dfb[f'{i}단계매도회원번호'] = _foreign_label
                    
        for i in range(1, 6):
            _foreign_label = []
            for j in dfb[f'{i}단계매수회원번호']:
                if j in foreign:
                    _foreign_label.append(1)
                else:
                    _foreign_label.append(0)
            dfb[f'{i}단계매수회원번호'] = _foreign_label


        # 상위거래원 외국인 증권사와 기관 컬럼 분리
        dfb_split = pd.DataFrame(dfb['시간'])
        sellbuy = ['매도','매수']
        for j in sellbuy:
            for i in range(1,6):
                dfb_split_x = dfb[dfb[f'{i}단계{j}회원번호'] == 1][['시간',f'{i}단계{j}체결수량']]
                dfb_split_x = dfb_split_x.rename(columns={f'{i}단계{j}체결수량':f'{i}단계{j}체결수량_외국인'})
                dfb_split = pd.merge(dfb_split, dfb_split_x, on='시간', how='left')
                
                dfb_split_x = dfb[dfb[f'{i}단계{j}회원번호'] == 0][['시간',f'{i}단계{j}체결수량']]
                dfb_split_x = dfb_split_x.rename(columns={f'{i}단계{j}체결수량':f'{i}단계{j}체결수량_기관'})
                dfb_split = pd.merge(dfb_split, dfb_split_x, on='시간', how='left')

        dfb_split['외국인 매도수량'] = dfb_split['1단계매도체결수량_외국인'] + dfb_split['2단계매도체결수량_외국인'] + dfb_split['3단계매도체결수량_외국인'] + dfb_split['3단계매도체결수량_외국인'] + dfb_split['4단계매도체결수량_외국인'] + dfb_split['5단계매도체결수량_외국인'] 
        dfb_split['기관 매도수량'] = dfb_split['1단계매도체결수량_기관'] + dfb_split['2단계매도체결수량_기관'] + dfb_split['3단계매도체결수량_기관'] + dfb_split['3단계매도체결수량_기관'] + dfb_split['4단계매도체결수량_기관'] + dfb_split['5단계매도체결수량_기관'] 
        dfb_split['외국인 매수수량'] = dfb_split['1단계매수체결수량_외국인'] + dfb_split['2단계매수체결수량_외국인'] + dfb_split['3단계매수체결수량_외국인'] + dfb_split['4단계매수체결수량_외국인'] + dfb_split['5단계매수체결수량_외국인'] 
        dfb_split['기관 매수수량'] = dfb_split['1단계매수체결수량_기관'] + dfb_split['2단계매수체결수량_기관'] + dfb_split['3단계매수체결수량_기관'] + dfb_split['4단계매수체결수량_기관'] + dfb_split['5단계매수체결수량_기관'] 


        #모든 데이터를 합치기 위한 데이터 프레임 생성
        # 시작 시간과 끝 시간 설정 
        start_time = datetime.strptime('08:30:00', '%H:%M:%S') #문자열을 시간으로
        end_time = datetime.strptime('18:00:00', '%H:%M:%S')

        # 시간 데이터 타입으로 이루어진 리스트 생성
        time_list = []
        current_time = start_time
        while current_time <= end_time:
            time_list.append(current_time.time().strftime('%H%M%S')) 
            current_time += timedelta(seconds=1)

        dft = pd.DataFrame({'시간':time_list})

        before_decrease = len(dfa)
        dfa['체결가격'] = dfa['체결가격']/1000
        # 데이터 양 줄이기
        dfa = dfa[dfa['체결가격'] * dfa['거래량'] > 1000] # 거래량 X 체결가격이 100만원보다 작은 값을 개인으로 보고 제거
            # 계산 값이 너무 크면 계산값이 갑자기 마이너스가 되는 경우가 생김
        after_decrease = len(dfa)
        dfc = dfc[dfc['순매수수량']!=0] # 순매수수량이 0인 행 제거. 10월 5일분  852개


        # 데이터 프레임 합치기
        dft = pd.merge(dft, dfa[['시간','거래량', '체결가격']], on='시간', how='left')
        dft = pd.merge(dft, dfb_split[['시간','외국인 매도수량','기관 매도수량','외국인 매수수량','기관 매수수량']] , on='시간', how='left')
        dft = pd.merge(dft, dfc[['시간', '순매수수량']], on='시간', how='left')

        dft.drop('시간', axis=1, inplace=True)
        # '시간'컬럼 제외하고 다른 커럼의 값이 모두 Null인 경우 해당 행 제거
        dft.dropna(subset=dft.columns, how='all', inplace=True) 

        # Null을 0으로 채움
        dft.fillna(value=0, inplace=True)

        desired_length = 100000
        reduction = 0
        if len(dft) <= desired_length:
            # 부족한 행 수 계산
            num_rows_to_add = desired_length - len(dft)
        if len(dft) > desired_length and len(dft) <= 200000:
            result_df = pd.DataFrame()
            for col in dft.columns:
                result_df[col] = dft[col].iloc[::2].reset_index(drop=True) + dft[col].iloc[1::2].reset_index(drop=True)
            dft = result_df
            del result_df
            num_rows_to_add = desired_length - len(dft)
        if len(dft) > 200000 and len(dft) <= 300000:
            result_df = pd.DataFrame()
            for col in dft.columns:
                result_df[col] = dft[col].iloc[::3].reset_index(drop=True) + dft[col].iloc[1::3].reset_index(drop=True) + dft[col].iloc[2::3].reset_index(drop=True)
            dft = result_df
            del result_df
            num_rows_to_add = desired_length - len(dft)
        if len(dft) > 300000 and len(dft) <= 400000:
            result_df = pd.DataFrame()
            for col in dft.columns:
                result_df[col] = dft[col].iloc[::4].reset_index(drop=True) + dft[col].iloc[1::4].reset_index(drop=True) + dft[col].iloc[2::4].reset_index(drop=True) + dft[col].iloc[3::4].reset_index(drop=True)
            dft = result_df
            del result_df
            num_rows_to_add = desired_length - len(dft)

        dft = dft[['거래량', '체결가격','순매수수량','외국인 매도수량','외국인 매수수량','기관 매도수량','기관 매수수량']]

        dft['거래량'] = dft['거래량'] / 6827315
        dft['체결가격'] = dft['체결가격'] / 3056
        dft['순매수수량'] = (dft['순매수수량'] -(-1910596))/(1373542-(-1910596))
        dft['외국인 매도수량'] = dft['외국인 매도수량'] / 1469274
        dft['외국인 매수수량'] = dft['외국인 매수수량'] / 1946016
        dft['기관 매도수량'] = dft['기관 매도수량'] / 8381906
        dft['기관 매수수량'] = dft['기관 매수수량'] / 8446100

        empty_df = pd.DataFrame(0, index=range(num_rows_to_add), columns=dft.columns)

        dft = pd.concat([dft,empty_df], ignore_index=True)

        # 가장 마지막에 나오는 데이터 12개만 남기기
        dfy = dfy[-12:]

        # 데이터 타입 변경
        dfy[dfy.columns] = dfy[dfy.columns].astype(int)

        # 순수량 구하기
        dfy['누적매매 체결 순수량'] = dfy['Accumulated Bid Trading Volume'] - dfy['Accumulated Ask Trading Volume']

        # 필요없는 컬럼 제거
        drop_columns = ['Accumulated Ask Trading Volume','Accumulated Bid Trading Volume']
        dfy.drop(drop_columns, axis=1, inplace=True)

        foreign = dfy[dfy['Investor Code'] == 9000.0]['누적매매 체결 순수량'].item()

        dft['y'] = foreign

        # Parquet로 데이터프레임 저장 (경로 지정)
        dft.to_parquet(f'./Preprocessed_data/{above_file_name}_{date}.parquet', index=False, engine='pyarrow')
```

### step 3
- 스케일러에 쓸 정답 데이터(GT)의 절댓값의 최대값을 구한다.
- 훈련 데이터와 테스트 데이터를 구분하고 훈련데이터 범위에서 구한다(정답 유출 방지).


```python
dates = dates[:170] #9월까지를 훈련 데이터로 정의

all_foreign = []
for above_file_name in above_file_names: #모든 종목 순환
    for Y,date in zip(C101S_file_name, dates): #모든 날짜 순환

        dfy = pd.read_parquet(f'./DATA/{above_file_name}/' + Y) #장마감
        # 가장 마지막에 나오는 데이터 12개만 남기기
        dfy = dfy[-12:]

        # 데이터 타입 변경
        dfy[dfy.columns] = dfy[dfy.columns].astype(int)

        # 순수량 구하기
        dfy['누적매매 체결 순수량'] = dfy['Accumulated Bid Trading Volume'] - dfy['Accumulated Ask Trading Volume']

        # 필요없는 컬럼 제거
        drop_columns = ['Accumulated Ask Trading Volume','Accumulated Bid Trading Volume']
        dfy.drop(drop_columns, axis=1, inplace=True)

        foreign = dfy[dfy['Investor Code'] == 9000.0]['누적매매 체결 순수량'].item()

        all_foreign.append(foreign)
        
def normalize(s):
    return (s-(-6790662))/(13612549-(-6790662)) # max:13612549 # min:-6790662
def denormalize(s):
    return (s*(13612549-(-6790662))+(-6790662))
```

## Model Architecture 

### 1D CNN


```python
import torch
import torch.nn as nn  

class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()

        input_channels = 7

        layers = []
        for _ in range(15):
            output_channels = round(input_channels * 1.5)
            layers.append(nn.Conv1d(input_channels, output_channels, kernel_size=3, stride=2, padding=1)) 
            input_channels = output_channels

        self.layers = nn.Sequential(*layers) 

        self.avgpool = nn.AvgPool1d(kernel_size=4)
        self.fc = nn.Linear(output_channels, 1)

    def forward(self, x):
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.squeeze(x)
        x = self.fc(x)
        return x
```

- 1D CNN은 위 코드와 같이 설계했습니다.

![image](/assets/images/1D CNN stock forcasting images/CNN.png){: width="60%" height="60%"}

- 15층의 CNN layer와 linear haed를 사용했습니다.
- 컴퓨팅 리소스가 감당할 수 있도록 stride=2로 설정하여 feature의 사이즈를 줄여나갔습니다.

**학습을 위한 설정은 아래와 같습니다**
- batch size = 16
- learning rate = 0.00001
- Loss function = MSELoss()
- optimizer = optim.AdamW()
- Train data : 2023년 1월 ~ 9월 (7,480 set)
- Test data : 2023년 10월 (792 set)

## 시각화 및 성능평가

### 1D CNN 시각화
![image](/assets/images/1D CNN stock forcasting images/GT&Pred.png){: width="80" height="80"}

테스트 데이터셋으로 출력한 모델의 output과 Ground Truth를 시각화했습니다.\
결과는 그래프에서 보시다싶이 어느정도 경향성을 따라가지만 높은 예측율은 보여주지 않았습니다. 

### 성능평가
output이 scaling된 상태이기 때문에 -1~1 범위를 갖습니다. 이러한 범위의 값은 제곱하면 0으로 다가간다는 특성이 있기 때문에 절댓값을 취하는 MAE를 채택하였습니다.

테스트 데이터셋에 대하여 1D CNN과 LSTM의 MAE를 측정해보았을 때 결과는 아래와 같습니다.
- **1D CNN MAE : 0.0117**
- **LSTM MAE : 0.1179**

1D CNN이 LSTM보다 오차가 적은 것을 알 수 있습니다. 1D CNN의 성능 자체만 보면 그리 뛰어나지 않지만, LSTM보다 높은 성능을 보였슴으로 어느정도 유의미한 결과를 얻었다고 생각합니다.

# 회고

시계열 길이를 압축하고 scaling하는 과정에서 정보가 손상되어 성능의 저하가 많았던 것으로 예측됩니다. 특히, 주식 데이터의 특성상 값의 편차가 매우 크기 때문에 scaling을 했을 때 작은 값은 0에 매우 근접하여 정보가 손실(Underflow)되기 쉽습니다. 이로인해 학습이 원할하지 않았고 그 결과 예측값의 편차가 감소했다고 추측하고 있습니다.

팀원 모두가 가졌던 아쉬움은 부족한 컴퓨팅 리소스의 한계입니다. 거대한 데이터를 다룰 수 있는 환경이었다면 더 많은 실험과 더 다양한 모델, 여러 하이퍼 파라미터 튜닝을 실험할 수 있었겠지만 그러지 못한 것이 못내 아쉽습니다.

애초에 주식에 대한 예측은 성능을 기대하기 어려웠던 게 아닐까하는 생각도 듭니다. 학술연구 정보 서비스를 찾아 보았을 때 51%가 넘는 주식 예측력을 가진 모델을 찾기가 어려웠습니다. 다양한 모델과 하이퍼 파라미터 튜닝이 필요하나 주식 예측의 불확실성과 예측의 한계를 인식하였습니다.


