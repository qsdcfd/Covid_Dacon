# Dacon_Covid

## 회의 및 아카이빙 

[https://www.notion.so/9dfd78b8a1cd4f8891a9799fb9e7c313](https://www.notion.so/Dacon-61f6d385c1d148d29ee8a65b8e0a5dc1)

## 주관: Dacon

<br>

![image](https://user-images.githubusercontent.com/86671456/173595098-36e9e43e-3b20-4b95-a045-485425c103df.png)


## Abstract

| 분석명 |  
|:-----:|
|코로나 음성/양성 분류하기|

|  소스 데이터 |     데이터 입수 난이도    |      분석방법     |
|:------------------:| -----|:---------------:|
|train,unlabed| 하|   |

|  분석 적용 난이도  |     분석적용 난이도 사유    |      분석주기     | 분석결과 검증 Owner|
|:-----:| --------------------------------------- |:---------------:|----------------|
|상 |익숙하지 않은 음성테스트, 성별이 이진이 아닌 삼중으로 나눠짐   |Daily  | Dacon |



<br>

### Machine Learning Project 

**방식: 애자일 스프린트**

---

|  프로젝트 순서 |     Point    | 세부 내용 |  
|:------------------:| -----|------|
|문제 정의|해결할 점, 찾아내야할 점 |발열,두통 그리고 기침소리를 기반으로 음성/양성인 사람 찾기|
|데이터 수집|공개 데이터, 자체 수집, 제공된 데이터 |Dacon(train, test, unlabed|   
|데이터 전처리|문제에 따라서 처리해야할 방향 설정 |성별의 other의 의미, mfcc를 통해서 확진자/비확진자의 주파수 체크|
|Feature Engineering|모델 선정 혹은 평가 지표에 큰 영향|성별의 other, mfcc의 주파수 의미 찾기|
|연관 데이터 추가|추가 수집 |눈문, 관련 깃허브 탐색  |
|알고리즘 선택| 기본적, 현대적|MLP,CNN, ResNet50|   
|모델 학습|하이퍼파라미터,데이터 나누기 |Gradient Search CV기반으로 하어퍼파라미터 설정, train,val,test |
|모델 평가|확률|상위 10% |
|모델 성능 향상|성능 지표, 하이퍼파라미터, 데이터 리터러시 재수정 |  |

<br>

### Basic information

**공식기간: 2022.06.7 ~ 2022.07.08**


- 인원:이세현,[문석민](https://github.com/msmsm104/Dacon_covid19)
- 직책: 
- 데이터: 
- 주 역할:
- 보조 역할: 
- 추가 역할:
- 협업장소: Github, GoogleMeet
- 소통: Slack, Notion,Git project, Google OS
- 저장소: Github, Google Drive
- 개발환경: Visual studio code, Juypter Notebook, colab
- 언어 :python 3.8.x
- 라이브러리:Numpy,Pandas, Scikit-learn 1.1.x
- 시각화 라이브러리: Seaborn, Matplotlib, Plot,Plotly  
- 시각화 도구: Tableau, GA
- 웹 크롤링: 

<br>

#### 파일 설명

- feat: 기능 개발 관련
- fix: 오류 개선 혹은 버그 패치
- docs: 문서화 작업
- test: test 관련
- conf: 환경설정 관련
- build: 데이터 집산
- Definition: 프로젝트의 전반적인 문제 정의 및 내용 설정
- Data: 전처리 파일 및 모델링을 위한 파일
- models: 여러 모델들의 집합
- src :scripts
