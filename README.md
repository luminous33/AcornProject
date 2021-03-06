# AcornProject

## 09.17.2020 회의록
# == Topic Selection ==
### 1. 블록 체인 
  - 이유 
    1) 접근성이 생각보다 좋음
    2) 플랫폼 구현해주는 책 혹은 참고자료가 많음
    3) 배운 것을 다양하게 사용가능
    4) 신기술이라 희귀성 있음
    5) 병렬 처리, 서버 구현, ...
    6) 시각화, 시계열, 예측, 홈페이지 구축 가능
### 2. 유튜브
  - 이유
    1) 중간 프로젝트에 이어서 확장 가능
    2) 나타낼 수 있는 플랫폼이 많음
    3) 실시간으로 출력이 가능
### 3. 환경과 대체에너지
  - 이유
    1) 공공데이터 활용가능
    2) 기온상승으로 인한 남극 소멸 예측 가능
    3) 하나의 환경 플랫폼으로 다양한 자료 활용가능
    4) 화석에너지 비율, 쓰레기 배출량, 공기 오염도, 기온, ...등 다양한 변수를 활용하여
       시각적으로 출력이 가능.(사진, 영상, 그래프, 기사 등)
### 4. 캐글
  - 이유
    1) row데이터에 대한 접근성 용이
    2) 다양한 결론 도출 가능
    3) 프로젝트와 더불어 공모전 참여 가능
    4) 다른 사람들과 공유할 수 있어 피드백 용이
### 5. 주식 
  - 이유
    1) 현재 나와있는 플랫폼을 직접적으로 구현 가능
    2) 프로젝트 진행하며 아이디어 참고에 용이
    3) 웹페이지, 어플리케이션, S/W등 여러 플랫폼 가능
    4) 구직활동 시 경쟁력 있는 포트폴리오
    5) 유튜브와 더불어 오픈API 활용성이 큼

## Conclusion :
  - 주식, 유튜브, 환경과 대체에너지에 대한 자료수집(데이터 수집방법, 어떠한 플랫폼을 사용할 지, 결론 도출, 활용가능성... 등)
 <br/>
 <br/>
 <br/>
 
## 09-27-2020 회의록
-- 주제 : 주식 증권 데이터 분석( 웹페이지 구축 및 프로그램 설정 ) : **API를 활용하여 실시간 증권형태 및 딥러닝을 이용한 주가예측**
### 활용 프로그램
  -- Python(3.8.1), Django(3.0.2), MYsql(8.0) & MariaDB(10.5.1), Github, Notion

1. 디비구축
  - 마리아DB
  - 개별로 디비 구축 => 마지막 프레임워크 구축전에는 개별 디비로 활용 이후 웹구축시 하나의 디비로 통합 혹은 생성
2. 데이터수집
  - API
  - Yahoo Finance
3. 데이터분석( 이번주는 여기까지 하는 것으로 )
  - 상관분석
  - 금융이론(트레이딩 전략)
  - 각자 개별로 자신이 진행하는 트레이딩 전략 **메모**
4. 모델구축( 다음 회의에서 결정 )
  - 머신러닝
  - 딥러닝
5. 웹페이지 구축
  - Django활용(인성)
6. 프로그램 구동( 이후 작업 )
  - 자동매매 프로그램(인성)
 <br/>
 <br/>
 <br/>
 
## 10-04-2020
- 개별 트레이딩 전략 구현 및 실행
- 노션 사이트 업로드(볼린저 밴드까지)
<br/>
<br/>
<br/>

## 10-05-2020
### 웹 페이지 구축 
https://www.barchart.com/ 참고

1. 한국 기업 대상 주가 변동표
    - Close, Change, High, Low, Volume
2. 각 기업별 시각화 그래프 제공 
    - 주식 투자전략을 기반으로 매수, 매도 시점 알려줌
    - 지표 : 효율적 투자선, 샤프 지수, 볼린저 밴드 지표 등등 
3. 딥러닝 기반 주가 예측 정보 제공 
4. 자동매매 프로그램 (exe) 제공 
<br/>
<br/>
<br/>

## 10-08-2020
- Server 지정
- 웹페이지 구상
   - 메인페이지, 구축방법, 서브페이지 및 화면 디자인
- 툴 지정
- 트레이딩 전략 설정
- 딥러닝 모델 종류 선정

- 웹 구현 예시
![abc1](https://user-images.githubusercontent.com/63041717/95675742-63400d80-0bf4-11eb-8886-ce95ffef2953.PNG)
![abc2](https://user-images.githubusercontent.com/63041717/95675743-63d8a400-0bf4-11eb-9d17-0af210dda1cd.PNG)
![abc3](https://user-images.githubusercontent.com/63041717/95675744-6509d100-0bf4-11eb-9560-d9b8bdeaef23.PNG)
![abc4](https://user-images.githubusercontent.com/63041717/95675745-65a26780-0bf4-11eb-9e57-7a8783939084.PNG)
![abc5](https://user-images.githubusercontent.com/63041717/95675746-65a26780-0bf4-11eb-8627-bd839faa7924.PNG)
![abc6](https://user-images.githubusercontent.com/63041717/95675740-62a77700-0bf4-11eb-8bb2-6e7e96155865.PNG)
<br/>
<br/>
<br/>

## 10-11-2020
- 웹페이지 구상 수정
![abc1](https://user-images.githubusercontent.com/63041717/95695099-6a0c6600-0c70-11eb-9a83-a411129fea2f.PNG)
![abc2](https://user-images.githubusercontent.com/63041717/95695102-6b3d9300-0c70-11eb-9f8b-52a684d1a969.PNG)
![abc3](https://user-images.githubusercontent.com/63041717/95695094-68db3900-0c70-11eb-8166-7438c1d48e03.PNG)
![abc4](https://user-images.githubusercontent.com/63041717/95695101-6aa4fc80-0c70-11eb-8a37-90aead947d13.PNG)
![abc5](https://user-images.githubusercontent.com/63041717/95695087-67aa0c00-0c70-11eb-8fce-2aa3d09025fd.PNG)
![abc6](https://user-images.githubusercontent.com/63041717/95695098-68db3900-0c70-11eb-9cdd-8d08923bc963.PNG)
- 플랫폼 실제 구현 
- 딥러닝 모델 결정 : RNN,LSTM,NLP
