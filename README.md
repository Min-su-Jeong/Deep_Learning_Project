# Deep_Learning_Project
딥러닝 과목을 수강하면서 개발한 프로젝트 Repository입니다.

### Anomaly Detection for Determining Product Quaility of Injection Molding Machine Results
> 사출기 결과물의 양불 판정을 위한 이상치 검출  

<br/>

## Develop version
- Library
  - Python: 3.7.13
  - Numpy: 1.21.6
  - Pandas: 1.3.5
  - Scipy: 1.7.3
  - Scikit-learn: 1.0.2
  - Pyod: 1.0.7
  - Tensorflow-gpu: 2.3.0

- H/W
  - CPU : 11𝑡ℎ Gen Intel® Core™ i5-1135G7 @ 2.40GHz(8cpus) ~ 2.4GHz
  - RAM : 16GB
  - GPU : Internal – Intel® Iris® Xe Graphics, External – NVIDIA GeForce MX450
  
- S/W
  - OS: Windows 10
  - Develop Tool: Jupyter notebook(Anaconda)
  
  <br/>
  
## Detail Project
#### 데이터 공학에 사용한 방법
```
1) 회귀 분석
  ⦁ 잔차제곱합(RSS: Residual Sum of Squares)을 최소화하는 가중치 벡터를 구하는
    방법이며 Stepwise 기법이라고도 한다.
  ⦁ 속성 중 표준편차가 0이거나 문자인 데이터는 삭제하고 남은 속성의 이상치
    및 결측치를 제외한 후 결과(G/NG)에 영향을 가장 많이 끼치는 속성을 찾는 기법
  ⦁ 각각의 독립변수 x_{i}가 종속변수 y에 영향이 있는지 확인 가능하며 p-value
    값을 기준으로 무효한 feature들을 제거한다.

2) 독립 T 검정
  ⦁ 두 집단의 평균 차이를 검증하기 위한 방법
  ⦁ 각 feature의 GOOD/NOT_GOOD에 대한 영향력을 판단한다.
  ⦁ 귀무가설은 “집단에 따라 평균의 차이가 없다”가 되고 대립가설은 “집단에
    따라 평균의 차이가 있다.”가 된다. 여기서 집단에 따른 평균의 차이가 없다면 
    이는 해당 종속변수(피처)가 G와 NG에 유의하게 영향을 끼치지 않는다고 
    판단한다. 따라서 대립가설이 채택이 되어야 해당 피처가 결과물인 G/NG에
    유의미한 영향을 줄 수 있다고 판단되며 이를 위해 독립 T 검정의 유의 확률이 
    0.05보다 작아야 함을 의미한다.

3) 정규화 검정
  ⦁ 데이터 세트의 분포가 정규분포를 따르는지 검정하는 방법
  ⦁ 자료의 값들과 표준 정규점수와의 선형상관관계를 측정하여 표본이 정규 
    분포의 가정을 만족하는지 검정한다.
```

#### 모델 구조(Architecture)
<p align="center">
  <img src="https://user-images.githubusercontent.com/74342121/214363656-10dfc2ca-a5f1-4980-b2f3-0ca8d9f0f249.png" width="600" height="200">
</p>

```
⦁ Name: MO-GAAL

⦁ Feature
GAAL은 k개의 하위 생성기와 판별기로 구성된다. 
 1) 하위 생성기: 원본 데이터의 생성 메커니즘을 학습하려고 시도
 2) 판별기: 원본 데이터에서 생성된 데이터를 식별하려고 시도
 ⇒ 하위 생성기는 원래 데이터 세트 내부 또는 데이터 세트 근처에서 발생하는 정보를
    제공하는 잠재적 이상치를 점점 더 많이 생성할 수 있다. 하위 생성기에 의해 생성된
    잠재적 특이치의 수 제어와 결합하여 합리적인 기준 분포를 구성할 수 있다. 
    결과적으로 판별기는 비집중 정규 데이터에서 비집중 특이치를 분리할 수 있다.

⦁ Adventage
  1) SO-GAAL의 발전기가 모드 붕괴 문제에 빠지는 문제를 해결한 모델
  2) SO-GAAL의 네트워크 구조를 단일 발전기에서 목표가 다른 여러 발전기(MO-GAAL)로 확장 
  3) MO-GAAL(k=n)은 발전기를 여러 개 포함하고 있음
  4) 서로 다른 하위 생성기에 의해 생성된 서로 다른 수의 잠재적 특이치의 통합 
      ⇒ 전체 데이터 세트에 대한 합리적인 기준 분포를 생성
```

#### 파이프라인(Pipeline)
```
1. 데이터 수집
  ⦁ 사출기 결과물에 대한 csv 파일 제공
  ⦁ 약 100여 개의 Features
  
2. 전처리 및 EDA 과정 수행
  ⦁ Dataset(Train, test set) 병합
  ⦁ Label 이진화 수행
  ⦁ 결측치 확인 및 제거
  ⦁ Object type 제거
  ⦁ 회귀 분석(양/불 비율 동일: 랜덤 추출)
  ⦁ 정규화 검정 및 독립 T 검정 ⇒ 유의미한 특징 추출
  ⦁ StandardScaler 정규화
  ⦁ train-test 데이터 세트 split
   
3. 모델 학습
  ⦁ MO-GAAL 모델을 사용한 학습
  ⦁ Hyper-parameter tuning(k, stop_epochs, lr_d, lr_g, decay, etc)

4. 모델 결과 예측
  ⦁ 학습된 모델(MO-GAAL)을 기반으로 결과 예측(predict)
  ⦁ classification report를 사용한 결과 성능 확인(Macro avg: f1-score 기준)   
```

※ 이상치 검출에 대한 자세한 과정은 repository에 첨부된 <strong>『Anomaly Detection.ipynb』</strong> 파일 참조  
※ <strong>데이터 파일은 보안 상의 이유로 외부 유출이 불가하여 업로드 X</strong>
