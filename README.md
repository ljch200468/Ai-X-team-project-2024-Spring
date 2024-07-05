## Teamwork: Option A
### Youtube link:

**https://www.youtube.com/watch?v=vE79n678eJo**

### (1) Title
**두 개의 스트림 CNN-LSTM 알고리즘을 기반으로 한 굴착기 활동 인식**

### Note
두 명의 중국 유학생으로서, 한글 표현에 있어서 정확하지 않은 부분이 있을 수 있으니 양해 부탁드립니다.

### (2) Members
- **Member 1**:
  - **Name**: 이가성
  - **학생 ID**: 2022088313
  - **Department**: 건설환경공학과
  - **Email**: ljch200468@163.com
- **Member 2**:
  - **Name**: 구민제
  - **학생 ID**: 2022079089
  - **Department**: 기계공학과
  - **Email**: rjp3703@163.com

### (3) Proposal

#### 1. Motivation: Why are you doing this?
우리 팀의 주제 선택을 논의할 때, 우리는 먼저 "Ai+X: Deep Learning" 강좌에서 배운 딥러닝 관련 지식을 팀원들의 전공과 결합하여, 배운 지식을 실제 전공 문제 해결에 적용해보자는 취지로 정했습니다. 이를 위해 관련 자료와 학술 논문을 조사하고 팀원들의 다양한 전공 배경을 고려하여 두 가지 주제를 초기 선정했습니다:
1. **딥러닝 기반 컴퓨터 비전 알고리즘을 이용한 건설 기계의 3D 재구성**:
   - 이 연구 주제를 선택한 주된 이유는 팀원 구민제의 전공이 기계공학이기 때문입니다. 관련 자료와 논문을 조사한 후, 그는 건설 기계와 관련된 연구를 하고 싶어했습니다. 딥러닝 기반의 컴퓨터 비전 3D 재구성은 현재 산업계와 학계에서 매우 인기 있는 분야이기 때문에, 우리는 이를 하나의 대안 주제로 선택했습니다.
2. **Two-Stream CNN-LSTM 알고리즘을 이용한 굴삭기 활동 인식**:
   - 이 연구 주제를 선택한 주된 이유는 팀원 이가성의 전공이 건설환경공학이기 때문입니다. 그의 전공은 주로 토목 및 환경 공학 시공과 설계 관련 과정을 다룹니다. 관련 자료와 논문을 조사한 후, 구민제의 전공 내용을 종합적으로 고려하여, 우리는 CNN-LSTM 알고리즘을 기반으로 한 굴삭기 활동 인식을 연구 주제로 선택했습니다.
3. **최종적으로 관련 코드를 깊이 이해하고 컴파일 및 실행을 시도한 후, 코드 구현의 난이도와 예상 결과를 고려하여, 우리는 "Two-Stream CNN-LSTM 알고리즘을 이용한 굴삭기 활동 인식"을 우리 팀의 연구 주제로 최종 결정했습니다.**

#### 2. What do you want to see at the end?
Muti-Stream CNN-LSTM 알고리즘을 이용하여 다양한 굴삭기 활동을 정확하게 인식하고, 이를 통해 굴삭기의 생산성을 정밀하게 모니터링하며, 장비 배치 계획에 대한 신속한 결정을 내리는 것입니다.

### (4) Data Collection and Annotation

#### 1. Data Collection
아래 그림과 같이 5가지 다른 유형의 굴삭기 활동을 나타내는 총 5,000장의 사진을 수집하고 정리하였습니다. 이는 한양대학교 환경 및 토목 공학과의 조원석 교수의 연구실의 도움을 받아 ACIDb 및 CIS-Dataset의 자원을 활용하여 이루어졌습니다.

- [ACIDb](https://www.acidb.ca/)
- [CIS-Dataset](https://github.com/XZ-YAN/CIS-Dataset)

![image](https://github.com/ljch200468/Ai-X-team-project-2024-Spring/assets/170994864/b252a005-8b31-406d-a713-c39ae0b9c9fd)

**그림 1: 다양한 굴착기 활동을 포함하는 5가지 유형의 이미지 데이터셋입니다.**

#### 2. Data annotation
5000장의 이미지 중에서 ACIDb 및 CIS-Dataset 공공 데이터셋에 포함된 굴착기 활동 이미지는 이미 라벨링된 카테고리 정보가 있습니다. 이 부분 데이터에 대해서는 수작업 검사를 통해 사진 라벨 정보의 정확성을 확보했습니다. Jongwon Seo 교수의 연구실에서 제공하는 굴착기 활동 이미지는 카테고리 정보가 표시되어 있지 않아, 이 부분 데이터는 수작업으로 라벨을 붙였습니다. 원본 이미지 데이터셋의 라벨 정보는 아래 그림 2와 같습니다.


### (5) Methodology

#### 1. Explaining Your Choice of Algorithms (Methods)
관련 논문과 GitHub 자료를 검토한 후, 우리는 굴삭기 활동 인식 알고리즘으로 Two-Stream CNN-LSTM 알고리즘을 선택했습니다. 

Two-stream CNN-LSTM 네트워크를 이용한 굴착기 활동 인식 방법의 프레임워크는 그림 3과 같습니다. 구체적인 내용은 다음과 같습니다:
1. 먼저 굴착기 내부와 외부에 설치된 "조종실 카메라"와 "외부 카메라"를 이용하여 5가지 다른 굴착기 활동의 내부 및 외부 RGB 이미지를 획득합니다.
2. 그런 다음 OpenCV 알고리즘을 사용하여 동일 시점의 내부 및 외부 굴착기 활동 이미지를 하나의 이미지로 통합하여 Two-stream CNN-LSTM 네트워크 활동 인식 알고리즘의 입력 데이터로 사용합니다.
3. 이후 Two-stream CNN-LSTM 모델은 구체적인 굴착기 활동을 인식하는 역할을 담당하며 각 스트림의 합성곱 층은 특징 추출을 담당하고 LSTM은 두 스트림의 결합 출력에서 시퀀스 데이터의 장기 종속성을 인식합니다.
4. 마지막으로 다층 퍼셉트론 헤드(분류 층)는 완전 연결 층을 통해 LSTM 층의 결합 출력을 분류하고 훈련된 모델의 손실 및 정확도를 계산합니다.



#### 2. Source Code of Two-Stream CNN-LSTM Network Algorithm for Activity Recognition
굴착기 활동 인식을 위한 Two-stream CNN-LSTM 네트워크 알고리즘의 소스 코드 및 주석은 다음을 참조하십시오:

![image](https://github.com/ljch200468/Ai-X-team-project-2024-Spring/assets/170994864/2091142f-7963-4f87-8c60-1455bb23bb3b)

**그림 4. 두 개의 스트림을 사용하는 CNN-LSTM 네트워크 소스 코드 스크린샷**

![image](https://github.com/ljch200468/Ai-X-team-project-2024-Spring/assets/170994864/a53d7b70-2f7d-4ff1-8153-726d7f677ab7)

**그림 5. 두 개의 스트림을 사용하는 CNN-LSTM 네트워크 소스 코드--메인 함수.**

![image](https://github.com/ljch200468/Ai-X-team-project-2024-Spring/assets/170994864/61a0a8a0-99b7-4b04-83cf-596ece4922c8)

**그림 6. 두 개의 스트림을 사용하는 CNN-LSTM 네트워크 소스 코드--데이터셋.**

### (6) Evaluation & Analysis

#### 1. Performance Metrics
우리는 Two-stream CNN-LSTM 네트워크를 이용한 활동 인식 알고리즘의 성능을 평가하기 위해 흔히 사용되는 딥러닝 알고리즘 평가 지표인 Precision, Recall, Accuracy, 그리고 F1 Score를 사용합니다. 각 평가 지표의 의미와 공식은 다음의 공식 1-4에 나와 있습니다.

- Precision = TP / (TP + FP)  #1
- Recall = TP / (TP + FN)  #2
- Accuracy = (TP + TN) / (TP + FP + FN + TN)  #3
- F1 = 2 * (Precision * Recall) / (Precision + Recall)  #4

#### 2. Implementation Environment
해당 알고리즘은 Windows 10, 64비트 운영 체제에서 구현되었으며, 하드웨어 구성은 470 Intel(R) Xeon(R) Gold 6242R CPU @ 3.10 GHz 3.09 GHz (2) intel이 장착된 HP Z8 G4 워크스테이션입니다.

#### 3. Experimental Setup and Process
사용된 Two-stream CNN-LSTM 네트워크 알고리즘의 활성화 함수는 "ReLU"입니다. 모델의 성능을 지속적으로 최적화하기 위해 실험 과정에서 알고리즘의 하이퍼파라미터(epochs와 학습률)를 지속적으로 조정하였습니다. 훈련 과정은 그림 7과 같습니다.

![image](https://github.com/ljch200468/Ai-X-team-project-2024-Spring/assets/170994864/d605b387-829c-4043-8643-33c7144ded9c)

**그림 7 (a).훈련 과정 스크린샷**



하이퍼파라미터 튜닝을 기반으로 네트워크의 학습 하이퍼파라미터가 선택되었습니다: 학습률은 0.0001, 최소 배치 크기는 8, 학습 에포크 수는 30, LSTM 층의 은닉 유닛 수는 64입니다.

### (7) Experiment Results

#### 1. 훈련 및 테스트 데이터 세트의 혼동 행렬
제안된 방법의 훈련용 혼동 행렬은 그림 8(a)에 나와 있습니다. 개발된 모델은 97.36%의 높은 정밀도, 97.31%의 F1 점수 및 2.75%의 전체 손실을 보였습니다. 각 활동의 정밀도, 재현율 및 정확도는 공식(1-4)를 사용하여 계산되었으며, 그림 8(a)와 (b)에서 볼 수 있습니다. 혼동 행렬은 모델이 대부분의 활동에서 잘 훈련되었음을 보여줍니다. 결과는 유일하게 “평탄화” 활동이 주로 “접근” 및 “덤핑” 클래스로 잘못 분류된다는 것을 나타냈습니다. 반면, “접근”은 네 가지 활동을 “덤핑” 클래스로 잘못 분류하였고, “덤핑”은 “평탄화” 클래스로 하나의 활동만 잘못 분류했습니다. 다른 두 클래스인 “파기” 및 “유휴”도 매우 정확하게 훈련되었으며, 훈련 손실이 발생하지 않았습니다.

전체 결과는 제안된 연구 방법이 95.56%의 높은 정밀도와 95.29%의 정확도를 나타내었으며, 오차는 단 4.71%에 불과함을 보여줍니다.

### (8) Conclusion
본 연구는 두 개의 병렬 CNN 분기, LSTM 층 및 MLP 헤드로 구성된 두 개의 스트림 CNN-LSTM 네트워크를 사용하여 굴착기 활동을 인식합니다. 먼저, 다른 건설 현장에서 조종실 및 외부 비디오 프레임을 수집하고 비디오 프레임을 라벨링하고 동기화하여 입력 데이터를 준비합니다. 입력 데이터는 두 개의 스트림 CNN-LSTM 모델을 통해 처리되어 활동 인식을 위한 훈련 모델을 생성합니다. 다중 스트림 입력의 비디오 프레임에 대한 옵티컬 플로우를 추정하고, CNN의 두 분기가 비디오 프레임의 공간적 특징을 추출합니다. 두 개의 CNN 분기의 공간적 특징은 LSTM 층으로 전달되기 전에 결합됩니다. LSTM 층은 CNN 결합 출력의 시간적 특징을 학습하고, FC 층은 비디오 프레임의 시공간 특징을 기반으로 최종 예측을 하여 해당 굴착기 활동인 접근, 파기, 덤핑, 유휴 및 평탄화를 인식합니다. 제안된 방법은 두 개의 카메라로부터 비디오 데이터 입력을 동시에 사용하여 두 개의 스트림 CNN-LSTM 딥러닝 모델을 통해 활동 인식 정확도를 향상시켰습니다. 본 연구의 학습 및 테스트 데이터셋의 정확도는 각각 97.25% 및 95.29%였습니다.
본 연구는 제안된 방법이 굴착기 활동 인식을 효과적으로 향상시킬 수 있음을 성공적으로 입증하였으며, 이를 통해 안전성이 향상되고, 자원의 효율적 활용 및 토목 작업의 자동화가 가능해집니다.

### (9) Related Work (e.g. existing studies)

#### 1. 이 프로젝트를 수행하는 데 사용한 도구, 라이브러리, 블로그 또는 모든 문서
- **관련 논문**:
  - Roberts, D., and M. Golparvar-Fard. 2019. “End-to-end vision-based detection, tracking and activity analysis of earthmoving equipment filmed at ground level.” Autom. Constr., 105 (August 2018): 102811. Elsevier. https://doi.org/10.1016/j.autcon.2019.04.006.
  - Kim, J., and S. Chi. 2019a. “Action recognition of earthmoving excavators based on sequential pattern analysis of visual features and operation cycles.” Autom. Constr., 104 (March): 255–264. Elsevier. https://doi.org/10.1016/j.autcon.2019.03.025.

### (10) Division of Work and Roles of Team Members
- Member 1: 이가성 - Data collection, dataset processing, write-up, and YouTube clipping.
- Member 2: 구민제 - Code implementation, graph analysis, write-up, and YouTube recording.



