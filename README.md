# An Overview of Model-Agnostic Interpretation Methods

## 개요
- 모델의 종류에 관계 없는 해석 방법

## 해석 방법
### 1. Partial Dependence Plot (PDP)

- $X = X_s \cup X_c$  [S: Selected(관심 대상)인 feature set), C: 그 이외(marginal) feature set]
- $\hat{f}_{X_s}(X_s)=E_{X_c}[\hat{f}(X_s, X_c)] = \int{\hat{f}(X_s, X_c)\hat{f}_{X_c}(X_c)dX_c}={1\over n}\sum_i^n{\hat{f}(X_s, x_{ic})}$ (연속적인 변수) → (1-1)
- $\hat{f}_{X_s}(X_s)=E_{X_c}[\hat{f}(X_s, X_c)] = \sum{\hat{f}(X_s, X_c)\hat{f}_{X_c}(X_c)}={1\over n}\sum_i^n{\hat{f}(X_s, x_{ic})}$ (이산적인 변수) → (1-2)
- 예시
    - $X$=(키, 나이), $X_s$=(나이), $x_s$=30, $X_C$=(키), n: 데이터 개수
    
        | X | 150 | 160 | 170 | 180 |
        | --- | --- | --- | --- | --- |
        | $p(X_c)$ | 1/10 | 3/10 | 5/10 | 1/10 |
        | $f(X_s = 30, X_c)$ | 200 | 250 | 300 | 350 |
    - $\hat{f}(X_s = 30)={1 \over 20}*200+{3 \over 10}*250+{5 \over 10}*300+{1 \over 10}*350=270$→ (2)
    - 하지만, $p(X_c)$는 주어지지 않기 때문에 (2)는 (1)처럼 몬테카를로(샘플링) 방법으로 계산
    - 다른 나이값(..., 28, 29, 31, 32, ...)에 대해서도 마찬가지로 계산

### 2. Individual Conditional Expectation (ICE)

- 각 관측치의 dependence를 개별적으로 시각화
- 관심변수 $X_s$가 변화될 때 어떻게 예측값(target)이 변하는 지, 모든 train 데이터에 대해 보여줌
- 즉, PDP는 ICE의 각 line들의 평균
- 장점
    - 해석이 직관적이고 명확
    - PDP처럼 기댓값을 취하지 않기 때문에, 각 관측치에 대응되는 선을 그릴 수 있음
- 단점
    - 관측치 수가 많을 경우 너무 조밀하여 제대로 파악하기 어려울 수 있음
    - 그 외 단점은 PDP와 유사

### 3. Permutation Feature Importance

- 데이터 행렬 $X$($n$ x $p$)로 사전 학습된 모델 $\hat{f}$가 있을 때 (데이터 n개, 특성치 p개)
- 확인하고자 하는 특성치(j열)의 순서만을 셔플한 새 데이터 행렬 $X^{perm}$을 만들어 baseline과의 성능 차이를 feature importance($FI^j$)로 사용 ($j=1,2,...,p$)
- $FI^j=e^{perm}-e^{base}=L(y,\hat{f}(X^{perm}))-L(y,\hat{f}(X))$
- 또는 $FI^j=e^{perm}/e^{base}$
- 장점
    - $FI^j=e^{perm}/e^{base}$을 오류 비율로 정의할 경우 $FI$값이 정규화되어 서로 다른 문제끼리 비교 가능
    - Baseline 모델로 많이 쓰이는 Tree 계열 모델에 손쉽게 사용 가능
- 단점
    - 레이블이 있는 지도 학습에서만 사용 가능
    - Permute 시 비 현실적인 데이터의 발생 가능성
    - 무작위로 섞기 때문에 성능이 일관적이지 않음(Inconsistent)

### 4. LIME (Local Interpretable Model-agnostic Explanation)

- 모델 전체의 해석력보다 단일 관측치(혹은 데이터셋의 일부분)에 대한 모델 예측값 해석에 대해 초점을 둠
- Surrogate model: 원래 모델이 복잡하기 때문에 해석하기 쉬운 대리(surrogate) 모델을 보고 해석
    
    ![LIME의 핵심 아이디어](https://raw.githubusercontent.com/marcotcr/lime/master/doc/images/lime.png)
    
- 분홍색 부분은 모델 전체의 분류 기준을 나타냄 → 복잡하기 때문에 사람이 해석하기 쉽지 않음
- 빨간색으로 굵게 표시된 십자는 전체 데이터 셋 중 단일 데이터
- 시점을 국소적으로 축소하여, 타겟 단일 데이터와 주변에 분포된 다른 데이터들을 분류하는 해석하기 쉬운 모델(선형 회귀, 결정 트리)로 재해석
- 재해석 모델 $g$를 찾기 위한 최적화 문제를 다음과 같이 정의
    
    $\xi(x)=\argmin\limits_{g \in G} \mathcal{L}(f,g,\pi_x)+\Omega(g)$
    
    - $f$: 학습한 모델(black-box model)
    - $g$: 설명 가능한 쉬운 모델(선형 회귀, 결정 트리)
    - $\pi_x$: 타겟 주변에 분포된 다른 데이터들과의 유사도 측도($\pi_x=\exp(-{Distance(x,z)^2 \over \sigma^2})$
    - $\Omega(g)$: 모형의 복잡도(선형 회귀의 경우 $\beta$가 많은 모델, 결정 트리의 경우 깊이가 깊은 경우 복잡도가 높음)
- 설명이 가능하도록 쉬운 모델 $g$를 선택했으나, 기존 복잡한 모델 $f$의 형태에 맞춰진 복잡한 입력 $x$을 동일하게 사용할 수 없음
- 이를 위해 $x'$를 구하는데, $x'$는 입력 $x$를 feature단위로 나눌 수 있도록 만들어 놓은 상태.
- 예를 들어, 이미지의 경우 아래와 같이 조각들로 나눔(segmentation)
    
    ![이미지 segmentation](https://miro.medium.com/max/361/1*pmPM8BTEZH7rqKC9qwb7Lg.png)
    
- $x'$의 features 중 랜덤하게 선택한 것을 $z$라고 정의
- 예를 들어, 이미지의 경우 아래와 같이 나누어진 조각들 중 일부를 랜덤하게 선택
    
    ![랜덤 샘플링](https://miro.medium.com/max/875/1*wHLeYXmhK7h34yKb4Iq56A.png)
    
- 텍스트의 경우, "i love the dog"라는 입력이 있으면 "i love", "i the dog" 등과 같이 각 단어를 feature로 하여 랜덤하게 선택
- 시계열의 경우 이미 features 단위로 나누어져 있으므로 랜덤하게 선택(perturb)
- 각 feature의 선택 여부를 나타낸 것을 $z' \in {\{0, 1\}}^{d'}$
- $g$는 $z'$를 입력으로 사용
- 결과적으로 복잡한 모델 $f(z)$와 설명 가능한 모델 $g(z')$의 결과가 유사하길 원하기 때문에 Local-aware loss를 다음과 같이 정의
    
    $\mathcal{L}(f,g,\pi_x)=\sum\limits_{z,z'\in \mathcal{Z}}\pi_x(z)(f(z)-g(z'))^2$
    
- 학습된 $g$를 통해 features들의 중요도를 보고 모델의 결정을 설명
- 장점
    - Global한 해석이 아닌 개별 데이터에 대한 local 해석력 제공
    - Perturbation (features를 랜덤하게 선택하는 과정)
    - SHAP 보다 계산량이 적음
- 단점
    - 데이터 분포가 local에서도 매우 비선형적이면, local에서 선형성을 가정하는 LIME의 설명력에 한계를 갖게 됨
    - $\pi_x=\exp(-{Distance(x,z)^2 \over \sigma^2})$의 하이퍼파라미터에 따라서 샘플링 성능이 불안정함(Inconsistent)
    - Data 종류(이미지, 텍스트, ...)에 따라 perturbation 방식이 달라지므로 model-agnostic 방법이 갖는 유연성이 다소 퇴색됨

### 5. SHAP (Shapley Additive exPlanations)

- LIME과 기본적인 아이디어가 유사
- 복잡한 모델 $f$를 설명할 수 있는 단순한 모델 $g$를 찾는 것이 목표
- Additive Feature Attribute methods
    - Binary(0 or 1) 변수의 선형 결합으로 이루어진 Explanation 함수($g$)
        
        $g(z')=\varphi_0 + \sum_{i=1}^{M} \varphi_iZ'_i, where z' \in \{0,1\}^M, \varphi_i \in R$
        
- 게임에 참여한 인원들의 기여도를 평가하는 게임이론인 Shapley value를 사용함
- Shapley value는 게임에 참여한 모든 인원에 대한 조합 중에서, 특정 인원이 포함된 조합과 빠진 조합의 점수 차이를 의미함
- 예를 들어 A, B, C로 구성된 팀이 있고, A의 shapley value를 구하는 것이 목적이라고 하자
- 집합 $\{A,B,C\}$의 부분 집합 중 $\{A,B\}$의 게임 점수와 $\{B\}$의 게임 점수의 차이를 구해 A의 기여도를 구할 수 있음
- A가 포함된 모든 부분 집합에 대해 기여도를 구하고 가중 평균을 구함
- Shapley value, $\varphi_i = \sum\limits_{S\subseteq N/\{i\}}{|S|!(n-|S|-1)! \over n!}(f(S \cup \{i\})-f(S))$
- 위의 수식 중 $N$은 전체 부분 집합, $S$는 그 중 특정 플레이어(A)가 빠진 부분 집합을 의미
- Shapley value는 feature attribution이 원하는 3가지 조건을 만족함
    - Local accuracy: 각 팀원의 점수를 합하면, 전체 점수가 되어야 함
    - Missingness: 게임에 참여하지 않았다면, 개인 점수는 0점이어야 함
    - Consistency: 매번 같은 방식으로 게임했다면, 결과는 같아야 함
- 장점
    - 수학적 이론을 기반으로 Explantion model이 필요한 특징들을 만족함
    - local 뿐만 아니라 각 feature 별 SHAP mean으로 global explanation도 얻을 수 있음
- 단점
    - 모든 순열 조합을 계산해야 하기 때문에 느림 (이를 극복하기 위한 방법들이 제시되었음)
    - LIME과 동일하게 데이터의 유형 별로 다른 전처리가 필요함

---

출처

- [http://dmqm.korea.ac.kr/activity/seminar/297](http://dmqm.korea.ac.kr/activity/seminar/297)
- [https://christophm.github.io/interpretable-ml-book/](https://christophm.github.io/interpretable-ml-book/)
- [https://yjjo.tistory.com/3](https://yjjo.tistory.com/3)
- [https://towardsdatascience.com/interpretable-machine-learning-for-image-classification-with-lime-ea947e82ca13](https://towardsdatascience.com/interpretable-machine-learning-for-image-classification-with-lime-ea947e82ca13)