## 주의사항
아파트와 연립 다세대를 구분하여 테스트할경우 df_type0, df_type1데이터를 만든 이후 ```주택유형_encoded```컬럼도 지운다, 계절 컬럼은 계약월을 이용해 만든 컬럼이고, 계약월이 더 중요하다!
## 아파트 모델 
### ** 유형별(역, 공원...등) 최적의 거리 선택 과정 **

###   1. 유형별 모든 거리를 다 넣었을 때
```python
model, (X_train, X_test, y_train, y_test) = train_eval_xgb(df_type0.drop(['보증금/월세금','월세금/면적','월세금(만원)','보증금(만원)','주택유형_encoded'],axis=1),'월부담액',plot_feature_importance=True,plot_shap=False)
```
![image](https://github.com/user-attachments/assets/f5b042fd-627b-43e0-adb4-65e0b6e40183)

####  성능 확인

<pre>
Test RMSE: 0.3346
Test MAE: 0.1997
Test R^2: 0.9296
</pre>

### 2. 거리관련 features 다 뺀 모델 성능(ver. 아파트)

최적의 거리를 찾아가는 과정

```python
model, (X_train, X_test, y_train, y_test) = train_eval_xgb(df_type0.drop(['보증금/월세금','월세금/면적','월세금(만원)','보증금(만원)','주택유형_encoded'] + group_features,axis=1),'월부담액',plot_feature_importance=True,plot_shap=False)
```
####  성능 확인
<pre>
Test RMSE: 0.3266
Test MAE: 0.1947
Test R^2: 0.9329
</pre>

### 모델 설명

**아래 모델은 각 유형별 한개씩**
즉 hospital의 5개* station*2개 * 3 * 3 * 3 * 4  = 1080개의 조합 중 최고의 성능을 보여주는 상위의 5개 조합을 보여줌.

itertools.product()는 **두 개 이상의 iterable의 모든 가능한 데카르트 곱(Cartesian Product)**을 구할 때 사용해. 쉽게 말하면, 모든 가능한 조합(순서 중요)을 구해주는 함수를 이용하겠다!

```python
import itertools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# (예시) 각 그룹 변수 리스트 정의
hospital_group = ['병원_10km내_개수', '병원_3km내_개수','병원_0.5km내_개수','병원_1km내_개수','병원_0.2km내_개수']
station_group = ['500m_이내_역_개수', '1km_이내_역_개수']
restaurant_group = ['식당_0.2km내_개수','식당_0.5km내_개수','식당_1km내_개수']
park_group = ['공원_300m_이내_개수','공원_500m_이내_개수','공원_800m_이내_개수']
police_group = ['파출소/지구대_0.5km내_개수','파출소/지구대_1km내_개수','파출소/지구대_3km내_개수']
university_group = ['대학_0.2km내_개수','대학_0.5km내_개수','대학_1km내_개수','대학_2km내_개수']
기본변수리스트 = ['전용면적(㎡)','자치구코드','건축년도','층','법정동코드','아파트_거래수',
'계약개월수','계약월','35-64대인구비','연립다세대_거래수','0-19대인구비','20-34대인구비','65세이상_인구비율','외국인_비율']  # 입지 변수 제외한 주요 변수 리스트

group_lists = [
    hospital_group,
    station_group,
    restaurant_group,
    park_group,
    police_group,
    university_group
]

all_combinations = list(itertools.product(*group_lists))

results = []

for combo in all_combinations:
    feature_list = list(combo)
    cols_to_use = 기본변수리스트 + feature_list
    df_sub = df_type0[cols_to_use].copy()
    df_sub['월부담액'] = df_type0['월부담액']

    # 모델 학습 및 평가 (함수 이용)
    model, (X_train, X_test, y_train, y_test) = train_eval_xgb(
        df_sub,
        target_col='월부담액',
        plot_feature_importance=False,
        plot_shap=False
    )
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append({'features': feature_list, 'r2': r2, 'mse': mse})

# 결과 DataFrame 생성
results_df = pd.DataFrame(results)

# r2(성능) 기준 내림차순 정렬, 상위 5개 조합 확인
top_results = results_df.sort_values('r2', ascending=False).head()
print(top_results)
```

### 결과_나온 최고의 조합
features: ['병원_1km내_개수', '500m_이내_역_개수', '식당_0.5km내_개수', '공원_800m_이내_개수', '파출소/지구대_1km내_개수', '대학_2km내_개수']

```python
cols_to_add = ['병원_1km내_개수', '500m_이내_역_개수', '식당_0.5km내_개수', '공원_800m_이내_개수', '파출소/지구대_1km내_개수', '대학_2km내_개수']
X_input = df_type0.drop(['보증금/월세금', '월세금/면적', '월세금(만원)', '보증금(만원)', '주택유형_encoded'] + group_features, axis=1)

X_input[cols_to_add] = df_type0[cols_to_add] 

model, (X_train, X_test, y_train, y_test) = train_eval_xgb(X_input, '월부담액', plot_feature_importance=True, plot_shap=False)
```
####  성능 확인
<pre>
Test RMSE: 0.3211
Test MAE: 0.1940
Test R^2: 0.9352
</pre>

![image](https://github.com/user-attachments/assets/60ea49b4-b147-4baa-839f-7b0d5eae146b)

## 결론

features: ['병원_1km내_개수', '500m_이내_역_개수', '식당_0.5km내_개수', '공원_800m_이내_개수', '파출소/지구대_1km내_개수', '대학_2km내_개수'] 

사용했을 때 
**Test R^2: 0.9329  -> Test R^2: 0.9352**  
성능이 유의미하게 증가함

## 연립다세대 모델 
아파트와 동일한 과정 반복
### 1. 유형별 모든 거리를 다 넣었을 때(ver.연립다세대)
```python
model, (X_train, X_test, y_train, y_test) = train_eval_xgb(df_type1.drop(['보증금/월세금','월세금/면적','월세금(만원)','보증금(만원)','주택유형_encoded'],axis=1),'월부담액',plot_feature_importance=True,plot_shap=False)
```
####  성능 확인
<pre>
  Test RMSE: 0.1673
  Test MAE: 0.1058
  Test R^2: 0.8388
</pre>
![image](https://github.com/user-attachments/assets/8c561ea8-0f64-4832-9824-08097bd288bd)

### 2. 거리관련 features 다 뺀 모델 성능(ver. 연립다세대)
```python
# 연립 다세대 basci model
model, (X_train, X_test, y_train, y_test) = train_eval_xgb(df_type1.drop(['보증금/월세금','월세금/면적','월세금(만원)','보증금(만원)','주택유형_encoded'] + group_features,axis=1),'월부담액',plot_feature_importance=True,plot_shap=False)
```
####  성능 확인
<pre>
Test RMSE: 0.1663
Test MAE: 0.1057
Test R^2: 0.8406
</pre>

### 아까와 같은 모델에 df_type1(연립 다세대)로만 바꿈
```python
import itertools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# (예시) 각 그룹 변수 리스트 정의
hospital_group = ['병원_10km내_개수', '병원_3km내_개수','병원_0.5km내_개수','병원_1km내_개수','병원_0.2km내_개수']
station_group = ['500m_이내_역_개수', '1km_이내_역_개수']
restaurant_group = ['식당_0.2km내_개수','식당_0.5km내_개수','식당_1km내_개수']
park_group = ['공원_300m_이내_개수','공원_500m_이내_개수','공원_800m_이내_개수']
police_group = ['파출소/지구대_0.5km내_개수','파출소/지구대_1km내_개수','파출소/지구대_3km내_개수']
university_group = ['대학_0.2km내_개수','대학_0.5km내_개수','대학_1km내_개수','대학_2km내_개수']
기본변수리스트 = ['전용면적(㎡)','자치구코드','건축년도','층','법정동코드','아파트_거래수',
'계약개월수','계약월','35-64대인구비','연립다세대_거래수','0-19대인구비','20-34대인구비','65세이상_인구비율','외국인_비율']  # 입지 변수 제외한 주요 변수 리스트

group_lists = [
    hospital_group,
    station_group,
    restaurant_group,
    park_group,
    police_group,
    university_group
]

all_combinations = list(itertools.product(*group_lists))

results = []

for combo in all_combinations:
    feature_list = list(combo)
    cols_to_use = 기본변수리스트 + feature_list
    df_sub = df_type1[cols_to_use].copy()
    df_sub['월부담액'] = df_type1['월부담액']

    # 모델 학습 및 평가 (함수 이용)
    model, (X_train, X_test, y_train, y_test) = train_eval_xgb(
        df_sub,
        target_col='월부담액',
        plot_feature_importance=False,
        plot_shap=False
    )
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    results.append({'features': feature_list, 'r2': r2, 'rmse': rmse})

# 결과 DataFrame 생성
results_df = pd.DataFrame(results)

# r2(성능) 기준 내림차순 정렬, 상위 5개 조합 확인
top_results = results_df.sort_values('r2', ascending=False).head()
print(top_results)
```

### 결과_나온 최고의 조합
features: ['병원_1km내_개수', '1km_이내_역_개수', '식당_0.5km내_개수', '공원_500m_이내_개수', '파출소/지구대_0.5km내_개수', '대학_1km내_개수']

```python
cols_to_add = ['병원_1km내_개수', '1km_이내_역_개수', '식당_0.5km내_개수', '공원_500m_이내_개수', '파출소/지구대_0.5km내_개수', '대학_1km내_개수']
X_input = df_type1.drop(['보증금/월세금', '월세금/면적', '월세금(만원)', '보증금(만원)', '주택유형_encoded'] + group_features, axis=1)

X_input[cols_to_add] = df_type1[cols_to_add] 

model, (X_train, X_test, y_train, y_test) = train_eval_xgb(X_input, '월부담액', plot_feature_importance=True, plot_shap=False)
```
####  성능 확인
<pre>
Test RMSE: 0.1599
Test MAE: 0.1050
Test R^2: 0.8528
</pre>

![image](https://github.com/user-attachments/assets/1aa7c968-c508-4655-92bd-f57b3e76ee64)

## 결론

features: ['병원_1km내_개수', '1km_이내_역_개수', '식당_0.5km내_개수', '공원_500m_이내_개수', '파출소/지구대_0.5km내_개수', '대학_1km내_개수']

사용했을 때 
**Test R^2: 0.8406  -> Test R^2: 0.8528**  
성능이 유의미하게 증가함








## 최종모델 feature importance 분석

#### < feature importance 순위표 ver.아파트>
| 순위 | 아파트           |  
| -- | ------------ | 
| 1  | 전용면적         |
| 2  | 자치구 코드       | 
| 3  | 건축년도         | 
| 4  | 층            | 
| 5  | 법정동 코드       |
| 6  | 800m 이내 공원 수 | 
| 7  | 500m 이내 역 개수 | 
<해석>
1. 전용면적  -> 아파트는 면적에 따른 가격 체계가 명확(큰 면적- 비쌈)
2. 자치구 코드 -> 예를 들어 월세: 강남 > 노원 
3. 건축년도 -> 신축 아파트는 브랜드,편의성이 좋고 이에 따른 프리미엄이 붙음
4. 층 -> 같은 단지라도 층수에 따라 월세도 차이 남(전망, 소음...)
5. 법정동 코드 -> 자치구 코드와 동일
6.  주변 환경(공원,역) -> 고급 아파트 수요층은 쾌적한 환경에 더 민감함(역세권/ 숲세권)

#### < feature importance 순위표 ver.연립다세대>  
| 순위 | 연립다세대           |  
| -- | ------------ | 
| 1  | 건축년도             |
| 2  | 아파트_거래수        |
| 3  | 전용면적             |
| 4  | 자치구 코드          |
| 5  | 계약개월수           |
| 6  | 65세 이상 인구 비율  |
| 7  | 계약월              |
<해석>
1. 건축년도  -> 연립다세대는 아파트에 비해 노후 비율이 높아서 가격에 더 민감하게 반응
2. 아파트_거래수 -> 아파트와 대체재 관계이거나 해당 지역 부동산 시장의 열기가 반영된 거라 볼 수도 있겠음
3. 전용면적 -> 당연히 면적은 중요하지만 아파트보단 덜함
4. 자치구 코드 -> 마찬가지
5. 계약개월수 -> 개별 임대인에 따라 계약기간도 유동적이고 계약기간이 짧으면 월세가 비싸질 수 있다
6. 65세 이상 인구 비율 -> 연립은 고령층/저속득층 주거지로 많이 분포
7. 계약월 -> 학기 시작등 이사철에 가격이 더 비싸진다.
   
#### < 정리 : 아파트 VS 연립 다세대 비교>
| 항목       | 아파트         | 연립 다세대              |
| -------- | ----------- | ------------------- |
| 주요 영향    | 입지 + 물리적 조건 | 시점 + 사회 환경 + 시장 분위기 |
| 거래량 영향   | 없음 (독립적)         | 아파트 거래수 영향 있음       |
| 공원/역 중요도 | 중요          | 중요도 낮음              |
| 고령 인구 영향 | 없음          | 있음                  |

