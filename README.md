# My First Repository

깃허브에 처음 올려보는 저장소입니다!  
앞으로 코딩 연습, 공부, 프로젝트를 올릴 예정입니다.

#### 아파트 모델 

# 아파트에 관련한 feature importance 선정

```python
model, (X_train, X_test, y_train, y_test) = train_eval_xgb(df_type0.drop(['보증금/월세금','월세금/면적','월세금(만원)','보증금(만원)','주택유형_encoded'],axis=1),'월부담액',plot_feature_importance=True,plot_shap=False)
```
![image](https://github.com/user-attachments/assets/f5b042fd-627b-43e0-adb4-65e0b6e40183)

## 아파트에 관한 성능 파악
<pre>
Test RMSE: 0.3346
Test MAE: 0.1997
Test R^2: 0.929
</pre>

# 연립다세대에 관련한 feature importance 선정
```python
# 연립다세대대에 관련한 feature importance 선정
model, (X_train, X_test, y_train, y_test) = train_eval_xgb(df_type1.drop(['보증금/월세금','월세금/면적','월세금(만원)','보증금(만원)','주택유형_encoded'],axis=1),'월부담액',plot_feature_importance=True,plot_shap=False)
```
![image](https://github.com/user-attachments/assets/2057616d-2c88-4094-b7f0-47feb51b6119)
<pre>
  Test RMSE: 0.1673
Test MAE: 0.1058
Test R^2: 0.8388
</pre>

## 아파트와 연립다세대를 분리해서 하는게 ~~~ 해서 유의미한 거 같다!
### 그럼 이제 유형별(역,병원...등) 최적의 거리를 찾자

### 찾기 위한 모델 만듬( ver.아파트)
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

  ### 모델 설명
  ##### 
  **위 모델은 각 유형별 한개씩**
즉 hospital의 5개* station*2개 * 3 * 3 * 3 * 4  = 1080개의 조합 중 최고의 성능을 보여주는 상위의 5개 조합을 보여줌.

<pre>
  결과:
  features: ['병원_1km내_개수', '500m_이내_역_개수', '식당_0.5km내_개수', '공원_800m_이내_개수', '파출소/지구대_1km내_개수', '대학_2km내_개수']
r2: 0.9350, rmse: 0.3215
</pre>

### 아파트에 들어가면 좋을 피쳐들 더 정제하기??
```python
cols_to_add = ['병원_1km내_개수', '500m_이내_역_개수', '식당_0.5km내_개수', '공원_800m_이내_개수', '파출소/지구대_1km내_개수', '대학_2km내_개수']
X_input = df_type0.drop(['보증금/월세금', '월세금/면적', '월세금(만원)', '보증금(만원)', '주택유형_encoded'] + group_features, axis=1)

X_input[cols_to_add] = df_type0[cols_to_add] 

model, (X_train, X_test, y_train, y_test) = train_eval_xgb(X_input, '월부담액', plot_feature_importance=True, plot_shap=False)
```
![image](https://github.com/user-attachments/assets/ce5a47e4-5929-4f7c-b700-588d1c60747e)
