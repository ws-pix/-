# My First Repository

깃허브에 처음 올려보는 저장소입니다!  
앞으로 코딩 연습, 공부, 프로젝트를 올릴 예정입니다.

#### 아파트 모델 

# 아파트에 관련한 feature importance 선정

```python
model, (X_train, X_test, y_train, y_test) = train_eval_xgb(df_type0.drop(['보증금/월세금','월세금/면적','월세금(만원)','보증금(만원)','주택유형_encoded'],axis=1),'월부담액',plot_feature_importance=True,plot_shap=False)

![image](https://github.com/user-attachments/assets/f5b042fd-627b-43e0-adb4-65e0b6e40183)

