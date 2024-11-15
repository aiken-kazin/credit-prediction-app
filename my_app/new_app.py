import streamlit as st
import pickle
import numpy as np
import pandas as pd


st.title("Прогнозирование Просрочки по Кредиту")


with open('/app/my_app/model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('/app/my_app/kmeans_model.pkl', 'rb') as file:
    kmeans = pickle.load(file)


st.sidebar.header("Введите данные для прогноза:")

input1 = st.sidebar.number_input("Input 1 (e.g., days_90_queries)", value=0.0)
input2 = st.sidebar.number_input("Input 2 (e.g., ageolddate_f7)", value=0.0)
input3 = st.sidebar.number_input("Input 3 (e.g., f41_my)", value=0.0)
input4 = st.sidebar.number_input("Input 4 (e.g., age2)", value=0.0)
input5 = st.sidebar.number_input("Input 5 (e.g., active_crd_amount_to_limit)", value=0.0)
input6 = st.sidebar.number_input("Input 6 (e.g., sumlastvalue_7_f50)", value=0.0)
input7 = st.sidebar.number_input("Input 7 (e.g., avg_active_credit_limit)", value=0.0)
input8 = st.sidebar.number_input("Input 8 (e.g., total_amount)", value=0.0)
input9 = st.sidebar.number_input("Input 9 (e.g., max_del_history)", value=0.0)
input10 = st.sidebar.number_input("Input 10 (e.g., overdue_credit_contract)", value=0.0)



if st.sidebar.button("Сделать прогноз"):
    clus = kmeans.predict([[input2]])[0]


    data = [input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, float(clus)]
    data_df = pd.DataFrame([data], 
                           columns=['days_90_queries', 'ageolddate_f7', 'f41_my', 'age2',
                                    'active_crd_amount_to_limit', 'sumlastvalue_7_f50',
                                    'avg_active_credit_limit', 'total_amount', 
                                    'max_del_history', 'overdue_credit_contract', 'cluster'])

   
    try:
        predictions = model.predict(data_df)
        result = "Просрочка вероятна" if predictions[0] == 1 else "Просрочка маловероятна"
        st.success(f"Результат прогноза: {result}")
    except Exception as e:
        st.error(f"Произошла ошибка при прогнозировании: {e}")


