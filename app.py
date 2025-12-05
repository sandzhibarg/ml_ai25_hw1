import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_resource
def load_model():
    with open('model_pipeline.pickle', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
    return df_train

def preprocess_input(data, artifacts):
    """
    препроцессинг входных данных для предсказания
    """
    df = data.copy()
    
    # создаем признаки
    current_year = artifacts['current_year']
    df['car_age'] = current_year - df['year']
    df['km_per_year'] = df['km_driven'] / (df['car_age'] + 1)
    df['power_per_liter'] = df['max_power'] / (df['engine'] / 1000)
    df['power_per_liter'] = df['power_per_liter'].replace([np.inf, -np.inf], np.nan)
    df['brand'] = df['name'].str.split().str[0]
    df['is_premium'] = df['brand'].isin(artifacts['premium_brands']).astype(int)
    df['year_x_power'] = df['year'] * df['max_power']

    df = df.drop('name', axis=1)
    
    # заполняем пропуски
    for col, val in artifacts['medians_num'].items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    
    # разделяем на числовые и категориальные
    num_columns = artifacts['num_columns']
    cat_columns = artifacts['cat_columns']
    
    # one-hot encoding
    cat_encoded = artifacts['encoder'].transform(df[cat_columns])
    cat_encoded_df = pd.DataFrame(
        cat_encoded, 
        columns=artifacts['encoder'].get_feature_names_out(cat_columns)
    )
    
    # объединяем
    X = pd.concat([df[num_columns].reset_index(drop=True), 
                   cat_encoded_df.reset_index(drop=True)], axis=1)
    
    # скейлинг
    X_scaled = artifacts['scaler'].transform(X)
    
    return X_scaled

def main():
    st.set_page_config(page_title="предсказание цены авто", layout="wide")
    st.title("Предсказание стоимости автомобиля")
    
    try:
        artifacts = load_model()
    except FileNotFoundError:
        st.error("файл model_pipeline.pickle не найден!")
        return
    
    # меню
    page = st.sidebar.selectbox(
        "выберите раздел",
        ["EDA и графики", "предсказание цены", "веса модели"]
    )
    
    if page == "EDA и графики":
        st.header("Exploratory data analysis (EDA)")
        
        df = load_data()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("распределение цены")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df['selling_price'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_xlabel('цена')
            ax.set_ylabel('частота')
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.subheader("цена по типу топлива")
            fig, ax = plt.subplots(figsize=(8, 5))
            df.boxplot(column='selling_price', by='fuel', ax=ax)
            plt.suptitle('')
            ax.set_xlabel('тип топлива')
            ax.set_ylabel('цена')
            st.pyplot(fig)
        
        # корр матрица
        st.subheader("матрица корреляций")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)
        
    elif page == "предсказание цены":
        st.header("Предсказание цены автомобиля")
        
        input_method = st.radio("способ ввода данных", ["ручной ввод", "загрузить csv"])
        
        if input_method == "ручной ввод":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                name = st.text_input("название авто", "Maruti Swift VDi")
                year = st.number_input("год выпуска", 2000, 2024, 2015)
                km_driven = st.number_input("пробег (км)", 0, 500000, 50000)
                fuel = st.selectbox("тип топлива", ["Diesel", "Petrol", "CNG", "LPG"])
            
            with col2:
                seller_type = st.selectbox("тип продавца", ["Individual", "Dealer", "Trustmark Dealer"])
                transmission = st.selectbox("коробка передач", ["Manual", "Automatic"])
                owner = st.selectbox("владелец", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
                seats = st.number_input("кол-во мест", 2, 14, 5)
            
            with col3:
                mileage = st.number_input("расход (kmpl)", 5.0, 35.0, 18.0)
                engine = st.number_input("объем двигателя (cc)", 500, 5000, 1200)
                max_power = st.number_input("мощность (bhp)", 30.0, 400.0, 80.0)
            
            if st.button("предсказать цену"):
                input_data = pd.DataFrame([{
                    'name': name, 'year': year, 'km_driven': km_driven, 
                    'fuel': fuel, 'seller_type': seller_type, 
                    'transmission': transmission, 'owner': owner, 
                    'seats': seats, 'mileage': mileage, 
                    'engine': engine, 'max_power': max_power
                }])
                
                X_scaled = preprocess_input(input_data, artifacts)
                pred_log = artifacts['model'].predict(X_scaled)
                pred_price = np.expm1(pred_log)[0]
                
                price_formatted = f"{pred_price:,.0f}".replace(',', ' ')
                st.success(f"предсказанная цена: **{price_formatted}** рупий")
        
        else:
            uploaded_file = st.file_uploader("загрузите csv файл", type=['csv'])
            
            if uploaded_file is not None:
                input_df = pd.read_csv(uploaded_file)
                st.write("загруженные данные:", input_df.head())
                
                if st.button("предсказать"):
                    X_scaled = preprocess_input(input_df, artifacts)
                    pred_log = artifacts['model'].predict(X_scaled)
                    predictions = np.expm1(pred_log)
                    
                    result_df = input_df.copy()
                    result_df['predicted_price'] = predictions
                    st.write("результаты:", result_df)
                    
                    csv = result_df.to_csv(index=False)
                    st.download_button("скачать результаты", csv, "predictions.csv", "text/csv")
    
    else:
        st.header("Веса модели")
        
        feature_names = artifacts['feature_names']
        coefficients = artifacts['model'].coef_
        
        coef_df = pd.DataFrame({
            'признак': feature_names,
            'коэффициент': coefficients,
            'abs_коэффициент': np.abs(coefficients)
        }).sort_values('abs_коэффициент', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 12))
        colors = ['red' if c < 0 else 'green' for c in coef_df['коэффициент']]
        ax.barh(coef_df['признак'], coef_df['коэффициент'], color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('коэффициент')
        ax.set_title('важность признаков (коэффициенты ridge регрессии)')
        ax.grid(alpha=0.3, axis='x')
        st.pyplot(fig)
        
        st.subheader("топ-10 по важности")
        top_10 = coef_df.nlargest(10, 'abs_коэффициент')[['признак', 'коэффициент']]
        st.dataframe(top_10)

if __name__ == "__main__":
    main()
