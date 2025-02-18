# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:39:26 2025

@author: Tugce
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy import stats

# Session state initialization
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None

def load_data(file):
    """CSV dosyasını uygun karakter kodlamasıyla yükler."""
    encodings = ['utf-8', 'ISO-8859-1', 'Windows-1254', 'latin1']
    for encoding in encodings:
        try:
            df = pd.read_csv(file, encoding=encoding)
            return df
        except UnicodeDecodeError:
            continue
    st.error("Dosya açılamadı")
    return None

def handle_missing_data(df):
    st.subheader("Eksik Veri Analizi ve İşleme")
    
    # Eksik veri analizi kısmı
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        "Eksik Değer Sayısı": missing,
        "Yüzde (%)": missing_percent.round(2)
    })
    
    with st.expander("Eksik Veri Tablosu"): 
        st.write(missing_df[missing_df['Eksik Değer Sayısı'] > 0]) 
    
    with st.expander("Eksik Veri Görselleştirme"): 
        # Eksik değerlerin sayısını görselleştirme
        missing_df_plot = missing_df[missing_df['Eksik Değer Sayısı'] > 0]  # Sadece eksik veri olan sütunları al
        if not missing_df_plot.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_df_plot['Eksik Değer Sayısı'].plot(kind='barh', color='salmon', ax=ax)
            ax.set_title("Eksik Değer Sayıları", fontsize=16)
            ax.set_xlabel("Eksik Değer Sayısı", fontsize=12)
            ax.set_ylabel("Sütunlar", fontsize=12)
            st.pyplot(fig)
        else:
            st.info("Veride eksik değer bulunmamaktadır.")
    
    # Eksik veri işleme yöntemleri
    with st.expander("Eksik Veri Temizleme"):
        handling_method = st.radio(
            "Eksik veri işleme yöntemi seçin:",
            ["Silme", "Ortalama ile Doldur", "Mod ile Doldur", "Medyan ile Doldur"]
        )
        
        if handling_method == "Silme":
            df_clean = df.dropna()
        elif handling_method == "Ortalama ile Doldur":
            df_clean = df.fillna(df.mean(numeric_only=True))
        elif handling_method == "Mod ile Doldur":
            df_clean = df.fillna(df.mode().iloc[0])
        elif handling_method == "Medyan ile Doldur":
            df_clean = df.fillna(df.median(numeric_only=True))
            
        st.write(f"İşlenmiş veri boyutu: {df_clean.shape}")
        return df_clean
    return df



def hypothesis_testing(df):
    """Hipotez testleri yapar."""
    st.subheader("📊 Hipotez Testleri")
    
    # Anlamlılık düzeyi (alpha)
    alpha = 0.05

    # Test seçenekleri ve gereken değişken tipleri
    test_options = {
        "T-Testi": ["numeric", "numeric"],
        "ANOVA": ["numeric", "categorical"],
        "Ki-Kare": ["categorical", "categorical"]
    }
    
    # Test tipi seçimi
    selected_test = st.selectbox("Test tipi seçin:", list(test_options.keys()))
    
    # Sütun tiplerini belirle
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Değişken seçimi
    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox("Birinci değişken:", df.columns)
    
    with col2:
        # İkinci değişkenin tipine göre uygun sütunları filtrele
        expected_var2_type = test_options[selected_test][1]
        if expected_var2_type == "numeric":
            available_vars = numeric_cols
        else:
            available_vars = cat_cols
        
        # var1'i listeden çıkararak uygun seçenekleri oluştur
        available_vars_filtered = [col for col in available_vars if col != var1]
        
        if not available_vars_filtered:
            st.error("İkinci değişken için yeterli seçenek yok. Lütfen farklı bir test tipi veya veri seti kullanın.")
            return
        else:
            var2 = st.selectbox("İkinci değişken:", available_vars_filtered)
    
    # Ek kontrol: ANOVA için seçilen kategorik değişken en az 3 gruba sahip olmalı
    if selected_test == "ANOVA":
        if df[var2].nunique() < 3:
            st.error("ANOVA için en az 3 grup gereklidir. Seçilen kategorik değişken yeterli gruba sahip değil.")
            return
    elif selected_test == "Ki-Kare":
        if df[var1].nunique() < 2 or df[var2].nunique() < 2:
            st.error("Ki-Kare testi için her iki kategorik değişken de en az 2 kategoriye sahip olmalıdır.")
            return
    
    # Testi çalıştır butonu
    if st.button("Testi Çalıştır", key="hypothesis_test_button"):
        try:
            # Değişken tiplerini kontrol et
            var1_type = "numeric" if var1 in numeric_cols else "categorical"
            var2_type_actual = "numeric" if var2 in numeric_cols else "categorical"
            
            expected_var1_type, expected_var2_type = test_options[selected_test]
            if var1_type != expected_var1_type or var2_type_actual != expected_var2_type:
                st.error(f"Hata: {selected_test} için uygun değişken tipleri seçilmedi. "
                         f"Beklenen tipler: {expected_var1_type} ve {expected_var2_type}.")
                return
            
            if selected_test == "T-Testi":
                # İki bağımsız örnek için T-Testi
                result = stats.ttest_ind(df[var1].dropna(), df[var2].dropna())
                conclusion = "Anlamlı fark vardır." if result.pvalue < alpha else "Anlamlı fark yoktur."
                st.success(f"T-Test Sonucu:\nT-Statistic = {result.statistic:.3f}\nP-Value = {result.pvalue:.3f}\n"
                           f"H0 {'reddedildi' if result.pvalue < alpha else 'kabul edildi'}: {conclusion}")
            
            elif selected_test == "ANOVA":
                # ANOVA için, kategorik değişken en az 3 farklı gruba sahip olmalı
                if df[var2].nunique() < 3:
                    st.error("ANOVA için en az 3 grup gereklidir. Seçilen kategorik değişken yeterli gruba sahip değil.")
                    return

                # Grupları oluşturuyoruz: her grup, var1'in var2'ye göre dağılımını içerir
                groups = [group.dropna() for name, group in df.groupby(var2)[var1] if len(group.dropna()) > 0]
                
                # Eğer grupların sayısı 3'ten az ise hata döndür
                if len(groups) < 3:
                    st.error("ANOVA için en az 3 grup gereklidir.")
                    return
                
                f_val, p_val = stats.f_oneway(*groups)
                conclusion = "Anlamlı fark vardır." if p_val < alpha else "Anlamlı fark yoktur."
                st.success(f"ANOVA Sonucu:\nF-Value = {f_val:.3f}\nP-Value = {p_val:.3f}\n"
                           f"H0 {'reddedildi' if p_val < alpha else 'kabul edildi'}: {conclusion}")
            
            elif selected_test == "Ki-Kare":
                # Ki-Kare testi için kontenjans tablosunu oluşturuyoruz.
                contingency_table = pd.crosstab(df[var1], df[var2])
                if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                    st.error("Ki-Kare testi için kontenjans tablosunda en az 2 satır ve 2 sütun olmalıdır.")
                    return
                chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
                conclusion = "Anlamlı fark vardır." if p < alpha else "Anlamlı fark yoktur."
                st.success(f"Ki-Kare Sonucu:\nChi2 = {chi2:.3f}\nP-Value = {p:.3f}\nDegrees of Freedom = {dof}\n"
                           f"H0 {'reddedildi' if p < alpha else 'kabul edildi'}: {conclusion}")
        
        except Exception as e:
            st.error(f"Hata oluştu: {str(e)}")


def statistical_analysis(df):
    """İstatistiksel analiz ve görselleştirme"""
    st.subheader("📈 İstatistiksel Analiz")
    
    with st.expander("Temel İstatistikler"):
        st.write(df.describe(include='all').T)
    
    with st.expander("Dağılım Analizi"):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                try:
                    skew_val = df[numeric_cols].skew().mean().round(2)
                    st.metric("Ortalama Çarpıklık", skew_val)
                except Exception as e:
                    st.error(f"Çarpıklık hesaplanamadı: {str(e)}")
            
            with col2:
                try:
                    kurt_val = df[numeric_cols].kurt().mean().round(2)
                    st.metric("Ortalama Basıklık", kurt_val)
                except Exception as e:
                    st.error(f"Basıklık hesaplanamadı: {str(e)}")
            
            selected_col = st.selectbox("Dağılım için sütun seçin", numeric_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[selected_col], kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Sayısal sütun bulunamadı!")
    
    with st.expander("Korelasyon Analizi"):
        numeric_df = df.select_dtypes(include=np.number)
        if len(numeric_df.columns) >= 2:
            corr_matrix = numeric_df.corr()
            fig = px.imshow(corr_matrix, 
                          color_continuous_scale='RdBu_r',
                          title="Korelasyon Matrisi")
            st.plotly_chart(fig)
        else:
            st.warning("Korelasyon için yeterli sayısal sütun yok!")

def preprocess_data(df):
    """Veriyi makine öğrenmesi için hazırlar"""
    # Kategorik kodlama
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def ml_recommender(df, target_column=None):
    """ML öneri sistemi"""
    st.subheader("🤖 Makine Öğrenmesi Önerileri")
    
    problem_type = "Belirsiz"
    recommendations = []
    evaluation_metrics = []
    
    if target_column and target_column in df.columns:
        target_dtype = str(df[target_column].dtype)
        
        # Problem tipi belirleme
        if df[target_column].nunique() == 2:
            problem_type = "İkili Sınıflandırma"
            recommendations = [
                "Lojistik Regresyon", 
                "Random Forest",
                "XGBoost",
                "Destek Vektör Makineleri"
            ]
            evaluation_metrics = ["Doğruluk", "F1-Skoru", "ROC-AUC"]
        
        elif df[target_column].nunique() > 2 and target_dtype == 'object':
            problem_type = "Çoklu Sınıflandırma"
            recommendations = [
                "Random Forest",
                "Gradient Boosting",
                "Yapay Sinir Ağları",
                "LightGBM"
            ]
            evaluation_metrics = ["Makro F1-Skoru", "Karmaşıklık Matrisi"]
        
        elif target_dtype in ['int64', 'float64']:
            problem_type = "Regresyon"
            recommendations = [
                "Lineer Regresyon",
                "Karar Ağacı Regresyonu",
                "Gradient Boosting Regresyon",
                "ElasticNet"
            ]
            evaluation_metrics = ["RMSE", "R² Skoru", "MAE"]
    else:
        problem_type = "Kümeleme"
        recommendations = [
            "K-Means", 
            "DBSCAN",
            "Hiyerarşik Kümeleme",
            "GMM"
        ]
        evaluation_metrics = ["Silhouette Skoru", "Elbow Yöntemi"]
    
    # Öneri gösterimi
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Problem Tipi:** `{problem_type}`")
        st.markdown("**Önerilen Modeller:**")
        for model in recommendations:
            st.markdown(f"- {model}")
    
    with col2:
        st.markdown("**Değerlendirme Metrikleri:**")
        for metric in evaluation_metrics:
            st.markdown(f"- {metric}")
        
        st.markdown("**Ön İşleme Adımları:**")
        steps = [
            "Eksik veri temizliği",
            "Kategorik kodlama",
            "Özellik ölçeklendirme",
            "Veri normalizasyonu"
        ]
        for step in steps:
            st.markdown(f"- {step}")
    
    return problem_type

def main():
    st.title("📊 Akıllı Veri Analiz Sistemi")
    
    uploaded_file = st.file_uploader("CSV dosyası yükleyin", type=["csv"])
    
    if uploaded_file:
        raw_df = load_data(uploaded_file)
        if raw_df is not None:
            st.session_state.raw_df = raw_df.copy()
            
            # Eksik veri işleme
            processed_df = handle_missing_data(raw_df)
            
            # Veri ön işleme
            processed_df = preprocess_data(processed_df)
            st.session_state.processed_df = processed_df
            
            # Analiz bölümleri
            hypothesis_testing(raw_df)  # Ham veri ile hipotez testleri
            statistical_analysis(processed_df)  # İşlenmiş veri ile analiz
            
            st.markdown("---")
            target_column = st.selectbox(
                "Hedef Değişken Seçin (ML için):",
                [None] + processed_df.columns.tolist()
            )
            
            problem_type = ml_recommender(processed_df, target_column)
            
            if target_column and problem_type != "Belirsiz":
                st.markdown("---")
                st.subheader("🔍 Özellik Önem Analizi")
                
                X = processed_df.drop(columns=[target_column])
                y = processed_df[target_column]
                
                if problem_type in ["İkili Sınıflandırma", "Çoklu Sınıflandırma"]:
                    model = RandomForestClassifier()
                else:
                    model = RandomForestRegressor()
                
                model.fit(X, y)
                
                importance_df = pd.DataFrame({
                    'Özellik': X.columns,
                    'Önem': model.feature_importances_
                }).sort_values('Önem', ascending=False)
                
                fig = px.bar(importance_df, 
                           x='Önem', 
                           y='Özellik',
                           title="Özellik Önem Sıralaması")
                st.plotly_chart(fig)

if __name__ == "__main__":
    main()

