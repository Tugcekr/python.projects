# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:52:28 2025

@author: Tugce
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats
from scipy.stats import (shapiro, kstest, norm, ttest_1samp, ttest_ind, ttest_rel, levene, chi2_contingency, f_oneway)
from contextlib import redirect_stdout
import statsmodels.api as sm
from statsmodels.formula.api import ols


#yardımcı fonksiyon
def normalite_testleri(veri, secilen_sutun):
    #Shapiro ve kolmogorov smirnov testlerini uygular. kolmogorov daha detaylı test eder büyük veriler için daha uygundur
    #kolmogorov normallik dışınıda test edebilir. uniform vb. onlarada bakılır. shapiro genelde daha küçük veri setlerinde etkili.
    shapiro_stat, shapiro_p = shapiro(veri[secilen_sutun])
    ks_stat, ks_p = kstest(veri[secilen_sutun], 'norm')
    return shapiro_p, ks_p

# Sidebar – Sayfa Seçimi (Sıralama: Keşifsel Veri Analizi, Z Testi, T Testi, ANOVA & Ki-Kare Testleri, ML Öneri Sistemi)
sayfa_secimleri = ["Keşifsel Veri Analizi", "Z Testi", "T Testi", "ANOVA & Ki-Kare Testleri", "ML Öneri Sistemi"]
st.sidebar.title("Sayfa seçimi")
sayfa = st.sidebar.radio("Lütfen bir sayfa seçin:", sayfa_secimleri)

# -------------------- KEŞİFSEL VERİ ANALİZİ --------------------
if sayfa == "Keşifsel Veri Analizi":
    st.header("Keşifsel Veri Analizi")

    #yükleme kısmı
    dosya_turu = ["csv", "txt", "xls", "xlsx"]
    yuklenen_veri = st.file_uploader("Veri setinizi CSV, TXT veya Excel formatında yükleyiniz.", type=dosya_turu)
    if yuklenen_veri is not None:
        dosya_adi = yuklenen_veri.name.lower()
        if "csv" in dosya_adi:
            try:
                veri = pd.read_csv(yuklenen_veri, encoding='utf-8')
            except UnicodeDecodeError:
                veri = pd.read_csv(yuklenen_veri, encoding='ISO-8859-1')
        elif "txt" in dosya_adi:
            try:
                veri = pd.read_csv(yuklenen_veri, delimiter='\t', encoding='utf-8') #hata almamak için belirtildi.
            except UnicodeDecodeError: #karakter kodlaması hata verdiğinde kullanırız.
                veri = pd.read_csv(yuklenen_veri, delimiter='\t', encoding='ISO-8859-1')
        elif "xls" in dosya_adi or "xlsx" in dosya_adi:
            veri = pd.read_excel(yuklenen_veri)
        st.write("Veri Örneği:")
        st.write(veri.head())


        st.subheader("Yüklenen Veri Seti")
        st.write(veri)
        st.markdown(f"*Veri Seti:* {veri.shape[0]} satır, {veri.shape[1]} sütun.")

        # VERİ TEMİZLEME VE EKSKİ DEĞER İMPUTASYONU
        st.subheader("Veri Temizleme İşlemleri")
        if st.checkbox("Veri Temizleme Seçeneklerini Göster"):
            st.write("**Eksik Değerler:**")
            eksik_degerler = veri.isnull().sum()
            st.write(eksik_degerler)

            if st.checkbox("Eksik Değerleri Olan Satırları Kaldır"):
                veri = veri.dropna()
                st.write("Eksik değerleri olan satırlar kaldırıldı.")

            if st.checkbox("Eksik Değerleri Belirli Bir Değerle Doldur"):
                doldur_degeri = st.text_input("Eksik değerler için doldurulacak değeri giriniz (örn: 0 veya 'Bilinmiyor'):")
                if st.button("Belirli Değer ile Doldur"):
                    veri = veri.fillna(doldur_degeri)
                    st.write("Eksik değerler dolduruldu.")

            if st.checkbox("Yinelenen Satırları Kaldır"):
                veri = veri.drop_duplicates()
                st.write("Yinelenen satırlar kaldırıldı.")

            st.markdown("---")
            st.subheader("Eksik Değer İmputasyonu")
            # Sayısal değişkenler için imputasyon
            if st.checkbox("Eksik Değer İmputasyonu (Sayısal)"):
                impute_method = st.selectbox("İmpute Yöntemi Seçiniz (Sayısal):", 
                                             ["Belirli Değer", "Ortalama", "Medyan", "Mod"])
                if impute_method == "Belirli Değer":
                    doldur_degeri_sayi = st.text_input("Sayısal eksik değerler için doldurulacak değeri giriniz:")
                    if st.button("Sayısal Değerleri Doldur", key="num_fill"):
                        for col in veri.select_dtypes(include=['float64', 'int64']).columns:
                            try:
                                veri[col] = veri[col].fillna(float(doldur_degeri_sayi))
                            except:
                                st.write(f"{col} sütunu için dönüştürme hatası!")
                        st.write("Eksik sayısal değerler dolduruldu.")
                elif impute_method == "Ortalama":
                    for col in veri.select_dtypes(include=['float64', 'int64']).columns:
                        veri[col] = veri[col].fillna(veri[col].mean())
                    st.write("Eksik sayısal değerler ortalama ile dolduruldu.")
                elif impute_method == "Medyan":
                    for col in veri.select_dtypes(include=['float64', 'int64']).columns:
                        veri[col] = veri[col].fillna(veri[col].median())
                    st.write("Eksik sayısal değerler medyan ile dolduruldu.")
                elif impute_method == "Mod":
                    for col in veri.select_dtypes(include=['float64', 'int64']).columns:
                        veri[col] = veri[col].fillna(veri[col].mode()[0])
                    st.write("Eksik sayısal değerler mod ile dolduruldu.")

            # Kategorik değişkenler için imputasyon
            if st.checkbox("Eksik Değer İmputasyonu (Kategorik)"):
                impute_method_kat = st.selectbox("Kategorik İmpute Yöntemi:", ["Belirli Değer", "Mod"])
                if impute_method_kat == "Belirli Değer":
                    doldur_degeri_kat = st.text_input("Kategorik eksik değerler için doldurulacak değeri giriniz:")
                    if st.button("Kategorik Değerleri Doldur", key="cat_fill"):
                        for col in veri.select_dtypes(include='object').columns:
                            veri[col] = veri[col].fillna(doldur_degeri_kat)
                        st.write("Eksik kategorik değerler dolduruldu.")
                elif impute_method_kat == "Mod":
                    for col in veri.select_dtypes(include='object').columns:
                        veri[col] = veri[col].fillna(veri[col].mode()[0])
                    st.write("Eksik kategorik değerler mod ile dolduruldu.")
            st.markdown("---")

        # Genel veri bilgileri
        if st.checkbox("Veri Hakkında Genel Bilgileri Göster"):
            bilgi_buffer = io.StringIO()
            with redirect_stdout(bilgi_buffer):
                veri.info()
            st.text(bilgi_buffer.getvalue())

        if st.checkbox("Değişken Adlarını Göster"):
            st.write("Tüm Değişkenler:", veri.columns.tolist())

        if st.checkbox("İlk Beş Satırı Göster"):
            st.write(veri.head())

        if st.checkbox("Sondan 5 Satırı Göster"):
            st.write(veri.tail())

        if st.checkbox("Betimsel İstatistikleri Göster"):
            st.write(veri.describe().T)

        # Sayısal değişkenlerin dağılım kontrolü
        if st.checkbox("Dağılım Kontrolü"):
            sayisal_sutunlar = veri.select_dtypes(include=['float64', 'int64']).columns
            secilen_sutun = st.selectbox("İncelemek istediğiniz sayısal değişkeni seçin.", sayisal_sutunlar)
            if st.button("Histogram"):
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(veri[secilen_sutun], kde=True, bins=30, ax=ax)
                ax.set_title("Histogram")
                st.pyplot(fig)
            if st.button("Q-Q Plot"):
                fig, ax = plt.subplots(figsize=(8, 6))
                stats.probplot(veri[secilen_sutun], dist="norm", plot=ax)
                ax.set_title("Q-Q Plot")
                st.pyplot(fig)
            if st.button("Normal Dağılıma Uygunluk Testleri"):
                shapiro_p, ks_p = normalite_testleri(veri, secilen_sutun)
                st.write(f"Shapiro-Wilk testi: p-değeri = {shapiro_p:.4f}")
                if shapiro_p >= 0.05:
                    st.write("Yorum: Veri normal dağılıma uygun (H0 reddedilemez).")
                else:
                    st.write("Yorum: Veri normal dağılıma uymamaktadır (H0 reddedilir).")
                st.write(f"Kolmogorov-Smirnov testi: p-değeri = {ks_p:.4f}")
                if ks_p >= 0.05:
                    st.write("Yorum: Veri normal dağılıma uygun (H0 reddedilemez).")
                else:
                    st.write("Yorum: Veri normal dağılıma uymamaktadır (H0 reddedilir).")

        # Kategorik değişken grafikleri
        if st.checkbox("Kategorik Değişkenler için Grafikler"):
            kategorik_sutunlar = veri.select_dtypes(include='object').columns
            secilen_kategorik = st.selectbox("İncelemek istediğiniz kategorik değişkeni seçiniz.", kategorik_sutunlar)
            grafik_turu = st.selectbox("Grafik Türünü Seçin.", ["Bar Plot", "Pie Chart"])
            if grafik_turu == "Bar Plot":
                fig = px.histogram(veri, x=secilen_kategorik,
                                   title=f"{secilen_kategorik} - Bar Plot")
            else:
                fig = px.pie(veri, names=secilen_kategorik,
                             title=f"{secilen_kategorik} - Pie Chart")
            st.plotly_chart(fig)

        # Sayısal değişken grafikleri
        if st.checkbox("Sayısal Değişkenler için Grafikler"):
            sayisal_sutunlar = veri.select_dtypes(include=['float64', 'int64']).columns
            secilen_sayisal = st.selectbox("İncelemek istediğiniz sayısal değişkeni seçiniz.", sayisal_sutunlar)
            grafik_secim = st.selectbox("Grafik Türünü Seçin.", 
                                        ["Box Plot", "Violin Plot", "Scatter Plot", "Line Plot"])
            if grafik_secim == "Box Plot": #sns.boxplot ilede yapılabilirdi ancak px ile DAHA KOLAY DAHA AZ SATIRLA YAPTIM. PX DAHA KOLAY.
                fig = px.box(veri, y=secilen_sayisal,
                             title=f"{secilen_sayisal} - Box Plot",
                             width=1000, height=600)
                st.write(fig)
            elif grafik_secim == "Violin Plot":
                fig = px.violin(veri, y=secilen_sayisal, box=True, points="all",
                                title=f"{secilen_sayisal} - Violin Plot",
                                width=1000, height=600)
                st.write(fig)
            elif grafik_secim == "Scatter Plot":
                fig = px.scatter(veri, x=veri.index, y=secilen_sayisal,
                                 title=f"{secilen_sayisal} - Scatter Plot",
                                 width=1000, height=600)
                st.write(fig)
            elif grafik_secim == "Line Plot":
                fig = px.line(veri, x=veri.index, y=secilen_sayisal,
                              title=f"{secilen_sayisal} - Line Plot",
                              width=1000, height=600)
                st.write(fig)

        #korelasyon matrisi ve grafikler
        if st.checkbox("Sayısal Değişkenler Arası Korelasyon Matrisi"):
            corr_pearson = veri.select_dtypes(include=['float64', 'int64']).corr(method='pearson')
            corr_spearman = veri.select_dtypes(include=['float64', 'int64']).corr(method='spearman')
            st.subheader("Pearson Korelasyon Matrisi")
            st.write(corr_pearson)
            st.subheader("Spearman Korelasyon Matrisi")
            st.write(corr_spearman)
            grafik_secim2 = st.selectbox("Korelasyon Grafiği Türünü Seçin.", ["Pairplot", "Heatmap"])
            if grafik_secim2 == "Pairplot":
                pair_plot = sns.pairplot(veri.select_dtypes(include=['float64', 'int64'])) #ızgara grafiği
                plt.suptitle("Pairplot", y=1.02)
                st.pyplot(pair_plot.fig)
            else:
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(corr_pearson, annot=True, cmap="coolwarm", fmt=".2f", ax=ax) #annot her hücreye korelasyon katsayısını yazdırır, fmt 2 ondalık basamak için yazıldı.
                ax.set_title("Pearson Korelasyon Matrisi Heatmap")
                st.pyplot(fig)

        # İki sayısal değişken için regresyon analizi
        if st.checkbox("İki Sayısal Değişken için Regresyon Analizi"):
            sayisal_sutunlar = veri.select_dtypes(include=['float64', 'int64']).columns
            bagimsiz_deg = st.selectbox("Bağımsız Değişkeni Seçiniz", sayisal_sutunlar)
            bagimli_deg = st.selectbox("Bağımlı Değişkeni Seçiniz", sayisal_sutunlar)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.regplot(x=veri[bagimsiz_deg], y=veri[bagimli_deg], line_kws={'color': 'red'}, ax=ax)
            ax.set_title(f"{bagimli_deg} vs {bagimsiz_deg} - Regresyon Analizi")
            st.pyplot(fig)

# -------------------- Z TESTİ --------------------
elif sayfa == "Z Testi":
    st.title("Z Testi 📊")
    dosya_turu = ["csv", "txt", "xls", "xlsx"]
    yuklenen_veri = st.file_uploader("Veri setinizi CSV, TXT veya Excel formatında yükleyiniz.", type=dosya_turu)
    if yuklenen_veri is not None:
        dosya_adi = yuklenen_veri.name.lower()
        if "csv" in dosya_adi:
            veri = pd.read_csv(yuklenen_veri)
        elif "txt" in dosya_adi:
            veri = pd.read_csv(yuklenen_veri, delimiter='\t')
        elif "xls" in dosya_adi or "xlsx" in dosya_adi:
            veri = pd.read_excel(yuklenen_veri)
        st.write("Veri Örneği:")
        st.write(veri)

        st.subheader("Z-Testi Parametreleri")
        sayisal_sutunlar = veri.select_dtypes(include=['float64', 'int64']).columns
        if len(sayisal_sutunlar) > 0:
            ilk_sayisal = sayisal_sutunlar[0]
            pop_ortalama_default = float(veri[ilk_sayisal].mean())
            pop_std_default = float(veri[ilk_sayisal].std())
            orneklem_ortalama_default = pop_ortalama_default
            n_default = int(max(30, min(len(veri), 100)))
        else:
            pop_ortalama_default = 0.0
            pop_std_default = 1.0
            orneklem_ortalama_default = 0.0
            n_default = 30

        pop_ortalama = st.number_input("Popülasyon Ortalaması (μ)", value=pop_ortalama_default)
        pop_std = st.number_input("Popülasyon Standart Sapması (σ)", value=pop_std_default)
        orneklem_ortalama = st.number_input("Örneklem Ortalaması", value=orneklem_ortalama_default)
        n = st.number_input("Örneklem Büyüklüğü (n)", value=n_default, step=1)

        z_istatistik = (orneklem_ortalama - pop_ortalama) / (pop_std / np.sqrt(n))
        p_degeri = 2 * (1 - norm.cdf(np.abs(z_istatistik)))

        st.subheader("Z-Testi Sonuçları")
        st.write("Z-istatistiği:", z_istatistik)
        st.write("P-değeri:", p_degeri)
        if p_degeri < 0.05:
            st.write("Yorum: Sonuç anlamlıdır. Örneklem ortalaması ile popülasyon ortalaması arasında istatistiksel olarak anlamlı fark vardır (H0 reddedilir).")
        else:
            st.write("Yorum: Sonuç anlamlı değildir. Örneklem ortalaması ile popülasyon ortalaması arasında istatistiksel olarak anlamlı fark bulunmamaktadır (H0 reddedilemez).")

# -------------------- T TESTİ --------------------
elif sayfa == "T Testi":
    st.title("T Testi 📊")
    dosya_turu = ["csv", "txt", "xls", "xlsx"]
    yuklenen_veri = st.file_uploader("Veri setinizi CSV, TXT veya Excel formatında yükleyiniz.", type=dosya_turu)
    if yuklenen_veri is not None:
        dosya_adi = yuklenen_veri.name.lower()
        if "csv" in dosya_adi:
            veri = pd.read_csv(yuklenen_veri)
        elif "txt" in dosya_turu:
            veri = pd.read_csv(yuklenen_veri, delimiter='\t')
        elif "xls" in dosya_adi or "xlsx" in dosya_adi:
            veri = pd.read_excel(yuklenen_veri)
        st.write("Veri Örneği:")
        st.write(veri)

        # Tek Örneklem T Testi
        st.subheader("Tek Örneklem T Testi")
        sayisal_sutunlar = veri.select_dtypes(include=['float64', 'int64']).columns
        if len(sayisal_sutunlar) > 0:
            secilen_sutun = st.selectbox("Tek örneklem t-testi için değişkeni seçiniz:", sayisal_sutunlar)
            hipotez_degeri = st.number_input("Null Hipotezdeki Ortalama Değer", value=float(veri[secilen_sutun].mean()))
            if st.button("Tek Örneklem T Testi Uygula"):
                t_stat, p_val = ttest_1samp(veri[secilen_sutun], hipotez_degeri)
                st.write("T-istatistiği:", t_stat)
                st.write("P-değeri:", p_val)
                if p_val < 0.05:
                    st.write("Yorum: Sonuç anlamlıdır. Null hipotez reddedilmiştir,örneklem ortalaması hipotezde belirtilen değerden anlamlı derecede farklıdır.")
                else:
                    st.write("Yorum: Sonuç anlamlı değildir. Null hipotez reddedilemedi. Örneklem ortalaması, hipotezde belirtilen değere yakın kabul edilebilir.")
        else:
            st.write("Sayısal değişken bulunamadı.")

        # İki Örneklem Bağımsız T Testi
        st.subheader("İki Örneklem Bağımsız T Testi")
        sayisal_sutunlar = veri.select_dtypes(include=['float64', 'int64']).columns
        grup_sutunlar = veri.select_dtypes(include='object').columns
        if len(grup_sutunlar) > 0 and len(sayisal_sutunlar) > 0:
            test_sutunu = st.selectbox("Test edilecek sayısal değişkeni seçiniz:", sayisal_sutunlar)
            grup_sutunu = st.selectbox("Grup değişkenini seçiniz:", grup_sutunlar)
            gruplar = veri[grup_sutunu].unique()
            if len(gruplar) >= 2:
                grup1 = st.selectbox("Grup 1", gruplar, index=0)
                grup2 = st.selectbox("Grup 2", gruplar, index=1)
                guven_düzeyi = st.number_input("Güven Düzeyi (%)", value=95, step=1)
                if st.button("Bağımsız T Testi Uygula"):
                    veri_grup1 = veri[veri[grup_sutunu] == grup1][test_sutunu]
                    veri_grup2 = veri[veri[grup_sutunu] == grup2][test_sutunu]
                    levene_stat, levene_p = levene(veri_grup1, veri_grup2)
                    alpha = 1 - guven_düzeyi / 100
                    if levene_p > alpha:
                        t_stat, p_val = ttest_ind(veri_grup1, veri_grup2, equal_var=True) #iki grubun varyanslarını eşit kabul ettiğimiz durum için:
                        var_durumu = "Homojen"
                    else:
                        t_stat, p_val = ttest_ind(veri_grup1, veri_grup2, equal_var=False)
                        var_durumu = "Homojen Değil"
                    st.write("T-istatistiği:", t_stat)
                    st.write("P-değeri:", p_val)
                    st.write("Varyans Durumu:", var_durumu)
                    if p_val < 0.05:
                        st.write("Yorum: Gruplar arasında anlamlı fark vardır (H0 reddedilir).")
                    else:
                        st.write("Yorum: Gruplar arasında anlamlı fark bulunmamaktadır (H0 reddedilemez).")
            else:
                st.write("En az 2 grup gereklidir.")
        else:
            st.write("Bağımsız T Testi için uygun grup veya sayısal değişken bulunamadı.")

        # İki Örneklem Bağımlı T Testi
        st.subheader("İki Örneklem Bağımlı T Testi")
        sayisal_sutunlar = veri.select_dtypes(include=['float64', 'int64']).columns
        if len(sayisal_sutunlar) >= 2:
            degisken1 = st.selectbox("İlk Değişkeni Seçiniz:", sayisal_sutunlar, key="dependent1")
            degisken2 = st.selectbox("İkinci Değişkeni Seçiniz:", sayisal_sutunlar, key="dependent2")
            if st.button("Bağımlı T Testi Uygula"):
                t_stat, p_val = ttest_rel(veri[degisken1], veri[degisken2])
                st.write("T-istatistiği:", t_stat)
                st.write("P-değeri:", p_val)
                if p_val < 0.05:
                    st.write("Yorum: İki ölçüm arasında anlamlı fark vardır (H0 reddedilir).")
                else:
                    st.write("Yorum: İki ölçüm arasında anlamlı fark bulunmamaktadır (H0 reddedilemez).")
        else:
            st.write("Bağımlı T Testi için en az 2 sayısal değişken gereklidir.")

# -------------------- ANOVA & KI-KARE TESTLERİ --------------------
elif sayfa == "ANOVA & Ki-Kare Testleri":
    st.title("ANOVA & Ki-Kare Testleri")
    dosya_turu = ["csv", "txt", "xls", "xlsx"]
    yuklenen_veri = st.file_uploader("Veri setinizi yükleyiniz.", type=dosya_turu)
    if yuklenen_veri is not None:
        dosya_adi = yuklenen_veri.name.lower()
        if "csv" in dosya_adi:
            veri = pd.read_csv(yuklenen_veri)
        elif "txt" in dosya_adi:
            veri = pd.read_csv(yuklenen_veri, delimiter='\t')
        elif "xls" in dosya_adi or "xlsx" in dosya_adi:
            veri = pd.read_excel(yuklenen_veri)
        st.write("Veri Örneği:")
        st.write(veri.head())
        
        st.markdown("---")
        test_turu = st.radio("Test Türünü Seçin:", 
                             ["Tek Yönlü ANOVA", "Çift Yönlü ANOVA", "Ki-Kare Testi"])

        # Tek Yönlü ANOVA
        if test_turu == "Tek Yönlü ANOVA":
            st.subheader("Tek Yönlü ANOVA")
            sayisal_sutunlar = veri.select_dtypes(include=["float64", "int64"]).columns
            grup_sutunlar = veri.select_dtypes(include="object").columns
            if len(sayisal_sutunlar) > 0 and len(grup_sutunlar) > 0:
                secilen_sayisal = st.selectbox("Bağımlı (Sayısal) Değişkeni Seçiniz:", sayisal_sutunlar)
                secilen_grup = st.selectbox("Gruplama için Kategorik Değişkeni Seçiniz:", grup_sutunlar)
                gruplar = veri[secilen_grup].unique()
                if len(gruplar) < 3:
                    st.error("Tek yönlü ANOVA için en az 3 grup gereklidir. Seçtiğiniz değişkende yetersiz grup sayısı mevcut.")
                else:
                    if st.button("Tek Yönlü ANOVA Uygula"):
                        grup_listesi = [veri[veri[secilen_grup] == grup][secilen_sayisal].dropna().values for grup in gruplar]
                        f_stat, p_val = f_oneway(*grup_listesi)
                        st.write("ANOVA F-istatistiği:", f_stat)
                        st.write("P-değeri:", p_val)
                        if p_val < 0.05:
                            st.write("Yorum: Gruplar arasında anlamlı fark vardır (H0 reddedilir).")
                        else:
                            st.write("Yorum: Gruplar arasında anlamlı fark bulunmamaktadır (H0 reddedilemez).")
            else:
                st.write("Tek yönlü ANOVA için uygun sayısal veya kategorik değişken bulunamadı.")

        # Çift Yönlü ANOVA
        elif test_turu == "Çift Yönlü ANOVA":
            st.subheader("Çift Yönlü ANOVA")
            sayisal_sutunlar = veri.select_dtypes(include=['float64', 'int64']).columns
            grup_sutunlar = veri.select_dtypes(include='object').columns
            if len(sayisal_sutunlar) > 0 and len(grup_sutunlar) >= 2:
                dependent = st.selectbox("Bağımlı (Sayısal) Değişkeni Seçiniz:", sayisal_sutunlar)
                factor1 = st.selectbox("1. Faktörü Seçiniz:", grup_sutunlar, key="f1")
                factor2 = st.selectbox("2. Faktörü Seçiniz:", grup_sutunlar, key="f2")
                if factor1 == factor2:
                    st.error("Lütfen farklı iki faktör seçiniz.")
                else:
                    if st.button("Çift Yönlü ANOVA Uygula"):
                        form = f"{dependent} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})"
                        model = ols(form, data=veri).fit()
                        anova_table = sm.stats.anova_lm(model, typ=2)
                        st.write(anova_table)
                        st.write("Yorum: Tabloyu inceleyiniz. p-değerleri < 0.05 olan faktörler veya etkileşimler istatistiksel olarak anlamlıdır.")
            else:
                st.write("Çift yönlü ANOVA için yeterli sayıda kategorik değişken veya sayısal bağımlı değişken bulunamadı.")

        # Ki-Kare Testi
        elif test_turu == "Ki-Kare Testi":
            st.subheader("Ki-Kare Testi")
            grup_sutunlar = veri.select_dtypes(include='object').columns
            if len(grup_sutunlar) >= 2:
                cat1 = st.selectbox("1. Kategorik Değişkeni Seçiniz:", grup_sutunlar, key="chi1")
                cat2 = st.selectbox("2. Kategorik Değişkeni Seçiniz:", grup_sutunlar, key="chi2")
                if st.button("Ki-Kare Testi Uygula"):
                    contingency_table = pd.crosstab(veri[cat1], veri[cat2])
                    chi2, p, dof, expected = chi2_contingency(contingency_table)
                    st.write("Chi-Square İstatistiği:", chi2)
                    st.write("P-değeri:", p)
                    st.write("Serbestlik Derecesi:", dof)
                    st.write("Beklenen Değerler:", expected)
                    if p < 0.05:
                        st.write("Yorum: İki kategorik değişken arasında anlamlı ilişki vardır (H0 reddedilir).")
                    else:
                        st.write("Yorum: İki kategorik değişken arasında anlamlı ilişki bulunmamaktadır (H0 reddedilemez).")
            else:
                st.write("Ki-Kare testi için en az 2 kategorik değişken gerekmektedir.")



# -------------------- ML ÖNERİ SİSTEMİ --------------------
elif sayfa == "ML Öneri Sistemi":
    st.title("ML Öneri Sistemi")
    st.write("Bu bölümde, veri setinizin özelliklerine göre makine öğrenmesi yaklaşımı ve metrik önerileri sunulacaktır.")
    
    # Dosya türlerini tanımla ve veri seti yükle
    dosya_turu = ["csv", "txt", "xls", "xlsx"]
    yuklenen_veri = st.file_uploader("Veri setinizi yükleyiniz.", type=dosya_turu)
    
    if yuklenen_veri is not None:
        dosya_adi = yuklenen_veri.name.lower()
        if "csv" in dosya_adi:
            veri = pd.read_csv(yuklenen_veri)
        elif "txt" in dosya_adi:
            veri = pd.read_csv(yuklenen_veri, delimiter='\t')
        elif "xls" in dosya_adi or "xlsx" in dosya_adi:
            veri = pd.read_excel(yuklenen_veri)
        st.write("Veri Örneği:")
        st.write(veri.head())

   

        st.subheader("Problem Tipini Belirleme")
        hedef_deg = st.selectbox("Hedef değişken seçin (Yoksa 'Denetimsiz Öğrenme'yi seçin):", ["Denetimsiz Öğrenme"] + list(veri.columns))
        
        if hedef_deg == "Denetimsiz Öğrenme":
            st.subheader("Kümeleme (Clustering) Önerileri")
            st.markdown("""
            **Önerilen Algoritmalar:**
            - K-Means
            - DBSCAN
            - Hiyerarşik Kümeleme
            - Gaussian Mixture Models
            - OPTICS

            **Değerlendirme Metrikleri:**
            - Silhouette Skor
            - Davies-Bouldin İndeksi
            - Calinski-Harabasz İndeksi
            - Elbow Yöntemi (SSE)
            """)
        
        else:
            unique_values = veri[hedef_deg].nunique()
            dtype = veri[hedef_deg].dtype

            # Tip Belirleme Mantığı Geliştirildi
            if dtype in ['object', 'category']:
                if unique_values == 2:
                    problem_tipi = "Binary Sınıflandırma"
                elif 2 < unique_values <= 20:
                    problem_tipi = "Çoklu Sınıflandırma"
                else:
                    problem_tipi = "Yüksek Kardinaliteli Kategori: çok fazla eşsiz değer içerir."
            elif np.issubdtype(dtype, np.number):
                if unique_values <= 10 and veri[hedef_deg].apply(lambda x: float(x).is_integer()).all():
                    problem_tipi = "Çoklu Sınıflandırma" 
                else:
                    problem_tipi = "Regresyon"
            else:
                problem_tipi = "Bilinmeyen Tip"

            st.subheader(f"Tespit Edilen Problem Tipi: {problem_tipi}")
            
            if "Sınıflandırma" in problem_tipi:
                st.markdown("""
                **Önerilen Modeller:**
                - Lojistik Regresyon
                - XGBoost
                - LightGBM
                - CatBoost
                - Random Forest
                - SVM
                - KNN
                - Çok Katmanlı Perceptron(mlp)

                **Değerlendirme Metrikleri:**
                - Accuracy, Precision, Recall
                - F1-Score 
                - ROC-AUC 
                - Confusion Matrix
                """)
                if "Yüksek" in problem_tipi:
                    st.warning("Sınıf sayısını azaltmayı veya hiyerarşik yöntemleri düşünün.")

            elif problem_tipi == "Regresyon":
                st.markdown("""
                **Önerilen Modeller:**
                - Lineer Regresyon
                - Lasso Regresyon
                - Ridge Regresyon
                - Gradient Boosting Regresör
                - Support Vector Regresyon
                - ElasticNet
                - Neural Networks
                - Decision Tree Regressor
                - Random Forest Regressor

                **Değerlendirme Metrikleri:**
                - MAE, MSE, RMSE
                - R² Skor
                """)



