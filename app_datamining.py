import streamlit as st
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from mlxtend.frequent_patterns import apriori, association_rules

import matplotlib.pyplot as plt
import seaborn as sns

from io import BytesIO
import base64
import warnings
import nltk

from nltk.corpus import stopwords

warnings.filterwarnings("ignore")

nltk.download('stopwords', quiet=True)

# =====================================================
# CONFIGURACIÓN
# =====================================================

st.set_page_config(
    page_title="Simulador de Data Mining - BotiCura",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    text-align: center;
    color: white;
}

.main-header h1 {
    font-size: 2.3rem;
    font-weight: 800;
}

.kpi-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
}

.section-header {
    background: linear-gradient(90deg, #1e3a5f, #2d6a9f);
    color: white;
    padding: 0.8rem 1rem;
    border-radius: 8px;
    margin-top: 1rem;
    margin-bottom: 1rem;
    font-size: 1.1rem;
    font-weight: 700;
}

.insight-box {
    background-color: #f0f9ff;
    border-left: 5px solid #0ea5e9;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    margin-top: 1rem;
    margin-bottom: 1rem;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="main-header">
    <h1>🧠 Simulador de Data Mining - BotiCura</h1>
    <p>Segmentación · Clasificación · Asociación</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# UTILIDADES
# =====================================================

def to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

def download_btn(df, filename, label):
    data = to_excel_bytes(df)
    b64 = base64.b64encode(data).decode()
    href = f'''
    <a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}"
    download="{filename}"
    style="
    background:#0ea5e9;
    color:white;
    padding:8px 16px;
    border-radius:6px;
    text-decoration:none;
    font-weight:600;">
    ⬇️ {label}
    </a>
    '''
    st.markdown(href, unsafe_allow_html=True)

def insight(text):
    st.markdown(
        f"""
        <div class="insight-box">
        💡 <b>Insight:</b> {text}
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("## 📁 Cargar Datasets")

    empresa_nombre = st.text_input(
        "Empresa",
        value="BotiCura S.A.C."
    )

    uploaded_trans = st.file_uploader(
        "📦 Transacciones",
        type=["xlsx"]
    )

    uploaded_rev = st.file_uploader(
        "⭐ Reseñas",
        type=["xlsx"]
    )

    uploaded_cli = st.file_uploader(
        "👥 Clientes",
        type=["xlsx"]
    )

    st.markdown("---")
    st.markdown("""
    ### ℹ️ Contexto Empresarial
    💊 BotiCura S.A.C.
    Retail farmacéutico peruano.
    """)
    st.markdown("---")
    st.markdown("""
   © 2026 Desarrollado por Wilton Torvisco - Data Mining Simulator """)

# =====================================================
# VALIDACIÓN
# =====================================================

REQUIRED_TRANS = {
    "customer_id",
    "order_id",
    "order_date",
    "departamento",
    "distrito",
    "canal_venta",
    "categoria_producto",
    "product_name",
    "quantity",
    "unit_price",
    "discount",
    "total_amount",
    "payment_method",
    "campaign",
    "margin"
}

REQUIRED_REV = {
    "review_id",
    "customer_id",
    "canal_venta",
    "categoria_producto",
    "review_text",
    "sentiment",
    "rating",
    "departamento"
}

if not uploaded_trans or not uploaded_rev or not uploaded_cli:
    st.info("Carga los 3 datasets para comenzar.")
    st.stop()

# =====================================================
# CARGA
# =====================================================

try:
    df_t = pd.read_excel(uploaded_trans)
    df_r = pd.read_excel(uploaded_rev)
    df_c = pd.read_excel(uploaded_cli)
except Exception as e:
    st.error(f"Error leyendo archivos: {e}")
    st.stop()

# =====================================================
# VALIDACIÓN COLUMNAS
# =====================================================

missing_t = REQUIRED_TRANS - set(df_t.columns)
missing_r = REQUIRED_REV - set(df_r.columns)

if missing_t:
    st.error(f"Faltan columnas en Transacciones: {missing_t}")
    st.stop()

if missing_r:
    st.error(f"Faltan columnas en Reseñas: {missing_r}")
    st.stop()

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "🎯 Segmentación",
    "💬 Clasificación",
    "🔗 Asociación",
    "🗺️ Geográfico"
])

# =====================================================
# TAB 1 DASHBOARD
# =====================================================

with tab1:
    st.markdown(
        '<div class="section-header">📊 Dashboard ejecutivo</div>',
        unsafe_allow_html=True
    )

    total_ventas = df_t["total_amount"].sum()
    total_clientes = df_t["customer_id"].nunique()
    total_ordenes = df_t["order_id"].nunique()
    ticket_prom = df_t.groupby("order_id")["total_amount"].sum().mean()
    margen_total = df_t["margin"].sum()
    pct_online = ((df_t["canal_venta"] == "App").mean() * 100)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpis = [
        (c1, f"S/ {total_ventas:,.0f}", "Ventas"),
        (c2, f"{total_clientes:,}", "Clientes"),
        (c3, f"{total_ordenes:,}", "Órdenes"),
        (c4, f"S/ {ticket_prom:.1f}", "Ticket promedio"),
        (c5, f"S/ {margen_total:,.0f}", "Margen"),
        (c6, f"{pct_online:.1f}%", "App"),
    ]

    for col, val, label in kpis:
        col.markdown(
            f"""
            <div class="kpi-card">
                <h3>{val}</h3>
                <p>{label}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("Vista previa datasets")
    st.dataframe(df_t.head())

# =====================================================
# TAB 2 CLUSTERING
# =====================================================

with tab2:
    st.markdown(
        '<div class="section-header">🎯 Segmentación de Clientes</div>',
        unsafe_allow_html=True
    )

    n_clusters = st.slider("Número de clusters", 2, 8, 4)

    if st.button("Ejecutar Clustering"):
        feat = df_t.groupby("customer_id").agg(
            total_spent=("total_amount", "sum"),
            avg_spent=("total_amount", "mean"),
            order_count=("order_id", "nunique"),
            unique_categories=("categoria_producto", "nunique")
        ).reset_index()

        X = feat[["total_spent", "avg_spent", "order_count", "unique_categories"]]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Curva de Codo (Mantenida como mejora)
        st.markdown("### 📈 Curva de Codo (Elbow Method)")
        wcss = []
        K_range = range(1, 11)
        for k in K_range:
            km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            km_temp.fit(X_scaled)
            wcss.append(km_temp.inertia_)
            
        fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4))
        ax_elbow.plot(K_range, wcss, marker='o', linestyle='--', color='#0ea5e9')
        ax_elbow.set_title("Método del Codo para hallar el K óptimo")
        ax_elbow.set_xlabel("Número de clusters (K)")
        ax_elbow.set_ylabel("Inercia (WCSS)")
        ax_elbow.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig_elbow)
        st.markdown("---")

        # Ejecución del KMeans
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        feat["cluster"] = model.fit_predict(X_scaled)

        st.markdown("### 🧬 Distribución de Clusters")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=feat,
            x="total_spent",
            y="order_count",
            hue="cluster",
            palette="viridis",
            s=100,
            ax=ax
        )
        ax.set_title("Clusters de Clientes")
        st.pyplot(fig)

        resumen = feat.groupby("cluster").agg(
            Clientes=("customer_id", "count"),
            Gasto=("total_spent", "mean"),
            Frecuencia=("order_count", "mean")
        ).round(2)

        st.dataframe(resumen)

        insight(
            "Los segmentos permiten diseñar estrategias diferenciadas de fidelización, retención y cross-selling. Utiliza la curva de codo superior para justificar tu número óptimo de segmentos."
        )

        download_btn(feat, "segmentacion_clientes.xlsx", "Descargar Segmentación")

# =====================================================
# TAB 3 CLASIFICACIÓN (Solo Sentimiento)
# =====================================================

with tab3:
    st.markdown(
        '<div class="section-header">💬 Clasificación Predictiva (Análisis de Sentimiento)</div>',
        unsafe_allow_html=True
    )

    test_size = st.slider("Porcentaje prueba", 10, 40, 20)
    max_features = st.slider("Máx palabras", 100, 2000, 500)

    if st.button("Ejecutar Clasificación"):
        spanish_stopwords = stopwords.words('spanish')

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=spanish_stopwords,
            ngram_range=(1,2)
        )

        X = vectorizer.fit_transform(df_r["review_text"].fillna(""))
        y = df_r["sentiment"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42, stratify=y
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        # --- CORRECCIÓN: Guardar en memoria (session_state) ---
        st.session_state['modelo_nlp'] = model
        st.session_state['vector_nlp'] = vectorizer
        st.session_state['nlp_entrenado'] = True

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc*100:.1f}%")

        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        st.pyplot(fig_cm)

        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

    # --- SIMULADOR DE TEXTO (Fuera del botón para evitar reinicios) ---
    if st.session_state.get('nlp_entrenado', False):
        st.markdown("---")
        st.markdown("### 🧪 Simulador: Probar el algoritmo en vivo")
        texto = st.text_area("Escribe una reseña como si fueras un cliente de BotiCura:")
        
        if st.button("Predecir Sentimiento"):
            if texto:
                mod = st.session_state['modelo_nlp']
                vec = st.session_state['vector_nlp']
                pred = mod.predict(vec.transform([texto]))[0]
                prob = mod.predict_proba(vec.transform([texto]))[0].max()
                
                if pred.lower() == 'positivo':
                    st.success(f"🟢 **Sentimiento:** {pred} (Confianza: {prob*100:.1f}%)")
                elif pred.lower() == 'negativo':
                    st.error(f"🔴 **Sentimiento:** {pred} (Confianza: {prob*100:.1f}%)")
                else:
                    st.info(f"⚪ **Sentimiento:** {pred} (Confianza: {prob*100:.1f}%)")
            else:
                st.warning("Por favor, ingresa un texto válido.")

# =====================================================
# TAB 4 APRIORI
# =====================================================

with tab4:
    st.markdown(
        '<div class="section-header">🔗 Reglas de Asociación</div>',
        unsafe_allow_html=True
    )

    soporte = st.slider("Soporte mínimo", 0.001, 0.05, 0.01)

    if st.button("Ejecutar Apriori"):
        basket = (
            df_t.groupby(["order_id", "product_name"])["quantity"]
            .sum()
            .unstack()
            .fillna(0)
        )

        basket = (basket > 0).astype(int)

        frequent_items = apriori(basket, min_support=soporte, use_colnames=True)
        rules = association_rules(frequent_items, metric="lift", min_threshold=1)

        if rules.empty:
            st.warning("No se encontraron reglas")
        else:
            rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
            rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

            top_rules = rules.sort_values("lift", ascending=False).head(20)

            st.dataframe(
                top_rules[[
                    "antecedents",
                    "consequents",
                    "support",
                    "confidence",
                    "lift"
                ]]
            )

            insight(
                "Las reglas permiten diseñar promociones, combos y estrategias de cross-selling."
            )

            download_btn(top_rules, "reglas_asociacion.xlsx", "Descargar Reglas")

# =====================================================
# TAB 5 GEOGRÁFICO
# =====================================================

with tab5:
    st.markdown(
        '<div class="section-header">🗺️ Análisis Geográfico</div>',
        unsafe_allow_html=True
    )

    ventas_geo = df_t.groupby("distrito")["total_amount"].sum().sort_values(ascending=False).head(15)

    fig_geo, ax_geo = plt.subplots(figsize=(10,5))
    ventas_geo.plot(kind="bar", ax=ax_geo)
    ax_geo.set_title("Ventas por Distrito")
    st.pyplot(fig_geo)

    geo_full = df_t.groupby("distrito").agg(
        Ventas=("total_amount", "sum"),
        Ordenes=("order_id", "nunique"),
        Clientes=("customer_id", "nunique")
    ).round(2)

    st.dataframe(geo_full)

# =====================================================
# PREGUNTAS MBA
# =====================================================

st.markdown("---")
st.subheader("🎓 Preguntas guía")
st.markdown("""
1. ¿Qué segmento genera mayor margen?
2. ¿Qué distrito tiene clientes más rentables?
3. ¿Qué categorías impulsan cross-selling?
4. ¿Qué canal posee mejor satisfacción?
5. ¿Cómo impacta un falso positivo en la rentabilidad de las devoluciones?
6. ¿Qué campañas podrían aumentar recompra?
7. ¿Qué productos deberían venderse en combo?
""")

# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<hr>
<center>
🧠 <b>Data Mining - BotiCura</b><br>
Segmentación · Clasificación · Asociación<br>
Retail Farmacéutico Peruano
</center>
""", unsafe_allow_html=True)