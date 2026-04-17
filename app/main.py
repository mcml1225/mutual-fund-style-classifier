"""
Streamlit dashboard - With meaningful investment visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_acquisition import MutualFundDataCollector
from src.feature_engineering import FeatureEngineer
from src.clustering_model import StyleBoxClusterer
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Mutual Fund Style Classifier", layout="wide")

st.title("📈 Mutual Fund Style Classifier")
st.markdown("### Clasificación automática de fondos usando Machine Learning")

# Sidebar
st.sidebar.header("⚙️ Configuración")

collector = MutualFundDataCollector()

@st.cache_data
def load_data():
    df = collector.load_from_csv()
    if df.empty:
        st.error("No data found. Run: python preload_data_simple.py")
        st.stop()
    return df

with st.spinner("Cargando datos..."):
    raw_data = load_data()

if raw_data is not None and not raw_data.empty:
    # Procesar features
    engineer = FeatureEngineer(raw_data)
    engineer.calculate_returns().calculate_volatility().calculate_sharpe_ratio()
    features = engineer.create_features_matrix()
    
    # Escalar y clusterizar
    scaler = StandardScaler()
    feature_cols = ['volatility', 'momentum', 'sharpe_ratio', 'liquidity']
    X_scaled = scaler.fit_transform(features[feature_cols])
    
    n_unique = features['Ticker'].nunique()
    actual_clusters = min(6, n_unique)
    
    clusterer = StyleBoxClusterer(n_clusters=actual_clusters)
    results = clusterer.train(X_scaled, features)
    
    # ============ NUEVAS VISUALIZACIONES ============
    
    tab1, tab2, tab3 = st.tabs([
        "📊 Análisis de Estilo", 
        "📈 Rendimiento y Riesgo", 
        "🎯 Clasificación por Cluster"
    ])
    
    # ========== TAB 1: ANÁLISIS DE ESTILO ==========
    with tab1:
        st.header("Morningstar Style Box - Clasificación Automática")
        
        # Crear matriz 3x3 del Style Box
        if 'Style_Box' in results.columns:
            # Parsear las categorías
            caps = ['Large Cap', 'Mid Cap', 'Small Cap']
            styles = ['Value', 'Blend', 'Growth']
            
            # Crear matriz de conteo
            style_matrix = []
            for cap in caps:
                row = []
                for style in styles:
                    style_name = f"{cap} {style}"
                    count = len(results[results['Style_Box'] == style_name])
                    row.append(count)
                style_matrix.append(row)
            
            # Heatmap del Style Box
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=style_matrix,
                x=styles,
                y=caps,
                text=style_matrix,
                texttemplate='%{text}',
                textfont={"size": 20, "color": "white"},
                colorscale='RdYlGn',
                showscale=True,
                colorbar_title="Número de Fondos"
            ))
            
            fig_heatmap.update_layout(
                title="<b>Morningstar Style Box</b><br><sup>Distribución de fondos por estilo de inversión</sup>",
                xaxis_title="Estilo de Inversión",
                yaxis_title="Capitalización de Mercado",
                height=500,
                font=dict(size=14)
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Explicación
            with st.expander("📖 ¿Qué significa el Style Box?"):
                st.markdown("""
                - **Large Cap**: Empresas grandes y establecidas (Apple, Microsoft)
                - **Mid Cap**: Empresas medianas con potencial de crecimiento
                - **Small Cap**: Empresas pequeñas con alto potencial pero mayor riesgo
                - **Value**: Empresas infravaloradas (precio bajo relativo a fundamentales)
                - **Blend**: Mezcla de Value y Growth
                - **Growth**: Empresas con alto potencial de crecimiento
                """)
        
        # Gráfico de barras horizontal de estilos
        st.subheader("📊 Distribución por Estilo de Inversión")
        
        col1, col2 = st.columns(2)
        
        with col1:
            style_counts = results['Style_Box'].value_counts().head(8)
            fig_bar = px.bar(
                x=style_counts.values, 
                y=style_counts.index,
                orientation='h',
                color=style_counts.values,
                color_continuous_scale='Viridis',
                title="Fondos por Categoría de Estilo",
                labels={'x': 'Número de Fondos', 'y': 'Categoría'}
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Riesgo vs Retorno por estilo
            style_risk_return = results.groupby('Style_Box').agg({
                'volatility': 'mean',
                'avg_return': 'mean'
            }).reset_index()
            
            fig_scatter = px.scatter(
                style_risk_return,
                x='volatility',
                y='avg_return',
                text='Style_Box',
                size='volatility',
                color='Style_Box',
                title="Riesgo vs Retorno por Estilo",
                labels={
                    'volatility': 'Volatilidad (Riesgo)',
                    'avg_return': 'Retorno Promedio Diario'
                }
            )
            fig_scatter.update_traces(textposition='top center')
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    # ========== TAB 2: RENDIMIENTO Y RIESGO ==========
    with tab2:
        st.header("Análisis de Rendimiento y Riesgo")
        
        # Métricas clave por fondo
        st.subheader("📈 Comparativa de Fondos")
        
        # Crear tabla de rendimiento
        performance_data = results[['Ticker', 'volatility', 'momentum', 'sharpe_ratio', 'avg_return']].copy()
        performance_data.columns = ['Fondo', 'Volatilidad (Riesgo)', 'Momentum', 'Sharpe Ratio', 'Retorno Diario']
        performance_data['Sharpe Ratio'] = performance_data['Sharpe Ratio'].round(3)
        performance_data['Volatilidad (Riesgo)'] = performance_data['Volatilidad (Riesgo)'].round(4)
        performance_data['Retorno Diario'] = (performance_data['Retorno Diario'] * 100).round(2)
        
        st.dataframe(performance_data, use_container_width=True)
        
        # Gráfico de riesgo-retorno
        st.subheader("🎯 Mapa Riesgo-Retorno")
        
        fig_risk_return = px.scatter(
            results,
            x='volatility',
            y='avg_return',
            size='sharpe_ratio',
            color='Style_Box',
            hover_data=['Ticker'],
            title="<b>Eficiencia de Inversión</b><br><sup>Mayor Sharpe Ratio = Mejor retorno ajustado por riesgo</sup>",
            labels={
                'volatility': 'Volatilidad (Riesgo)',
                'avg_return': 'Retorno Promedio Diario',
                'sharpe_ratio': 'Sharpe Ratio'
            }
        )
        
        # Líneas de referencia
        fig_risk_return.add_hline(y=results['avg_return'].mean(), line_dash="dash", line_color="gray", 
                                   annotation_text="Retorno Promedio")
        fig_risk_return.add_vline(x=results['volatility'].mean(), line_dash="dash", line_color="gray",
                                   annotation_text="Riesgo Promedio")
        
        st.plotly_chart(fig_risk_return, use_container_width=True)
        
        # Top performers
        st.subheader("🏆 Top Fondos por Sharpe Ratio")
        top_funds = results.nlargest(5, 'sharpe_ratio')[['Ticker', 'sharpe_ratio', 'volatility', 'Style_Box']]
        top_funds.columns = ['Fondo', 'Sharpe Ratio', 'Volatilidad', 'Estilo']
        
        for i, row in top_funds.iterrows():
            st.metric(
                label=f"{row['Fondo']} - {row['Estilo']}",
                value=f"Sharpe: {row['Sharpe Ratio']:.3f}",
                delta=f"Vol: {row['Volatilidad']:.3f}"
            )
    
    # ========== TAB 3: CLASIFICACIÓN POR CLUSTER ==========
    with tab3:
        st.header("Clasificación Automática por Cluster")
        
        # Mostrar asignación de clusters
        cluster_assignment = results[['Ticker', 'Cluster', 'Style_Box', 'volatility', 'momentum']].drop_duplicates()
        cluster_assignment = cluster_assignment.sort_values('Cluster')
        
        st.dataframe(cluster_assignment, use_container_width=True)
        
        # Características de cada cluster
        st.subheader("📊 Perfil de cada Cluster")
        
        cluster_profile = results.groupby('Cluster').agg({
            'volatility': 'mean',
            'momentum': 'mean',
            'sharpe_ratio': 'mean',
            'Ticker': 'count'
        }).round(3)
        
        cluster_profile.columns = ['Volatilidad Media', 'Momentum Medio', 'Sharpe Ratio Medio', 'N° Fondos']
        
        # Formatear para mejor visualización
        st.dataframe(cluster_profile, use_container_width=True)
        
        # Interpretación de clusters
        st.subheader("💡 Interpretación de Clusters")
        
        for cluster in sorted(results['Cluster'].unique()):
            cluster_data = results[results['Cluster'] == cluster]
            avg_vol = cluster_data['volatility'].mean()
            avg_mom = cluster_data['momentum'].mean()
            
            # Determinar perfil del cluster
            if avg_vol > results['volatility'].mean():
                riesgo = "🔴 Alto Riesgo (Small/Mid Cap)"
            else:
                riesgo = "🟢 Bajo Riesgo (Large Cap)"
            
            if avg_mom > results['momentum'].mean():
                estilo = "🚀 Crecimiento (Growth)"
            else:
                estilo = "💰 Valor (Value)"
            
            with st.expander(f"Cluster {cluster} - {riesgo} | {estilo}"):
                st.write(f"**Fondos en este cluster:** {', '.join(cluster_data['Ticker'].unique())}")
                st.write(f"**Volatilidad promedio:** {avg_vol:.4f}")
                st.write(f"**Momentum promedio:** {avg_mom:.4f}")
                st.write(f"**Sharpe Ratio promedio:** {cluster_data['sharpe_ratio'].mean():.3f}")
    
    # Footer con explicación
    st.markdown("---")
    st.markdown("""
    ### 📖 Metodología
    
    | Métrica | Qué mide | Relación con inversiones |
    |---------|----------|------------------------|
    | **Volatilidad** | Riesgo del fondo | Small Cap > Mid Cap > Large Cap |
    | **Momentum** | Tendencia de crecimiento | Growth > Blend > Value |
    | **Sharpe Ratio** | Retorno ajustado por riesgo | Más alto = mejor eficiencia |
    | **Liquidez** | Facilidad de trading | Fondos grandes = más líquidos |
    
    **K-Means Clustering** agrupa fondos con características similares para recrear el Morningstar Style Box.
    """)

else:
    st.error("No se pudieron cargar los datos")