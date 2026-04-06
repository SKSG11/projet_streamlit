"""
Dashboard Streamlit - Segmentation de Transactions Bancaires
Projet 3 - Licence 3 DSBD

Analyser et regrouper les transactions selon leurs caractéristiques
afin d’identifier des comportements atypiques.
Ce dashboard permet d'analyser et de regrouper les transactions bancaires selon leurs caractéristiques
afin d’identifier des comportements atypiques.
"""

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt

## La page

st.set_page_config(
    page_title="Segmentation Transactions Bancaires",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

## Chargement des données

@st.cache_data
def load_data():
    """
    Charge les données des transactions bancaires
    
    Retourne:
        DataFrame pandas avec les transactions
    """
    try:
        df = pd.read_csv('creditcard.csv')
        return df
    except FileNotFoundError:
        st.error(" Fichier creditcard.csv non trouvé. Veuillez vérifier le chemin.")
        st.stop()

## Prépa des données

@st.cache_data
def preprocess_data(df):
    """
    Prépare les données pour l'analyse
    
    Args:
        df: DataFrame brut
        
    Retourne:
        DataFrame nettoyé et enrichi
    """
    # Copie du dataframe original
    df_clean = df.copy()
    
    # Conversion du temps en heures
    df_clean['Time_Hours'] = df_clean['Time'] / 3600
    
    # Catégorisation des montants
    df_clean['Amount_Category'] = pd.cut(
        df_clean['Amount'], 
        bins=[0, 10, 50, 100, 500, float('inf')],
        labels=['Très faible', 'Faible', 'Moyen', 'Élevé', 'Très élevé']
    )
    
    # Ajout de statistiques dérivées
    df_clean['Log_Amount'] = np.log1p(df_clean['Amount'])
    
    return df_clean

# SEGMENTATION K-MEANS

@st.cache_data
def perform_clustering(df, n_clusters, features_to_use):
    """
    Applique l'algorithme K-Means pour segmenter les transactions
    
    Args:
        df: DataFrame des transactions
        n_clusters: Nombre de clusters souhaités
        features_to_use: Liste des colonnes à utiliser pour le clustering
        
    Retourne:
        DataFrame avec les labels de clusters et le modèle KMeans
    """
    # Sélection des features
    X = df[features_to_use].copy()
    
    # Normalisation des données (très important pour K-Means!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Application de K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Ajout des clusters au DataFrame
    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters
    
    return df_clustered, kmeans, scaler, X_scaled

## Analyse cluster

def analyze_clusters(df_clustered):
    """
    Analyse les caractéristiques de chaque cluster
    
    Args:
        df_clustered: DataFrame avec les labels de clusters
        
    Retourne:
        DataFrame avec les statistiques par cluster
    """
    cluster_stats = df_clustered.groupby('Cluster').agg({
        'Amount': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'Time': ['mean', 'std'],
        'Log_Amount': 'mean'
    }).round(2)
    
    return cluster_stats

## Visualisation PCA
def create_pca_visualization(X_scaled, clusters, n_clusters):
    """
    Crée une visualisation PCA des clusters
    
    Args:
        X_scaled: Données normalisées
        clusters: Labels des clusters
        n_clusters: Nombre de clusters
        
    Retourne:
        Figure Plotly
    """
    # Réduction à 2 dimensions avec PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Création du DataFrame pour Plotly
    df_pca = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': clusters
    })
    
    # Graphique interactif
    fig = px.scatter(
        df_pca, 
        x='PC1', 
        y='PC2', 
        color='Cluster',
        title='Visualisation des Clusters (Projection PCA)',
        labels={'PC1': 'Composante Principale 1', 'PC2': 'Composante Principale 2'},
        color_continuous_scale='viridis'
    )
    
    fig.update_traces(marker=dict(size=5, opacity=0.6))
    fig.update_layout(height=500)
    
    return fig

## Interface Principale

# En-tête
st.markdown('<h1 class="main-header"> Dashboard de Segmentation des Transactions Bancaires</h1>', 
            unsafe_allow_html=True)
st.markdown("**Objectif** : Analyser et regrouper les transactions selon leurs caractéristiques pour identifier des comportements atypiques")

# Chargement des données
df = load_data()
df = preprocess_data(df)

# Sidebar

st.sidebar.title("Paramètres de Segmentation")
st.sidebar.markdown("---")

# Sélection du nombre de clusters
n_clusters = st.sidebar.slider(
    "Nombre de clusters (segments)",
    min_value=2,
    max_value=10,
    value=4,
    help="Plus il y a de clusters, plus la segmentation est fine"
)

# Sélection des features pour le clustering
st.sidebar.markdown("### Variables pour la segmentation")
use_amount = st.sidebar.checkbox("Montant (Amount)", value=True)
use_time = st.sidebar.checkbox("Temps (Time)", value=True)
use_pca_features = st.sidebar.checkbox("Variables PCA (V1-V28)", value=True)

# Construction de la liste des features
features_to_use = []
if use_amount:
    features_to_use.append('Amount')
if use_time:
    features_to_use.append('Time')
if use_pca_features:
    features_to_use.extend([f'V{i}' for i in range(1, 29)])

if len(features_to_use) == 0:
    st.sidebar.error("Veuillez sélectionner au moins une variable!")
    st.stop()

st.sidebar.markdown(f"**{len(features_to_use)} variables sélectionnées**")
st.sidebar.markdown("---")

# Filtres sur les données
st.sidebar.markdown("### Filtres des données")
amount_range = st.sidebar.slider(
    "Plage de montants (€)",
    float(df['Amount'].min()),
    float(df['Amount'].max()),
    (float(df['Amount'].min()), float(df['Amount'].max()))
)

# Application des filtres
df_filtered = df[(df['Amount'] >= amount_range[0]) & (df['Amount'] <= amount_range[1])]

## Statistiques globales


st.markdown('<h2 class="sub-header">Statistiques Globales</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Volume de transactions",
        value=f"{len(df_filtered):,}",
        delta=f"{len(df_filtered) - len(df):,}" if len(df_filtered) != len(df) else None
    )

with col2:
    st.metric(
        label="Montant moyen",
        value=f"{df_filtered['Amount'].mean():.2f}€"
    )

with col3:
    st.metric(
        label="Montant médian",
        value=f"{df_filtered['Amount'].median():.2f}€"
    )

with col4:
    st.metric(
        label="Montant total",
        value=f"{df_filtered['Amount'].sum():,.0f}€"
    )

## Exploration des données

st.markdown('<h2 class="sub-header">Exploration des Données</h2>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Distributions", "Analyse temporelle", "Données brutes"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution des montants
        fig_amount = px.histogram(
            df_filtered, 
            x='Amount',
            nbins=50,
            title='Distribution des Montants',
            labels={'Amount': 'Montant (€)', 'count': 'Nombre de transactions'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_amount.update_layout(height=400)
        st.plotly_chart(fig_amount, use_container_width=True)
    
    with col2:
        # Box plot des montants par catégorie
        fig_box = px.box(
            df_filtered,
            y='Amount',
            x='Amount_Category',
            title='Montants par Catégorie',
            labels={'Amount': 'Montant (€)', 'Amount_Category': 'Catégorie'},
            color='Amount_Category'
        )
        fig_box.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

with tab2:
    # Analyse temporelle
    fig_time = px.scatter(
        df_filtered.sample(min(1000, len(df_filtered))),  # Échantillon pour performance
        x='Time_Hours',
        y='Amount',
        title='Transactions dans le temps',
        labels={'Time_Hours': 'Temps (heures)', 'Amount': 'Montant (€)'},
        color='Amount',
        color_continuous_scale='viridis'
    )
    fig_time.update_layout(height=400)
    st.plotly_chart(fig_time, use_container_width=True)

with tab3:
    st.dataframe(df_filtered.head(100), use_container_width=True)

##Segmentation

st.markdown('<h2 class="sub-header">Segmentation des Transactions</h2>', unsafe_allow_html=True)

# Lancement du clustering
with st.spinner('Calcul de la segmentation en cours...'):
    df_clustered, kmeans, scaler, X_scaled = perform_clustering(
        df_filtered, 
        n_clusters, 
        features_to_use
    )

st.success(f"Segmentation réussie : {n_clusters} clusters identifiés")

# Visualisation PCA
st.markdown("### Visualisation des Clusters")
fig_pca = create_pca_visualization(X_scaled, df_clustered['Cluster'].values, n_clusters)
st.plotly_chart(fig_pca, use_container_width=True)

# Statistiques des clusters
st.markdown("### Statistiques par Cluster")
cluster_stats = analyze_clusters(df_clustered)
st.dataframe(cluster_stats, use_container_width=True)

# Distribution des clusters
col1, col2 = st.columns(2)

with col1:
    # Taille des clusters
    cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
    fig_pie = px.pie(
        values=cluster_counts.values,
        names=[f'Cluster {i}' for i in cluster_counts.index],
        title='Répartition des transactions par cluster',
        hole=0.3
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    # Montant moyen par cluster
    avg_amounts = df_clustered.groupby('Cluster')['Amount'].mean().sort_index()
    fig_bar = px.bar(
        x=[f'Cluster {i}' for i in avg_amounts.index],
        y=avg_amounts.values,
        title='Montant Moyen par Cluster',
        labels={'x': 'Cluster', 'y': 'Montant moyen (€)'},
        color=avg_amounts.values,
        color_continuous_scale='blues'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

##Interprétation

st.markdown('<h2 class="sub-header">Analyse Interprétative des Profils</h2>', unsafe_allow_html=True)

# Analyse automatique des profils
st.markdown("### Description des Profils Transactionnels")

for cluster_id in range(n_clusters):
    cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
    
    # Calcul des caractéristiques
    n_transactions = len(cluster_data)
    avg_amount = cluster_data['Amount'].mean()
    median_amount = cluster_data['Amount'].median()
    std_amount = cluster_data['Amount'].std()
    
    # Détermination du profil
    if avg_amount < df_filtered['Amount'].quantile(0.25):
        profil = "**Profil : Petites transactions**"
        description = f"Ce cluster contient {n_transactions} transactions de faible montant (moyenne: {avg_amount:.2f}€). Il peut s'agir de paiements quotidiens ou de micro-transactions."
        risque = "Risque faible"
    elif avg_amount > df_filtered['Amount'].quantile(0.75):
        profil = "**Profil : Transactions importantes**"
        description = f"Ce cluster regroupe {n_transactions} transactions de montant élevé (moyenne: {avg_amount:.2f}€). Nécessite une surveillance accrue."
        risque = "🟡 Risque modéré - Surveiller"
    else:
        profil = "**Profil : Transactions standard**"
        description = f"Ce cluster comprend {n_transactions} transactions de montant moyen (moyenne: {avg_amount:.2f}€). Comportement transactionnel normal."
        risque = "🟢 Risque faible"
    
    # Ajout de l'analyse de variabilité
    if std_amount > avg_amount * 0.8:
        variabilite = "Forte variabilité des montants (comportement hétérogène)"
    else:
        variabilite = "Variabilité modérée (comportement homogène)"
    
    # Affichage
    with st.expander(f"**Cluster {cluster_id}** - {n_transactions} transactions ({n_transactions/len(df_filtered)*100:.1f}%)"):
        st.markdown(profil)
        st.write(description)
        st.markdown(f"**{risque}**")
        st.markdown(f"- Montant moyen: **{avg_amount:.2f}€**")
        st.markdown(f"- Montant médian: **{median_amount:.2f}€**")
        st.markdown(f"- Écart-type: **{std_amount:.2f}€**")
        st.markdown(f"- {variabilite}")
