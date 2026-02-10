import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from fpdf import FPDF
import tempfile

# --- FONCTION DE GÉNÉRATION PDF ---
def generer_pdf(dataframe):
    # Création du PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Titre
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Rapport de Performance RSE - Synthèse", 0, 1, 'C')
    pdf.ln(10)

    # Infos Générales
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Nombre de filiales analysées : {len(dataframe)}", 0, 1)
    
    croissance_moy = dataframe['Croissance_Clients'].mean()
    co2_moy = dataframe['Evolution_CO2'].mean()
    
    pdf.cell(0, 10, f"Croissance Moyenne Clients : {croissance_moy:.2f}%", 0, 1)
    pdf.cell(0, 10, f"Evolution Moyenne CO2 : {co2_moy:.2f}%", 0, 1)
    pdf.ln(10)

    # Détails par Filiale
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Détail par Filiale", 0, 1)
    pdf.set_font("Arial", size=10)
    
    for index, row in dataframe.iterrows():
        # Encodage latin-1 pour gérer les accents basiques
        nom = row['Société'].encode('latin-1', 'replace').decode('latin-1')
        txt = f"- {nom} : Croissance {row['Croissance_Clients']:.1f}% | CO2 {row['Evolution_CO2']:.1f}% | Intensite {row['Intensite']:.2f} t/pax"
        pdf.cell(0, 8, txt, 0, 1)
        
    return pdf.output(dest='S').encode('latin-1')

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Cockpit RSE - Voyageurs du Monde", layout="wide")

st.title("🌍 Cockpit de Performance Carbone & IA")
st.markdown("Analysez le passé, simulez le futur et laissez l'IA segmenter les filiales. **Glissez votre rapport ci-dessous.**")

# --- 1. IMPORTATION ET CHARGEMENT ---
st.sidebar.header("📥 Données")
uploaded_file = st.sidebar.file_uploader("Glissez votre fichier Excel ou CSV", type=["xlsx", "csv"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, header=1) 
                if 'Société' not in df.columns:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, header=0)
            except:
                df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("✅ Rapport chargé !")
    except Exception as e:
        st.error(f"Erreur de lecture : {e}")
        st.stop()
else:
    # DONNÉES DE DÉMO
    st.info("👆 En attente de fichier... Chargement de la démo.")
    data = {
        'Société': ['Terres d\'Aventure', 'Allibert Trekking', 'Voyageurs du Monde', 'Comptoir des Voyages', 'Nomade Aventure', 'Bivouac', 'Original Travel'],
        'Clients_N': [34479, 26913, 39496, 29109, 14381, 1500, 3183],
        'Clients_N_1': [34093, 26962, 39198, 29048, 14195, 1400, 2853],
        'CO2_N': [48053000, 32203000, 85222000, 57221000, 28492000, 2500000, 8175000],
        'CO2_N_1': [52996000, 31958000, 99687000, 64170000, 31698000, 2600000, 8004000]
    }
    df = pd.DataFrame(data)

# --- 2. MOTEUR DE CALCUL & NETTOYAGE ---
try:
    # Standardisation
    col_mapping = {
        'Nb pax 2025': 'Clients_N', 'Nb pax 2024': 'Clients_N_1',
        'CO2 2025': 'CO2_N', 'CO2 2024': 'CO2_N_1', 'Société': 'Société'
    }
    df = df.rename(columns=col_mapping)

    # Conversion nombres
    cols_to_numeric = ['Clients_N', 'Clients_N_1', 'CO2_N', 'CO2_N_1']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Nettoyage
    df = df.dropna(subset=['CO2_N', 'Clients_N', 'Société']) 
    df = df[~df['Société'].astype(str).str.contains('Total', case=False, na=False)]
    df = df[~df['Société'].astype(str).str.contains('Société', case=False, na=False)]

    # Calculs KPIs
    df['Croissance_Clients'] = ((df['Clients_N'] - df['Clients_N_1']) / df['Clients_N_1']) * 100
    df['Evolution_CO2'] = ((df['CO2_N'] - df['CO2_N_1']) / df['CO2_N_1']) * 100
    df['Intensite'] = (df['CO2_N'] / df['Clients_N']) / 1000 # En Tonnes
    df['Decouplage'] = df['Evolution_CO2'] - df['Croissance_Clients']
    
except Exception as e:
    st.error(f"❌ Erreur de structure de fichier : {e}")
    st.stop()

# --- 3. BOUTON PDF (SIDEBAR) ---
st.sidebar.divider()
st.sidebar.header("🖨️ Export")
if st.sidebar.button("Générer le Rapport PDF"):
    try:
        pdf_bytes = generer_pdf(df)
        st.sidebar.download_button(
            label="📥 Télécharger le PDF",
            data=pdf_bytes,
            file_name="Rapport_RSE_Groupe.pdf",
            mime="application/pdf"
        )
        st.sidebar.success("PDF généré !")
    except Exception as e:
        st.sidebar.error(f"Erreur PDF: {e}")

# --- 4. BANDEAU DE KPIS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Filiales Analysées", len(df))
col2.metric("Croissance Moy.", f"{df['Croissance_Clients'].mean():.1f}%")
col3.metric("Baisse CO2 Moy.", f"{df['Evolution_CO2'].mean():.1f}%")
decoupling = df['Evolution_CO2'].mean() - df['Croissance_Clients'].mean()
col4.metric("Score Découplage", f"{decoupling:.1f}", delta_color="inverse")

st.divider()

# --- 5. ONGLETS D'ANALYSE ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Analyse Découplage", "🎯 Matrice Stratégique", "📋 Données Détaillées", "🔮 Simulateur 2030", "🤖 Segmentation IA"])

# ONGLET 1 : DÉCOUPLAGE
with tab1:
    st.subheader("Qui grandit sans polluer ?")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Société'], y=df['Croissance_Clients'], name='Croissance Clients (%)', marker_color='#2ecc71'))
    fig.add_trace(go.Bar(x=df['Société'], y=df['Evolution_CO2'], name='Évolution CO2 (%)', marker_color='#e74c3c'))
    fig.update_layout(barmode='group', height=450, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ONGLET 2 : MATRICE
with tab2:
    st.subheader("Matrice de Performance (Intensité en t/pax)")
    fig_matrix = px.scatter(df, 
        x="Croissance_Clients", y="Evolution_CO2", size="CO2_N", color="Intensite",
        hover_name="Société", text="Société", color_continuous_scale="RdYlGn_r", size_max=60,
        labels={"Intensite": "Intensité (tCO2e/pax)"}, title="Positionnement Stratégique")
    
    fig_matrix.add_hline(y=0, line_dash="dot", opacity=0.5)
    fig_matrix.add_vline(x=0, line_dash="dot", opacity=0.5)
    fig_matrix.add_shape(type="rect", x0=0, y0=0, x1=100, y1=-100, line=dict(color="Green", width=0), fillcolor="Green", opacity=0.1)
    st.plotly_chart(fig_matrix, use_container_width=True)

# ONGLET 3 : DATA
with tab3:
    st.dataframe(df[['Société', 'Clients_N', 'CO2_N', 'Intensite', 'Decouplage']].style.format({
        'Intensite': '{:.3f} t/pax', 'Decouplage': '{:.1f}', 'CO2_N': '{:,.0f} kg'
    }))

# ONGLET 4 : SIMULATEUR
with tab4:
    st.header("🔮 Simulateur de Trajectoire")
    col_sim1, col_sim2 = st.columns(2)
    hyp_croissance = col_sim1.slider("Hypothèse Croissance Clients / an", 0.0, 10.0, 2.0, 0.5) / 100
    hyp_reduction = col_sim2.slider("Hypothèse Réduction Intensité / an", -10.0, 0.0, -4.0, 0.5) / 100

    years = [2025, 2026, 2027, 2028, 2029, 2030]
    proj_co2 = [df['CO2_N'].sum()]
    proj_pax = [df['Clients_N'].sum()]
    
    for i in range(5):
        new_pax = proj_pax[-1] * (1 + hyp_croissance)
        proj_pax.append(new_pax)
        current_intensity = (proj_co2[-1] / proj_pax[-2])
        proj_co2.append(new_pax * (current_intensity * (1 + hyp_reduction)))

    df_proj = pd.DataFrame({'Année': years, 'CO2 Projeté (kg)': proj_co2})
    var_globale = ((proj_co2[-1] - proj_co2[0]) / proj_co2[0]) * 100
    
    st.metric("Émission CO2 en 2030", f"{proj_co2[-1]/1000:,.0f} Tonnes", f"{var_globale:.1f}% vs 2025", delta_color="inverse")
    
    fig_proj = go.Figure()
    fig_proj.add_trace(go.Scatter(x=years, y=df_proj['CO2 Projeté (kg)'], mode='lines+markers', name='Trajectoire', line=dict(color='red', width=3)))
    fig_proj.add_hline(y=proj_co2[0]*0.7, line_dash="dot", annotation_text="Objectif -30%", line_color="green")
    st.plotly_chart(fig_proj, use_container_width=True)

# ONGLET 5 : IA CLUSTERING
with tab5:
    st.header("🤖 Segmentation Automatique (K-Means)")
    
    # Préparation IA
    X = df[['Croissance_Clients', 'Evolution_CO2', 'Intensite']].copy().fillna(0)
    
    if len(df) >= 3:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X)
        
        # Attribution des labels
        centers = df.groupby('Cluster')[['Evolution_CO2']].mean()
        sorted_clusters = centers.sort_values('Evolution_CO2').index
        label_map = {sorted_clusters[0]: "🌟 Leaders (Modèles)", sorted_clusters[1]: "⚖️ Stables (Moyenne)", sorted_clusters[2]: "⚠️ À Risque (Hausse CO2)"}
        df['Nom_Cluster'] = df['Cluster'].map(label_map)
        
        fig_cluster = px.scatter(df, x="Croissance_Clients", y="Evolution_CO2", color="Nom_Cluster",
            size="CO2_N", hover_name="Société", symbol="Nom_Cluster", title="Cartographie IA des Profils", height=500)
        fig_cluster.add_hline(y=0, line_dash="dot")
        fig_cluster.add_vline(x=0, line_dash="dot")
        st.plotly_chart(fig_cluster, use_container_width=True)
    else:
        st.warning("Il faut au moins 3 filiales pour lancer l'IA.")