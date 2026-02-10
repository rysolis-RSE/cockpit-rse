import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from fpdf import FPDF

# --- FONCTION PDF ---
def generer_pdf(dataframe, cout_total):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Rapport de Performance RSE & Financier", 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Nombre de filiales : {len(dataframe)}", 0, 1)
    pdf.cell(0, 10, f"Risque Financier Estime : {cout_total:,.0f} Euros", 0, 1) # Ajout du coût
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    for index, row in dataframe.iterrows():
        nom = row['Société'].encode('latin-1', 'replace').decode('latin-1')
        txt = f"- {nom} : Croissance {row['Croissance_Clients']:.1f}% | CO2 {row['Evolution_CO2']:.1f}%"
        pdf.cell(0, 8, txt, 0, 1)
    return pdf.output(dest='S').encode('latin-1')

# --- CONFIGURATION ---
st.set_page_config(page_title="Cockpit RSE - Voyageurs du Monde", layout="wide")
st.title("🌍 Cockpit RSE : Impact Carbone & Financier")

# --- 1. CHARGEMENT ---
st.sidebar.header("📥 Données")
uploaded_file = st.sidebar.file_uploader("Fichier Excel/CSV", type=["xlsx", "csv"])

if uploaded_file:
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
    except:
        st.error("Erreur de fichier")
        st.stop()
else:
    # DÉMO
    data = {
        'Société': ['Terres d\'Aventure', 'Allibert Trekking', 'Voyageurs du Monde', 'Comptoir des Voyages', 'Nomade Aventure'],
        'Clients_N': [34479, 26913, 39496, 29109, 14381],
        'Clients_N_1': [34093, 26962, 39198, 29048, 14195],
        'CO2_N': [48053000, 32203000, 85222000, 57221000, 28492000],
        'CO2_N_1': [52996000, 31958000, 99687000, 64170000, 31698000]
    }
    df = pd.DataFrame(data)

# --- 2. CALCULS ---
try:
    col_mapping = {'Nb pax 2025': 'Clients_N', 'Nb pax 2024': 'Clients_N_1', 'CO2 2025': 'CO2_N', 'CO2 2024': 'CO2_N_1', 'Société': 'Société'}
    df = df.rename(columns=col_mapping)
    for col in ['Clients_N', 'Clients_N_1', 'CO2_N', 'CO2_N_1']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['CO2_N', 'Clients_N'])
    df = df[~df['Société'].astype(str).str.contains('Total|Société', case=False, na=False)]

    df['Croissance_Clients'] = ((df['Clients_N'] - df['Clients_N_1']) / df['Clients_N_1']) * 100
    df['Evolution_CO2'] = ((df['CO2_N'] - df['CO2_N_1']) / df['CO2_N_1']) * 100
    df['Intensite'] = (df['CO2_N'] / df['Clients_N']) / 1000
except:
    st.error("Erreur de colonnes")
    st.stop()

# --- 3. NOUVEAU : MODULE FINANCE (SIDEBAR) ---
st.sidebar.divider()
st.sidebar.header("💶 Simulateur Taxe Carbone")
st.sidebar.info("Simulez l'impact financier d'une taxe carbone européenne.")

# Curseurs
prix_tonne = st.sidebar.slider("Prix de la Tonne (€)", 0, 200, 80, 10)
quota_gratuit = st.sidebar.slider("Part Gratuite (Quota) %", 0, 100, 20, 10) / 100

# Calcul Financier
total_co2_tonnes = df['CO2_N'].sum() / 1000
emissions_taxables = total_co2_tonnes * (1 - quota_gratuit)
cout_total = emissions_taxables * prix_tonne

st.sidebar.metric(
    "Coût Estime / an",
    f"{cout_total:,.0f} €".replace(",", " "),
    f"{prix_tonne}€ / tonne",
    delta_color="inverse"
)

# Bouton PDF mis à jour avec le prix
if st.sidebar.button("📄 Télécharger Rapport"):
    pdf_bytes = generer_pdf(df, cout_total)
    st.sidebar.download_button("📥 PDF Prêt", pdf_bytes, "Rapport_Financier.pdf", "application/pdf")

# --- 4. VISUALISATIONS ---
col1, col2, col3 = st.columns(3)
col1.metric("Filiales", len(df))
col2.metric("Total CO2 Groupe", f"{total_co2_tonnes:,.0f} t")
col3.metric("Impact Financier", f"{cout_total:,.0f} €", delta="Risque", delta_color="inverse")

st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Performance", "🔮 IA & Futur", "📋 Données"])

with tab1:
    st.subheader("Matrice Stratégique")
    fig = px.scatter(df, x="Croissance_Clients", y="Evolution_CO2", size="CO2_N", color="Intensite",
        hover_name="Société", color_continuous_scale="RdYlGn_r", title="Positionnement (Vert = Bien)")
    fig.add_hline(y=0, line_dash="dot"); fig.add_vline(x=0, line_dash="dot")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🤖 Segmentation IA")
    X = df[['Croissance_Clients', 'Evolution_CO2', 'Intensite']].fillna(0)
    if len(df) >= 3:
        kmeans = KMeans(n_clusters=3, n_init=10).fit(X)
        df['Cluster'] = kmeans.labels_
        st.write("L'IA a identifié 3 groupes de performance distincts (0, 1, 2).")
        st.bar_chart(df.set_index('Société')['Cluster'])
    else:
        st.warning("Pas assez de données pour l'IA.")

with tab3:
    st.dataframe(df)
