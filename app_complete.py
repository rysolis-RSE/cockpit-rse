import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from fpdf import FPDF
import tempfile

# ==============================================================================
# 1. CERVEAU DU CHATBOT (IA Symbolique)
# ==============================================================================
def reponse_chatbot(question, df):
    question = question.lower()
    
    # Scénario 1 : Le Meilleur
    if "meilleur" in question or "top" in question or "champion" in question:
        best = df.loc[df['Decouplage'].idxmax()]
        return f"🏆 Le champion du découplage est **{best['Société']}**.\n\nSon score est de {best['Decouplage']:.1f} pts (Baisse CO2: {best['Evolution_CO2']:.1f}% pour une croissance de {best['Croissance_Clients']:.1f}%)."
    
    # Scénario 2 : Le Pire
    elif "pire" in question or "mauvais" in question or "retard" in question:
        worst = df.loc[df['Decouplage'].idxmin()]
        return f"⚠️ La filiale la plus en retard est **{worst['Société']}**.\n\nSon CO2 a augmenté de {worst['Evolution_CO2']:.1f}% alors que sa croissance n'est que de {worst['Croissance_Clients']:.1f}%."
    
    # Scénario 3 : Les Moyennes
    elif "moyenne" in question:
        avg_co2 = df['Evolution_CO2'].mean()
        avg_croissance = df['Croissance_Clients'].mean()
        return f"📊 **Moyennes du Groupe** :\n- Croissance Clients : {avg_croissance:.1f}%\n- Évolution CO2 : {avg_co2:.1f}%"
    
    # Scénario 4 : Recherche par nom de société
    else:
        found = False
        for societe in df['Société']:
            if str(societe).lower() in question:
                data = df[df['Société'] == societe].iloc[0]
                return f"🔎 **Focus sur {societe}** :\n- Croissance Clients : {data['Croissance_Clients']:.1f}%\n- Évolution CO2 : {data['Evolution_CO2']:.1f}%\n- Intensité : {data['Intensite']:.2f} t/pax\n- Score Découplage : {data['Decouplage']:.1f}"
        
        return "🤖 Je suis votre Assistant RSE. Posez-moi des questions comme : **'Qui est le meilleur ?'**, **'Quelle est la moyenne ?'**, ou **'Donne moi les chiffres de Terdav'**."

# ==============================================================================
# 2. FONCTION DE GÉNÉRATION PDF (Avec Finance)
# ==============================================================================
def generer_pdf(dataframe, cout_financier):
    pdf = FPDF()
    pdf.add_page()
    
    # En-tête
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Rapport de Pilotage RSE & Financier", 0, 1, 'C')
    pdf.ln(10)

    # Résumé Financier
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Nombre de filiales auditees : {len(dataframe)}", 0, 1)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(255, 0, 0) # Rouge pour l'argent
    pdf.cell(0, 10, f"RISQUE FINANCIER (Taxe Carbone) : {cout_financier:,.0f} Euros".replace(",", " "), 0, 1)
    pdf.set_text_color(0, 0, 0) # Retour au noir
    pdf.ln(5)
    
    # Résumé Carbone
    croissance_moy = dataframe['Croissance_Clients'].mean()
    co2_moy = dataframe['Evolution_CO2'].mean()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Croissance Moyenne Activite : {croissance_moy:.2f}%", 0, 1)
    pdf.cell(0, 10, f"Evolution Moyenne CO2 : {co2_moy:.2f}%", 0, 1)
    pdf.ln(10)

    # Tableau Détaillé
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Detail par Filiale", 0, 1)
    pdf.set_font("Arial", size=10)
    
    for index, row in dataframe.iterrows():
        # Gestion basique des caractères spéciaux pour le PDF
        nom = str(row['Société']).encode('latin-1', 'replace').decode('latin-1')
        txt = f"- {nom} : Croissance {row['Croissance_Clients']:.1f}% | CO2 {row['Evolution_CO2']:.1f}% | Intensite {row['Intensite']:.2f} t/pax"
        pdf.cell(0, 8, txt, 0, 1)
        
    return pdf.output(dest='S').encode('latin-1')

# ==============================================================================
# 3. CONFIGURATION ET CHARGEMENT
# ==============================================================================
st.set_page_config(page_title="Cockpit RSE - Voyageurs du Monde", layout="wide")

st.title("🌍 Cockpit de Pilotage RSE & IA")
st.markdown("Analysez le passé, simulez le futur, évaluez le risque financier et discutez avec vos données.")

# --- CHARGEMENT DES DONNÉES ---
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
    # DONNÉES DE DÉMO (Pour que l'appli ne soit pas vide au démarrage)
    st.info("👆 En attente de fichier... Chargement de la démo.")
    data = {
        'Société': ['Terres d\'Aventure', 'Allibert Trekking', 'Voyageurs du Monde', 'Comptoir des Voyages', 'Nomade Aventure', 'Bivouac', 'Original Travel'],
        'Clients_N': [34479, 26913, 39496, 29109, 14381, 1500, 3183],
        'Clients_N_1': [34093, 26962, 39198, 29048, 14195, 1400, 2853],
        'CO2_N': [48053000, 32203000, 85222000, 57221000, 28492000, 2500000, 8175000],
        'CO2_N_1': [52996000, 31958000, 99687000, 64170000, 31698000, 2600000, 8004000]
    }
    df = pd.DataFrame(data)

# ==============================================================================
# 4. MOTEUR DE CALCUL (NETTOYAGE & KPIs)
# ==============================================================================
try:
    # A. Standardisation des colonnes
    col_mapping = {
        'Nb pax 2025': 'Clients_N', 'Nb pax 2024': 'Clients_N_1',
        'CO2 2025': 'CO2_N', 'CO2 2024': 'CO2_N_1', 'Société': 'Société'
    }
    df = df.rename(columns=col_mapping)

    # B. Conversion en nombres
    cols_to_numeric = ['Clients_N', 'Clients_N_1', 'CO2_N', 'CO2_N_1']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # C. Nettoyage des lignes vides/totaux
    df = df.dropna(subset=['CO2_N', 'Clients_N', 'Société']) 
    df = df[~df['Société'].astype(str).str.contains('Total', case=False, na=False)]
    df = df[~df['Société'].astype(str).str.contains('Société', case=False, na=False)]

    # D. Calculs KPIs
    df['Croissance_Clients'] = ((df['Clients_N'] - df['Clients_N_1']) / df['Clients_N_1']) * 100
    df['Evolution_CO2'] = ((df['CO2_N'] - df['CO2_N_1']) / df['CO2_N_1']) * 100
    df['Intensite'] = (df['CO2_N'] / df['Clients_N']) / 1000 # Conversion en Tonnes
    df['Decouplage'] = df['Evolution_CO2'] - df['Croissance_Clients']
    
except Exception as e:
    st.error(f"❌ Erreur de structure de fichier : {e}")
    st.stop()

# ==============================================================================
# 5. SIDEBAR : MODULE FINANCE & EXPORT PDF
# ==============================================================================
st.sidebar.divider()
st.sidebar.header("💶 Risque Financier")
st.sidebar.info("Simulateur de Taxe Carbone (Shadow Pricing)")

# Paramètres financiers
prix_tonne = st.sidebar.slider("Prix de la Tonne CO2 (€)", 0, 200, 80, 5)
quota_gratuit = st.sidebar.slider("Quota Gratuit (%)", 0, 100, 20, 10) / 100

# Calcul du coût
total_co2_tonnes = df['CO2_N'].sum() / 1000
emissions_taxables = total_co2_tonnes * (1 - quota_gratuit)
cout_total = emissions_taxables * prix_tonne

# Affichage du coût
st.sidebar.metric(
    label="Coût Annuel Estimé",
    value=f"{cout_total:,.0f} €".replace(",", " "),
    delta=f"Base: {prix_tonne}€/t",
    delta_color="inverse"
)

st.sidebar.divider()
st.sidebar.header("🖨️ Export")

if st.sidebar.button("Générer le Rapport PDF"):
    try:
        pdf_bytes = generer_pdf(df, cout_total)
        st.sidebar.download_button(
            label="📥 Télécharger le PDF",
            data=pdf_bytes,
            file_name="Rapport_RSE_Financier.pdf",
            mime="application/pdf"
        )
        st.sidebar.success("PDF prêt !")
    except Exception as e:
        st.sidebar.error(f"Erreur PDF: {e}")

# ==============================================================================
# 6. DASHBOARD CENTRAL (KPIS & ONGLETS)
# ==============================================================================

# --- KPIs GLOBAUX ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Filiales Analysées", len(df))
col2.metric("Croissance Moy.", f"{df['Croissance_Clients'].mean():.1f}%")
col3.metric("Baisse CO2 Moy.", f"{df['Evolution_CO2'].mean():.1f}%")
col4.metric("Score Découplage", f"{(df['Evolution_CO2'].mean() - df['Croissance_Clients'].mean()):.1f}", delta_color="inverse")

st.divider()

# --- LES 6 ONGLETS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Analyse Découplage", 
    "🎯 Matrice Stratégique", 
    "📋 Données Détaillées", 
    "🔮 Simulateur 2030", 
    "🤖 Segmentation IA", 
    "💬 Chatbot RSE"
])

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

# ONGLET 4 : SIMULATEUR 2030
with tab4:
    st.header("🔮 Simulateur de Trajectoire")
    st.markdown("Jouez avec les hypothèses pour voir si le groupe atteindra ses objectifs.")
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

# ONGLET 5 : IA CLUSTERING (VERSION LUXE)
with tab5:
    st.header("🤖 Segmentation Automatique (K-Means)")
    
    # Préparation IA
    X = df[['Croissance_Clients', 'Evolution_CO2', 'Intensite']].copy().fillna(0)
    
    if len(df) >= 3:
        # 1. Calcul des clusters
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X)
        
        # 2. L'IA analyse qui est le "bon" et le "mauvais" élève
        # On calcule la moyenne de CO2 pour chaque groupe pour savoir lequel pollue le moins
        centers = df.groupby('Cluster')[['Evolution_CO2']].mean()
        sorted_clusters = centers.sort_values('Evolution_CO2').index
        
        # 3. On donne des noms intelligents (Mapping)
        label_map = {
            sorted_clusters[0]: "🌟 Leaders (Modèles)", 
            sorted_clusters[1]: "⚖️ Stables (Moyenne)", 
            sorted_clusters[2]: "⚠️ À Risque (Hausse CO2)"
        }
        df['Nom_Cluster'] = df['Cluster'].map(label_map)
        
        # 4. Affichage du Graphique avec les nouveaux noms
        fig_cluster = px.scatter(df, x="Croissance_Clients", y="Evolution_CO2", color="Nom_Cluster",
            size="CO2_N", hover_name="Société", symbol="Nom_Cluster", 
            title="Cartographie IA des Profils", height=500,
            color_discrete_map={
                "🌟 Leaders (Modèles)": "#2ecc71", # Vert
                "⚖️ Stables (Moyenne)": "#f1c40f", # Jaune
                "⚠️ À Risque (Hausse CO2)": "#e74c3c" # Rouge
            }
        )
        fig_cluster.add_hline(y=0, line_dash="dot")
        fig_cluster.add_vline(x=0, line_dash="dot")
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        st.success("L'algorithme a identifié les 3 profils automatiquement selon leur performance CO2.")
    else:
        st.warning("Il faut au moins 3 filiales pour lancer l'IA.")

# ONGLET 6 : CHATBOT
with tab6:
    st.header("💬 Assistant Virtuel")
    st.markdown("Posez une question : *'Qui est le meilleur ?'*, *'Donne moi les chiffres de Terdav'*")
    
    # Initialisation de l'historique
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Affichage des anciens messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Zone de saisie
    if prompt := st.chat_input("Votre question..."):
        # Affiche la question utilisateur
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Calcule et affiche la réponse IA
        response = reponse_chatbot(prompt, df)
        
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

