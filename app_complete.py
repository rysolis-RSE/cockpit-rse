import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from fpdf import FPDF

# --- CERVEAU DU CHATBOT (IA Symbolique) ---
def reponse_chatbot(question, df):
    question = question.lower()
    if "meilleur" in question or "top" in question or "champion" in question:
        best = df.loc[df['Decouplage'].idxmax()]
        return f"🏆 Le champion du découplage est **{best['Société']}**.\n\nSon score est de {best['Decouplage']:.1f} pts (Baisse CO2: {best['Evolution_CO2']:.1f}% pour une croissance de {best['Croissance_Clients']:.1f}%)."
    elif "pire" in question or "mauvais" in question or "retard" in question:
        worst = df.loc[df['Decouplage'].idxmin()]
        return f"⚠️ La filiale la plus en retard est **{worst['Société']}**.\n\nSon CO2 a augmenté de {worst['Evolution_CO2']:.1f}% alors que sa croissance n'est que de {worst['Croissance_Clients']:.1f}%."
    elif "moyenne" in question:
        avg_co2 = df['Evolution_CO2'].mean()
        avg_croissance = df['Croissance_Clients'].mean()
        return f"📊 **Moyennes du Groupe** :\n- Croissance Clients : {avg_croissance:.1f}%\n- Évolution CO2 : {avg_co2:.1f}%"
    else:
        # Recherche par nom de société
        found = False
        for societe in df['Société']:
            if societe.lower() in question:
                data = df[df['Société'] == societe].iloc[0]
                return f"🔎 **Focus sur {societe}** :\n- Croissance Clients : {data['Croissance_Clients']:.1f}%\n- Évolution CO2 : {data['Evolution_CO2']:.1f}%\n- Intensité : {data['Intensite']:.2f} t/pax\n- Score Découplage : {data['Decouplage']:.1f}"
        
        return "🤖 Je suis programmé pour répondre aux questions sur : le **meilleur**, le **pire**, la **moyenne**, ou une **société** spécifique (ex: 'Terdav')."

# --- FONCTION PDF ---
def generer_pdf(dataframe, cout_financier, annee):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Rapport RSE & Financier - {annee}", 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Nombre de filiales : {len(dataframe)}", 0, 1)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"RISQUE FINANCIER ESTIME : {cout_financier:,.0f} Euros".replace(",", " "), 0, 1)
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Détail par Filiale", 0, 1)
    pdf.set_font("Arial", size=10)
    for index, row in dataframe.iterrows():
        nom = str(row['Société']).encode('latin-1', 'replace').decode('latin-1')
        txt = f"- {nom} : Croissance {row['Croissance_Clients']:.1f}% | CO2 {row['Evolution_CO2']:.1f}% | Intensite {row['Intensite']:.2f} t/pax"
        pdf.cell(0, 8, txt, 0, 1)
    return pdf.output(dest='S').encode('latin-1')

# --- CONFIGURATION ---
st.set_page_config(page_title="Cockpit RSE - Voyageurs du Monde", layout="wide")
st.title("🌍 Cockpit de Pilotage RSE & IA")

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
        st.error("Erreur lecture fichier"); st.stop()
else:
    # Démo
    data = {
        'Société': ['Terres d\'Aventure', 'Allibert Trekking', 'Voyageurs du Monde', 'Comptoir des Voyages', 'Nomade Aventure', 'Bivouac', 'Original Travel'],
        'Clients_N': [34479, 26913, 39496, 29109, 14381, 1500, 3183],
        'Clients_N_1': [34093, 26962, 39198, 29048, 14195, 1400, 2853],
        'CO2_N': [48053000, 32203000, 85222000, 57221000, 28492000, 2500000, 8175000],
        'CO2_N_1': [52996000, 31958000, 99687000, 64170000, 31698000, 2600000, 8004000]
    }
    df = pd.DataFrame(data)

# --- 2. CALCULS ---
try:
    col_mapping = {'Nb pax 2025': 'Clients_N', 'Nb pax 2024': 'Clients_N_1', 'CO2 2025': 'CO2_N', 'CO2 2024': 'CO2_N_1', 'Société': 'Société'}
    df = df.rename(columns=col_mapping)
    for col in ['Clients_N', 'Clients_N_1', 'CO2_N', 'CO2_N_1']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['CO2_N', 'Clients_N', 'Société'])
    df = df[~df['Société'].astype(str).str.contains('Total|Société', case=False, na=False)]
    
    df['Croissance_Clients'] = ((df['Clients_N'] - df['Clients_N_1']) / df['Clients_N_1']) * 100
    df['Evolution_CO2'] = ((df['CO2_N'] - df['CO2_N_1']) / df['CO2_N_1']) * 100
    df['Intensite'] = (df['CO2_N'] / df['Clients_N']) / 1000
    df['Decouplage'] = df['Evolution_CO2'] - df['Croissance_Clients']
except:
    st.error("Erreur colonnes"); st.stop()

# --- 3. FINANCE & PDF (SIDEBAR) ---
st.sidebar.divider()
st.sidebar.header("💶 Risque Financier")
prix_tonne = st.sidebar.slider("Prix Tonne CO2 (€)", 0, 200, 80, 10)
quota_gratuit = st.sidebar.slider("Quota Gratuit (%)", 0, 100, 20, 10) / 100
cout_total = (df['CO2_N'].sum() / 1000) * (1 - quota_gratuit) * prix_tonne
st.sidebar.metric("Coût Estime", f"{cout_total:,.0f} €".replace(",", " "), delta_color="inverse")

st.sidebar.divider()
if st.sidebar.button("Générer PDF"):
    try:
        pdf_bytes = generer_pdf(df, cout_total, 2025)
        st.sidebar.download_button("📥 Télécharger PDF", pdf_bytes, "Rapport_RSE.pdf", "application/pdf")
        st.sidebar.success("OK !")
    except Exception as e: st.sidebar.error(f"Erreur: {e}")

# --- 4. KPIs ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Filiales", len(df))
col2.metric("Croissance Moy.", f"{df['Croissance_Clients'].mean():.1f}%")
col3.metric("Baisse CO2 Moy.", f"{df['Evolution_CO2'].mean():.1f}%")
col4.metric("Score Découplage", f"{(df['Evolution_CO2'].mean() - df['Croissance_Clients'].mean()):.1f}", delta_color="inverse")

st.divider()

# --- 5. ONGLETS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Performance", "🎯 Matrice", "📋 Data", "🔮 Simulateur", "🤖 Segmentation", "💬 Chatbot"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Société'], y=df['Croissance_Clients'], name='Croissance %', marker_color='#2ecc71'))
    fig.add_trace(go.Bar(x=df['Société'], y=df['Evolution_CO2'], name='CO2 %', marker_color='#e74c3c'))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.scatter(df, x="Croissance_Clients", y="Evolution_CO2", size="CO2_N", color="Intensite", hover_name="Société", color_continuous_scale="RdYlGn_r")
    fig.add_hline(y=0, line_dash="dot"); fig.add_vline(x=0, line_dash="dot")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.dataframe(df)

with tab4:
    st.header("🔮 Simulation 2030")
    c1, c2 = st.columns(2)
    h_croissance = c1.slider("Croissance / an", 0.0, 10.0, 2.0) / 100
    h_reduction = c2.slider("Réduction Intensité / an", -10.0, 0.0, -4.0) / 100
    years = [2025, 2026, 2027, 2028, 2029, 2030]
    proj_co2 = [df['CO2_N'].sum()]
    proj_pax = [df['Clients_N'].sum()]
    for i in range(5):
        new_pax = proj_pax[-1] * (1 + h_croissance)
        proj_pax.append(new_pax)
        current_intensity = (proj_co2[-1] / proj_pax[-2])
        proj_co2.append(new_pax * (current_intensity * (1 + h_reduction)))
    
    st.metric("CO2 en 2030", f"{proj_co2[-1]/1000:,.0f} t", f"{((proj_co2[-1]-proj_co2[0])/proj_co2[0])*100:.1f}%", delta_color="inverse")
    fig_proj = px.line(x=years, y=proj_co2, markers=True, title="Trajectoire CO2 Absolue")
    fig_proj.add_hline(y=proj_co2[0]*0.7, line_dash="dot", annotation_text="Objectif -30%", line_color="green")
    st.plotly_chart(fig_proj, use_container_width=True)

with tab5:
    st.header("🤖 Clustering K-Means")
    X = df[['Croissance_Clients', 'Evolution_CO2', 'Intensite']].fillna(0)
    if len(df) >= 3:
        kmeans = KMeans(n_clusters=3, n_init=10).fit(X)
        df['Cluster'] = kmeans.labels_
        st.write("L'IA a détecté 3 profils de filiales (0, 1, 2).")
        fig_clust = px.scatter(df, x="Croissance_Clients", y="Evolution_CO2", color=df['Cluster'].astype(str), hover_name="Société")
        st.plotly_chart(fig_clust, use_container_width=True)

with tab6:
    st.header("💬 Assistant Virtuel")
    st.markdown("Posez une question : *'Qui est le meilleur ?'*, *'Donne moi les chiffres de Terdav'*")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Votre question..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        response = reponse_chatbot(prompt, df)
        
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

