import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from pandas.api.types import is_numeric_dtype
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import statsmodels.api as sm

st.set_page_config(page_title='Hit Songs', page_icon="🎵", layout="centered")
st.title('🎵 Hit Songs')

DATA_PATH = 'data/spotify.csv'


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def fill_nan(df: pd.DataFrame):
    for c in df.columns:
        if df[c].isna().any():
            if is_numeric_dtype(df[c]):
                df.fillna({c: df[c].mean()}, inplace=True)
            else:
                df.fillna({c: df[c].mode().iloc[0]}, inplace=True)


def remove_outliers(data, col):
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return data[(data[col] >= lower) & (data[col] <= upper)]


def amplitudine(x):
    return x.max() - x.min()


def style_plot(fig, ax):
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="white")
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

try:
    df_init = load_data(DATA_PATH)
    st.success('✅ Dataset incarcat cu succes')
except Exception as e:
    st.error('❌ Nu se poate incarca datasetul. Verifica path-ul!')
    st.exception(e)
    st.stop()

df = df_init.copy()
fill_nan(df)

if 'tempo' in df.columns:
    nr_initial = len(df)
    df = remove_outliers(df, 'tempo')
    nr_final = len(df)

if 'popularity' in df.columns:
    df['hit'] = (df['popularity'] > 70).astype(int)
else:
    st.error("Coloana 'popularity' nu exista in dataset")
    st.stop()

df['hit_label'] = df['hit'].map({0: 'non-HIT', 1: 'HIT'})

top_melodii = df[['track_name', 'artists', 'album_name', 'track_genre', 'popularity']] \
    .sort_values('popularity', ascending=False) \
    .drop_duplicates(subset=['track_name', 'artists']) \
    .head(5)

st.sidebar.title('🎵 Top 5 melodii populare')
for i, (_, row) in enumerate(top_melodii.iterrows(), start=1):
    st.sidebar.markdown(f"### #{i} {row['track_name']}")
    st.sidebar.write(f"**Artist:** {row['artists']}")
    st.sidebar.write(f"**Album:** {row['album_name']}")
    st.sidebar.write(f"**Gen:** {row['track_genre']}")
    st.sidebar.write(f"**Popularitate:** {row['popularity']}")
    st.sidebar.divider()

csv = df.to_csv(index=False).encode('utf-8')
st.sidebar.subheader('Export')
st.sidebar.download_button(
    label='Descarca datasetul prelucrat',
    data=csv,
    file_name='spotify_prelucrat.csv',
    mime='text/csv'
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Date & Curatare",
    "Analiza",
    "Transformari",
    "Modele",
])

with tab1:
    st.subheader('Preview dataset')
    st.write(f'Shape: {df_init.shape[0]} randuri si {df_init.shape[1]} coloane')
    coloane_preview = [c for c in ['track_name', 'artists', 'album_name', 'track_genre', 'popularity'] if
                       c in df_init.columns]
    st.dataframe(df_init[coloane_preview].head(10), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader('Analiza valorilor lipsa din setul de date')
    total = df_init.isnull().sum().sort_values(ascending=False)
    percent = (df_init.isnull().sum() * 100 / df_init.isnull().count()).sort_values(ascending=False)
    missing = pd.concat([total, percent], axis=1, keys=['Total', 'Procent'])
    with st.expander('Vezi analiza valorilor lipsa'):
        st.dataframe(missing.reset_index(), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader('Tratarea valorilor lipsa')
    st.write(f'Valori lipsa ramase dupa prelucrare: {df.isnull().sum().sum()}')

    st.divider()

    st.subheader('Tratarea valorilor extreme')
    if 'tempo' in df.columns:
        col1, col2 = st.columns(2)
        col1.metric('Observatii inainte', nr_initial)
        col2.metric('Observatii dupa', nr_final)
    else:
        st.warning("Coloana 'tempo' nu exista in dataset")

with tab2:
    st.subheader('Statistici')
    coloane_statistici = [c for c in ['popularity', 'danceability', 'energy', 'tempo', 'valence'] if c in df.columns]
    with st.expander('Vezi statisticile descriptive'):
        st.dataframe(df[coloane_statistici].describe().round(2), use_container_width=True)

    st.divider()

    st.subheader('Gruparea si agregarea datelor')
    hit_stats = df.groupby('hit_label').agg({
        'popularity': ['mean', 'count', 'max', 'min'],
        'danceability': 'mean',
        'energy': 'mean',
        'tempo': 'mean'
    }).reset_index()

    hit_stats.columns = [
        'Tip melodie',
        'Popularitate medie',
        'Numar melodii',
        'Popularitate maxima',
        'Popularitate minima',
        'Danceability medie',
        'Energy medie',
        'Tempo mediu'
    ]

    col1, col2, col3 = st.columns(3)
    col1.metric('Total melodii', len(df))
    col2.metric('Numar HIT-uri', int((df['hit'] == 1).sum()))
    col3.metric('Numar non-HIT-uri', int((df['hit'] == 0).sum()))

    st.dataframe(hit_stats, use_container_width=True, hide_index=True)

    if 'track_genre' in df.columns:
        genre_stats = df.groupby('track_genre').agg({
            'popularity': ['mean', 'count'],
            'duration_ms': 'mean'
        }).sort_values(('popularity', 'mean'), ascending=False).head(10)

        st.write('Top 10 genuri dupa popularitatea medie')
        st.dataframe(genre_stats.reset_index(), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader('Utilizare functiilor de grup')
    result = df.groupby('hit_label')['energy'].apply(amplitudine)
    st.dataframe(result.reset_index(name='amplitudine_energy'), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader('Reprezentari grafice')
    fig, ax = plt.subplots(figsize=(5, 3))
    df['hit_label'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Distributia melodiilor HIT vs non-HIT')
    plt.xticks(rotation=0)
    style_plot(fig, ax)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

    if 'track_genre' in df.columns:
        top_genuri = df['track_genre'].value_counts().head(10)
        top_genuri_df = top_genuri.reset_index()
        top_genuri_df.columns = ['Gen', 'Numar melodii']
        st.write('Top 10 genuri dupa numarul de melodii')
        st.dataframe(top_genuri_df, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader('Corelatia dintre variabilele numerice')
    corr = df.select_dtypes(include=[np.number]).corr()
    with st.expander('Vizualizeaza matricea de corelatie'):
        st.dataframe(corr, use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(corr, aspect='auto')
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8, color='white')
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=8, color='white')
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')
    style_plot(fig, ax)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

with tab3:
    st.subheader('Codificarea datelor')
    if 'track_genre' in df.columns:
        encoder = LabelEncoder()
        df['track_genre_encoded'] = encoder.fit_transform(df['track_genre'])
        st.dataframe(
            df[['track_genre', 'track_genre_encoded']].drop_duplicates().head(10),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("Coloana 'track_genre' nu exista in dataset")

    st.divider()

    st.subheader('Selectbox alegerea genului')
    if 'track_genre' in df.columns:
        gen_selectat = st.selectbox(
            'Alege un gen muzical',
            ['Toate'] + sorted(df['track_genre'].unique().tolist())
        )

        if gen_selectat != 'Toate':
            df_filtered = df[df['track_genre'] == gen_selectat]
        else:
            df_filtered = df.copy()

        coloane_filtrate = [c for c in ['track_name', 'artists', 'album_name', 'track_genre', 'popularity'] if
                            c in df_filtered.columns]
        st.dataframe(df_filtered[coloane_filtrate].head(10), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader('Metode de scalare')
    coloane_scalare = ['danceability', 'energy', 'tempo', 'valence']
    coloane_scalare = [c for c in coloane_scalare if c in df.columns]

    if len(coloane_scalare) > 0:
        scaler_standard = StandardScaler()
        scaler_minmax = MinMaxScaler()

        df_standardizat = pd.DataFrame(
            scaler_standard.fit_transform(df[coloane_scalare]),
            columns=[c + '_std' for c in coloane_scalare]
        )

        df_minmax = pd.DataFrame(
            scaler_minmax.fit_transform(df[coloane_scalare]),
            columns=[c + '_minmax' for c in coloane_scalare]
        )

        col1, col2 = st.columns(2)
        with col1:
            st.write('Exemplu StandardScaler')
            st.dataframe(df_standardizat.head(10), use_container_width=True, hide_index=True)

        with col2:
            st.write('Exemplu MinMaxScaler')
            st.dataframe(df_minmax.head(10), use_container_width=True, hide_index=True)

with tab4:
    st.subheader('Scikit-learn: clusterizare KMeans')
    cluster_features = ['danceability', 'energy', 'valence', 'tempo']
    cluster_features = [c for c in cluster_features if c in df.columns]

    if len(cluster_features) >= 2:
        scaler = StandardScaler()
        X_cluster = scaler.fit_transform(df[cluster_features])

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X_cluster)

        cluster_counts = df['cluster'].value_counts().sort_index().reset_index()
        cluster_counts.columns = ['Cluster', 'Numar melodii']
        st.dataframe(cluster_counts, use_container_width=True, hide_index=True)

        cluster_profile = df.groupby('cluster')[cluster_features].mean().round(2)
        st.write('Profilul mediu al fiecarui cluster')
        st.dataframe(cluster_profile, use_container_width=True)

    st.divider()

    st.subheader('Scikit-learn: regresie logistica')
    model_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'tempo']
    model_features = [c for c in model_features if c in df.columns]

    if len(model_features) > 0:
        X = df[model_features]
        y = df['hit']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(max_iter=2000, class_weight='balanced')
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        coloane_afisare = [c for c in ['track_name', 'artists', 'track_genre', 'popularity'] if c in df.columns]
        info_test = df.loc[X_test.index, coloane_afisare].copy()

        rezultate_test_final = info_test.copy()
        rezultate_test_final['hit_real'] = y_test.values
        rezultate_test_final['hit_prezis'] = y_pred
        rezultate_test_final['prob_hit'] = y_proba

        hituri_prezise = rezultate_test_final[
            rezultate_test_final['hit_prezis'] == 1
            ].sort_values('prob_hit', ascending=False)

        st.metric('Acuratete model', f'{accuracy_score(y_test, y_pred):.4f}')

        col1, col2 = st.columns(2)

        with col1:
            st.write('Matrice de confuzie')
            conf_df = pd.DataFrame(
                confusion_matrix(y_test, y_pred),
                index=['Real non-HIT', 'Real HIT'],
                columns=['Prezis non-HIT', 'Prezis HIT']
            )
            st.dataframe(conf_df, use_container_width=True)

        with col2:
            st.write("Exemple melodii prezise ca HIT")
            if len(hituri_prezise) > 0:
                tabel_hit = hituri_prezise[
                    [c for c in ['track_name', 'artists', 'track_genre', 'popularity', 'prob_hit'] if
                     c in hituri_prezise.columns]
                ].head(10).copy()
                tabel_hit['prob_hit'] = (tabel_hit['prob_hit'] * 100).round(2)
                tabel_hit = tabel_hit.rename(columns={'prob_hit': 'Probabilitate HIT (%)'})
                st.dataframe(tabel_hit, use_container_width=True, hide_index=True)
            else:
                st.info("Modelul nu a prezis nicio melodie ca HIT pe setul de test.")

        with st.expander("Vezi raportul complet de clasificare"):
            raport_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
            st.dataframe(raport_df, use_container_width=True)

    st.divider()

    st.subheader('Statsmodels: regresie multipla')
    ols_features = ['danceability', 'energy', 'tempo', 'valence']
    ols_features = [c for c in ols_features if c in df.columns]

    if len(ols_features) > 0 and 'popularity' in df.columns:
        X_ols = df[ols_features]
        X_ols = sm.add_constant(X_ols)
        y_ols = df['popularity']

        model_ols = sm.OLS(y_ols, X_ols).fit()
        st.write(f'R-squared: {model_ols.rsquared:.4f}')
        st.caption('Modelul estimeaza cat de bine pot explica variabilele audio popularitatea melodiei.')

        with st.expander("Vezi sumarul complet al regresiei multiple"):
            st.text(model_ols.summary().as_text())
