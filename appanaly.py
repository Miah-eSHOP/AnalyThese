import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from scipy.stats import kruskal
from scipy.stats import linregress
from scipy.stats import t
import statsmodels.api as sm
from statsmodels.formula.api import ols

from io import BytesIO
import base64

# Fonction pour charger les données
def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.xls') or file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        raise Exception("Format de fichier non supporté. Veuillez choisir un fichier CSV ou Excel.")
    return df

# Fonction pour ajouter des annotations de pourcentage sur les barres
def add_percentages(ax):
    total = sum([bar.get_height() for bar in ax.patches])
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# Fonction pour obtenir le lien de téléchargement de l'image
def get_image_download_link(fig, filename, text):
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

 #Fonction pour tracer les graphiques pour une variable explicative
def plot_single_variable_chart(df, chart_type, x, hue=None):
    fig, ax = plt.subplots()
    if chart_type == 'Bar':
        sns.countplot(data=df, x=x, hue=hue, ax=ax, order=df[x].value_counts().index.sort_values())
        add_percentages(ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    elif chart_type == 'Histogramme':
        df_sorted = df.sort_values(by=x)
        sns.histplot(data=df_sorted, x=x, hue=hue, multiple="stack", ax=ax, bins=10, discrete=True)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
        add_percentages(ax)
        
    elif chart_type == 'Camembert':
        counts = df[x].value_counts()
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
        ax.axis('equal')#
    elif chart_type == 'Ligne':
        if x in df.columns:
            sns.lineplot(data=df, x=df.index, y=x, ax=ax)
    elif chart_type == 'Box Plot':
        if hue:
            sns.boxplot(data=df, x=x, y=hue, ax=ax)
        else:
            sns.boxplot(data=df, x=x, ax=ax)
        ax.set_xlabel(x)
        ax.set_ylabel(hue if hue else x)
        ax.set_title(f"Box Plot de {x}" + (f" par {hue}" if hue else ""))
    else:
        st.error("Type de graphique non supporté.")
    
    ax.set_title(f"Répartition pour {x}")
    ax.set_ylabel("Nombre d'individus")
    ax.set_xlabel(x)
    
    return fig, ax


# Fonction pour la distribution de la population
def population_distribution(df):
    st.title("Répartition de la Population d'Étude")

    # Affichage des données
    st.write("Aperçu des données :")
    st.write(df.head())

    # Sidebar pour la sélection des variables et du type de graphique
    st.sidebar.header("Sélection des Variables et du Type de Graphique")
    explanatory_variables = st.sidebar.multiselect("Choisissez les variables explicatives", df.columns, key='explanatory_variables_dist')
    chart_type = st.sidebar.selectbox("Choisissez le type de graphique", ["Bar", "Histogramme", "Camembert", "Ligne", 'Box Plot'], key='chart_type_dist')

    # Vérification si les variables explicatives sont sélectionnées
    if len(explanatory_variables) == 0:
        st.warning("Sélectionnez au moins une variable explicative.")
        return

    if len(explanatory_variables) > 2:
        st.warning("Vous ne pouvez sélectionner que deux variables explicatives.")
        return

    # Diagramme de répartition de la population
    st.header("Répartition de la Population")
    
    # Cas où une seule variable explicative est sélectionnée
    if len(explanatory_variables) == 1:

        # Message pour sélectionner une deuxième variable explicative
        st.info("Vous pouvez sélectionner une deuxième variable explicative pour une analyse bivarée.")
        try:
            fig, ax = plot_single_variable_chart(df, chart_type, x=explanatory_variables[0])
            ax.set_title(f"Répartition de la Population selon {explanatory_variables[0]}")
            ax.set_ylabel("Nombre d'individus")
            ax.set_xlabel(explanatory_variables[0])
            st.pyplot(fig)

        except KeyError as e:
            st.error(f"Erreur lors du chargement des données : {e}")
    
    # Cas où deux variables explicatives sont sélectionnées
    elif len(explanatory_variables) == 2:
        # Message pour sélectionner une deuxième variable explicative
        st.info("Vous avez sélectionné deux variables explicatives.")
        try:
            fig, ax = plot_single_variable_chart(df, chart_type, x=explanatory_variables[0], hue=explanatory_variables[1])
            ax.set_title(f"Répartition de la Population selon {explanatory_variables[0]} et {explanatory_variables[1]}")
            ax.set_ylabel("Nombre d'individus")
            ax.set_xlabel(explanatory_variables[0])
            st.pyplot(fig)
        
        except KeyError as e:
            st.error(f"Erreur lors du chargement des données : {e}")
    
    # Cas où deux variables explicatives sont sélectionnées
    elif len(explanatory_variables) == 2:
        # Message pour sélectionner une deuxième variable explicative
        st.info("Vous avez sélectionné deux variables explicatives.")
        try:
            fig, ax = plot_single_variable_chart(df, chart_type, x=explanatory_variables[0], hue=explanatory_variables[1])
            ax.set_title(f"Répartition de la Population selon {explanatory_variables[0]} et {explanatory_variables[1]}")
            ax.set_ylabel("Nombre d'individus")
            ax.set_xlabel(explanatory_variables[0])
            st.pyplot(fig)
        
        except KeyError as e:
            st.error(f"Erreur lors du chargement des données : {e}")

    # Lien de téléchargement de la figure
    if len(explanatory_variables) >= 1:
        st.markdown(get_image_download_link(fig, "repartition_population.png", "Télécharger la figure"), unsafe_allow_html=True)
    
# Fonction pour vérifier si une colonne est numérique
def is_numeric(series):
    return pd.api.types.is_numeric_dtype(series)

# Fonction pour la prévalence des cas positifs de la maladie
def prevalence(df):
    st.title("Prévalence des cas positifs de la maladie")
    # Initialisation de explanatory_variable_type
    explanatory_variable_type = None
    # Affichage des données
    st.write("Aperçu des données :")
    st.write(df.head())

    # Sidebar pour la sélection des variables et du type de graphique
    st.sidebar.header("Sélection des Variables et du Type de Graphique")
    target_variable = st.sidebar.selectbox("Choisissez la variable cible", df.columns, key='target_variable_prev')
    target_values = df[target_variable].unique().tolist()
    positive_value = st.sidebar.selectbox("Valeur représentant 'Positive'", target_values, key='positive_value_prev')
    
    # Sélection des variables explicatives avec messages d'information
    st.sidebar.info("Sélectionnez jusqu'à deux variables explicatives.")
    explanatory_variables = st.sidebar.multiselect("Choisissez les variables explicatives", df.columns, key='explanatory_variables_prev')
    if len(explanatory_variables) > 2:
        st.sidebar.warning("Vous ne pouvez sélectionner que deux variables explicatives.")
        explanatory_variables = explanatory_variables[:2]

    chart_type = st.sidebar.selectbox("Choisissez le type de graphique", ["Bar", "Histogramme", "Camembert", "Ligne",'Box Plot'], key='chart_type_prev')

    # Vérification si les variables sont sélectionnées
    if not target_variable or not explanatory_variables or not positive_value:
        st.warning("Sélectionnez une variable cible, une ou deux variables explicatives, et définissez les valeurs discrètes pour 'Positive'.")
        return

    negative_values = [val for val in target_values if val != positive_value]
    if not negative_values:
        st.warning("Il faut au moins une valeur pour 'Negative'.")
        return
    negative_value = negative_values[0]

    # Prévalence globale
    st.header("Prévalence Globale")
    total_cases = df.shape[0]
    positive_cases = df[df[target_variable] == positive_value][target_variable].count()
    global_prevalence = (positive_cases / total_cases) * 100
    st.write(f"Prévalence globale de {target_variable} = {global_prevalence:.2f}%")

    # Calculer la prévalence des cas positifs de la variable cible selon l'une ou deux variables explicatives
    table_pivot = df.groupby(explanatory_variables)[target_variable].value_counts().unstack(fill_value=0)
    table_pivot['Positive_total'] = table_pivot[positive_value].sum()  # Total des cas positifs

    # Gestion des valeurs nulles dans le dénominateur
    table_pivot['Prevalence_relative'] = np.where(table_pivot['Positive_total'] != 0,
                                                  table_pivot[positive_value] / table_pivot['Positive_total'] * 100,
                                                  np.nan)

    # Affichage du tableau de prévalence relative
    st.header("Prévalence des cas positifs par groupe")
    st.write(table_pivot)

    # Tests univariés et bivariés appropriés
    if len(explanatory_variables) == 1:
        explanatory_variable_type = df[explanatory_variables[0]].dtype
    
    if explanatory_variable_type == 'object':
        st.info("Test de Chi-square pour variables catégorielles:")
        # Effectuer un test de Chi-square pour les variables catégorielles
        chi2, p, dof, expected = stats.chi2_contingency(pd.crosstab(df[explanatory_variables[0]], df[target_variable]))
        st.write(f"Chi-square : {chi2}")
        st.write(f"P-value : {p}")
        if p < 0.05:
            st.write("Interprétation : La p-value est inférieure à 0.05, donc nous rejetons l'hypothèse nulle. Il y a une relation significative entre les variables.")
        else:
            st.write("Interprétation : La p-value est supérieure à 0.05, donc nous ne pouvons pas rejeter l'hypothèse nulle. Il n'y a pas suffisamment de preuves pour suggérer une relation significative entre les variables.")
    
    elif explanatory_variable_type in ['int64', 'float64']:
        st.info("Test de l'indépendance de Student (t-test) pour variable continue :")
        # Effectuer un t-test indépendant pour les variables continues
        group1 = df[df[target_variable] == df[target_variable].unique()[0]][explanatory_variables[0]]
        group2 = df[df[target_variable] == df[target_variable].unique()[1]][explanatory_variables[0]]
        
        t_stat, p_val = stats.ttest_ind(group1, group2)
        st.write(f"t-statistique : {t_stat}")
        st.write(f"P-value : {p_val}")
        if p_val < 0.05:
            st.write("Interprétation : La p-value est inférieure à 0.05, donc nous rejetons l'hypothèse nulle. Il y a une différence significative entre les moyennes des groupes.")
        else:
            st.write("Interprétation : La p-value est supérieure à 0.05, donc nous ne pouvons pas rejeter l'hypothèse nulle. Il n'y a pas suffisamment de preuves pour suggérer une différence significative entre les moyennes des groupes.")
        
        st.info("Test ANOVA pour variable continue :")
        # Vérifier si la colonne est numérique
        if not is_numeric(df[explanatory_variables[0]]):
            st.error("La variable explicative doit être numérique pour effectuer un test ANOVA.")
            return
        # Effectuer un test ANOVA pour les variables continues
        f_stat, p_val_anova = stats.f_oneway(group1, group2)
        st.write(f"F-statistique : {f_stat}")
        st.write(f"P-value ANOVA : {p_val_anova}")
        if p_val_anova < 0.05:
            st.write("Interprétation : La p-value ANOVA est inférieure à 0.05, donc nous rejetons l'hypothèse nulle. Il y a une différence significative entre les moyennes des groupes.")
        else:
            st.write("Interprétation : La p-value ANOVA est supérieure à 0.05, donc nous ne pouvons pas rejeter l'hypothèse nulle. Il n'y a pas suffisamment de preuves pour suggérer une différence significative entre les moyennes des groupes.")

    elif len(explanatory_variables) == 2:
        explanatory_variable_type_1 = df[explanatory_variables[0]].dtype
        explanatory_variable_type_2 = df[explanatory_variables[1]].dtype
        if explanatory_variable_type_1 == 'object' and explanatory_variable_type_2 == 'object':
            st.info("Test de l'indice de contingency pour variables catégorielles:")
            # Effectuer un test de l'indice de contingency pour les variables catégorielles
            contingency_table = pd.crosstab(df[explanatory_variables[0]], df[explanatory_variables[1]])
            contingency_table = contingency_table.values
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            st.write(f"Chi-square : {chi2}")
            st.write(f"P-value : {p}")
            if p < 0.05:
                st.write("Interprétation : La p-value est inférieure à 0.05, donc nous rejetons l'hypothèse nulle. Il y a une relation significative entre les variables.")
            else:
                st.write("Interprétation : La p-value est supérieure à 0.05, donc nous ne pouvons pas rejeter l'hypothèse nulle. Il n'y a pas suffisamment de preuves pour suggérer une relation significative entre les variables.")
        elif (explanatory_variable_type_1 == 'object' and explanatory_variable_type_2 in ['int64', 'float64']) or (explanatory_variable_type_2 == 'object' and explanatory_variable_type_1 in ['int64', 'float64']):
            st.info("Test de Kruskal-Wallis pour variable catégorielle et continue:")
            # Vérifier si la colonne est numérique
            if not is_numeric(df[explanatory_variables[1]] if explanatory_variable_type_1 == 'object' else df[explanatory_variables[0]]):
                st.error("La variable explicative continue doit être numérique pour effectuer un test de Kruskal-Wallis.")
                return
            # Supprimer les valeurs non numériques ou manquantes
            df_cleaned = df.dropna(subset=[explanatory_variables[0], explanatory_variables[1], target_variable])
            df_cleaned[explanatory_variables[1] if explanatory_variable_type_1 == 'object' else explanatory_variables[0]] = pd.to_numeric(df_cleaned[explanatory_variables[1] if explanatory_variable_type_1 == 'object' else explanatory_variables[0]], errors='coerce')
            df_cleaned = df_cleaned.dropna(subset=[explanatory_variables[1] if explanatory_variable_type_1 == 'object' else explanatory_variables[0]])
            # Effectuer un test de Kruskal-Wallis pour les variables catégorielle et continue
            groups = []
            for group_name, group_data in df_cleaned.groupby(explanatory_variables[0]):
                groups.append(group_data[target_variable].values)
            h_stat, p = stats.kruskal(*groups)
            st.write(f"Kruskal-Wallis H-statistique : {h_stat}")
            st.write(f"P-value : {p}")
            if p < 0.05:
                st.write("Interprétation : La p-value est inférieure à 0.05, donc nous rejetons l'hypothèse nulle. Il y a une différence significative entre les groupes.")
            else:
                st.write("Interprétation : La p-value est supérieure à 0.05, donc nous ne pouvons pas rejeter l'hypothèse nulle. Il n'y a pas suffisamment de preuves pour suggérer une différence significative entre les groupes.")
        elif explanatory_variable_type_1 in ['int64', 'float64'] and explanatory_variable_type_2 in ['int64', 'float64']:
            st.info("Test de régression linéaire pour variables continues:")
            # Vérifier si les colonnes sont numériques
            if not is_numeric(df[explanatory_variables[0]]) or not is_numeric(df[explanatory_variables[1]]):
                st.error("Les deux variables explicatives doivent être numériques pour effectuer une régression linéaire.")
                return
            # Supprimer les valeurs non numériques ou manquantes
            df_cleaned = df.dropna(subset=[explanatory_variables[0], explanatory_variables[1], target_variable])
            df_cleaned[explanatory_variables[0]] = pd.to_numeric(df_cleaned[explanatory_variables[0]], errors='coerce')
            df_cleaned[explanatory_variables[1]] = pd.to_numeric(df_cleaned[explanatory_variables[1]], errors='coerce')
            df_cleaned = df_cleaned.dropna(subset=[explanatory_variables[0], explanatory_variables[1]])
            # Effectuer un test de régression linéaire pour les variables continues
            model = stats.linregress(df_cleaned[explanatory_variables[0]], df_cleaned[explanatory_variables[1]])
            st.write(f"Pente : {model.slope}")
            st.write(f"Intercept : {model.intercept}")
            st.write(f"R-valeur : {model.rvalue}")
            st.write(f"P-value : {model.pvalue}")
            if model.pvalue < 0.05:
                st.write("Interprétation : La p-value est inférieure à 0.05, donc nous rejetons l'hypothèse nulle. Il y a une relation linéaire significative entre les variables.")
            else:
                st.write("Interprétation : La p-value est supérieure à 0.05, donc nous ne pouvons pas rejeter l'hypothèse nulle. Il n'y a pas suffisamment de preuves pour suggérer une relation linéaire significative entre les variables.")

    # Tracer le graphique de la prévalence relative
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == 'Bar':
            sns.barplot(data=table_pivot.reset_index(), x=explanatory_variables[0], y='Prevalence_relative', hue=explanatory_variables[1] if len(explanatory_variables) == 2 else None, ax=ax)
            ax.set_xlabel(", ".join(explanatory_variables))
            ax.set_ylabel("Prévalence relative (%)")
            ax.set_title(f"Prévalence relative des cas positifs de {positive_value}")
            ax.legend(title=explanatory_variables[1] if len(explanatory_variables) == 2 else None)
            add_percentages(ax)  # Ajouter les pourcentages au-dessus des barres

        elif chart_type == 'Histogramme':
            sns.histplot(data=table_pivot.reset_index(), x=explanatory_variables[0], weights='Prevalence_relative', bins=10, ax=ax, discrete=True)
            ax.set_xlabel(explanatory_variables[0])
            ax.set_ylabel("Prévalence relative (%)")
            ax.set_title(f"Prévalence relative des cas positifs de {positive_value}")
            add_percentages(ax)  # Ajouter les pourcentages au-dessus des barres

        elif chart_type == 'Camembert':
            counts = table_pivot['Prevalence_relative']
            ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
            ax.set_ylabel("")
            ax.set_title(f"Prévalence relative des cas positifs de {positive_value}")
            ax.axis('equal')

        elif chart_type == 'Ligne':
            sns.lineplot(data=table_pivot.reset_index(), x=explanatory_variables[0], y='Prevalence_relative', hue=explanatory_variables[1] if len(explanatory_variables) == 2 else None, ax=ax)
            ax.set_xlabel(", ".join(explanatory_variables))
            ax.set_ylabel("Prévalence relative (%)")
            ax.set_title(f"Prévalence relative des cas positifs de {positive_value}")
        
        # Nouveau type de graphique Box Plot
        elif chart_type == 'Box Plot':
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.boxplot(data=table_pivot.reset_index(), x='Prevalence_relative', y=explanatory_variables[0], hue=explanatory_variables[1] if len(explanatory_variables) == 2 else None, ax=ax, width=1)
            sns.stripplot(data=table_pivot.reset_index(), y=explanatory_variables[0], x='Prevalence_relative', hue=explanatory_variables[1] if len(explanatory_variables) == 2 else None, ax=ax, color=".1")
            ax.set_xlabel(((pd.DataFrame(table_pivot)).columns[-1]))
            ax.set_ylabel(((table_pivot.index.names)))
            ax.set_title(f"Box Plot des cas positifs de {positive_value}")

        else:
            st.error("Type de graphique non supporté.")
            return

        # Affichage du graphique
        st.pyplot(fig)

    except KeyError as e:
        st.error(f"Erreur lors du tracé du graphique : {e}")

    # Lien de téléchargement de la figure
    st.markdown(get_image_download_link(fig, "prevalence.png", "Télécharger la figure"), unsafe_allow_html=True)



# Fonction pour déterminer si une série est qualitative
def is_qualitative(series):
    return series.dtype == 'object' or series.dtype.name == 'category'

# Fonction pour calculer l'intervalle de confiance pour la moyenne
def confidence_interval_mean(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    alpha = 1 - confidence
    t_statistic = t.ppf(1 - alpha / 2, df=n - 1)
    margin_of_error = t_statistic * (std_dev / np.sqrt(n))
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    return lower_bound, upper_bound

# Fonction pour calculer l'intervalle de confiance pour une valeur individuelle
def calculate_confidence_interval(value, count, confidence=0.95):
    alpha = 1 - confidence
    t_critical = t.ppf(1 - alpha / 2, df=count - 1)
    margin_error = t_critical * (value / np.sqrt(count))
    lower_bound = value - margin_error
    upper_bound = value + margin_error
    return lower_bound, upper_bound

# Fonction pour les statistiques descriptives
def statistics(df):
    st.title("Statistiques Descriptives")
    
    st.write("Aperçu des données :")
    st.write(df.head())

    # Sidebar pour la sélection des statistiques et des variables explicatives
    st.sidebar.header("Sélection des Statistiques et des Variables Explicatives")
    statistic_method = st.sidebar.selectbox("Choisissez la méthode statistique", ["Describe", "Fréquence", "Grouper"], key='stat_method')
    explanatory_variables = st.sidebar.multiselect("Choisissez les variables explicatives", df.columns, key='explanatory_variables_stats')

    if not explanatory_variables:
        st.warning("Sélectionnez au moins une variable explicative.")
        return

    st.header("Résultats des Statistiques")
    
    if statistic_method == "Describe":
        st.write(df[explanatory_variables].describe())
    
    elif statistic_method == "Fréquence":
        for var in explanatory_variables:
            st.subheader(f"Fréquence de {var}")

            sn = df[var].value_counts().fillna(0.001)
            sp = df[var].value_counts(normalize=True).fillna(0.001) * 100

            # Calcul des intervalles de confiance pour chaque valeur
            sn_intervals = sn.apply(lambda x: calculate_confidence_interval(x, sn.sum()))
            sp_intervals = sp.apply(lambda x: calculate_confidence_interval(x, sp.sum()))

            # Création des DataFrames pour l'affichage avec intervalles de confiance
            sn_df = pd.DataFrame({
                sn.index.names[0]: sn.index,
                'Nombre': sn.values,
                'Intervalle de confiance': [f"[{low:.2f}, {high:.2f}]" for (low, high) in sn_intervals]
            })

            sp_df = pd.DataFrame({
                sp.index.names[0]: sp.index,
                'Pourcentage': sp.round(2).values,
                'Intervalle de confiance': [f"[{low:.2f}, {high:.2f}]" for (low, high) in sp_intervals]
            })

            # Affichage des tableaux côte à côte avec les bornes inférieures et supérieures correctement différenciées
            col1, col2 = st.columns(2)

            with col1:
                st.write("Nombre")
                st.write(sn_df[[sn.index.names[0], 'Nombre', 'Intervalle de confiance']])

            with col2:
                st.write("Pourcentage")
                st.write(sp_df[[sp.index.names[0], 'Pourcentage', 'Intervalle de confiance']])
    
    elif statistic_method == "Grouper":
        group_by_vars = st.sidebar.multiselect("Choisissez les variables de regroupement", df.columns, key='group_by_vars')
        if not group_by_vars:
            st.warning("Sélectionnez au moins une variable pour le regroupement.")
            return
        
        st.subheader("Résultats du regroupement")
        aggregation_methods = {}
        for var in explanatory_variables:
            if is_qualitative(df[var]):
                st.warning(f"La variable '{var}' est qualitative. Recommandation : Utilisez 'count' ou 'size' pour cette variable.")
                aggregation_methods[var] = ['count', 'size']
            else:
                aggregation_methods[var] = ['mean', 'sum', 'min', 'max', 'size']
        
        grouped_df = df.groupby(group_by_vars).agg(aggregation_methods)
        st.write(grouped_df)

# Fonction principale pour l'application
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Aller à", ["Distribution de la Population", "Prévalence des Cas Positifs", "Statistiques"])

    uploaded_file = st.sidebar.file_uploader("Charger un fichier CSV ou Excel", type=['csv', 'xls', 'xlsx'])

    if uploaded_file is None:
        st.info("Veuillez charger un fichier.")

    else:
        try:
            df = load_data(uploaded_file)

            if page == "Distribution de la Population":
                population_distribution(df)
            elif page == "Prévalence des Cas Positifs":
                prevalence(df)
            elif page == "Statistiques":
                statistics(df)

        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {str(e)}")

    st.sidebar.title("À propos de")
    st.sidebar.info(
        "Cette application permet d'explorer la répartition de la population d'étude et la prévalence des cas positifs de la maladie."
    )
    st.sidebar.info(
        "Elle utilise des graphiques interactifs et des tests statistiques pour faciliter l'analyse exploratoire des données."
    )

if __name__ == "__main__":
    main()

