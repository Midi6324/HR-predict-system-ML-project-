"""
Script de Prédiction d'Attrition des Employés
==============================================

Ce script permet à une personne RH de prédire le risque de départ des employés
en utilisant le modèle ML entraîné.

Usage:
    python predict_attrition.py

Le script :
1. Charge automatiquement les données depuis le dossier 'data_rh/'
2. Effectue le prétraitement complet
3. Charge le modèle ML entraîné
4. Génère des prédictions pour tous les employés
5. Sauvegarde les résultats dans un fichier CSV

Prérequis:
- Les fichiers de données doivent être dans le dossier 'data_rh/'
- Un modèle entraîné doit être présent dans 'saved_models/'
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
import shutil
from datetime import datetime
from glob import glob


def print_banner():
    """Affiche la bannière du programme"""
    print("=" * 70)
    print("🏢 SYSTÈME DE PRÉDICTION D'ATTRITION DES EMPLOYÉS")
    print("=" * 70)
    print()


def extract_zip_data(data_root="data_rh/"):
    """Extrait le fichier ZIP s'il existe"""
    print("📦 Étape 1/5 - Extraction des données...")
    
    zip_file = os.path.join(data_root, 'in_out_time.zip')
    if os.path.exists(zip_file):
        print(f"   Extraction de {zip_file}...")
        shutil.unpack_archive(zip_file, data_root)
        print("   ✅ Extraction terminée")
    else:
        print("   ℹ️ Pas de fichier ZIP à extraire")
    print()


def create_time_features(df_time, prefix='in'):
    """Créer des features agrégées à partir des heures d'arrivée/départ"""
    date_cols = [col for col in df_time.columns if col != 'EmployeeID']
    
    # Convertir en datetime
    df_values = df_time[date_cols].apply(pd.to_datetime, errors='coerce')
    
    # Extraire l'heure en format numérique
    df_hours = df_values.apply(lambda x: x.dt.hour + x.dt.minute/60.0)
    
    # Calculer des statistiques agrégées
    features = pd.DataFrame()
    features['EmployeeID'] = df_time['EmployeeID']
    features[f'{prefix}_avg_hour'] = df_hours.mean(axis=1)
    features[f'{prefix}_std_hour'] = df_hours.std(axis=1)
    features[f'{prefix}_min_hour'] = df_hours.min(axis=1)
    features[f'{prefix}_max_hour'] = df_hours.max(axis=1)
    features[f'{prefix}_missing_days'] = df_values.isna().sum(axis=1)
    
    return features


def load_and_preprocess_data(data_root="data_rh/"):
    """Charge et prétraite les données RH"""
    print("📊 Étape 2/5 - Chargement et fusion des données...")
    
    # Vérifier la présence des fichiers requis
    required_files = [
        'general_data.csv',
        'employee_survey_data.csv',
        'manager_survey_data.csv',
        'in_time.csv',
        'out_time.csv'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(data_root, f))]
    
    if missing_files:
        print(f"   ❌ Fichiers manquants: {missing_files}")
        print(f"   ⚠️ Assurez-vous que tous les fichiers CSV sont dans le dossier '{data_root}'")
        sys.exit(1)
    
    # Charger les fichiers
    print("   Chargement des fichiers CSV...")
    df1 = pd.read_csv(os.path.join(data_root, 'general_data.csv'))
    df2 = pd.read_csv(os.path.join(data_root, 'employee_survey_data.csv'))
    df3 = pd.read_csv(os.path.join(data_root, 'manager_survey_data.csv'))
    df4 = pd.read_csv(os.path.join(data_root, 'in_time.csv'))
    df5 = pd.read_csv(os.path.join(data_root, 'out_time.csv'))
    
    print(f"   - general_data.csv: {df1.shape[0]} employés")
    print(f"   - employee_survey_data.csv: {df2.shape[0]} employés")
    print(f"   - manager_survey_data.csv: {df3.shape[0]} employés")
    print(f"   - in_time.csv: {df4.shape[0]} employés")
    print(f"   - out_time.csv: {df5.shape[0]} employés")
    
    # Renommer la première colonne des fichiers temporels
    df4 = df4.rename(columns={'Unnamed: 0': 'EmployeeID'})
    df5 = df5.rename(columns={'Unnamed: 0': 'EmployeeID'})
    
    # Créer les features temporelles
    print("   Création des features temporelles...")
    in_features = create_time_features(df4, prefix='arrival')
    out_features = create_time_features(df5, prefix='departure')
    
    time_features = in_features.merge(out_features, on='EmployeeID', how='inner')
    time_features['avg_work_hours'] = (
        time_features['departure_avg_hour'] - time_features['arrival_avg_hour']
    )
    
    print(f"   - {time_features.shape[1]-1} features temporelles créées")
    
    # Fusionner tous les DataFrames
    print("   Fusion des données...")
    result = df1.merge(df2, on='EmployeeID', how='inner') \
                .merge(df3, on='EmployeeID', how='inner') \
                .merge(time_features, on='EmployeeID', how='inner')
    
    # Garder une copie de l'EmployeeID pour le rapport final
    employee_ids = result['EmployeeID'].copy()
    
    # Supprimer les colonnes inutiles
    columns_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeID']
    columns_existing = [col for col in columns_to_drop if col in result.columns]
    
    if columns_existing:
        result = result.drop(columns_existing, axis=1)
    
    # Supprimer la colonne Attrition si elle existe (pour les nouvelles prédictions)
    if 'Attrition' in result.columns:
        result = result.drop('Attrition', axis=1)
    
    print(f"   ✅ Dataset fusionné: {result.shape[0]} lignes × {result.shape[1]} colonnes")
    print()
    
    return result, employee_ids


def find_latest_model(models_dir="saved_models"):
    """Trouve le modèle le plus récent dans le dossier"""
    print("🔍 Étape 3/5 - Recherche du modèle entraîné...")
    
    if not os.path.exists(models_dir):
        print(f"   ❌ Le dossier '{models_dir}' n'existe pas")
        print("   ⚠️ Veuillez d'abord entraîner un modèle avec le notebook ML_Project_Complete.ipynb")
        sys.exit(1)
    
    # Chercher tous les fichiers .joblib
    model_files = glob(os.path.join(models_dir, "*.joblib"))
    
    if not model_files:
        print(f"   ❌ Aucun modèle trouvé dans '{models_dir}'")
        print("   ⚠️ Veuillez d'abord entraîner un modèle avec le notebook ML_Project_Complete.ipynb")
        sys.exit(1)
    
    # Prendre le plus récent
    latest_model = max(model_files, key=os.path.getmtime)
    
    # Chercher le fichier info correspondant
    info_file = latest_model.replace('.joblib', '_info.json')
    
    print(f"   ✅ Modèle trouvé: {os.path.basename(latest_model)}")
    
    # Charger les infos si disponibles
    model_info = None
    if os.path.exists(info_file):
        with open(info_file, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        print(f"   📋 Type de modèle: {model_info['model_name']}")
        print(f"   📅 Date d'entraînement: {model_info['date_training']}")
        print(f"   📊 Performance (F1-Score): {model_info['performance']['f1_score']:.4f}")
    
    print()
    return latest_model, model_info


def load_model(model_path):
    """Charge le modèle ML"""
    print("📥 Étape 4/5 - Chargement du modèle...")
    
    try:
        model = joblib.load(model_path)
        print("   ✅ Modèle chargé avec succès")
        print()
        return model
    except Exception as e:
        print(f"   ❌ Erreur lors du chargement du modèle: {e}")
        sys.exit(1)


def make_predictions(model, X, employee_ids):
    """Fait des prédictions sur les données"""
    print("🎯 Étape 5/5 - Génération des prédictions...")
    
    try:
        # Prédictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        # Créer le DataFrame de résultats
        results = pd.DataFrame({
            'EmployeeID': employee_ids,
            'RisqueDepart': predictions,
            'ProbabiliteDepart': probabilities,
            'NiveauRisque': pd.cut(
                probabilities,
                bins=[0, 0.3, 0.6, 1.0],
                labels=['🟢 Faible', '🟡 Moyen', '🔴 Élevé']
            ),
            'Action': pd.cut(
                probabilities,
                bins=[0, 0.3, 0.6, 1.0],
                labels=['Aucune', 'Surveillance', 'Action immédiate']
            )
        })
        
        # Trier par probabilité décroissante
        results = results.sort_values('ProbabiliteDepart', ascending=False)
        
        print(f"   ✅ {len(results)} prédictions générées")
        print()
        
        return results
    
    except AttributeError as e:
        if '_fill_dtype' in str(e) or 'SimpleImputer' in str(e):
            print(f"   ❌ Erreur de compatibilité de version scikit-learn")
            print()
            print("   🔧 Solutions possibles:")
            print("   1. Réentraîner le modèle avec le notebook ML_Project_Complete.ipynb")
            print("   2. OU downgrader scikit-learn: pip install scikit-learn==1.7.2")
            print()
            sys.exit(1)
        else:
            print(f"   ❌ Erreur lors des prédictions: {e}")
            sys.exit(1)
    
    except Exception as e:
        print(f"   ❌ Erreur lors des prédictions: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def display_summary(results):
    """Affiche un résumé des prédictions"""
    print("=" * 70)
    print("📊 RÉSUMÉ DES PRÉDICTIONS")
    print("=" * 70)
    print()
    
    total = len(results)
    at_risk = results['RisqueDepart'].sum()
    high_risk = (results['NiveauRisque'] == '🔴 Élevé').sum()
    medium_risk = (results['NiveauRisque'] == '🟡 Moyen').sum()
    low_risk = (results['NiveauRisque'] == '🟢 Faible').sum()
    
    print(f"👥 Total des employés analysés: {total}")
    print(f"⚠️ Employés à risque de départ: {at_risk} ({at_risk/total*100:.1f}%)")
    print()
    
    print("📈 Distribution par niveau de risque:")
    print(f"   🔴 Risque élevé (>60%):   {high_risk:3d} employés ({high_risk/total*100:.1f}%)")
    print(f"   🟡 Risque moyen (30-60%): {medium_risk:3d} employés ({medium_risk/total*100:.1f}%)")
    print(f"   🟢 Risque faible (<30%):  {low_risk:3d} employés ({low_risk/total*100:.1f}%)")
    print()
    
    print("🚨 TOP 10 - Employés à risque élevé:")
    print("-" * 70)
    top_10 = results.head(10)
    for idx, row in top_10.iterrows():
        print(f"   {row['EmployeeID']:>6} | Probabilité: {row['ProbabiliteDepart']:>5.1%} | {row['NiveauRisque']} | {row['Action']}")
    print()


def save_results(results, output_file=None):
    """Sauvegarde les résultats dans un fichier CSV"""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"predictions_attrition_{timestamp}.csv"
    
    results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"💾 Résultats sauvegardés: {output_file}")
    print()
    
    # Créer aussi un rapport Excel si openpyxl est disponible
    try:
        excel_file = output_file.replace('.csv', '.xlsx')
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Onglet principal avec tous les résultats
            results.to_excel(writer, sheet_name='Prédictions', index=False)
            
            # Onglet résumé
            summary = pd.DataFrame({
                'Métrique': [
                    'Total employés',
                    'Risque élevé (>60%)',
                    'Risque moyen (30-60%)',
                    'Risque faible (<30%)',
                    'Employés à risque'
                ],
                'Valeur': [
                    len(results),
                    (results['NiveauRisque'] == '🔴 Élevé').sum(),
                    (results['NiveauRisque'] == '🟡 Moyen').sum(),
                    (results['NiveauRisque'] == '🟢 Faible').sum(),
                    results['RisqueDepart'].sum()
                ]
            })
            summary.to_excel(writer, sheet_name='Résumé', index=False)
        
        print(f"📊 Rapport Excel créé: {excel_file}")
        print()
        
    except ImportError:
        print("ℹ️ Pour générer un rapport Excel, installez openpyxl: pip install openpyxl")
        print()


def main():
    """Fonction principale"""
    print_banner()
    
    # Configuration
    data_root = "data_rh/"
    models_dir = "saved_models"
    
    # 1. Extraire le ZIP si nécessaire
    extract_zip_data(data_root)
    
    # 2. Charger et prétraiter les données
    X, employee_ids = load_and_preprocess_data(data_root)
    
    # 3. Trouver le modèle le plus récent
    model_path, model_info = find_latest_model(models_dir)
    
    # 4. Charger le modèle
    model = load_model(model_path)
    
    # 5. Faire les prédictions
    results = make_predictions(model, X, employee_ids)
    
    # 6. Afficher le résumé
    display_summary(results)
    
    # 7. Sauvegarder les résultats
    save_results(results)
    
    print("=" * 70)
    print("✅ TRAITEMENT TERMINÉ AVEC SUCCÈS")
    print("=" * 70)
    print()
    print("💡 Prochaines étapes:")
    print("   1. Consultez le fichier CSV généré pour voir tous les résultats")
    print("   2. Concentrez-vous sur les employés à risque élevé (>60%)")
    print("   3. Mettez en place des actions de rétention pour les employés prioritaires")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Programme interrompu par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
