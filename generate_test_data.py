"""
Script pour générer des données de test RH
Ces données sont nouvelles et n'ont jamais été vues par le modèle
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Configuration
np.random.seed(123)  # Seed différent pour avoir des données vraiment nouvelles
n_employees = 50  # Générer 50 nouveaux employés pour le test
start_id = 10000  # Commencer à ID 10000 pour être sûr qu'ils n'existent pas

# Créer le dossier data_rh s'il n'existe pas
os.makedirs('data_rh', exist_ok=True)

print("🔄 Génération de données de test RH...")
print(f"   Nombre d'employés: {n_employees}")
print(f"   IDs: {start_id} à {start_id + n_employees - 1}")
print()

# ============================================================================
# 1. GENERAL DATA
# ============================================================================
print("📊 Génération: general_data.csv")

employee_ids = list(range(start_id, start_id + n_employees))

general_data = pd.DataFrame({
    'Age': np.random.randint(22, 60, n_employees),
    'Attrition': np.random.choice(['Yes', 'No'], n_employees, p=[0.16, 0.84]),
    'BusinessTravel': np.random.choice(['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'], 
                                       n_employees, p=[0.15, 0.70, 0.15]),
    'Department': np.random.choice(['Sales', 'Research & Development', 'Human Resources'], 
                                   n_employees, p=[0.30, 0.65, 0.05]),
    'DistanceFromHome': np.random.randint(1, 30, n_employees),
    'Education': np.random.choice([1, 2, 3, 4, 5], n_employees, p=[0.10, 0.25, 0.35, 0.20, 0.10]),
    'EducationField': np.random.choice(['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources'],
                                       n_employees, p=[0.41, 0.27, 0.10, 0.09, 0.08, 0.05]),
    'EmployeeCount': 1,
    'EmployeeID': employee_ids,
    'Gender': np.random.choice(['Male', 'Female'], n_employees, p=[0.60, 0.40]),
    'JobLevel': np.random.choice([1, 2, 3, 4, 5], n_employees, p=[0.35, 0.30, 0.20, 0.10, 0.05]),
    'JobRole': np.random.choice([
        'Sales Executive', 'Research Scientist', 'Laboratory Technician',
        'Manufacturing Director', 'Healthcare Representative', 'Manager',
        'Sales Representative', 'Research Director', 'Human Resources'
    ], n_employees),
    'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_employees, p=[0.32, 0.46, 0.22]),
    'MonthlyIncome': np.random.randint(20000, 200000, n_employees),
    'NumCompaniesWorked': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_employees, p=[0.12, 0.18, 0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.05, 0.04]),
    'Over18': 'Y',
    'PercentSalaryHike': np.random.randint(11, 26, n_employees),
    'StandardHours': 8,
    'StockOptionLevel': np.random.choice([0, 1, 2, 3], n_employees, p=[0.63, 0.18, 0.12, 0.07]),
    'TotalWorkingYears': np.random.randint(0, 40, n_employees),
    'TrainingTimesLastYear': np.random.choice([0, 1, 2, 3, 4, 5, 6], n_employees, p=[0.05, 0.05, 0.28, 0.30, 0.15, 0.12, 0.05]),
    'YearsAtCompany': np.random.randint(0, 40, n_employees),
    'YearsSinceLastPromotion': np.random.randint(0, 15, n_employees),
    'YearsWithCurrManager': np.random.randint(0, 17, n_employees),
})

general_data.to_csv('data_rh/general_data.csv', index=False)
print(f"   ✅ {len(general_data)} lignes générées")

# ============================================================================
# 2. EMPLOYEE SURVEY DATA
# ============================================================================
print("📊 Génération: employee_survey_data.csv")

employee_survey = pd.DataFrame({
    'EmployeeID': employee_ids,
    'EnvironmentSatisfaction': np.random.choice([1, 2, 3, 4], n_employees, p=[0.22, 0.24, 0.25, 0.29]),
    'JobSatisfaction': np.random.choice([1, 2, 3, 4], n_employees, p=[0.20, 0.22, 0.30, 0.28]),
    'WorkLifeBalance': np.random.choice([1, 2, 3, 4], n_employees, p=[0.07, 0.23, 0.56, 0.14]),
})

employee_survey.to_csv('data_rh/employee_survey_data.csv', index=False)
print(f"   ✅ {len(employee_survey)} lignes générées")

# ============================================================================
# 3. MANAGER SURVEY DATA
# ============================================================================
print("📊 Génération: manager_survey_data.csv")

manager_survey = pd.DataFrame({
    'EmployeeID': employee_ids,
    'JobInvolvement': np.random.choice([1, 2, 3, 4], n_employees, p=[0.07, 0.23, 0.58, 0.12]),
    'PerformanceRating': np.random.choice([3, 4], n_employees, p=[0.85, 0.15]),
})

manager_survey.to_csv('data_rh/manager_survey_data.csv', index=False)
print(f"   ✅ {len(manager_survey)} lignes générées")

# ============================================================================
# 4. IN_TIME DATA (heures d'arrivée)
# ============================================================================
print("📊 Génération: in_time.csv")

# Générer 262 jours de données (environ 1 an de travail)
dates = pd.date_range(start='2024-01-01', periods=262, freq='B')  # Business days
date_columns = [d.strftime('%Y-%m-%d') for d in dates]

in_time_data = {'Unnamed: 0': employee_ids}

for date_col in date_columns:
    # Heure d'arrivée entre 7h et 10h, avec quelques absences (20%)
    times = []
    for _ in range(n_employees):
        if np.random.random() < 0.20:  # 20% d'absences
            times.append(np.nan)
        else:
            hour = np.random.randint(7, 10)
            minute = np.random.randint(0, 60)
            times.append(f'{date_col} {hour:02d}:{minute:02d}:00')
    in_time_data[date_col] = times

in_time_df = pd.DataFrame(in_time_data)
in_time_df.to_csv('data_rh/in_time.csv', index=False)
print(f"   ✅ {len(in_time_df)} employés × {len(date_columns)} jours")

# ============================================================================
# 5. OUT_TIME DATA (heures de départ)
# ============================================================================
print("📊 Génération: out_time.csv")

out_time_data = {'Unnamed: 0': employee_ids}

for date_col in date_columns:
    # Heure de départ entre 17h et 21h
    times = []
    for i in range(n_employees):
        # Si absent le matin, absent le soir aussi
        if pd.isna(in_time_data[date_col][i]):
            times.append(np.nan)
        else:
            hour = np.random.randint(17, 22)
            minute = np.random.randint(0, 60)
            times.append(f'{date_col} {hour:02d}:{minute:02d}:00')
    out_time_data[date_col] = times

out_time_df = pd.DataFrame(out_time_data)
out_time_df.to_csv('data_rh/out_time.csv', index=False)
print(f"   ✅ {len(out_time_df)} employés × {len(date_columns)} jours")

# ============================================================================
# RÉSUMÉ
# ============================================================================
print()
print("=" * 70)
print("✅ GÉNÉRATION TERMINÉE")
print("=" * 70)
print()
print(f"📁 Dossier: data_rh/")
print(f"📊 Fichiers créés:")
print(f"   - general_data.csv")
print(f"   - employee_survey_data.csv")
print(f"   - manager_survey_data.csv")
print(f"   - in_time.csv")
print(f"   - out_time.csv")
print()
print(f"👥 {n_employees} employés de test avec IDs {start_id}-{start_id + n_employees - 1}")
print()
print("💡 Ces données peuvent maintenant être utilisées avec predict_attrition.py")
print()
