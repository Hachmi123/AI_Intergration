import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
from datetime import datetime

class SalaryPredictorTND:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_version = "v3.0-TND"
        self.metrics = {}
        
    def prepare_features(self, df):
        """Préparation des variables d'entrée"""
        X = df.copy()

        # NO MORE education_level → new unified LEVEL
        level_map = {
            'Beginner': 0,
            'Intermediate': 1,
            'Advanced': 2,
            'Expert': 3
        }
        X['level_encoded'] = X['level'].map(level_map).fillna(1)

        # Localisation encodée
        top_locations = ['Tunis', 'Sfax', 'Sousse', 'Nabeul', 'Monastir', 'Bizerte', 'Gabes', 'Remote']
        X['location_encoded'] = X['location'].apply(
            lambda x: top_locations.index(x) if x in top_locations else len(top_locations)
        )

        # Nombre de compétences
        X['skills_count'] = X['skills'].apply(
            lambda x: len(x.split(',')) if isinstance(x, str) else len(x) if isinstance(x, list) else 0
        )

        # Encodage du poste actuel
        common_titles = [
            'Ingénieur Logiciel', 'Data Scientist', 
            'Chef de Projet', 'Analyste Données', 
            'Technicien Réseau'
        ]
        X['title_encoded'] = X['current_title'].apply(
            lambda x: common_titles.index(x) if x in common_titles else len(common_titles)
        )

        # NEW : heures par semaine
        X['hours_per_week'] = X.get('hours_per_week', 40)
        X['hours_per_week'] = X['hours_per_week'].clip(10, 70)

        features = [
            'years_experience',
            'level_encoded',
            'location_encoded',
            'skills_count',
            'title_encoded',
            'hours_per_week'
        ]
        self.feature_names = features

        return X[features]
    
    def train(self, data_path='data/salary_data_tnd.csv'):
        """Entraîne le modèle de prédiction des salaires (TND)"""
        try:
            df = pd.read_csv(data_path)
            print(f"✅ Dataset chargé : {len(df)} enregistrements")

            X = self.prepare_features(df)
            y = df['salary']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model = RandomForestRegressor(
                n_estimators=120,
                max_depth=18,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train_scaled, y_train)
            
            y_pred = self.model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            
            self.metrics = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'cv_mae': cv_mae,
                'training_date': datetime.now().isoformat(),
                'model_version': self.model_version,
                'feature_importance': feature_importance
            }
            
            print("\n=== Résultats de l'entraînement ===")
            print(f"MAE : {mae:.2f} TND")
            print(f"RMSE : {rmse:.2f} TND")
            print(f"R² : {r2:.4f}")
            print(f"MAE (Cross-Validation) : {cv_mae:.2f} TND")
            
            self.save_model()
            self.save_metrics()
            
            return self.metrics
        except Exception as e:
            print(f"❌ Erreur entraînement : {e}")
            return None
    
    def predict(self, features):
        """Prédit un salaire en TND"""
        if self.model is None:
            self.load_model()

        if 'hours_per_week' not in features:
            features['hours_per_week'] = 40

        input_df = pd.DataFrame([features])
        prepared = self.prepare_features(input_df)
        scaled = self.scaler.transform(prepared)
        prediction_tnd = self.model.predict(scaled)[0]
        return prediction_tnd
    
    def save_model(self):
        os.makedirs('models', exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'version': self.model_version,
            'metrics': self.metrics
        }, 'models/salary_model_tnd.joblib')
        print("💾 Modèle enregistré !")
    
    def load_model(self):
        try:
            data = joblib.load('models/salary_model_tnd.joblib')
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.model_version = data['version']
            self.metrics = data.get('metrics', {})
            print("📦 Modèle chargé !")
            return True
        except Exception as e:
            print(f"Erreur chargement modèle : {e}")
            return False
    
    def save_metrics(self):
        os.makedirs('models', exist_ok=True)
        with open('models/model_metrics_tnd.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)

def generate_sample_data_tnd():
    """Génère dataset réaliste VERSION LEVEL"""
    np.random.seed(42)
    n = 1200
    
    locations = ['Tunis', 'Sfax', 'Sousse', 'Nabeul', 'Monastir', 'Bizerte', 'Gabes', 'Remote', 'Kef']
    titles = ['Ingénieur Logiciel', 'Data Scientist', 'Chef de Projet', 'Analyste Données', 'Technicien Réseau']
    skills_pool = ['Python', 'SQL', 'Java', 'Linux', 'Docker', 'Kubernetes', 'Machine Learning', 'React']
    levels = ['Beginner', 'Intermediate', 'Advanced', 'Expert']
    
    data = {
        'years_experience': np.random.exponential(5, n).clip(0, 25),
        'level': np.random.choice(levels, n, p=[0.25, 0.45, 0.25, 0.05]),
        'location': np.random.choice(locations, n),
        'current_title': np.random.choice(titles, n),
        'skills': [','.join(np.random.choice(skills_pool, np.random.randint(2, 6), replace=False)) for _ in range(n)],
        'hours_per_week': np.random.randint(30, 50, n)
    }

    # Bonus salaires
    base_salary = 1200
    exp_bonus = data['years_experience'] * 160 + (data['years_experience'] ** 1.4) * 5
    
    level_bonus = {'Beginner': 0, 'Intermediate': 300, 'Advanced': 700, 'Expert': 1500}
    title_premium = {
        'Ingénieur Logiciel': 700,
        'Data Scientist': 1100,
        'Chef de Projet': 1300,
        'Analyste Données': 650,
        'Technicien Réseau': 500
    }
    location_multiplier = {
        'Tunis': 1.15, 'Sfax': 1.10, 'Sousse': 1.05, 'Nabeul': 1.10,
        'Monastir': 1.00, 'Bizerte': 0.95, 'Gabes': 0.90, 'Remote': 0.85, 'Kef': 0.90
    }

    skills_bonus = [len(sk.split(',')) * 150 for sk in data['skills']]
    hours_bonus = (data['hours_per_week'] - 40) * 20

    data['salary'] = (
        base_salary +
        exp_bonus +
        [level_bonus[l] for l in data['level']] +
        [title_premium[t] for t in data['current_title']] +
        skills_bonus +
        hours_bonus
    ) * [location_multiplier[l] for l in data['location']] + np.random.normal(0, 120, n)

    df = pd.DataFrame(data)
    df['salary'] = df['salary'].clip(1000, 7500)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/salary_data_tnd.csv', index=False)
    print(f"📁 Dataset généré : {len(df)} lignes")
    print(f"Salaire moyen : {df['salary'].mean():.2f} TND")
    
    return df

# -------------------------
# INITIALISATION
# -------------------------

predictor = SalaryPredictorTND()

if not os.path.exists('models/salary_model_tnd.joblib'):
    if not os.path.exists('data/salary_data_tnd.csv'):
        print("📊 Génération du dataset...")
        generate_sample_data_tnd()
    print("🚀 Entraînement du modèle...")
    predictor.train()
else:
    print("📦 Chargement modèle existant...")
    predictor.load_model()
