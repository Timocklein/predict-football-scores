import requests
import json
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# API call
seasons = list(range(2010, 2025))
base_url = "https://v3.football.api-sports.io/fixtures"
headers = {
    'x-rapidapi-key': '032f30b055dabf54cdaf3fe312fcf931',
    'x-rapidapi-host': 'v3.football.api-sports.io'
}

all_rows = []
for season in seasons:
    teams_url = f"https://v3.football.api-sports.io/teams?league=88&season={season}"
    response = requests.get(teams_url, headers=headers)
    data = json.loads(response.text)
    team_ids = [item["team"]["id"] for item in data["response"]]
    for team_id in team_ids:
        url = f"{base_url}?team={team_id}&season={season}"
        response = requests.get(url, headers=headers)
        data = response.json()
        fixtures = data.get("response", [])
        for fixture in fixtures:
            f = fixture.get("fixture", {})
            league = fixture.get("league", {})
            teams = fixture.get("teams", {})
            goals = fixture.get("goals", {})
            row = {
                "season": season,
                "fixture_id": f.get("id"),
                "date": f.get("date"),
                "referee": f.get("referee"),
                "venue": f.get("venue", {}).get("name"),
                "league": league.get("name"),
                "round": league.get("round"),
                "home_team": teams.get("home", {}).get("name"),
                "away_team": teams.get("away", {}).get("name"),
                "home_goals": goals.get("home"),
                "away_goals": goals.get("away"),
                "home_winner": teams.get("home", {}).get("winner"),
                "away_winner": teams.get("away", {}).get("winner"),
                "status_long": f.get("status", {}).get("long"),
                "status_short": f.get("status", {}).get("short"),
            }
            all_rows.append(row)

df = pd.DataFrame(all_rows)

# Data cleaning en feature engineering
df_cleaned = df.copy()
df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])
df_cleaned['year'] = df_cleaned['date'].dt.year
df_cleaned['month'] = df_cleaned['date'].dt.month
df_cleaned['day'] = df_cleaned['date'].dt.day
df_cleaned['hour'] = df_cleaned['date'].dt.hour
df_cleaned['minute'] = df_cleaned['date'].dt.minute
df_cleaned = df_cleaned[df_cleaned['status_long'] == 'Match Finished']
df_cleaned = df_cleaned.drop(columns=['season', 'fixture_id', 'date', 'home_goals', 'away_goals', 'status_long', 'status_short', 'away_winner'])
df_cleaned['city'] = df_cleaned['venue'].str.extract(r'\((.*?)\)')
df_cleaned['venue'] = df_cleaned['venue'].str.replace(r'\s*\(.*?\)', '', regex=True)
mapping = {None: 'draw', True: 'win', False: 'loss'}
df_cleaned['home_winner'] = df_cleaned['home_winner'].replace(mapping)
df_cleaned = df_cleaned.rename(columns={'home_winner': 'result'})
df_cleaned = df_cleaned[['year', 'month', 'day', 'hour', 'minute', 'venue', 'city', 'round', 'league', 'home_team', 'away_team', 'referee', 'result']]
df_cleaned.to_csv("df_cleaned.csv", index=False)

# Label encoding
df_final = df_cleaned.copy()
labelencode_columns = ['venue', 'city', 'round', 'league', 'home_team', 'away_team', 'referee', 'result']
encoder = LabelEncoder()
for column in labelencode_columns:
    df_final[column] = encoder.fit_transform(df_final[column])

# Split data en train model
X = df_final.drop(columns=['result'])
y = df_final['result']

X_train_val, X_holdout, y_train_val, y_holdout = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train_val, y_train_val, test_size=0.20, random_state=42)

# Sla de data op
X_holdout.to_csv("X_holdout.csv", index=False)
y_holdout.to_csv("y_holdout.csv", index=False)

# K-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Baseline model
dc = DummyClassifier(strategy='stratified', random_state=42)
scores = cross_validate(dc, X_train, y_train, cv=kf, scoring='accuracy')
print(f"Mean accuracy: {scores['test_score'].mean()}")

# Random Forest model
params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 15, 20, 25, None],
    'min_samples_split': [20, 30, 40, 50],
    'min_samples_leaf': [20, 30, 40, 50],
    'n_estimators': [100, 150, 200, 250, 300]
}

rf = RandomForestClassifier(n_estimators=100, random_state=42)
random_search = RandomizedSearchCV(rf, param_distributions=params, n_iter=25, scoring='accuracy', n_jobs=-1, cv=kf, random_state=42)
random_search.fit(X_train, y_train)

best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Best hyperparameters: {random_search.best_params_}')
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Sla model en encoder op
joblib.dump(best_rf, 'best_rf_model.pkl')
joblib.dump(encoder, 'label_encoder.pkl')

# Maak een confusion matrix
matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['draw', 'loss', 'win'], yticklabels=['draw', 'loss', 'win'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("graphs/confusion_matrix.png")
plt.show()

# Maak een feature importance plot
df_importances = pd.DataFrame({'feature': X.columns,'importance': best_rf.feature_importances_})
df_importances = df_importances.sort_values('importance', ascending=True)
plt.figure(figsize=(10,5))
sns.barplot(x='importance', y='feature', data=df_importances)
plt.title('Feature Importances')
plt.savefig("graphs/feature_importances.png")
plt.show()