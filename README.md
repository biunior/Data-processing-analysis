### [README.md](file:///c%3A/Users/Yvain/Desktop/FTask/analyse/programme%20analyse%20last/README.md)

This script, `utils_yvain.py`, contains utility functions for analyzing movement data in behavioral experiments. It provides tools to compute various metrics related to reaction time, movement, and target interaction. Below is a summary of its purpose and key functionalities.

---

### **Purpose**
The script processes movement data stored in pandas DataFrames to compute metrics such as reaction time, movement distances, and times to specific events (e.g., crossing a trigger or entering a target). It also includes functions for data augmentation, such as adding velocity columns and marking specific events.

---

### **Key Functionalities**
1. **Data Augmentation**:
   - Adds time (`t`) and velocity (`vx`, `vy`) columns to the DataFrame.
   - Marks events like movement start, trigger crossing, and target entry.

2. **Metrics Computation**:
   - **Reaction Time (RT)**: Time until movement starts.
   - **Movement to Trigger (Rt-Trig)**: Time from movement start to trigger crossing.
   - **Trigger to Target Time (TTTarg)**: Time from trigger crossing to entering the target.
   - **Total Movement Time**: Duration of the entire movement.
   - **Distances**: Computes distances for various phases (e.g., start to trigger, trigger to target).

3. **Target Interaction**:
   - Identifies when the cursor enters the target circle.
   - Computes metrics like time spent in the target and distance to the target center.

4. **Categorical Outcomes**:
   - Determines trial outcomes (e.g., success, failure, or lost).

---

### **Usage**
The script is designed to be used as part of a larger analysis pipeline. It processes trial data and computes metrics for each trial, which can then be saved or summarized for further analysis.

---

### **Dependencies**
- `pandas`
- `numpy`
- `shapely`
- `sklearn`
- `matplotlib`

---

### **Notes**
- The script includes several TODOs for further improvements, such as refining movement criteria and handling edge cases.
- Some functions are specialized for specific experimental setups and may require adaptation for other use cases.




## 📊 Variables analysées

| **Nom** | **Définition** | **Critères / Ancien nom** |
|--------|----------------|-----------------------------|
| **Reaction Time (RT)** | Du début de l'essai jusqu'au début du mouvement | Vy ≥ 1p/0.01s pendant 0.2s, ou ≥ 200p/1s pendant 0.1s. Si RT < 0.2s → problème. *(ancien nom : ta)* |
| **Movement to Trigger (Rt-Trig)** | Du début du mouvement jusqu’au trigger | *(ancien nom : dt)* |
| **Trigger Time (TrigT)** | Temps total jusqu'au trigger | RT + MtT *(ancien nom : tr1)* |
| **Trigger Distance (TrigD)** | Distance entre le début et le trigger | - |
| **Trigger to Target Time (TargT)** | Du trigger à l'entrée dans la cible | Si feedback : 1ère ligne d'entrée. Sinon, dépend de l'état (perdu, pas d'entrée, etc.) *(ancien nom : tc)* |
| **Trigger to Target Distance (TargD)** | Distance du trigger à l'entrée dans la cible | "no target enter" si aucune entrée |
| **Time Target to Stop (StopT)** | De l'entrée cible jusqu’à l’arrêt du mouvement | Seulement si feedback ; sinon jusqu’à la fin de l’essai *(ancien nom : ts)* |
| **Target to Stop Distance (StopD)** | Distance entre entrée cible et arrêt mouvement | Sinon : distance jusqu’à la fin de l’essai |
| **Total Time (TotT)** | Temps total de l’essai | Début mouvement → arrêt ou fin d’essai |
| **Total Distance (TotD)** | Distance totale parcourue | Jusqu’à arrêt mouvement, sinon fin essai |
| **Final Distance (FD)** | Distance entre souris et centre cible à l’arrêt | Sinon : distance à la fin de l’essai *(ancien nom : Dist_final)* |
| **Time to Adjust (TA)** | Temps entre trigger et Vx max vers la cible | Vx max entre trigger et trigger+1s *(ancien nom : tr2)* |
| **VxMax** | Vitesse maximale dans la direction de la cible | - |
| **Catégorie (Réussite / Espace / Échec)** | Réussite si la cible est atteinte à la fin | Échec sinon ou si touche espace / hors cible |

## ⚙️ Cas spéciaux

- **Feedback activé** : certaines mesures dépendent de la détection de la cible.
- **Perdu** : Si aucune entrée cible et dépassement des bords ou >3s sans mouvement → essai considéré comme perdu.

## 📁 Structure du fichier Python

Le fichier Python associé permet de :
- Charger les données
- Détecter les étapes du mouvement
- Calculer les variables listées ci-dessus
- Gérer les cas spéciaux (feedback, mouvement non arrêté, etc.)

---

> N’hésite pas à ouvrir une **issue** ou faire une **pull request** si tu veux améliorer l’analyse ou ajouter des métriques !


