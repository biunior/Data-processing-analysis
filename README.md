# Tutoriel : Création d'un environnement virtuel, installation des dépendances et exécution d'un script en Python
Ce tutoriel vous guidera à travers les étapes pour créer un environnement virtuel,
installer les dépendances d'un fichier requirements.txt et exécuter un script Python nommé main.py sur Windows 10.
Le tutoriel suppose que vous n'avez aucune connaissance préalable en programmation Python.

## Étape 1 : Installation de Python
Avant de commencer, assurez-vous que Python est installé sur votre système.
Si vous ne l'avez pas encore installé, vous pouvez suivre les étapes suivantes :

Rendez-vous sur le site officiel de Python à l'adresse https://www.python.org/downloads/.
Téléchargez le programme d'installation de la dernière version stable de Python ( 3.9 pour toi Yvain !) pour Windows.
Exécutez le programme d'installation et suivez les instructions à l'écran.
Assurez-vous de cocher la case "Add Python to PATH" lorsqu'elle vous est proposée.
Une fois l'installation terminée, vous pouvez passer à l'étape suivante.

## Étape 2 : Création d'un environnement virtuel
Un environnement virtuel est une isolation de l'espace de travail Python,
qui vous permet d'installer des packages spécifiques sans interférer avec les autres projets.
Voici comment créer un environnement virtuel :

Ouvrez une fenêtre de commande (Invite de commandes) en appuyant sur la touche Windows + R,
puis en tapant `cmd` suivi de Entrée.

Naviguez vers le répertoire (dossier) où vous souhaitez créer votre environnement virtuel en utilisant la commande `cd` 
(par exemple, `cd chemin/vers/mon_projet`).

Une fois dans le répertoire souhaité,
exécutez la commande suivante pour créer un nouvel environnement virtuel nommé env :

```
python -m venv env
```
Cela créera un répertoire env contenant l'environnement virtuel.

Activez l'environnement virtuel en exécutant les commandes suivantes :

```
Set-ExecutionPolicy Unrestricted -Scope Process (si jamais erreur parce que windows bloquait)
```
```
env\Scripts\activate
```

Vous verrez que le nom de votre environnement virtuel apparaît dans l'invite de commandes,
indiquant que vous êtes maintenant dans l'environnement virtuel.

Vous avez maintenant créé et activé votre environnement virtuel. Passons à l'étape suivante.

## Étape 3 : Installation des dépendances
Dans cette étape, nous allons installer les dépendances nécessaires à partir d'un fichier requirements.txt. 
Ce fichier contient une liste de packages Python nécessaires à votre projet.

Assurez-vous que vous vous trouvez toujours dans votre environnement virtuel 
(vérifiez que l'environnement virtuel est activé dans l'invite de commandes).
Regardez l'invite de commandes qui s'affiche.
Si un environnement virtuel est activé,
vous verrez généralement son nom entre parenthèses ou crochets avant le chemin du répertoire actuel.
Par exemple :

```
(env) C:\chemin\vers\mon_projet>
```
Dans cet exemple, l'environnement virtuel nommé "env" est activé.

Dans le répertoire du projet se trouve un ficher requirements.txt
Aller dans le même dossier que ce fichier.

Exécutez la commande suivante pour installer les dépendances :

```
pip install -r requirements.txt
```

Cette commande parcourra le fichier requirements.txt et installera automatiquement toutes les dépendances listées.

<div style="border: 2px solid red; padding: 10px;">
    Il manque quelques packagse à installer en plus du requirements.txt. Il faut donc exécuter également les commandes suivantes:

```

    pip install scikit-learn
    pip install shapely
    
```

    
</div>


## Étape 4 : utilisation du script : 

<div style="border: 2px solid red; padding: 10px;">
copier les données à traiter dans un dossier data
et exécuter le script qui va générer un fichier nommé updated_resume_resultats.csv

```
    python main.py data new
```
ou 

```
    python main.py data legacy 
```
en fonction de si le format des données est l'ancien ou le nouveau. 
</div>

Lorsque que vous lancez le script on vous demandera d'entrer les paramètres pour la détection des cassures
(qui seront sauvegardés pour chaque tracé dans updated_resume_resultats.csv).
Voici une idée d'ordre de grandeur pour ces paramètres:

| Paramètre       | valeur example  | description     |
|-----------------|-----------------|-----------------|
| $v_{min}$       | 10             | vitesse minimum pour qu'une cassure soit détectée     |
| Angle minimum   | 60    | angle minimum pour qu'une cassure soit détectée |
| Interval de temps | 0.01    |  interval de temps entre position enregistré dans le fichier coordonnées |
| temps minimum | 0.1    | durée minimum entre deux détections de cassures  |
| disstance minimum | 10   | distance minimum entre deux détections de cassures |
| fenêtre d'angle | 2    |  nombre de temps surlequel l'anlge est calculé (e.g., fenêtre d'angle = 4, l'angle sera caluculé sur 2*4 + 1 temps) |



# ANNEXE
## comment naviguer dans les dossiers et les fichiers avec l'invite de commande windows.
L'invite de commande Windows permet de naviguer dans les dossiers et les fichiers de votre système d'exploitation.
Voici quelques commandes de base pour vous aider à naviguer :

`dir` : Affiche la liste des fichiers et dossiers présents dans le répertoire courant.
`cd` : Change de répertoire (dossier). Vous pouvez utiliser cette commande de différentes manières :
`cd` chemin/vers/dossier : Accède au dossier spécifié.
`cd .. `: Remonte d'un niveau dans l'arborescence des dossiers.
`cd \ `: Accède au répertoire racine.
`cd`: Accède au répertoire utilisateur (équivalent à cd C:\Users\nom_utilisateur).
`cd .. `: Remonte d'un niveau dans l'arborescence des dossiers.
`cd` : Accède au répertoire utilisateur (équivalent à cd C:\Users\nom_utilisateur).
`mkdir` : Crée un nouveau dossier. Par exemple, mkdir nom_dossier crée un dossier avec le nom spécifié.
`del` : Supprime un fichier. Par exemple, del nom_fichier.extension supprime le fichier spécifié.
`rmdir` : Supprime un dossier vide. Par exemple, rmdir nom_dossier supprime le dossier spécifié.
`exit` : Ferme l'invite de commande.

Voici comment utiliser ces commandes pour naviguer dans les dossiers et les fichiers :

Ouvrez l'invite de commande en appuyant sur la touche Windows + R, puis en tapant cmd suivi de Entrée.
Pour afficher la liste des fichiers et dossiers présents dans le répertoire courant, tapez dir et appuyez sur Entrée.
Utilisez la commande cd suivie du chemin vers le dossier souhaité pour accéder à ce dossier. Par exemple, cd Documents vous emmènera dans le dossier "Documents".
Pour remonter d'un niveau dans l'arborescence des dossiers, tapez cd .. et appuyez sur Entrée.
Utilisez la commande mkdir pour créer un nouveau dossier. Par exemple, mkdir NouveauDossier créera un dossier nommé "NouveauDossier" dans le répertoire courant.
Pour supprimer un fichier, utilisez la commande del suivie du nom du fichier et de son extension. Par exemple, del fichier.txt supprimera le fichier "fichier.txt".
Utilisez la commande rmdir pour supprimer un dossier vide. Par exemple, rmdir DossierVide supprimera le dossier vide "DossierVide".
Pour quitter l'invite de commande, tapez exit et appuyez sur Entrée.
Ces commandes de base vous permettront de naviguer dans les dossiers et de gérer les fichiers à l'aide de l'invite de commande Windows.