# 15-fev-24 / Jeudi:

Selection des 13 classes les plus representees  (<2K paintings) \
loader dans un iterateur tf avec scaling \
baseline model


Issue faced: \
Dataset subfolder selection dans l'iterateur tf\
Model training too large \
Cannot upload dataset to collab

To be done de son coté:\
baseline model

To be done samedi: \
diviser les tasks \
distincteur ? Check dnas chaque style: \
  Taille/Ratio des tableaux ? Couleurs ? Luminosité ?

To be looked into:\
Data augmentation


# 17-fev-24 / Samedi:

Creation d'un fichier .py pour le moduler ensuite.
Separation du code en plein de fonctions

Entrainement d'un modele basique, il overfit et ne fait pas mieux que baseline.

Impossibilité de charger tout le dataset en numparray ou dataframe (il faut 25go de ram)\
A la place, creation de l'array style par style possible.


Creation d'un GCP_PROJECT, d'un BUCKET et d'une INSTANCE VM
Installation python et quelques dependencies basiques dans VM
Upload de tout le dataset dans VM.



to be done:

Status à enlever dans cubisme
Nettoyer le dataset !

Creer un tableau moyen par style
Creer histogram montrant intensité pixel moyenne par style (et la variance ?)
Creer piechart montrant couleurs moyennes par style (et la variance ?)
