# Create our autonomous robot

Le code s'appuis sur des models déja utiliser.
Plusieurs phase dans la réalisation de ce projet.

## specifications

* Robot operating only on a pedestrian lane and cycle path
* Using google map to have GPS's pointway
* Have a smaller ai model as possible
* Have to take considerations of all road signalitics :
  * Speed limitation
  * Road lane
  * Human and object collision
  * Sheet collision (on the road)

Le projet ce découpe en plusieurs phases de conceptions.
La deuxieme phase est la plus complexe a mettre en oeuvre et détermine a suite des phases suivante.

### Première phase

  Perception de l'environnement (profondeur de carte, segmentation des textures, détection des objects ,suiveur de ligne)
  
### Deuxième phase

Création de la vue 3D bird'eye view (BEV) avec une vue simplifiée sous forme de vecteur.

### Troisième phase

  La sortie du BEV passe dans l'entrée d'un transformer et aussi les différents capteurs GPS et pathway de google map. Cela permet de faire une prediction du les points de route future.

### Create our ai model

First objectif :
From carla software simulation environment :

* Scripte simulation with automatic and manuel
* AI model (ARchitecture and dataflow)
* Architecture of training model
* Scripte of evaluation performance model

### Hardware specification

* 2 camera min 1080p
* GPS
* Acces to google map (online or offline)
* Robot

<figure>
<img title="illustration of robot" src="Images/Robot_illustration.jpg">
  <figcaption>illustration of robot </figcaption>
</figure>

### Robot architecture

<figure>
<img title="Ai model architecture" src="Ai Model Architecture/Model_Architecture.png">
  <figcaption>Ai model architecture </figcaption>
</figure>

<figure>
  <img title="Robot architecture" src="Ai Model Architecture/Robot_Architecture.png">
  <figcaption>Robot architecture </figcaption>
</figure>



### how to create Map
Follow the link:
Tesla model explaine:

méthode 4 Pillars : https://www.thinkautonomous.ai/blog/autonomous-vehicle-architecture/
[ Perception -> Localization -> Planning -> Control ]
this methode is very good for only for little system

https://www.thinkautonomous.ai/blog/tesla-end-to-end-deep-learning/
https://www.thinkautonomous.ai/blog/occupancy-networks/?ref=thinkautonomous.ai
https://carla.readthedocs.io/en/latest/build_windows/#windows-build
