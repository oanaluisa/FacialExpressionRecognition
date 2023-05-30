# FacialExpressionRecognition
 
Open cv - read image in BGR
Matplotlib - read image in RGB

Data - https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
%matplotlib inline
 
from glob import glob

Conda activate faceexpr
 
Cd C:\face expressions
 
Start flask - cd C:\face expressions\flask
Python main.py

TEORIE

TEORIE

CNN - tip special de retea neuronala cu mai multe straturi - inspirata de mecanismul sistemlui optic al creaturilor vii

- compus din blocuri simple/multiple de straturi convolutionale/pooling, straturi complet conectate, strat de iesire

MODEL CNN = combinatie 2 componente - partea de extractie a caracteristicilor
					- parte de clasificare

- straturi convolutionale + pooling => extrag caracteristici (ochi, nas, frunte, gura);
	- invata caracteristicile construindu-se unul peste altul = primele straturi detecteaza margini, urmatoarele combina pt a 
detecta forme, iar straturile urm imbina aceste informatii pt a deduce o clasa/emotie

- straturile complet conectate => actioneaza ca un clasificator asupra caracteristicilor si atribuie o probabilitate ca
imaginea de intrare  sa apartina unei anumite clase
	- invata cum sa foloseasca aceste caracteristici produse de convolutii pt a clasifica corect imaginile


CNN nu stie nicio emotie - vazand mai multe imagini cu o emotie anume invata sa detecteze caracterstici


A. STRAT CONVOLUTIE - partea centrala a unei arhitecturi CNN (numar filtre, dimensiunea filtrului)

- convolutia = OPERATIE LINIARA CARE IMPLICA MULTIPLICAREA UNUI SET DE GREUTATI CU INTRAREA.

- multiplicarea este realizate intre o matrice de date de intrare si o matrice bidimensionala de greutati (filtru/kernel (3,3))
	- multiplicarea intre o sectiune a matricei de intrare de dimensiunea filtrului si filtru => o singura valoare

= filtru se aplica sistematic fiecarei parti surprapuse sau sectiune de la stanga la dreapta, de sus in jos
- daca filtru detecteaza un anumit tip de caracteristica, atunci aplicarea filtrului in mod sistematic pe intreaga imagine
permite filtrului posibilitatea de a descoperi acea caracteristica oriunde in imagine

= rezultatul straturilor convolutionale este o matrice bidimensionala de valori de iesire care reprezinta o filtrare a
matricei de intrare  === "harta caractersiticilor" (indica locatia si puterea fiecarei caracteristici detectate intr-o imagine



B. STRAT POOLING - rezumarea prezentei caracteristicilor in sectiuni ale hartii de caracteristici
- preia cate o regiune de dimensiuni reduse din outputul straturilor convolutionale, asupra caruia aplica un filtru bidimensional (2,2) pt
a sumariza caracteristicile situate in regiunea acoperita de filtru si a produce o singura iesire

- folosit pt a REDUCE NR DE PARAMETRII DE INVATAT si pt a REDUCE CALCULELE EFECTUATE IN RETEA

MAX POOLING -- preia CEA MAI MARE VALORARE a pixelilor dintr-o regiune => outputul dupa fiecare strat de max pooling este o harta de
caracterisitici care contine cele mai proeminente atribute ale hartii de caract. rezultate din strat conv.



C. STRAT DROPOUT - folosit pt a SETA IN MOD ALEATORIU UNITATILE PRIMITE CA INPUT CU 0 CU O RATA (param 0.5) la fiecare pas din timpul
antrenamentului
- restul de intrari sunt marite cu 1/1-rata (0.5) => suma tuturor intrarilor neschimbata

- anuleaza contributia unor neuroni

- reteaua este mai capabila sa generalizeze mai bine si probabilitatea sa apara supra-adaptare este redusa



D. STRAT FLATTEN - TRANSFORMA FORMATUL IMAGINILOR DINTR-UN TABLOU BIDIMENSIONAL IN UNIDIMENSIONAL
- reformeaza datele


F. STRAT FULLY CONNECTED - PREIA CA INPUT HARTA DE CARACT. APLATIZATA DE FLATTEN SI EFECTUEAZA OPERATII ASUPRA FIECARUI NEURON PT A GENERA
UN OUTPUT


G. STRAT IESIRE - SOFTMAX - CREEAZA UN VECTOR A CAROR VALORI ADUNATE = 1, CA SA POATA FI INTERPRETATE CA PROBABILITATI


-> fc de activare -> decide ce inforrmatii ale modelului ar trebui pastrate
"ReLu" - returneaza 0 pt orice valoare negativa, pt valoare pozitiva returneaza valoarea


"categorical_crosstropy" - imagine poate apartine doar unei clase
