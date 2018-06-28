# sheel

\documentclass[12pt,a4paper]{article}
\usepackage[latin2]{inputenc}
\usepackage{graphicx}
\usepackage{ulem}
\usepackage{amsmath}
\begin{document}
\textbf{Red Wine Qaulity Dataset}



\textbf{Introduction:}



This dataset is downloaded from Kaggle website. This is a small dataset 
involving 11 variables related to red wine quality.

Fixed Acidity: Acids in red wine involved in red wine

Volatile Acidity: Amount of acetic acid in wine when in large quantity 
leads to unpleasant taste

Critic Acid: Usually in small quantities and adds freshness and flavor 
to red wine

Residual Sugar: Amount of Sugar remaining once fermentation stops

Chlorides: Amount of salt in red wine

Free Sulfur Dioxide: Free from sulphur dioxide 

Total sulphur dioxide: Amount of free and bound forms of SO2

Density: Density of water close to water

pH: Describes how acidic or basic the wine is

sulphates: Wine additive which contributes to sulphur dioxide gas

Alcohol: percent of alcohol content in wine

Quality: The output variable which predicts how good the wine is (6.5 
and above = Good)

 

\textbf{Working:}



I have applied the KNN algorithm which predicts the data depending of 
its neighbors. 

1. Imported various libraries Numpy, Pandas, Sci-Kit Learn, Matlpotlib

2. Read the data from csv file 

3. Since the data is free on missing values scaled and fitted the data 
for normalization

4. Imported the preprocessing for testing and training the dataset

5. Applied the KNN algorithm to predict the outcome

6. Mapped with confusion matrix for accuracy of the predicted outcome

\end{document}
