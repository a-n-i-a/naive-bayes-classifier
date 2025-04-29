# 🪻 Naive Bayes Algorithm used to clasify a type of Iris
This simple program is meant to classify a type of Iris given by the user (Sepal Width, Sepal Length, Petal Width and Petal Length - given as Doubles) using
Naive Bayes Algorithm 

### ✏️ Naive Bayes Algoritghm 
As the name suggests, Naive Bayes Algorithm is naive - meaning it assumes that all features of iris (in this case) are independent. Suprisingly, even though 
this assumption is highly unrealistic, Naive Bayes Algorithm does a pretty good job at clasifying the data.

#### 🧞‍♂️ Deeper Dive: Smoothing 
In Naive Bayes Algorithm, there is a technique used to remove the chance of zero probability - called Smoothing. It is achieved by multiplying standard deviation
by a 1.1, small enough to reduce the zero probability and not to disturb as much the final output.
