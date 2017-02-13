# Living_Indicator
Learning based model where google street-view images are scored on the scale of 0-10 using computer vision and machine learning algorithms.

##ABSTRACT :

As we move forward in the twenty first century, we have witnessed the growth of Artificial

Intelligence greater than ever. From handwriting recognition to humanoid robots, there are

numerous fields on which extensive researches are being carried out. Our society, as we know

today, is transforming itself under the hood of Machine learning. One such area that has potential

to leverage the offerings of these bleeding edge technologies is Property Evaluation.

In this Thesis Project I will be exploring the application of Computer Vision and Machine

Learning algorithms to evaluate a community/property based on the images obtained from

Google Street-view Images. The idea is to assign the property with a rating based on the

appearance, and other factors like location of the house, year of construction, built in area etc.,

that influence the value of the property.

##INTRODUCTION:

The core of this project lies within the task of processing images. This will be done using the

famous Machine Learning method called Convolutional Neural Network (CNN) and other

machine learning algorithms. Convolutional neural network (CNN, or ConvNet) is a type of

artificial neural network which is inspired and designed based on the organization of the

mammal visual cortex. Individual neuron unit are active in a restricted space called receptive

field. These receptive fields of individual neuron units overlap with each other and convolution

is applied over the whole set. The layered neuron design of the model facilitates least amount of

preprocessing.

The trained model that is then coupled with other aspects of the property such as number of

bedrooms, price, built in area, year built etc. which acts as the indicator of the value of the

property. However, since quantifying so many features in a single glance is rather difficult, we

resort to dimensionality reduction using Principal Component Analysis (PCA) to get a 0-10 scale

rating on the property. Principal components analysis is a technique of picking smaller number of

uncorrelated variables ("principal components") from a large set of data. The goal of principal

components analysis is to explain the maximum amount of variance with the fewest number of

principal components (features).

Once the model is trained with the Google street-view Images and the corresponding Zillow API

data, we validate the accuracy and performance of the model using the cross validation method

and make observations regarding the errors resulting from the model.


##PROBLEM STATEMENT:

Evaluate Living Index which is a measure of wellbeing of a community, from the Google street-
view images based on the components/properties of the image.


##OBJECTIVES

1. Extract data from Zillow using Zillow API.

2. Extract corresponding Google Street-View images using Google Street-view API.

3. Design and develop a continuous score for evaluating the property based on Zillow API data as features using Principal Component Analysis.

4. Train the model using Convoluted Neural Network and other Machine Learning algorithms.

5. Validate and document results/errors, performance of the model.
