# Introduction

Computer vision is significantly a vast domain. There are always new techniques with different kinds of approaches. It really took off in 2014ish. With all the deep learning models seeing immense performance. 


## Before computer vision was more of a researc topic
Around 2014-2018, the landscape was mostly filled with individual repos mostly published by researchers. Some of the prominent ones were yolo, faster-rcnn. And people used these repos with their data to solve their specific use-case. There were very little companies trying to build end to end computer vision platforms. This was a very hard problem as the research was still advancing(actually it still is in 2025) at a fast pace. 

## How the landscape changed
Soon, the landscape filled up with a lot of companies both open-source and closed-source helps in either one part of this pipeline or as an end to end solution. 
And one of these open source libs was from the OpenMMLab which released some amazing repos like mmdetection, mmpose, and many more. 

## Emergence of computer vision platforms

## OpenMMLab
The whole overview of mmlab can be best described in this picture. There is a single base which acts as a backbone for all the other repos. The other repos expand on the base in terms or models, datasets, losses, metrics. The team has kept it very flexible for people to customize the models or any other parts of the code. 

# Installation
This is something quite some people have faced problems with. Normally I believe it is quite hard to install the packages locally. They each do have different kinds of dependcies and it can be quite a hassle to deal with them. Either use the docker version from them, or try to build one for a specific repo. 

There is a docker file here that includes mmengine, mmdetection, mmpose and mmsegmentation all in one environment. 
The flow is to have all the repos within the working directory directly and overlay mmengine template on it. 
Why do I do that this way? Because this keeps my code separate from the mm_repos code. It makes it easier to debug and also change version later on, if I want a next version of a specific repo. 
And it makes it easier to share the codebase with others, because essentially only the overlay is shared. After a while, the org will itself have a good set of modules/classes which can be shared and easily tested. 

-> The tutorial follows this procedure from now on and it will try to push the user to write code in mmengine_custom and not in the repos directly. This can be hard initially but is considered one of the good practise. 

# mmengine

## Runner

## Hooks

## Registry

## Configs

# A simple example
Lets take a simple example of classification of food dataset. A very simple dataset of different kinds of fruits and vegetables. Classifying them. 

# Writing a dataset loader(Assignment 1.1.1)

# Writing a custom model(Assignment 1.1.2)

# Writing a custom hook(Assignment 1.1.3)

# Putting it all in a config file(Assignment 1.1.4)


# Conclusion


