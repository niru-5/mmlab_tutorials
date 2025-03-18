# OpenMMLab Tutorial Series

This repository contains a series of tutorials on using various OpenMMLab repositories for computer vision tasks. We'll explore different components of the OpenMMLab ecosystem step by step.


[Why I created this repo?](#why-i-created-this-repo)


## Tutorial Structure

1. **MMEngine** - The foundational library
    - [Lesson 1: mmengine](lessons/lesson_1/lesson_1_1.md)
   This is one of the fundamental libraries of OpenMMLab. It has some design concepts which are implemented quite well. The concepts seem simple enough, but it takes some time to getting used to write the code in that style. It might seem a lot of work for small gains initially, but it does pay off(if you build models quite often).

2. **MMDetection** - Object Detection Framework
   Still Under Construction. 

3. **MMSegmentation** (Coming Soon)

5. **Additional OpenMMLab Projects** (Planned)
   - MMPose
   - MMRotate
   - MMTracking
   - MMDeploy (This closes the loop by deploying the models to the edge/cloud)

## Getting Started

The tutorials are designed to be followed sequentially, as later sections may build upon concepts introduced earlier. 
I will try to cover more of the good practices and tips I follow. I will also cover some theory on algorithms, but I would expect the user to have some prior knowledge of the topic(like how some of the object detection algorithms work).
- Theoretical background
- Practical examples
- Code implementations
- Best practices and tips

## Prerequisites

- Basic Python programming knowledge
- Understanding of deep learning concepts
- CUDA-capable GPU (recommended, can run on colab too(Need to test it))
- Docker (Highly Recommended)

## Setup
My setup involves using a custom docker image. Once the docker image is built, I run it with the command "sleep infinity" to keep the container running. And I mount the volumes(using bind mount) to the docker container. The work is done in the volumes and I can access it from the host machine. I also won't loose any data even if the container is deleted.
I use dev-containers to attach my vs-code/cursor to the docker container.
More on this in the [Lesson 1](lessons/lesson_1/lesson_1_1.md)


# Open to Feedback
Drop in issues or suggestions. I will try to respond to them as soon as possible.


# Why I created this repo? 
Aren't there enough medium blogs about mmdetection on the internet? 
Yes, there are. 
But, I did find them to be a bit more surfacial and not a very in-depth guide. They did help me in finding the amazing repos at openmmlab, but when I started working with mmdetection or mmsegmentation, the learning curve was a bit steep. 
I did see quite some redditors being confused or annoyed with the whole process of setting up the environment or understanding the code structure. And there are some other reditors who did understand, there is a lot of value in mastering this codebase. It is hard but it is worth it. 
I did go through a part of this journey, where I was amazed with the amount of algorithms that were covered. It was initially a bit easy to run some algorithms, and later I did have some frustration in making sense of everything. I don't think I have mastered it yet, but I do feel like I have a decent knowledge of the way things work. The goal is to take some of my sufferings and make them as a nice series of tutorials for others to try and learn. And also, it is a bit of a learning experience for me as well to write these tutorials. 