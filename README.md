# Task 

- The task in this project is to obtain a cookiecutter template from an existing hydra_lightning template 
- Other task is to replace MNIST data training with CIFAR10 
- Also, to integrate COG for inferencing and timm in order to train any model. 

## Inferencing 
- Inferencing can be done by doing cog.predict 

## Things done
- In order to incorporate CIFAR10, i had to create cifar10.yaml for both model and datamodule. As well as make changes to src/models/cifar10_module.py and src/data/cifar10_datamodule.py
