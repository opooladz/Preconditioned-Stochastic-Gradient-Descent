# Preconditioned-Stochastic-Gradient-Descent

For paper Main and Appendix see [Curvature-Informed SGD via General Purpose Lie-Group Preconditioners](./Curvature_Informed_SGD_via_General_Purpose_Lie_Group_Preconditioners_Main___Appendix.pdf)

### Quick Guide

```shell
python3 psgd_cifar10.py --experiment cifar10 --optimizer PSGD_XMat
```
You can pick from the following CIFAR10 ```--experiment```:
* Standard: ```cifar10```
* Class Imballanced:  ```imb```
* NTK Attacked: ```attacked```
* Noisy Label:  ```noisy```
* Blurred:  ```blurred```

You can control if you want to change dataloaders on the fly by setting a ```--stage2``` dataset.
For example:

```shell
python3 psgd_cifar10.py --experiment blurred --stage2 cifar10 --epoch_concept_switch 100  --optimizer PSGD_XMat --lr_scheduler exp
```

will train a ResNet-18 for 100 epochs and then switch to training with standard clean cifar10 data. Note lr_scheduler of exp is to be consistant with Critical Learning Period's paper and does not yield the best results. For best results use ```--lr_scheduler cos```

and 

```shell
python3 psgd_cifar10.py --experiment blurred --stage2 cifar10 --epoch_concept_switch 100  --optimizer SGD --num_runs 5
```
will train a ResNet-18 for 100 epochs and then switch to training with standard clean cifar10 data using SGD

For NTK Attacked dataset you need to [download](https://drive.google.com/drive/folders/1OD54_gK6wnhyVwQGnHs7vIsKVOL-48zd?usp=share_link) and set the path via the  ```--data_root``` argument.

If you want to run the Noisy experiments from the paer that uses proir information run the ```psgd_cifar10_noisy_label.py``` code.




### Dataset Setup:
Download the Neural Tangent Generalization Attacks [Dataset](https://drive.google.com/drive/folders/1OD54_gK6wnhyVwQGnHs7vIsKVOL-48zd?usp=share_link) and put it in the datasets folder.

Follow other readme files for each respective folder

