# Preconditioned-Stochastic-Gradient-Descent




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

If you want to run the Noisy experiments that uses proir information run the ```psgd_cifar10_noisy_label.py``` code.


### Experiment Observations 
#### Noisy Label : Pure Memorization without Generalization 
  * PSGD gets test acc 78% avg over 5 nets with ~45% train acc over noisy labels. 
  * SGD gets test acc 23% avg over 5 nets with ~44% train acc over noisy labels.
    * 4/5 get 10% test acc at 200 epochs with 99.99% confidence in predictions 
      * Pure Memorization -- Simply overfit the train set but with no generalization to the test set 10% accuracy on test. 
      * With a bad teacher seems the best they can do is memorize; since they get 10% acc on test set with super high confidence
    * 1/5 gets 77% test acc at 200 epochs with 99.99% confidence in predictions 
      * Lucky Initilization -- actually super smart and can learn/generalize even given a teacher thats wrong 54% of the time
    * both have ~44% acc on noisy labeled train set 
    
#### Blurred: Clear indication of PSGD retaining neuro-plasticity vs SGD.
  * Train for 100 epochs of blur and then for another 100 with standard: 
    * PSGD recover test accuracy of 93.5% with cosine lr sched
      *a 2% decrease compared to no deficit a ~1% decrease for SGD
    * With exp decay lr sched and removing the deficit at 100 epochs:
      * PSGD got
      * While the reported numbers for SGD was abount 84% 


### Dataset Setup:
Download the Neural Tangent Generalization Attacks [Dataset](https://drive.google.com/drive/folders/1OD54_gK6wnhyVwQGnHs7vIsKVOL-48zd?usp=share_link) and put it in the datasets folder.

### TODO
* Integrate Trace of FID
* Integrate entropy max margin and forgetting 
* Add RL Experiments & Results
* add SimCLR Experiments & Results
* add ConvMix Experiments & Results
* add ViT Experiments & Results
* add NAS Experiments & Results
