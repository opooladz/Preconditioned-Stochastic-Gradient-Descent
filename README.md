# Preconditioned-Stochastic-Gradient-Descent
A repo based on [XiLin Li's PSGD repo](https://github.com/lixilinx/psgd_torch) and extends some of the experiments.




### Quick Guide

```shell
python3 psgd_cifar10.py --experiment cifar10 --optimizer XMat
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
python3 psgd_cifar10.py --experiment blurred --stage2 cifar10 --epoch_concept_switch 100  --optimizer XMat
```

will train a ResNet-18 for 100 epochs and then switch to training with standard clean cifar10 data.

For NTK Attacked dataset you need to [download](https://drive.google.com/drive/folders/1OD54_gK6wnhyVwQGnHs7vIsKVOL-48zd?usp=share_link) and set the path via the  ```--data_root``` argument.

If you want to run the Noisy experiments that uses proir information run the ```psgd_cifar10_noisy_label.py``` code.


### Experiment Observations 
#### Noisy Label : Gifted Kid Syndrome 
  * PSGD gets test acc 78% avg over 5 nets with ~45% train acc over noisy labels. 
  * SGD gets test acc 23% avg over 5 nets with ~44% train acc over noisy labels.
    * 4/5 get 10% test acc at 200 epochs with 99.99% confidence in predictions 
      * Gifted Kid Syndrome -- most kids are not that smart but can learn with a good teacher. 
      * With a bad teacher seems the best they can do is memorize; since they get 10% acc on test set with super high confidence
    * 1/5 gets 77% test acc at 200 epochs with 99.99% confidence in predictions 
      * Lucky Gifted Kid -- actually super smart and can learn/generalize even given a teacher thats wrong 54% of the time
    * both have ~44% acc on noisy labeled train set 
    
#### Blurred: Clear indication of PSGD retaining neuro-plasticity vs SGD.
  * Train for 100 epochs of blur and then for another 100 with standard: 
    * PSGD recover test accuracy of 93.5% -- a 2% decrease compared to a ~11% decrease for SGD


### Dataset Setup:
Download the Neural Tangent Generalization Attacks [Dataset](https://drive.google.com/drive/folders/1OD54_gK6wnhyVwQGnHs7vIsKVOL-48zd?usp=share_link) and put it in the datasets folder.

### TODO
* Integrate Trace of FID
* Integrate entropy max margin and forgetting 
* Add GPT experiments 
* LLamA -- G or Meta's new LLM is around the same size as GPT2 would be REALLY nice if we could get results for that since its newer...
* Integrate SplitResNet results and do analysis...
* Lets get threshSGD integrated -- use subset instead of zeroing out the loss....
* do threshPSGD tests


### Ideas
* Coresets via trace of FIM?
* so many more things.... 


### Stats to colletct
* Forgettability 
* entropy 
* margin 
* Tr(FIM) both ways 
* Spearman r between entropy/margin orderings and forgettability score
* everything else we talked about
