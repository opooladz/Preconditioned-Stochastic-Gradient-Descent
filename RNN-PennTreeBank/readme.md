## Base ReadME from AdaBelief 

We can do other things like run with GRU...
```shell
 python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 200 --save PTB.pt --when 20 40 60 100 120 140 160 180 --clip 1.0 --beta1 0.9 --beta2 0.999 --optimizer psgd --lr 3 --eps 1e-16 --eps_sqrt 0.0 --nlayer 1 --run 0 --model GRU --wdrop 0.5 --wdecay 0  

```

QRNNs not supported yet but would be nice to show QRNN + PSGD can match other RNNs

## run experiments

#### 1-layer LSTM
```python run_all_layer1.py```

#### 2-layer LSTM
```python run_all_layer2.py```

#### 3-layer LSTM
```python run_all_layer3.py```

## Visualization
see ```LSTM_test.ipynb```
