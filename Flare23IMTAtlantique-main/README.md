# Flare23IMTAtlantique
submission of IMT Atlantique attention for FLARE23 challenge

author: Wu panpan,An Peng,Xu yurou

## training 
```python
python3 flare23-train.py -i input_folder -o output_folder
```

for training, note that ```input_folder``` should contain both ```TrainingImg/``` and ```TrainingMask/``` folders

## docker creation 
- copy epoch.pth in the weights folder
```python
sudo docker build -t war .
```
```python
sudo docker save -o war.tar.gz war
```

## inference on test FLARE23 data
```python
sudo docker image load < war.tar.gz
- docker container run --gpus "device=0" --name war --rm \
-v $PWD/inputs/:/workspace/inputs/ \
-v $PWD/TeamName_outputs/:/workspace/outputs/ \ 
war:latest /bin/bash -c "sh predict.sh"
```


