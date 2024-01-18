# MLCSP Data  


# Enviroment 
```
conda create -n mlcsp python=3.10
conda activate mlcsp
mamba install pytorch::pytorch torchvision torchaudio -c pytorch
mamba install pandas 
```

```
cat data/PM_single_truth_10000.txt | sed -e 's/  */,/g' > data/PM_single_truth_10000.csv 
```