# Getting the Dataset and Metadata 

The data can be obtained from Chad Spooner's [Cycliostationay Signal Processing](https://cyclostationary.blog/2023/02/02/psk-qam-cochannel-data-set-for-modulation-recognition-researchers-cspb-ml-2023/) blog. His blog has more of a description of the data than you will find here. Once you download the data, the folder structure in `data` should look like this:

```
data/PM_One_Batch_1/
data/PM_One_Batch_2/
...
data/PM_One_Batch_10/
data/PM_Two_Batch_1/
data/PM_Two_Batch_2/
...
data/PM_Two_Batch_10/
data/PM_single_truth_10000.csv
data/PM_two_truth_10000.csv
```

Note that the truth files need to be generated from the original files on the cycliostationary blog. This can be done with the following command:

```
cat data/PM_single_truth_10000.txt | sed -e 's/  */,/g' > data/PM_single_truth_10000.csv 
```

| Mod-Type | Mod-Variant | Signal |
| --- | --- | --- | 
| 1 | 1 | BPSK |
| 1 | 2 | QPSK (4QAM) |
| 1 | 3 | 8PSK |
| 2 | 2 | 4QAM (QPSK) |
| 2 | 4 | 16QAM |
| 2 | 6 | 64QAM |
| 3 | 1 | SQPSK |
| 3 | 2 | MSK |
| 3 | 3 | GMSK |



