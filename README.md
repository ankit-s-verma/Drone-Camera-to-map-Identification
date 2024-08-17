# Drone-Camera-to-map-Identification
PyTorch implementation of Drone Camera to map Identification.

## Libraries Used
- Pandas
- Numpy
- Torch, Torch.nn
- DataLoader
- MatplotLib
- TQDM
- interp1d
- Rotation, Slerp

```bash
pip install pandas numpy torch scipy
```

## Dataset
The dataset is available in the below addresses:

- [drive.google.com](https://drive.google.com/file/d/1HfuZYnSdeCiFsqkP57Jn9i_y22kpQ7xp/view)  
- [pan.baidu.com](https://pan.baidu.com/s/1wj5YeMah2N7Olka7MoeJEg) (password: y9dg)

The format of files are as follow:
```shell
Data
|--data_train
|   |--0
|     |--SenseINS.csv
|   |--...
|--data_val
|   |--0
|     |--SenseINS.csv
|   |--...
|--data_test
|   |--0
|     |--SenseINS.csv
|   |--...
|--Sense_INS_Data.md     // The format description of SenseINS.csv
```
