# Official implementation of "LUSS-AE"
This is the implementation of the paper "Leveraging Unsupervised and Self-Supervised Learning for Video Anomaly Detection" (VISAPP 2023).

## Dependencies
* Python 3.6
* PyTorch >= 1.7.0 
* Numpy
* Sklearn
* Use conda to install dependecies from requirements.yml

## Datasets
Download the datasets from their respective pages and put into ``dataset`` folder, like ``./dataset/ped2/``, ``./dataset/avenue/``, ``./dataset/shanghai/``

## Training
```bash
git clone https://github.com/devashishlohani/luss-ae_vad
```

* Running with default settings
```bash
python main.py --dataset_type ped2
```
Select --dataset_type from ped2, avenue, or shanghai.
For more details, check main.py

