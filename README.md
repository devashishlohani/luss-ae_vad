# Official implementation of "LUSS-AE"
This is the implementation of our paper "Leveraging Unsupervised and Self-Supervised Learning for Video Anomaly Detection" (VISAPP 2023).

## Cloning repository
```bash
git clone https://github.com/devashishlohani/luss-ae_vad
```

## Dependencies
* Python 3.6
* PyTorch >= 1.7.0 
* Numpy
* Sklearn

Use conda to install dependencies from requirements.yml

## Datasets
* USCD Ped2 [[dataset](https://drive.google.com/file/d/1w1yNBVonKDAp8uxw3idQkUr-a9Gj8yu1/view?usp=sharing)]
* CUHK Avenue [[dataset](https://drive.google.com/file/d/1q3NBWICMfBPHWQexceKfNZBgUoKzHL-i/view?usp=sharing)]
* ShanghaiTech [[dataset](https://drive.google.com/file/d/1rE1AM11GARgGKf4tXb2fSqhn_sX46WKn/view?usp=sharing)]

Download the datasets and put into ``dataset`` folder, like ``./dataset/ped2/``, ``./dataset/avenue/``, ``./dataset/shanghai/``

## Training
```bash
python train.py --dataset_type shanghai
```
Select --dataset_type from ped2, avenue, or shanghai.

For more details, check train.py

## Testing
* First fetch the pre-trained models from [drive](https://drive.google.com/drive/folders/1KhfgOfTxhwagr9A41QCNeDXne7pKbSxz?usp=sharing)
* Place the model in ``exp`` folder like ``./exp/shanghai/pre_trained/model_best.pth``
* Run the test script as shown below
```bash
python test.py --test_batch_size 4 --num_workers_test 4 --dataset_type shanghai --model_path exp/shanghai/pre_trained/model_best.pth 
```
For more details, check arguments in test.py

## Bibtex
```
@inproceedings{lohani2023leveraging,
  title={Leveraging Unsupervised and Self-Supervised Learning for Video Anomaly Detection},
  author={Lohani, Devashish and Crispim-Junior, Carlos F and Barth{\'e}lemy, Quentin and Bertrand, Sarah and Robinault, Lionel and Tougne, Laure},
  booktitle={18th International Conference on Computer Vision Theory and Applications},
  volume={5},
  pages={132--143},
  year={2023},
  organization={SCITEPRESS-Science and Technology Publications}
}
```
