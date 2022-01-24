![logo](data/logo.png)

[![CodeFactor](https://www.codefactor.io/repository/github/everlookneversee/bts_dp_mri/badge)](https://www.codefactor.io/repository/github/everlookneversee/bts_dp_mri)
![GitHub](https://img.shields.io/github/license/EverLookNeverSee/BTS_DP_MRI)
![GitHub branch checks state](https://img.shields.io/github/checks-status/EverLookNeverSee/BTS_DP_MRI/main)
![GitHub language count](https://img.shields.io/github/languages/count/EverLookNeverSee/BTS_DP_MRI)
![GitHub top language](https://img.shields.io/github/languages/top/EverLookNeverSee/BTS_DP_MRI)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/EverLookNeverSee/BTS_DP_MRI)
![Lines of code](https://img.shields.io/tokei/lines/github/EverLookNeverSee/BTS_DP_MRI)
![GitHub all releases](https://img.shields.io/github/downloads/EverLookNeverSee/BTS_DP_MRI/total)
![GitHub issues](https://img.shields.io/github/issues-raw/EverLookNeverSee/BTS_DP_MRI)
![GitHub pull requests](https://img.shields.io/github/issues-pr-raw/EverLookNeverSee/BTS_DP_MRI)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/EverLookNeverSee/BTS_DP_MRI)
![GitHub contributors](https://img.shields.io/github/contributors/EverLookNeverSee/BTS_DP_MRI)
![GitHub last commit](https://img.shields.io/github/last-commit/EverLookNeverSee/BTS_DP_MRI)

## Authors
* Research: [Dr. Golestan Karami](https://www.linkedin.com/in/golestan-karami-45984938/)
* Development: [Milad Sadeghi DM](https://elns.ir)


## Sample Pipeline
**Pre-processing:**  
**Desired dataset structure:**  
Input data shape: 64x64x10
```
.
└── Train
    ├── 100
    │   ├── mask_100.nii
    │   ├── MD_100.nii
    │   ├── rCBV_100.nii
    ├── 101
    │   ├── mask_101.nii
    │   ├── MD_101.nii
    │   ├── rCBV_101.nii
```
Do preprocessing on dataset:
```shell
python preprocessing.py --verbose --dataset <path to dataset directory>
```
This will create a new directory called *npy_files*, then creates specific folders for
each sample we have in dataset and saves *image.npy* and *mask.py* files in these folders.  
* Output data shape:  
    * Image: 64x64x16x2     
    * Mask: 64x64x16x3
```
npy_files/
├── 0
│   ├── mask_0.npy
│   └── image_0.npy
├── 1
│   ├── mask_1.npy
│   └── image_1.npy
```
**Training the model:**
```shell
python train.py --verbose --dataset <path to npy_files> --learning_rate 0.0001 --batch_size 2 --epochs 100 --save ./
```
After completing training process, command above will save entire trained model as a *.pb* file
, save training process history details as a *.csv* file and plot training process diagrams.


## Results
**Train:**  

| Fold    |  Accuracy      |   IOU Score   |   Loss        |   Precision  |   Recall   |   F1-Score   |
|  :----: |    :----:      |   :----:      |  :----:       |    :----:    |   :----:   |   :----:     |
| 1       | 0.968775       |   0.897440    |  -1.903993    |   0.968788   |  0.968766  |  0.968777    |
| 2       | 0.970905       |   0.903380    |  -1.730110    |   0.970907   |  0.970904  |  0.970905    |
| 3       | 0.972417       |   0.907598    |  -1.769276    |   0.972422   |  0.972415  |  0.972419    |
| 4       | 0.972910       |   0.910098    |  -1.937584    |   0.972916   |  0.972903  |  0.972909    |
| 5       | 0.973350       |   0.912430    |  -1.882656    |   0.973355   |  0.973342  |  0.973349    |
| 6       | 0.972398       |   0.907724    |  -1.793769    |   0.972407   |  0.972395  |  0.972401    |
| 7       | 0.973002       |   0.905900    |  -1.848724    |   0.973010   |  0.972997  |  0.973004    |
| 8       | 0.971803       |   0.906942    |  -1.939411    |   0.971807   |  0.971801  |  0.971804    |

## License
This project licensed under the MIT License - see the [LICENSE](LICENSE) file for more details.


## References
1. [Optimized U-Net for Brain Tumor Segmentation](https://arxiv.org/abs/2110.03352)
2. [Optimized High Resolution 3D Dense-U-Net Network for Brain and Spine Segmentation](https://www.mdpi.com/2076-3417/9/3/404)
3. [A novel fully automated MRI-based deep-learning method for classification of IDH mutation status in brain gliomas](https://pubmed.ncbi.nlm.nih.gov/31637430/)
