[![Security Status](https://www.murphysec.com/platform3/v31/badge/1682423519889874944.svg)](https://www.murphysec.com/console/report/1682423519839543296/1682423519889874944)

# Solution of Team saltfish for NeurIPS 2022 Cell Segmentation Challenge

This repository is the official implementation of our paper [MAUNet: Modality-Aware Anti-Ambiguity U-Net for
Multi-Modality Cell Segmentation](TBA). 

You can reproduce our method as follows step by step:



## Environments and Requirements

Our development environments:

| System                  | Ubuntu 22.04.1 LTS                         |
| ----------------------- | ------------------------------------------ |
| CPU                     | Intel(R) Xeon(R) CPU E5-2695 v4 @ 2.10GHz  |
| RAM                     | 16*×*4GB; 2.67MT/s                         |
| GPU(number and type)    | Two NVIDIA Titan RTX 24G                   |
| CUDA version            | 11.3                                       |
| Programming language    | Python 3.10.4                              |
| Deep learning framework | Pytorch (Torch 1.11.0, torchvision 0.12.0) |
| Specific dependencies   | monai 0.9.0                                |

To install requirements:

```setup
pip install -r requirements.txt
```



## Dataset

-  [NeurIPS 2022 Cell Segmentation Challenge official dataset](https://neurips22-cellseg.grand-challenge.org/dataset/). 



## Preprocessing

- Preprocess for images and labels:

  ```
  python data_preprocess.py -i <path_to_original_data> -o <path_to_processed_data>
  ```

- Preprocess for distance transform:

  ```
  python dist_transform_preprocess.py --data_path <path_to_processed_data> --data_ori_path <path_to_original_data> --save_path <path_to_distance_transform>
  ```



## Training

To train the models in the paper, run this command respectively:

- resnet50 backbone:

  ```
  python train.py --data_path <path_to_processed_data> --weight_path <path_to_distance_transform> --model_name "res50_unetr"
  ```

- resnet50wide backbone:

  ```
  python train.py --data_path <path_to_processed_data> --weight_path <path_to_distance_transform> --model_name "res50wt_unetr"
  ```

Then we get two models saved in "./work_dir/res50_unetr" and "./work_dir/res50wt_unetr" respectively.

To fine-tune the model on a customized dataset, please preprocess the dataset with the above command first , then run this command:

- resnet50 backbone:

  ```
  python train.py --data_path <path_to_processed_data> --weight_path <path_to_distance_transform> --model_name "res50_unetr" --pre_trained_model_path ./work_dir/res50_unetr
  ```

- resnet50wide backbone:

  ```
  python train.py --data_path <path_to_processed_data> --weight_path <path_to_distance_transform> --model_name "res50wt_unetr" --pre_trained_model_path ./work_dir/res50wt_unetr
  ```



## Trained Models

You can download models trained on the above dataset with the above code here:

- [res50_unetr](https://github.com/Woof6/neurips22-cellseg_saltfish/releases/tag/pth1) 
- [res50wt_unetr](https://github.com/Woof6/neurips22-cellseg_saltfish/releases/tag/pth)



## Inference

If you have gotten two trained models saved  in "./work_dir/res50_unetr" and "./work_dir/res50wt_unetr" respectively, you can infer the testing cases by running this command:

```python
python predict.py -i <path_to_data> -o <path_to_inference_results>
```

Docker  container link:[Docker Hub](https://hub.docker.com/repository/docker/woof4/saltfish/general)

```
docker container run --gpus "device=0" -m 28G --name saltfish --rm -v $PWD/CellSeg_Test/:/workspace/inputs/ -v $PWD/saltfish_seg/:/workspace/outputs/ saltfish:latest /bin/bash -c "sh predict.sh"
```

[Google Drive](https://drive.google.com/drive/folders/1ZL3EOpp4XkkagBgQkPV2FyH1zKSFYb95?usp=sharing) jupyter notebook for inference on colab



## Evaluation

To compute the evaluation metrics, run:

```eval
python eval_f1.py --seg_path <path_to_inference_results> --gt_path <path_to_ground_truth>
```



## Results

Our method achieves the following performance on tunning set of [NeurIPS 2022 Cell Segmentation Challenge](https://neurips22-cellseg.grand-challenge.org/):

- Performance evaluation

  | Method | Res50  | Res50wt | Complete model |
  | ------ | ------ | ------- | -------------- |
  | F1     | 0.8162 | 0.8211  | 0.8250         |

- Efficiency evaluation

  | Resolution   | Docker inference time(s) | GPU Memory(MiB) |
  | ------------ | ------------------------ | --------------- |
  | 640*×*480    | 9.67                     | 2884            |
  | 1024*×*1024  | 10.35                    | 3024            |
  | 3000*×*3000  | 18.56                    | 5886            |
  | 8415*×*10496 | 74.16                    | 8592            |




