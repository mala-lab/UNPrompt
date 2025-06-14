Code for "Zero-shot Generalist Graph Anomaly Detection with Unified Neighborhood Prompts"([UMPrompt](https://arxiv.org/pdf/2410.14886)) ([Supplementary](https://github.com/mala-lab/UNPrompt/blob/main/UNPrompt_Supplementary.pdf))

## Get Started
To run the code, the following packages are required to be installed:

-python==3.8.19

-torch==1.13.1

-dgl==1.0.1+cu117

## Datasets
Disney and Weibo are in the dataset folder. The other datasets can be found at https://drive.google.com/drive/folders/1qcDBcVdcfAr_q5VOXBYagtnhA_r3Mm3Z?usp=drive_link

## Train
To get the results with Facebook as the training graph, just run the following command:

     sh run.sh

## Citation
Please acknowledge our work via the following bibliography if you find our work/code useful:
```bibtex
@article{niu2024zero,
  title={Zero-shot Generalist Graph Anomaly Detection with Unified Neighborhood Prompts},
  author={Niu, Chaoxi and Qiao, Hezhe and Chen, Changlu and Chen, Ling and Pang, Guansong},
  journal={arXiv preprint arXiv:2410.14886},
  year={2024}
}
