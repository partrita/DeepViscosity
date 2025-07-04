# DeepViscosity
DeepViscosity는 고농도 단일클론항체 점도 등급(낮음 <= 20 cP, 높음 > 20 cP)을 예측하기 위해 개발된 앙상블 딥러닝 ANN 모델입니다. 이 모델은 DeepSP 대리 모델에서 얻은 30개의 공간 속성(설명자)을 학습 특징으로 활용했습니다. 229개의 mAb를 기반으로 학습되었습니다.

# DeepViscosity를 사용하여 낮거나 높은 등급의 점도를 예측하는 방법

## 옵션 1 - Google Colab 노트북
- DeepViscosity_input.csv 형식에 따라 입력 파일을 준비합니다.
- 노트북 파일 DeepViscosity_predictor.ipynb를 실행합니다.
- 입력된 시퀀스에 대한 DeepViscosity 등급(및 DeepSP 공간 속성)이 채워지고 csv 파일(DeepViscosity_classes.csv)에 저장됩니다.

## 옵션 2 - Linux 환경
- 환경을 만들고 필요한 패키지를 설치합니다.
	- conda create -n deepViscosity python=3.9.13
	- source activate deepViscosity
	- conda install -c bioconda anarci
	- pip install keras==2.11.0 tensorflow-cpu==2.11.0 scikit-learn==1.0.2 pandas numpy==1.26.4 joblib dill

- DeepViscosity_input.csv 형식에 따라 입력 파일을 준비합니다.
- 파이썬 파일 deepviscosity_predictor.py를 실행합니다.
- 입력된 시퀀스에 대한 DeepViscosity 등급(및 DeepSP 공간 속성)이 확보되어 csv 파일(DeepViscosity_classes.csv)에 저장됩니다.

# 인용
Lateefat A. Kalejaye, Jia-Min Chu, I-En Wu, Bismark Amofah, Amber Lee, Mark Hutchinson, Chacko Chakiath, Andrew Dippel, Gilad Kaplan, Melissa Damschroder, Valentin Stanev, Maryam Pouryahya, Mehdi Boroumand, Jenna Caldwell, Alison Hinton, Madison Kreitz, Mitali Shah, Austin Gallegos, Neil Mody and Pin-Kuang Lai (2025). Accelerating high-concentration monoclonal antibody development with large-scale viscosity data and ensemble deep learning. mAbs, 17(1). [https://doi.org/10.1080/19420862.2025.2483944](https://www.tandfonline.com/doi/full/10.1080/19420862.2025.2483944)
