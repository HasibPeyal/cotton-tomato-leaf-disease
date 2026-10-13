# plant-disease-xai
Explainable AI-based deep learning approach for plant disease classification using CNN and hybrid CNN-SVM models with Grad-CAM, SHAP, and LIME interpretability.

Paper Title: 
Plant Disease Classifier: Detection of Dual-Crop Diseases Using Lightweight 2D CNN Architecture
Official implementation of the paper published in IEEE Access, 2023.

Paper: 10.1109/ACCESS.2023.3320686

A lightweight 2D CNN for classifying 14 classes of tomato and cotton leaf diseases, with Grad-CAM explainability and an Android deployment.

Overview
This project implements a custom lightweight 2D CNN to classify 14 classes — 12 diseased and 2 healthy — across tomato and cotton crops. The model is trained on a merged dataset of PlantVillage (tomato) and Cotton-Leaf-Infection (cotton) images. It outperforms pretrained models (VGG16, VGG19, InceptionV3, MobileNet, MobileNetV2) while using significantly fewer parameters. Grad-CAM is used to explain model decisions, and the trained model is deployed in an Android application (Plant Disease Classifier) via TFLite for real-time smartphone-assisted diagnosis.

Classes (14)
Tomato (10)

Bacterial Spot

Early Blight

Late Blight

Leaf Mold

Septoria Leaf Spot

Spider Mites

Target Spot

Mosaic Virus

Yellow Leaf Curl Virus

Healthy Tomato

Cotton (4)

Bacterial Blight

Curl Virus

Fusarium Wilt

Healthy Cotton

Model Architecture
Layer (Type)	Output Shape	Params
Input Layer	(150, 150, 3)	0
L1 (Conv2D)	(None, 148, 148, 16)	448
max_pooling2d (MaxPooling)	(None, 74, 74, 16)	0
L2 (Conv2D)	(None, 72, 72, 32)	4,640
max_pooling2d (MaxPooling)	(None, 36, 36, 32)	0
L3 (Conv2D)	(None, 34, 34, 64)	18,496
max_pooling2d (MaxPooling)	(None, 17, 17, 64)	0
flatten (Flatten)	(None, 18496)	0
dense1 (Dense)	(None, 512)	9,470,464
batch_normalization	(None, 512)	2,048
dropout	(None, 512)	0
dense (Output)	(None, 14)	7,182
Total params: 9,503,278
Trainable params: 9,502,254
Non-trainable params: 1,024

Dataset
Merged dataset: PlantVillage (tomato) + Cotton-Leaf-Infection (cotton)

Image size: 150 × 150 (proposed CNN); 224 × 224 (transfer learning models)

Classes: 14

Training images: 11,366

Testing images: 1,327

Split ratio: ~9:1

Per-class distribution (training / testing):

Class	Train	Test
Tomato — Bacterial Spot	1000	100
Tomato — Early Blight	1000	100
Tomato — Healthy	1000	100
Tomato — Late Blight	1000	100
Tomato — Leaf Mold	1000	100
Tomato — Septoria Leaf Spot	1000	100
Tomato — Spider Mites	1000	100
Tomato — Target Spot	1000	100
Tomato — Mosaic Virus	1000	100
Tomato — Yellow Leaf Curl Virus	1000	100
Cotton — Bacterial Blight	358	89
Cotton — Curl Virus	335	84
Cotton — Fusarium Wilt	336	84
Cotton — Healthy	341	85
Total	11,366	1,327
Data Augmentation (6 types): rotation (40°), width shift (0.2), height shift (0.2), shear (0.2), zoom (0.2), horizontal flip.
After augmentation: training set expanded 6× to 68,196 images.

Tech Stack
Language: Python 3.7

Deep Learning: TensorFlow, Keras

Data Processing: NumPy

Evaluation: Scikit-learn, mlxtend

Visualization: Matplotlib

Explainability: Grad-CAM

Deployment: TFLite, Android

Training Configuration
Parameter	Value
Learning Rate	0.001
Activation	ReLU
Batch Size	32
Loss Function	Categorical Cross-Entropy
Optimizer	Adam
Epochs	300 (early stopping at 98% val_accuracy)
Results
Comparison with transfer learning models:

Model	Accuracy	Precision	Recall	F1-Score	AUC
VGG16	81.39%	86%	85%	85%	99.23%
VGG19	82.97%	86%	83%	82%	99.12%
InceptionV3	92.31%	93%	92%	92%	99.79%
MobileNet	83.65%	90%	88%	88%	99.63%
MobileNetV2	57.50%	86%	81%	82%	99.18%
Proposed 2D CNN	97.36%	97%	97%	97%	99.93%
Key highlights:

Misclassifications out of 1,327 test images: Proposed model only 51, vs. 98–246 for transfer learning models

Average inference time on Android: 4.84 ms (range 4–7 ms), near real-time

Model size: 108 MB (~3× smaller than VGG16/VGG19, ~4× smaller than InceptionV3)

Training time: 114 min vs. 184–1046 min for transfer learning models

Explainability: Grad-CAM
Grad-CAM (Gradient-weighted Class Activation Mapping) is applied to the last convolutional layer (L3) to visualize which regions of the leaf influenced the model's prediction. A heatmap is generated and superimposed on the original image, highlighting the responsible disease region and supporting transparent, trustworthy AI-based diagnosis.

Android Application
The trained model is converted to TFLite and deployed in an Android app named Plant Disease Classifier.

Detects 14 classes (12 diseased + 2 healthy)

Inference: 4–7 ms per image (avg. 4.84 ms) with GPU acceleration

Tested on Xiaomi Poco F3 (Android 11, Snapdragon 870 5G)

Detection confidence: 0.997–1.0

Usage
1. Clone the repository
bash
git clone https://github.com/HasibPeyal/cotton-tomato-leaf-disease.git
cd cotton-tomato-leaf-disease
2. Install dependencies
bash
pip install -r requirements.txt
3. Run the notebook
Open the .ipynb file in Jupyter Notebook or Kaggle.

4. Train the model
The notebook includes training code with 6-type data augmentation and early stopping.

5. Generate Grad-CAM visualizations
Run the Grad-CAM cells to visualize model attention on test images.

Requirements
text
tensorflow
keras
numpy
scikit-learn
matplotlib
mlxtend
Citation
If you use this code or dataset in your research, please cite:

bibtex
@article{peyal2023plant,
  title={Plant Disease Classifier: Detection of Dual-Crop Diseases Using Lightweight 2D CNN Architecture},
  author={Peyal, Hasibul Islam and Nahiduzzaman, Md. and Pramanik, Md. Abu Hanif and Syfullah, Md. Khalid and Shahriar, Saleh Mohammed and Sultana, Abida and Ahsan, Mominul and Haider, Julfikar and Khandakar, Amith and Chowdhury, Muhammad E. H.},
  journal={IEEE Access},
  volume={11},
  pages={110627--110643},
  year={2023},
  doi={10.1109/ACCESS.2023.3320686}
}
Author
Hasibul Islam Peyal
Assistant Professor, Dept. of CSE
International Standard University
Email: hasibpeyal9@gmail.com

License
This project is licensed under the MIT License — see the LICENSE file for details.

Acknowledgments
Tomato dataset: PlantVillage

Cotton dataset: Cotton-Leaf-Infection

Grad-CAM implementation adapted from Keras documentation

This work was supported by Qatar National Library (QNL).
