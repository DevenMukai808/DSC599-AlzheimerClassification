# Alzheimer Classification Model
Senior Capstone Project for DSC 599

Creating a machine learning model that can binary classify between Demented and Non Demented MRI scans. Also, using a Grad-CAM (Gradient-weighted Class Activation Mapping, generate a heatmap to display regions with the highest activation and strongest influence on the model.

To download the binary classification dataset used in this project, run these lines of code in Google Colab (also found in the notebook file):
!pip install -q --upgrade gdown
!gdown --folder https://drive.google.com/drive/folders/16JAUFq5YJI677ZL2xwEXwHLa_N8W8AUI -O /content/data


Dataset Reference:
Dataset Link: https://www.kaggle.com/datasets/ninadaithal/imagesoasis
Acknowledgement: “Data were provided 1-12 by OASIS-1: Cross-Sectional: Principal Investigators: D. Marcus, R, Buckner, J, Csernansky J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382”
