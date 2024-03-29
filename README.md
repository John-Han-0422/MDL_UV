# Multimodal deep learning for national-wide mapping of urban vegetation evolution in China with 30-years Landsat archive
This project contains two deep learning methods for urban vegetation estimation: a multimodal deep learning (MDL) and a deep neural network with progressively growing neurons (pDNN).
# Our environments
python 3.8  
torch 2.0.1  
tensorboardX 2.6  
scikit-learn 1.2.2  
numpy 1.24.1  
pandas 2.0.2
# Input
Five categories of features are used for model training:
1. reflective bands: blue, green, red, NIR, and SWIR1, and SWIR2  
2. vegetation indices: NDVI and EVI  
3. texture variables based on reflective bands of red, green, blue, and NIR: variance, contrast, entropy, angular second moment, inverse difference moment, and correlation  
4. topographical variables: regional elevation and  slop  
5. time labels  
Texture and topographical variables were standardized before input into the models.
