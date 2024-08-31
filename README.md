# Functional Connectivity Classifier 

This project aims to classify subjects based on functional connectivity derived from fMRI data using a deep learning approach. Specifically, the project explores the classification of subjects into different gender (male vs. female) by analyzing fMRI data during movie watching. The project leverages functional connectivity matrices extracted from fMRI data as input features for a neural network classifier.

## Project Overview

The experiment uses the Developmental fMRI Dataset, which consists of fMRI scans of children (ages 3-13) and young adults (ages 18-39) while they watched movies. This dataset is ideal for demonstrating how machine learning models can be trained to classify participants based on brain connectivity patterns.

## Results

**1. Superior Frontal Gyrus Medial - Caudate**
   - **Roles**: Decision-making, reward processing.
   - **Significance**: Correlation differences may explain how males and females handle emotions and make decisions differently.

**2. Superior Temporal Sulcus - Angular Gyrus**
   - **Roles**: Social cognition, language processing.
   - **Significance**: Variations might reflect how males and females perceive social cues and process language.

**3. Inferior Occipital Gyrus - Superior Parietal Lobule**
   - **Roles**: Visual processing, spatial awareness.
   - **Significance**: Differences may affect visual-spatial abilities, showing how males and females handle visual information.

**4. Superior Parietal Lobule Anterior - Parieto-Occipital Sulcus**
   - **Roles**: Spatial orientation, visual processing.
   - **Significance**: Disparities could reveal gender differences in spatial skills and visual attention.

**5. Superior Frontal Gyrus Medial - Dorsomedial Prefrontal Cortex**
   - **Roles**: Social cognition, self-referential thought.
   - **Significance**: Variations might indicate differences in self-reflection and understanding others’ perspectives between genders.

![image](https://github.com/user-attachments/assets/63b349a3-07d3-479d-b3e2-2ed2e542225a)

### Key Steps in the Experiment:

1. **Data Loading and Preprocessing**:
   - The fMRI data is downloaded from the Nilearn library, which fetches the Developmental fMRI dataset.
   - The data includes functional MRI Nifti images, confounds, and phenotypic information such as age, age group, gender, and handedness.

2. **Feature Extraction**:
   - Functional connectivity matrices are computed for each subject using the `ConnectivityMeasure` from Nilearn. The `MultiNiftiMapsMasker` is utilized to extract time-series data from predefined brain regions (using an atlas) and compute correlations between them to create connectivity matrices.

3. **Data Preparation for Machine Learning**:
   - The connectivity matrices are vectorized and used as input features for the classification model.
   - The target variable (e.g., gender) is encoded, and the data is split into training and testing sets.

4. **Modeling**:
   - A neural network model is defined with multiple hidden layers, batch normalization, and dropout for regularization. The architecture of the network is optimized using Optuna, an optimization framework that tunes hyperparameters like the number of hidden layers, learning rate, and dropout rate.

5. **Training and Evaluation**:
   - The model is trained using the training dataset and evaluated on the testing dataset. Metrics such as log-loss, accuracy, AUC-ROC, and Precision-Recall AUC are computed to assess the model's performance.
   - Early stopping is implemented to prevent overfitting by stopping the training process when no improvement is observed after a certain number of epochs.

6. **Feature Importance Analysis**:
   - The importance of different brain regions (features) is analyzed by perturbing each feature and measuring its impact on the model's loss. This analysis helps identify which brain regions are most critical in differentiating between classes.

### Key Findings:

- **Functional Connectivity Differences**: 
  - The results indicate that specific brain regions and their connections play a significant role in distinguishing between different classes, such as children vs. adults or males vs. females. 
  - For instance, connections involving the **Superior Frontal Gyrus** and **Superior Temporal Sulcus** were found to be among the top features influencing the classification.

- **Neuroscience Insights**:
  - The study highlights the utility of functional connectivity as a biomarker for age and gender classification. 
  - The **Superior Frontal Gyrus** is associated with higher cognitive functions, including working memory and decision-making, which might explain its role in differentiating age groups.
  - The **Superior Temporal Sulcus** is involved in social cognition, such as understanding others' intentions, which might contribute to gender-specific connectivity patterns.

## Requirements

- Python 3.x
- Libraries: `nilearn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `torch`, `optuna`, `scikit-learn`

## How to Run

1. Clone the repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Run the script `main.py` to execute the data processing, model training, and evaluation steps.

## Conclusion

This project demonstrates how deep learning can be applied to neuroimaging data for classifying subjects based on functional connectivity patterns. The insights gained from feature importance analysis contribute to our understanding of how different brain regions are involved in these classifications.
