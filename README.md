# Functional Connectivity Classifier 

This project aims to classify subjects based on functional connectivity derived from fMRI data using a deep learning approach. Specifically, the project explores the classification of subjects into different gender (male vs. female) by analyzing fMRI data during movie watching. The project leverages functional connectivity matrices extracted from fMRI data as input features for a neural network classifier.

## Project Overview

The experiment uses the Developmental fMRI Dataset, which consists of fMRI scans of children (ages 3-13) and young adults (ages 18-39) while they watched movies. This dataset is ideal for demonstrating how machine learning models can be trained to classify participants based on brain connectivity patterns.

---

### 1. **Superior Temporal Sulcus with Angular Gyrus – Superior Parietal Lobule Anterior**
   **Importance: 0.005392**
   
   - **Superior Temporal Sulcus (STS) with Angular Gyrus**: 
     - **STS** is a key region for **social cognition**, including interpreting **body language**, **facial expressions**, and **theory of mind** (understanding others' mental states). It also plays a role in **language comprehension**.
     - The **Angular Gyrus** integrates different types of sensory information, including **visual, auditory, and spatial information**, and is linked to **language processing** and **social cognition**.
   
   - **Superior Parietal Lobule (Anterior)**:
     - This region is involved in **spatial attention** and **sensory integration**. It helps in **coordinating movements** and understanding the position of one's body in space, important for motor control and planning.

   **Function of the Connection**: 
   This connection between the **STS/Angular Gyrus** and the **Superior Parietal Lobule Anterior** could reflect the brain's integration of **social information with spatial awareness**. During the film, this connection might allow individuals to integrate **social cues** (from observing characters) with an understanding of **spatial relationships** in the scene, indicating possible differences in how males and females interpret these aspects during the film.



### 2. **Superior Parietal Lobule Posterior – Intraparietal Sulcus (Left Hemisphere)**
   **Importance: 0.005281**

   - **Superior Parietal Lobule (Posterior)**: 
     - The posterior part of the **Superior Parietal Lobule** is heavily involved in **visuospatial processing**, including the integration of sensory inputs for **object manipulation** and **attention to objects in space**.
   
   - **Intraparietal Sulcus (Left Hemisphere)**: 
     - The **Intraparietal Sulcus (IPS)** helps coordinate **eye movements**, **spatial attention**, and **hand-eye coordination**. It is critical in **processing numerical and spatial information** and plays a role in understanding quantities and relationships in space.

   **Function of the Connection**: 
   The **Superior Parietal Lobule Posterior** and **Left IPS** connection suggests gender differences in **visuospatial processing** and **attention to objects in space**. These regions work together to allow individuals to allocate **visual attention** and **coordinate actions** in space, possibly suggesting that males and females use different strategies for interpreting spatial scenes or navigating the visual environment during the film.



### 3. **Cerebellum I-V – Precentral Gyrus Medial**
   **Importance: 0.005185**

   - **Cerebellum (Lobules I-V)**: 
     - These lobules of the cerebellum are involved in **fine motor control**, **balance**, and the timing of **voluntary movements**. The cerebellum also contributes to **cognitive functions**, including **timing and coordination** of cognitive processes.
   
   - **Precentral Gyrus (Medial)**: 
     - This region is part of the **primary motor cortex**, specifically controlling the movement of the trunk and limbs. It's essential for **initiating voluntary movements**.

   **Function of the Connection**: 
   This connection likely reflects gender differences in **motor planning** and **bodily coordination**. The **Cerebellum** and **Precentral Gyrus** are both involved in coordinating **movement and motor control**, so this connection might indicate that males and females process **motor responses** differently when reacting to film scenes, such as preparing for action or responding to physical stimuli.



### 4. **Insula Antero-Superior – Precentral Gyrus Medial**
   **Importance: 0.005047**

   - **Insula (Antero-Superior)**: 
     - The anterior insula is crucial for **emotional processing**, **interoception** (the perception of internal bodily states), and **awareness of emotions** like pain or disgust. It is also linked to **empathy** and the **subjective experience of emotions**.
   
   - **Precentral Gyrus (Medial)**: 
     - As part of the **primary motor cortex**, this region is responsible for initiating movements in the body's midline (such as the trunk or upper body).

   **Function of the Connection**: 
   This connection between the **insula** and the **precentral gyrus** may point to **gender-specific differences** in how emotions influence **motor responses**. For example, females might have stronger integration of emotional states (insula) with physical expressions (precentral gyrus), leading to more pronounced emotional reactions in their motor responses, while males might show different patterns of emotional regulation and expression.


### 5. **Intraparietal Sulcus (Right Hemisphere) – Lingual Gyrus**
   **Importance: 0.004649**

   - **Intraparietal Sulcus (IPS, Right Hemisphere)**: 
     - The right IPS is involved in **spatial attention** and **visual processing**, particularly in **3D perception** and the manipulation of objects in space. It plays a role in **tracking objects** and **coordinating eye movements**.
   
   - **Lingual Gyrus**: 
     - This region is part of the **visual cortex** and is responsible for **processing visual information**, particularly **high-level visual processing** such as recognizing **letters, words, shapes, and colors**.

   **Function of the Connection**: 
   This connection suggests that males and females might differ in how they **process visual stimuli** in terms of **attention** and **object recognition**. The **right IPS** and **Lingual Gyrus** working together help with **tracking visual information**, interpreting shapes, and coordinating visual attention, so the differences in this connection could reflect how each gender processes complex visual information differently during the film, potentially affecting how visual scenes are perceived.

![image](https://github.com/user-attachments/assets/8afde7d4-9ef1-4476-94f3-6d899ea8d302)



---
## Key Steps in the Experiment:

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

## Summary of Findings:
- **Social Cognition and Spatial Awareness**: The connection between the **Superior Temporal Sulcus** and **Superior Parietal Lobule** highlights possible differences in how males and females process **social cues** and **spatial information**.
- **Visuospatial and Motor Processing**: Connections involving the **Superior Parietal Lobule**, **Intraparietal Sulcus**, and **Cerebellum** suggest gender-specific strategies for **spatial attention**, **movement coordination**, and **motor planning**.
- **Emotion and Movement Integration**: The **insula** and **precentral gyrus** connection indicates that gender differences in **emotional regulation** might influence **motor activity**, reflecting varying emotional responses to the film's content.
- **Visual Attention and Recognition**: The connection between the **right IPS** and **Lingual Gyrus** hints at gender-specific differences in **visual attention** and **high-level visual processing**, particularly in how visual details are interpreted and acted upon.

## Requirements

- Python 3.x
- Libraries: `nilearn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `torch`, `optuna`, `scikit-learn`

## How to Run

1. Clone the repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Run the script `main.py` to execute the data processing, model training, and evaluation steps.

## Conclusion

This project demonstrates how deep learning can be applied to neuroimaging data for classifying subjects based on functional connectivity patterns. The insights gained from feature importance analysis contribute to our understanding of how different brain regions are involved in these classifications.
