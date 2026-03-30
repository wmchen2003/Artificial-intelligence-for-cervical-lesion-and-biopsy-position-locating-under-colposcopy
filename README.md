# CLEARIS

**Artificial intelligence integrating diagnostic clinical feature recognition for cervical lesion and biopsy position locating under colposcopy**

\---

## Overview

This repository contains the key Python implementation of the **DIC** framework, developed for **colposcopy-assisted cervical lesion diagnosis and biopsy localization**.

The codebase focuses on:

* Multi-class cervical lesion classification (Normal, LSIL, HSIL, Cancer)
* Lesion region segmentation and precise localization
* Biopsy position recommendation to support targeted biopsy

⚠️ **Note:**
The images used in this study and the basis for their feature selection have been described in detail in the manuscript. This repository is based on the selected features for downstream modeling and analysis.



## Methodological Framework

The Python implementation of SIC includes the following components:

### 1\. Model Development and Optimization

* Deep learning models are constructed for joint classification and segmentation tasks
* Multi-task learning framework integrates lesion classification and region localization
* **5-fold cross-validation** is applied to improve model robustness and generalization

\---

### 2\. Model Evaluation

* Model performance is assessed using:
* Accuracy, precision, recall, and F1-score for classification
* Dice coefficient and IoU for segmentation
* Localization accuracy for biopsy position recommendation
* External validation can be performed on independent colposcopy datasets

\---

### 3\. Structuration and Visualization

* The model outputs six key clinical features associated with cervical lesions, as defined in the study
* Lesion boundaries are delineated using irregular curves to accurately reflect the morphology of the lesion
* Visualization includes:

  * Predicted lesion grade (Normal, LSIL, HSIL, Cancer)
  * Recognized colposcopic features (e.g., acetowhite epithelium, punctuation, mosaicism, etc.)
  * Delineated lesion boundaries overlaid on original colposcopy images
  * Structured result

\---

## Outputs

The pipeline generates:

* Per-image classification results (Normal, LSIL, HSIL, Cancer)
* Recognition of six clinically meaningful colposcopic features
* Irregular-curve delineation of lesion regions
* Composite visual outputs showing:

  * Original colposcopy image
  * Overlaid lesion boundaries
  * Annotated feature information
  * Lesion grade label
* Structured output files suitable for clinical documentation and publication

\---

## Disclaimer

This model is intended for research purposes only. Further validation in multi-center and prospective cohorts is required before clinical application. The lesion delineation and feature recognition results are for reference only and should be interpreted by qualified clinicians.

\---

