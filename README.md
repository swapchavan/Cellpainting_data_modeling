# Predicting Chemical-induced Liver Toxicity Using High Content Imaging Phenotypes and Chemical Descriptors: A Random Forest Approach
_Swapnil Chavan, Nikolai Scherbak, Magnus Engwall, and Dirk Repsilber_

_Chem. Res. Toxicol. August, 2020
https://doi.org/10.1021/acs.chemrestox.9b00459_

**Abstract**

Hepatotoxicity is a major reason for the withdrawal or discontinuation of drugs from clinical trials. Thus better tools are needed to filter potential hepatotoxic drugs early in drug discovery. Our study demonstrates utilization of HCI phenotypes, chemical descriptors, and both combined (hybrid) descriptor to construct random forest classifiers (RFC) for the prediction of hepatotoxicity. HCI data published by Broad Institute, provided HCI phenotypes for about 30000 samples in multiple replicates. Phenotypes belonging to 346 chemicals which were tested in up to eight replicates, were chosen as a basis for our analysis. We then constructed individual RFC models for HCI phenotypes, chemical descriptors, and hybrid (chemical and HCI) descriptors. The model that was constructed using selective hybrid descriptors showed high predictive performance with 5-fold cross-validation (CV) balanced accuracy (BA) at 0.71, whereas within given applicability domain (AD), independent test set and external test set predictions BA were equal to 0.61 and 0.60, respectively. The model constructed using chemical descriptors showed a similar predictive performance with 5-fold CV BA equal to 0.66, test set prediction BA within AD equal to 0.56, and external test set prediction BA within AD equal to 0.50. In conclusion, the hybrid and chemical descriptor-based models presented here, should be worth being considered as a new tool for filtering hepatotoxic molecules during compound prioritization in drug discovery.

**Code**

1) Input data : avg_variables_346_inst_table.csv

2) RF model building, Y-randomization, McNemar test : RF_model_script.r

3) Applicability domain :  AD_KNN_IS_HCI_model.ipynb, AD_KNN_IS_Hybrid.ipynb, AD_KNN_IS_chemical_model.ipynb
