# DeepHealth

## AWE Dataset

The AWE dataset contains 7 files, "Normal", "Fault-1", "Fault-2", "Fault-3", "Fault-4", "Fault-5" respectively represent six classes of health conditions, and "Label" represents the corresponding labels for each health condition. Each data file is composed of 1000×4000 sampling points of vibration signals, where 1000 denote the data acquisition duration (i.e., 1000 seconds) and 4000 denote the sampling frequency.

Notably, The data are collected from real industrial facilities via destructive experiments, and each fault condition are formed by artificial destruction. Here, "Normal" indicates that the equipment is operated without any faults. "Fault-M" indicates that M gaskets are added to the upper bearing to make a M×3mm concentricity deviation between the upper bearing and lower bearing such that make the fault condition manually. Detailed description of the destructive experiment and data collection process can be found in the original paper.

## Code Description

There are four python files under the Code document, namely, "DH-1", "Data_DH_1", "DH-2", and "Data_DH_2". "DH-1" and "DH-2" are corresponding to the model structure of the DeepHealth paper, and "Data_DH_1" and "Data_DH_2" are the data preprocessing methods for the corresponding models.

Experiment environment: 
   IDE: PyCharm 
   Language: Python 3.6
   Tensorflow: 1.14.0
   

Lastly, we really appreciate the joint efforts of our partner company (including the engineers of the factory) who provide sufficient time and fire-new equipment for the destructive experiment. 
