# DeepHealth

Welcome to DeepHealth, a learning-based instant predictive maintenance framework, created for open-source dataset in a paper accepted by IEEE Transactions on Industrial Informatics.

## AWE dataset

The AWE dataset contains 7 files, "Normal", "Fault-1", "Fault-2", "Fault-3", "Fault-4", "Fault-5" respectively represent six classes of health conditions, and "Label" represents the corresponding labels for each health condition. Each data file is composed of 1000×4000 sampling points of vibration signals, where 1000 denote the data acquisition duration (i.e., 1000 seconds) and 4000 denote the sampling frequency.

Notably, The data are collected from real industrial facilities via destructive experiments, and each fault condition are formed by artificial destruction. Here, "Normal" indicates that the equipment is operated without any faults. "Fault-M" indicates that M gaskets are added to the upper bearing to make a M×3mm concentricity deviation between the upper bearing and lower bearing such that make the fault condition manually. Detailed description of the destructive experiment and data collection process can be found in the original paper.

## Code description

There are four python files under the Code document, namely, "DH-1", "Data_DH_1", "DH-2", and "Data_DH_2". "DH-1" and "DH-2" are corresponding to the model structure of the original paper, and "Data_DH_1" and "Data_DH_2" are the data preprocessing methods for the corresponding models. As described in the original paper, taking the predicted sensor sequences of "DH-2" as the newly samples, and then feed these samples into "DH-1", the corresponding health conditions of future moments can be identified rapidly, thereby completing the future health condition prediction and achieving the instant intelligent predictive maintenance functionalities.

Experiment environment: <br>
      &#160; &#160; &#160; &#160;IDE: PyCharm <br>
      &#160; &#160; &#160; &#160;Language: Python 3.6 <br>
      &#160; &#160; &#160; &#160;Tensorflow: 1.14.0 <br>

## Paper citation

If you use the AWE dataset or found it helpful, please cite the following paper. We are really appreciate your citation. <br>
@article{zhang2020deephealth,<br>
  &#160; &#160; &#160; &#160;title={{DeepHealth}: A self-attention based method for instant intelligent predictive maintenance in industrial {Internet} of things},<br>
  &#160; &#160; &#160; &#160;author={Zhang, Weiting and Yang, Dong and Xu, Youzhi and Huang, Xuefeng and Zhang, Jun and Gidlund, Mikael},<br>
  &#160; &#160; &#160; &#160;journal={IEEE Transactions on Industrial Informatics},<br>
  &#160; &#160; &#160; &#160;year={Early Access, Oct. 2020, doi: 10.1109/TII.2020.3029551},<br>
  &#160; &#160; &#160; &#160;publisher={IEEE}<br>
}<br>

<br>
Lastly, we really appreciate the joint efforts of our partner company (including the engineers of the factory) who provide sufficient time and fire-new equipment for the destructive experiment. 
