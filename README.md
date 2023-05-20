# GEORGIA: a Graph neural network based EmulatOR for Glacial Isostatic Adjustment

This repository contains the Python code base for Lin et al., : "GEORGIA: a Graph neural network based EmulatOR for Glacial Isostatic Adjustment". 

**Project abstract:**
> Glacial isostatic adjustment (GIA) modelling is not only useful for understanding past relative sea-level change but also for projecting future sea-level change due to ongoing land deformation. However, GIA model predictions are subject to ranges of uncertainties, including poorly-constrained global ice history. An effective way to reduce this uncertainty is to perform data-model comparisons over a large ensemble of possible ice histories, which is often prohibited by the limited computation resources. Here we address this problem by building a statistical GIA emulator that can mimic the behaviour of a physics-based GIA model (assuming a single 1-D Earth rheology) while being computationally cheap to evaluate. Based on deep learning algorithms, our emulator shows 0.54 m mean absolute error on 150 out-of-sample testing data with <0.5 seconds emulation time. Using this emulator, two illustrative applications related to calculate barystatic sea level are provided for use by the sea-level community. 

If you have any questions, comments, or feedback on this work or code, please [contact Yucheng](mailto:yucheng.lin@durham.ac.uk)

## Installation

Because GEORGIA contains some data files that are larger than 25 Mb, it should be downloaded with git lfs package, an open source Git extension for versioning large files. It can be downloaded through [this link](https://git-lfs.com/), and to install:
```
cd ~/Downloads/git-lfs-3.3.0/
sudo ./install.sh
git lfs install
```
Once you installed git lfs, GEORGIA can be downloaded and installed by by:
```
git lfs clone https://github.com/yc-lin-geo/Georgia_GIA.git
cd GEROGIA_GIA

```
