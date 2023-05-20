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
cd Gerogia_GIA/
pip install -r requirements.txt
```
It should be noted that GEROGIA was written and tested with Python 3.9.7 and Jupyter Notebook 5.4.0.

## File Descriptions
* **[GEORGIA_Tutorial.ipynb](./GEORGIA_Tutorial.ipynb)** - A notebook contains the tutorial to use GEORGIA to emulate RSL. It includes information about loading and preparing data for GEROGIA, visulising spatial temporal emulation error (Figures 2, 3 and S1 in the paper). Two illustrative examples of using GEORGIA to investigate palaeo sea-level problems are also provided. 
* **data/healpix16_coord.csv** - A csv file contains coordinate for 16-degree Hierarchical Equal Area isoLatitude Pixelation (Healpix) of a sphere.
* **data/heal16_input_mean.npy** - Mean ice history used to normalise input data ($\mu_{I}$).
* **data/heal16_input_std.npy** - Standard deviation of ice history that is used to normalise input data ($\sigma_{I}$).
* **data/heal16_output_mean.npy** - Mean relative sea-level change hisotry used to normalise output data ($\mu_{RSL}$).
* **data/heal16_output_std.npy** - Standard deviation of relative sea-level change history that is used to normalise output data ($\sigma_{RSL}$).
* **data/healpix_test_input_norm.npy** - 150-member normalised input data from the testing set.
* **data/healpix_test_output_norm.npy** - 150-member normalised output data from the testing set.
* **data/ice_0_healpix16.npy** - Modern ice thickness expressed in 16-degree Healpix.
* **data/modern_topo_healpix16.npy** - Modern topography data expressed in 16-degree Healpix.

## Co-authors
* **Pippa Whitehouse**
* **Andrew Valentine**
* **Sarah Woodroffe**

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details

## Acknowledgements
We thank Roger Creel and Lev Tarasov for their constructive comments that significantly improve this paper. The authors thank Glenn A. Milne for providing the code used to perform the GIA modelling, Parviz Ajourlou and Ryan Love for useful discussion. Y.L. was supported by a China Scholarship Council - Durham University joint scholarship. A.P.V acknowledges support from the Australian Research Council under grant DP200100053. This work made use of the facilities of the N8 Centre of Excellence in Computationally Intensive Research (N8 CIR) provided and funded by the N8 research partnership and EPSRC (Grant No. EP/T022167/1). The Centre is co-ordinated by the Universities of Durham, Manchester and York. The authors acknowledge the collaborative research opportunities created by PALSEA, a working group of the International Union for Quaternary Sciences (INQUA) and Past Global Changes (PAGES), which in turn received support from the Swiss Academy of Sciences and the Chinese Academy of Sciences. 