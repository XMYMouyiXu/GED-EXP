# Approximating the Graph Edit Distance with Compact Neighborhood Representations
Source code for the paper: Bause, F., Permann, C. & Kriege, N.M. Approximating the Graph Edit Distance with Compact Neighborhood Representations. ECML PKDD, Lecture Notes in Computer
Science 14945 (2024). 

## Usage
The algorithms contained in this package can be used to replicate the results in the paper. 
The results of the experiments (with the *MUTAG* dataset) can be replicated by running *app/GEDExperiments*.

For the results with the other datasets, download datasets *MSRC_9*, *NCI1*, *Letter-med*, and *PTC_FM* and adjust the variables *allDatasets*, *datasets* and *dss* accordingly (see code comments).

After running the experiments, the results can be found in results (for more details on the individual files and format see the *results/readme.txt*).

A thorough description of all the methods can be found in our paper and the corresponding references.  

    
## Datasets
Only datasets *MUTAG* and *proteincom_sampled* are in this repo as an example.
Datasets in the required format are available from the website [TUDatasets: A collection of benchmark datasets for graph classification and regression.](https://chrsmrrs.github.io/datasets/).
The full dataset *Protein_Complexes* can be found [here](https://github.com/BiancaStoecker/complex-similarity-evaluation/tree/master/simulated_complexes/true_constraints).  

## Terms and conditions
When using our code please cite our paper [Approximating the Graph Edit Distance with Compact Neighborhood Representations](https://link.springer.com/chapter/10.1007/978-3-031-70362-1_18):
Bause, F., Permann, C. & Kriege, N.M. Approximating the Graph Edit Distance with Compact Neighborhood Representations. ECML PKDD (2024). 


## Contact information
If you have any questions, please contact [Franka Bause](https://dm.cs.univie.ac.at/team/person/112939/).
