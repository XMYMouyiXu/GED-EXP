# Graph Edit Distance Estimation: A New Heuristic and A Holistic Evaluation of Learning-based Methods
## Introduction
Graph edit distance (GED) is an important metric for measuring the distance or similarity between two graphs. It is defined as the minimum number of edit operations required to transform one graph into another. Computing the exact GED between two graphs is an NP-hard problem. With the success of deep learning across various application domains, graph neural networks have also been recently utilized to predict the GED between graphs. However, the existing studies on learning-based methods have two significant limitations. (1) The development of deep learning models for GED prediction has been explored in various research fields (e.g., databases, machine learning, information retrieval, and computer vision), yet cross-field evaluations have been quite limited. (2) More importantly, all these advancements have relied on a very simple combinatorial heuristic baseline, with their models shown to outperform it. In this paper, we aim to bridge this knowledge gap. We first conduct a holistic review of the existing learning-based methods, categorizing them into non-interpretable and interpretable GED prediction approaches, while highlighting their overarching design principles and relationships among these models. Secondly, we propose a simple yet effective combinatorial heuristic algorithm App-BMao for GED estimation, inspired by an existing exact GED computation algorithm. Our algorithm App-BMao provides interpretable GED estimation, which is always no smaller than the exact GED value. Extensive empirical evaluations on three widely used datasets show that our new heuristic algorithm App-BMao outperforms all existing learning-based approaches for both interpretable and non-interpretable GED prediction.

## Dataset
The datasets can be downloaded [here](https://drive.google.com/file/d/1Dwtki6O6T6KgfIXXdcIiqisCxl_nEJFD/view?usp=sharing). It includes datasets for our APP-BMao, datasets for GEDGNN, datasets from pyg and datasets for SDTED.
After downloaded, extract it in the main directory.

In `simGNN`, `GENN` and `EGSC`, when parsing the dataset argument, for IMDBMulti dataset, `new_IMDB` refers to IMDB dataset with our generated ged values. `IMDBMulti` refers to the IMDB dataset from PyG.

In GREED, the dataset names are `GED_AIDS700nef`, `GED_LINUX` and `GED_IMDBMulti`.

## How to run
Please refer to the README file in each model directory.