## Predicting Ground-Based Air Temperature

With the spirit of reproducible research, this repository contains all the codes required to produce the results in the manuscript:

> H. Wang, M. S. Pathan, Y. H. Lee, and S. Dev, Predicting Ground-Based Air Temperature, *under review*.


### Executive summary
Climate change is a phenomenon that can affect many departments including health, development, and planning. In this reseach, we perform forecasting of ground-based air temperature. We evaluate the performance of our approach by computing the RMSE values. 


### Code
All codes are writeen in `python3`.
+ `forecasting-example1.py`: Computes the forecasting performance and plots the result for sample example 1.
+ `forecasting-example2.py`: Computes the forecasting performance and plots the result for sample example 2.
+ `benchmarking.py`: Computes the performance of our proposed method, along with other benchmarking methods. 


### Results
The results are stored in the folder `./results/`.
+ `prediction-index20.PDF`: Plot of forecasting example 1.
+ `prediction-index70.PDF`: Plot of forecasting example 2.
+ `comparison.txt`: Text file that details the performance of the various benchmarking methods.


### Datasets
The dataset used in our case study can be found in the folder `./data/`.
