
## WeatherBench loading tool

### Intro
___
Spatio-temporal data about weather conditions has complex nature - it includes _stochastic component, seasonality, trend_ 
and all these components differs in dependence of spatial distribution. 
That is why prediction task for such data is non-trivial. 

Also, spatio-temporal series are often represented as a video sequence and combine the task of predicting natural data with 
the task of predicting video for number of further frames.

To evaluate machine learning models for predictive tasks on natural data, the WeatherBench benchmark was created.
In [publication](https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2020MS002203) authors explains principles 
of data collection and presents quality metrics for baseline machine learning models. Data consists of atmosphere 
parameters with three types of grid resolution  - 1.40625, 2.8125, 5.625 degrees. It is reprojected and interpolated variables, 
downloaded from ERA5 reanalysis archive. Instructions and description for raw data authors have on their 
[GitHub page](https://github.com/pangeo-data/WeatherBench).   

Source WeatherBench storage as **http** is available via [link](https://dataserv.ub.tum.de/s/m1524895?path=%2F), as **ftp:** ftp://m1524895:m1524895@dataserv.ub.tum.de/.


### About loader
___
This package include tools for **WeatherBench** benchmark files loading and processing for video prediction (spatio-temporal prediction) task.
Original benchmark is presented as .zip archives with number of files in NetCDF4 format. It has metadata about projection and georeference 
about existing time steps for each matrix in file. This format is widespread for climatological data. But to 
work with deep learning methods (compatibility with PyTorch for example) that is **not** suitable and **not** easy to understand.

To make data preparation for prediction task more clear and easy to use this package was implemented. It automatically downloads files of 
benchmark and overwrites it into easy to read numpy matrices. 

> To see what the data is in NetCDF format [Panoply](https://www.giss.nasa.gov/tools/panoply/download/) util can be used.
> It allows to watch through .nc file structure, see available variables and make simple georeferenced visualizations of data.

For matrices dynamical convertion into dataset (features-target) [TorchCNNBuilder](https://github.com/ChrisLisbon/TorchCNNBuilder)
functions was used. 
In original paper authors use specific quality metrics such as **Weighted Root Mean Squared Error (WRMSE)** and 
**Weighted Anomaly Correlation Coefficient (WACC)**. Both are weighted by latitude - the error weight is maximized 
at the equator and zero at polar latitudes. This is due to the peculiarities of the projection. Loader has its [implementations](Metrics.py) as PyTorch module for loss calculation
as in function over numpy arrays for quality assessment. 


### How to use - download files
___

Description of available sources and short names of variables in NetCDF files are presented below (from [paper](https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2020MS002203)):
![SegmentLocal](media/var_table.png "segment")

[```BenchLoader.py```](BenchLoader.py) contains class ```WeatherBenchDownloader``` for loading benchmark files via ftp server, and its loading into memory as numpy multidimensional array with time marked in file name.

Name of each variable and its resolution should be specified directly to download files because of its large size:


    from BenchLoader import WeatherBenchDownloader

    loader = WeatherBenchDownloader(download_folder='files_path').download(resolution='5.625deg', sources=['temperature'])

Unpacking .zip files can be produced automatically (if ```unpack=True``` in ```WeatherBenchDownloader.download``` method) or manually if there are a need to work with filed separately.

Rewriting unpacked .nc to .npy matrices can be done as follows:

    loader.rewrite_to_npy()

If .nc files was unpacked manually and the class is just initialized folder with netCDF files should be specified: 
    
    loader = WeatherBenchDownloader(download_folder='files_path')
    loader.rewrite_to_npy(nc_folder='path_to_nc_files')

In this case ```matrices``` directory will be created in ```download_folder```. There will be located matrices for each time step in format ```'%Y%m%d_%H'```. 
If variable include multiple pressure levels, in folder ```matrices``` will be located subdirectories marked as level value. 
Inside each subdirectory matrices for time steps can be found.
    
### How to use - dataset initialization
___

After rewriting files as numpy matrices with time step mark we can call them by required date and form dataset for deep models
training and inference. Class [```NpyLoader```](BenchLoader.py) helps to load spatio-temporal series for selected period:

```
from BenchLoader import NpyLoader

matrices_path = f'D:/WeatherBench/5.625deg/matrices/vorticity_5.625deg/500'
loader = NpyLoader(matrices_path)
dates, data = loader.load_data(('20100101', '20151231'), step='7D', norm=False)
```
Time step frequency can be specified as ```'step'``` parameter in [format](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).
In benchmark data presented with hourly frequency, 24 steps (matrices) per day.  Flag ```'norm'``` regulate min/max normalization of final 
spatio-temporal series. Parameters of normalization are displayed in logs and saved in class body. 

___
Spatio-temporal series is a time series in which each time step presented as number of georeferenced values - matrix. For dataset preparation in
_features - target_ format prehistory size and forecast horizon should be specified. Prehistory size determines the number of steps _K_ (taking into account the time frequency) 
which are used as features to predict _N_ time steps ahead - forecast horizon. Sliding window steps through spatio-temporal series and form dataset from all available elements. 

![SegmentLocal](media/dataset_form.png "segment")

This algorithm implemented in a similar way to the lagged transformation for time series, we import it from 
[TorchCNNBuilder](https://github.com/ChrisLisbon/TorchCNNBuilder/blob/main/torchcnnbuilder/preprocess/time_series.py):
```
from torchcnnbuilder.preprocess.time_series import multi_output_tensor

forecast_len = 52
prehistory_len = 104

dataset = multi_output_tensor(data, forecast_len, prehistory_len)
```

Output dataset is an exemplar of ```torch.Dataset``` class and can be used for models based on PyTorch. 




