from BenchLoader import WeatherBenchDownloader

path_to_save = 'D:/WeatherBench_test'

w = WeatherBenchDownloader(download_folder=path_to_save)
w.download(resolution='5.625deg', sources=['2m_temperature'])
w.rewrite_to_npy(f'{path_to_save}/5.625deg/netcdf')