import ftplib
import os
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


class NpyLoader:
    def __init__(self, matrices_path: str, log_file: str = None):
        """
        Class for loading data from WeatherBench
        :param matrices_path: string with .npy matrices folder
        """
        self.min_val = None
        self.max_val = None
        self.log_file = log_file
        self.source_path = matrices_path

    def normalization(self, matrix: np.ndarray):
        self.max_val = np.nanmax(matrix)
        self.min_val = np.nanmin(matrix)
        print(f'Normalize data, max_value = {self.max_val}, min_value = {self.min_val}')
        if self.log_file is not None:
            df = pd.DataFrame()
            df['min'] = [self.min_val]
            df['max'] = [self.max_val]
            df.to_csv(self.log_file, index=False)
        return (matrix - self.min_val) / (self.max_val - self.min_val)

    def reverse_normalization(self, matrix: np.ndarray):
        print(f'Reverse data normalization to max_value = {self.max_val}, min_value = {self.min_val}')
        matrix = matrix * (self.max_val - self.min_val) + self.min_val
        return matrix

    def load_data(self, period: tuple[str, str], step: str, norm: bool = False):
        """
        Function for loading data from WeatherBench
        :param norm: is there need to normalize data to range (0, 1)
        :param period: tuple with dates in format %Y%m%d
        :param step: step of data in format "{int}D", "{int}h"
        :return: time spatial series [T, H, W]
        """
        matrices = []
        period = pd.date_range(period[0], period[1], freq=step)
        req_names = [p.strftime('%Y%m%d_%H.npy') for p in period]
        for file in req_names:
            try:
                print(f'Load date {file}')
                matrix = np.load(f'{self.source_path}/{file}')
            except Exception:
                raise Exception(f'Failed to load {file} as npy. Check out the directory {self.source_path}')
            matrices.append(matrix)

        matrices = np.array(matrices)
        if norm:
            matrices = self.normalization(matrices)
        return period, matrices


class WeatherBenchDownloader:
    def __init__(self, download_folder: str):
        """
        Class for downloading .zip files of WeatherBench storage
        ftp url named in official documentation https://mediatum.ub.tum.de/1524895
        """
        self.url = 'dataserv.ub.tum.de'
        self.login = 'm1524895'
        self.password = 'm1524895'
        self.loaded_files = []
        self.root_folder = download_folder

    @property
    def resolutions(self):
        return ['1.40625deg', '2.8125deg', '5.625deg']

    @property
    def sources(self):
        return ['10m_u_component_of_wind',
                '10m_v_component_of_wind',
                '2m_temperature',
                'geopotential',
                'potential_vorticity',
                'relative_humidity',
                'specific_humidity',
                'temperature',
                'toa_incident_solar_radiation',
                'total_cloud_cover',
                'total_precipitation',
                'u_component_of_wind',
                'v_component_of_wind',
                'vorticity']

    @property
    def target_variables(self):
        return {
            '10m_u_component_of_wind': 'u10',
            '10m_v_component_of_wind': 'v10',
            '2m_temperature': 't2m',
            'geopotential': 'z',
            'potential_vorticity': 'pv',
            'relative_humidity': 'r',
            'specific_humidity': 'q',
            'temperature': 't',
            'toa_incident_solar_radiation': 'tisr',
            'total_cloud_cover': 'tcc',
            'total_precipitation': 'tp',
            'u_component_of_wind': 'u',
            'v_component_of_wind': 'v',
            'vorticity': 'vo'
        }

    def _get_target_var(self, filename: str):
        """
        Function for searching matches in file name and variables dictionary
        :param filename: string with file name
        :return: string with target variable in nc for file
        """
        source = '_'.join(filename.split('_')[:-2])
        return self.target_variables[source]

    def download(self, resolution: str, sources: list, unpack: bool = True):
        """
        Function for downloading selected variables with specified resolution as .zip archives
        Zip archives include spatio-temporal data in .nc format
        Function unpack files if flag 'unpack' is True
        :param unpack: flag for indicating automatic unzip files
        :param resolution: string with resolution of selected data (self.resolutions)
        :param sources: string with name of selected variable (self.sources)
        """
        if resolution not in self.resolutions:
            raise Exception(f'Wrong resolution set, {self.resolutions} are available')
        if not os.path.exists(f'{self.root_folder}/{resolution}'):
            os.makedirs(f'{self.root_folder}/{resolution}')
        # overwrite root folder as working only with one resolution at a time
        self.root_folder = f'{self.root_folder}/{resolution}'
        for source in sources:
            if source not in self.sources:
                raise Exception(f'Wrong source {source}, {self.sources} are available')

            path_to_file = f'{resolution}/{source}'
            filename = f'{source}_{resolution}.zip'
            ftp = ftplib.FTP(self.url)
            ftp.login(self.login, self.password)
            ftp.cwd(path_to_file)
            print(f'Load file {filename}')
            ftp.retrbinary("RETR " + filename, open(self.root_folder + '\\' + filename, 'wb').write)
            ftp.quit()
            print(f'{filename} loaded to {self.root_folder}')
            self.loaded_files.append(f'{self.root_folder}/{filename}')
        if unpack:
            self.unpack()

    def unpack(self):
        """
        Function for unzip downloaded files
        """
        for file in self.loaded_files:
            path_to_unpack = f'{self.root_folder}/netcdf/{Path(file).stem}'
            print(f'Unpack file {file} to {path_to_unpack}')
            if not os.path.exists(path_to_unpack):
                os.makedirs(path_to_unpack)
            with zipfile.ZipFile(file, "r") as zip_ref:
                names = zip_ref.namelist()
                for name in names:
                    success = False
                    for i in range(10):
                        if not success:
                            try:
                                print(f'unpack {name}')
                                zip_ref.extract(name, path_to_unpack)
                                success = True
                            except Exception as e:
                                print(f'Failed unzip {name}, try:{i + 1}')
                                pass
                    if not success:
                        print(f'Failed unzip {name}, please unpack it manually')

    def rewrite_to_npy(self, nc_folder: str = None):
        import netCDF4 as nc
        """
        Function for converting .nc files of benchmark to .npy matrices files
        :param nc_folder: optional parameter if files for rewriting are not located in benchmark root
        """
        if nc_folder is None:
            nc_folder = f'{self.root_folder}/netcdf/'
        for source_folder in os.listdir(nc_folder):
            npy_folder = f'{self.root_folder}/matrices/{source_folder}'
            if not os.path.exists(npy_folder):
                os.makedirs(npy_folder)
            for nc_file in os.listdir(f'{nc_folder}/{source_folder}'):
                target_var = self._get_target_var(nc_file)
                ds = nc.Dataset(f'{nc_folder}/{source_folder}/{nc_file}')
                file_start_date = datetime.strptime(ds.variables['time'].units,
                                                    'hours since %Y-%m-%d %H:%M:%S')  # date of time steps numeration start
                try:
                    var = np.array(ds.variables[target_var])
                except Exception:
                    raise Exception(
                        f'Failed to load variable {target_var}, list of variables in {nc_file}: {ds.variables.keys()}')
                time = np.array(ds.variables['time']).astype(float)
                time = [file_start_date + relativedelta(hours=t) for t in time]
                for i, t in enumerate(time):
                    name = t.strftime('%Y%m%d_%H.npy')

                    # print log each 10 time steps
                    if i % 10 == 0:
                        print(name)

                    # if level in file
                    if len(var.shape) == 4:
                        for l in range(var.shape[1]):
                            level_path = npy_folder + '/' + str(np.array(ds['level'])[l])
                            if not os.path.exists(level_path):
                                os.makedirs(level_path)
                            matrix = var[i, l, :, :]
                            np.save(f'{level_path}/{name}', matrix)

                    # if single level in file
                    elif len(var.shape) == 3:
                        matrix = var[i, :, :]
                        np.save(f'{npy_folder}/{name}', matrix)
                    else:
                        raise Exception(f'Unexpected dims: {var.shape} - len: {len(var.shape)}')
