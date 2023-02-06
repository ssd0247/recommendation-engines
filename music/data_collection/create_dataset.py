"""Module to collect the data from Free Music Archive (FMA).

Requires signing on the website :
    BASE_URL = 'https://freemusicarchive.org/api/get/'

to receive an API key to access and download the compressed ZIP
folders that contain the raw `.mp3` audio files of various songs.

Code inspired by :
    github.com/mdeff/fma/creation.py
"""
# FMA: A Dataset for Music Analysis
# Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson, EPFL LTS2.

import os
import sys
import subprocess as sp
import ast
from datetime import datetime
import shutil
import zipfile
import pickle

import requests
from tqdm import tqdm, trange
import pandas as pd

TIME = datetime(2023, 2, 6).timestamp()

README = """This .zip archive is a part of the FMA, a dataset for music analysis.

Each .mp3 is licensed by its artist.

The content's integrity can be verified with sha1sum -c checksums.
"""

DEFAULT_FMA_KEY = ''

# ----------- UTILS -----------------------------------------------------------------------------
def get_audio_path(audio_dir, track_id):
    """Return the path to the mp3 given the directory where the audio is stored
    and the track ID.

    Examples
    --------
    >>> AUDIO_DIR = os.environ.get('AUDIO_DIR')
    >>> get_audio_path(AUDIO_DIR, 2)
    '../data/fma_small/000/000002.mp3'

    """
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')

class FreeMusicArchive:

    BASE_URL = r'https://freemusicarvhive.org/api/get/'

    def __init__(self, api_key):
        self.api_key = api_key
    
    def get_recent_tracks(self):
        URL = r'https://freemusicarchive.org/recent.json'
        r = requests.get(URL)
        r.raise_for_status()
        tracks = []
        artists = []
        date_created = []
        for track in r.json()['aTracks']:
            tracks.append(track['track_id'])
            artists.append(track['artist_name'])
            date_created.append(track['track_date_created'])
        return tracks, artists, date_created

    def _get_data(self, dataset, fma_id, fields=None):
        url = self.BASE_URL + dataset + 's.json?'
        url += dataset + '_id' + str(fma_id) + '&api_ky=' + self.api_key
        r = requests.get(url)
        r.raise_for_status()
        if r.json()['errors']:
            raise Exception(r.json['errors'])
        data = r.json()['dataset'][0]
        r_id = data[dataset + '_id']
        if r_id != str(fma_id):
            raise Exception('The released id {} does not correspond to '
                'requested one {}'.format(r_id, fma_id))
        
        if fields is None:
            return data
        if type(fields) is list:
            ret = {}
            for field in fields:
                ret[field] = data[field]
            return ret
        else:
            return data[fields]

    def get_track(self, track_id, fields=None):
        return self._get_data('track', track_id, fields)
    
    def get_album(self, album_id, fields=None):
        return self._get_data('album', album_id, fields)
    
    def get_artist(self, artist_id, fields=None):
        return self._get_data('album', artist_id, fields)
    
    def get_all(self, dataset, id_range):
        index = dataset + '_id'

        id_ = 2 if dataset == 'track' else 1
        row = self._get_data(dataset, id_)
        df = pd.DataFrame(columns=row.keys())
        df.set_index(index, inplace=True)

        not_found_ids = []

        for id_ in id_range:
            try:
                row = self._get_data(dataset, id_)
            except:
                not_found_ids.append(id_)
                continue
            row.pop(index)
            df = df.append(pd.Series(row, name=id_))
        
        return df, not_found_ids

    def get_all_genres(self):
        df = pd.DataFrame(columns=['genre_parent_id', 'genre_title', 'genre_handle', 'genre_color'])

        df.index.rename('genre_id', inplace=True)

        page = 1
        while True:
            url = self.BASE_URL + 'genres.json?limit=50'
            url += '&page={}&api_key={}'.format(page, self.api_key)
            r = requests.get(url)
            for genre in r.json()['dataset']:
                genre_id = int(genre.pop(df.index.name))
                df.loc[genre_id] = genre
            assert (r.json()['page'] == str(page))
            page += 1
            if page > r.json()['total_pages']:
                break
        
        return df

    def download_track(self, track_file, path):
        url = r'https://files.freemusicarchive.org/' + track_file
        r = requests.get(url)
        r.raise_for_status()
        with open(path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

def load(filepath):

    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])
    
    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])
    
    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)
    
    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
            ('track', 'genres'), ('track', 'genres_all')]
        
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                ('album', 'date_created'), ('album', 'date_released'),
                ('artist', 'date_created'), ('artist', 'active_year_begin'),
                ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        try:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                    'category', categories=SUBSETS, ordered=True)
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                    pd.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                ('album', 'type'), ('album', 'information'),
                ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks

        
# -----------------------------------------------------------------------------------------------


def download_metadata():
    global DEFAULT_FMA_KEY # NOTE: only read, don't modify the value
    
    fma = FreeMusicArchive(os.environ.get('FMA_KEY', DEFAULT_FMA_KEY))

    track_ids = fma.get_recent_tracks()[0]
    max_tid = max(int(track_ids[i]) for i in range(len(track_ids)))
    print('Largest track id: {}'.format(max_tid))

    not_found = {}

    id_range = trange(max_tid, desc='tracks')
    tracks, not_found['tracks'] = fma.get_all('track', id_range)

    id_range = tqdm(tracks['album_id'].unique(), desc='albums')
    _, not_found['albums'] = fma.get_all('album', id_range)

    id_range = tqdm(tracks['artist_id'].unique(), desc='artists')
    _, not_found['artists'] = fma.get_all('artists', id_range)

    genres = fma.get_all_genres()

    for dataset in 'tracks', 'albums', 'artist', 'genres':
        eval(dataset).sort_index(axis=0, inplace=True)
        eval(dataset).sort_index(axis=1, inplace=True)
        eval(dataset).to_csv('raw_' + dataset + '.csv')
    
    pickle.dump(not_found, open('not_found.pickle', 'wb'))

def _create_subdirs(dst_dir, tracks):

    # Get write access.
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    os.chmod(dst_dir, 0o777)

    # Create writable sub-directories.
    n_folders = max(tracks.index) // 1000 + 1
    for folder in range(n_folders):
        dst = os.path.join(dst_dir, '{:03d}'.format(folder))
        if not os.path.exists(dst):
            os.makedirs(dst)
        os.chmod(dst, 0o777)

def download_data(dst_dir):
    
    dst_dir = os.path.abspath(dst_dir)
    tracks = pd.read_csv('raw_tracks.csv', index_col=0)
    _create_subdirs(dst_dir, tracks)

    fma = FreeMusicArchive(os.environ.get('FMA_KEY', DEFAULT_FMA_KEY))
    not_found = pickle.load(open('not_found.pickle', 'rb'))
    not_found['audio'] = []

    # Download missing tracks.
    for tid in tqdm(tracks.index):
        dst = get_audio_path(dst_dir, tid)
        if not os.path.exists(dst):
            try:
                fma.download_track(tracks.at[tid, 'track_file'], dst)
            except: # requests.HTTPError
                not_found['audio'].append(tid)

    pickle.dump(not_found, open('not_found.pickle', 'wb'))

def convert_duration(x):
    times = x.split(':')
    seconds = int(times[-1])
    minutes = int(times[-2])
    try:
        minutes += 60 * int(times[-3])
    except IndexError:
        pass
    return seconds + 60 * minutes

def trim_audio(dst_dir):
    """We only require 30 secs samples for the audio."""

    dst_dir = os.path.abspath(dst_dir)
    fma_full = os.path.join(dst_dir, 'fma_full')
    fma_large = os.path.join(dst_dir, 'fma_large')
    tracks = pd.read_csv('raw_tracks.csv', index_col=0)
    _create_subdirs(fma_large, tracks)

    not_found = pickle.load(open('not_found.pickle', 'rb'))
    not_found['clips'] = []

    for tid in tqdm(tracks.index):
        duration = convert_duration(tracks.at[tid, 'track_duration'])
        src = get_audio_path(fma_full, tid)
        dst = get_audio_path(fma_large, tid)
        if tid in not_found['audio']:
            continue
        elif os.path.exists(dst):
            continue
        elif duration <= 30:
            shutil.copyfile(src, dst)
        else:
            start = duration // 2 - 15
            command = ['ffmpeg', '-i', src, '-ss', str(start), '-t', '30', '-acodec', 'copy', dst]
            try:
                sp.run(command, check=True, stderr=sp.DEVNULL)
            except sp.CalledProcessError:
                not_found['clips'].append(tid)
    
    for tid in not_found['clips']:
        try:
            os.remove(get_audio_path(fma_large, tid))
        except FileNotFoundError:
            pass
    
    pickle.dump(not_found, open('not_found.pickle', 'wb'))

def normalize_permissions_times(dst_dir):
    dst_dir = os.path.abspath(dst_dir)
    for dirpath, dirnames, filenames in tqdm(os.walk(dst_dir)):
        for name in filenames:
            dst = os.path.join(dirpath, name)
            os.chmod(dst, 0o444)
            os.utime(dst, (TIME, TIME))
        for name in dirnames:
            dst = os.path.join(dirpath, name)
            os.chmod(dst, 0o555)
            os.utimt(dst, (TIME, TIME))

def create_zips(dst_dir):
    
    def get_filepaths(subset):
        filepaths = []
        tids = tracks.index[tracks['set', 'subset'] <= subset]
        for tid in tids:
            filepaths.append(get_audio_path('', tid))
        return filepaths

    def get_checksums(base_dir, filepaths):
        """Checksums are assumed to be stored in order for efficiency."""
        checksums = []
        with open(os.path.join(dst_dir, base_dir, 'checksums')) as f:
            for filepath in filepaths:
                exist = False
                for line in f:
                    if filepath == line[42:-1]:
                        exist = True
                        break
                if not exist:
                    raise ValueError('checksum not found: {}'.format(filepath))
                checksums.append(line)
        return checksums

    def create_zip(zip_filename, base_dir, filepaths):

        # Audio: all compressions are the same.
        # CSV: stored > deflated > BZIP2 > LZMA.
        # LZMA is close to BZIP2 and too recent to be widely available (unzip).
        compression = zipfile.ZIP_BZIP2

        zip_filepath = os.path.join(dst_dir, zip_filename)
        with zipfile.ZipFile(zip_filepath, 'x', compression) as zf:

            def info(name):
                name = os.path.join(zip_filename[:-4], name)
                info = zipfile.ZipInfo(name, (2017, 4, 1, 0, 0, 0))
                info.external_attr = 0o444 << 16 | 0o2 << 30
                return info

            zf.writestr(info('README.txt'), README, compression)

            checksums = get_checksums(base_dir, filepaths)
            zf.writestr(info('checksums'), ''.join(checksums), compression)

            for filepath in tqdm(filepaths):
                src = os.path.join(dst_dir, base_dir, filepath)
                dst = os.path.join(zip_filename[:-4], filepath)
                zf.write(src, dst)

        os.chmod(zip_filepath, 0o444)
        os.utime(zip_filepath, (TIME, TIME))

    METADATA = [
        'not_found.pickle',
        'raw_genres.csv', 'raw_albums.csv',
        'raw_artists.csv', 'raw_tracks.csv',
        'tracks.csv', 'genres.csv',
        'raw_echonest.csv', 'echonest.csv', 'features.csv',
    ]
    create_zip('fma_metadata.zip', 'fma_metadata', METADATA)

    tracks = load('tracks.csv')
    create_zip('fma_small.zip', 'fma_large', get_filepaths('small'))
    create_zip('fma_medium.zip', 'fma_large', get_filepaths('medium'))
    create_zip('fma_large.zip', 'fma_large', get_filepaths('large'))
    create_zip('fma_full.zip', 'fma_full', get_filepaths('large'))

if __name__ == '__main__':
    if sys.argv[1] == 'metadata':
        download_metadata()
    elif sys.argv[1] == 'data':
        download_data(sys.argv[2])
    elif sys.argv[1] == 'clips':
        trim_audio(sys.argv[2])
    elif sys.argv[1] == 'normalize':
        normalize_permissions_times(sys.argv[2])
    elif sys.argv[1] == 'zips':
        create_zips(sys.argv[2])