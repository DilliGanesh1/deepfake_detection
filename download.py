#!/usr/bin/env python
""" Downloads FaceForensics++ and Deep Fake Detection public data release
Example usage:
    python download_ff.py /path/to/output -d all -c raw -t videos
"""
import argparse
import os
import urllib.request
import tempfile
import time
import sys
import json
import ssl
from tqdm import tqdm
from os.path import join

# URLs and filenames
FILELIST_URL = 'misc/filelist.json'
DEEPFEAKES_DETECTION_URL = 'misc/deepfake_detection_filenames.json'
DEEPFAKES_MODEL_NAMES = ['decoder_A.h5', 'decoder_B.h5', 'encoder.h5']

# Parameters
DATASETS = {
    'original_youtube_videos': 'misc/downloaded_youtube_videos.zip',
    'original_youtube_videos_info': 'misc/downloaded_youtube_videos_info.zip',
    'original': 'original_sequences/youtube',
    'DeepFakeDetection_original': 'original_sequences/actors',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceShifter': 'manipulated_sequences/FaceShifter',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
}
ALL_DATASETS = ['original', 'DeepFakeDetection_original', 'Deepfakes',
                'DeepFakeDetection', 'Face2Face', 'FaceShifter', 'FaceSwap',
                'NeuralTextures']
COMPRESSION = ['raw', 'c23', 'c40']
TYPE = ['videos', 'masks', 'models']
SERVERS = ['EU', 'EU2', 'CA']

def parse_args():
    parser = argparse.ArgumentParser(
        description='Downloads FaceForensics v2 public data release.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('output_path', type=str, help='Output directory.')
    parser.add_argument('-d', '--dataset', type=str, default='all',
                        help='Which dataset to download',
                        choices=list(DATASETS.keys()) + ['all'])
    parser.add_argument('-c', '--compression', type=str, default='raw',
                        help='Compression degree of videos',
                        choices=COMPRESSION)
    parser.add_argument('-t', '--type', type=str, default='videos',
                        help='File type to download',
                        choices=TYPE)
    parser.add_argument('-n', '--num_videos', type=int, default=None,
                        help='Number of videos to download')
    parser.add_argument('--server', type=str, default='EU',
                        help='Server to download from',
                        choices=SERVERS)
    return parser.parse_args()

def get_server_url(server):
    server_urls = {
        'EU': 'http://canis.vc.in.tum.de:8100/',
        'EU2': 'http://kaldir.vc.in.tum.de/faceforensics/',
        'CA': 'http://falas.cmpt.sfu.ca:8100/'
    }
    return server_urls.get(server, server_urls['EU'])

def download_files(filenames, base_url, output_path, report_progress=True):
    os.makedirs(output_path, exist_ok=True)
    if report_progress:
        filenames = tqdm(filenames)
    for filename in filenames:
        try:
            download_file(base_url + filename, join(output_path, filename))
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

def reporthook(count, block_size, total_size):
    if count == 0:
        reporthook.start_time = time.time()
        return
    duration = time.time() - reporthook.start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\rProgress: {percent}%, {progress_size/(1024*1024)} MB, {speed} KB/s, {duration:.0f} seconds")
    sys.stdout.flush()
reporthook.start_time = 0

def download_file(url, out_file, report_progress=False):
    out_dir = os.path.dirname(out_file)
    if not os.path.isfile(out_file):
        with tempfile.NamedTemporaryFile(delete=False, dir=out_dir) as tmp_file:
            try:
                # Bypass SSL certificate verification
                context = ssl._create_unverified_context()
                
                if report_progress:
                    urllib.request.urlretrieve(url, tmp_file.name, reporthook=reporthook)
                else:
                    urllib.request.urlretrieve(url, tmp_file.name, context=context)
                
                os.rename(tmp_file.name, out_file)
            except Exception as e:
                os.unlink(tmp_file.name)
                raise RuntimeError(f"Download failed: {e}")
    else:
        print(f'Skipping existing file: {out_file}')

def main():
    args = parse_args()

    # TOS Confirmation
    print('Confirm FaceForensics Terms of Use')
    input('Press Enter to continue, or CTRL-C to exit.')

    server_url = get_server_url(args.server)
    base_url = server_url + 'v3/'
    tos_url = server_url + 'webpage/FaceForensics_TOS.pdf'
    deepfakes_model_url = base_url + 'manipulated_sequences/Deepfakes/models/'

    os.makedirs(args.output_path, exist_ok=True)

    c_datasets = [args.dataset] if args.dataset != 'all' else ALL_DATASETS
    c_type = args.type
    c_compression = args.compression
    num_videos = args.num_videos

    for dataset in c_datasets:
        dataset_path = DATASETS[dataset]

        # Handle special dataset cases
        if 'original_youtube_videos' in dataset:
            suffix = '' if 'info' not in dataset_path else 'info'
            print(f'Downloading original youtube videos{" info" if suffix else ""}.')
            
            download_file(
                base_url + '/' + dataset_path,
                join(args.output_path, f'downloaded_videos{suffix}.zip'),
                report_progress=True
            )
            continue

        # Regular dataset download
        print(f'Downloading {c_type} of dataset "{dataset_path}"')

        # Fetch file list
        try:
            if 'DeepFakeDetection' in dataset_path or 'actors' in dataset_path:
                filepaths = json.loads(urllib.request.urlopen(base_url + '/' + DEEPFEAKES_DETECTION_URL).read())
                filelist = filepaths['actors'] if 'actors' in dataset_path else filepaths['DeepFakesDetection']
            elif 'original' in dataset_path:
                file_pairs = json.loads(urllib.request.urlopen(base_url + '/' + FILELIST_URL).read())
                filelist = [item for pair in file_pairs for item in pair]
            else:
                file_pairs = json.loads(urllib.request.urlopen(base_url + '/' + FILELIST_URL).read())
                filelist = [f'{"_".join(pair)}' for pair in file_pairs]
                if c_type != 'models':
                    filelist += [f'{"_".join(pair[::-1])}' for pair in file_pairs]

            # Limit videos if specified
            if num_videos is not None and num_videos > 0:
                print(f'Downloading first {num_videos} videos')
                filelist = filelist[:num_videos]

            # Determine download paths
            dataset_videos_url = f'{base_url}{dataset_path}/{c_compression}/{c_type}/'
            dataset_mask_url = f'{base_url}{dataset_path}/masks/videos/'

            if c_type == 'videos':
                output_path = join(args.output_path, dataset_path, c_compression, c_type)
                print(f'Output path: {output_path}')
                filelist = [f'{filename}.mp4' for filename in filelist]
                download_files(filelist, dataset_videos_url, output_path)

            elif c_type == 'masks':
                output_path = join(args.output_path, dataset_path, 'videos')
                print(f'Output path: {output_path}')
                
                if 'original' in dataset:
                    print('Skipping masks for original data.')
                    continue
                
                if 'FaceShifter' in dataset:
                    print('Masks not available for FaceShifter.')
                    continue

                filelist = [f'{filename}.mp4' for filename in filelist]
                download_files(filelist, dataset_mask_url, output_path)

            elif c_type == 'models':
                if dataset != 'Deepfakes':
                    print('Models only available for Deepfakes.')
                    continue

                output_path = join(args.output_path, dataset_path, c_type)
                print(f'Output path: {output_path}')

                for folder in tqdm(filelist):
                    model_url = f'{deepfakes_model_url}{folder}/'
                    folder_output_path = join(output_path, folder)
                    download_files(DEEPFAKES_MODEL_NAMES, model_url, folder_output_path, report_progress=False)

        except Exception as e:
            print(f"Error processing dataset {dataset}: {e}")

if __name__ == "__main__":
    main()