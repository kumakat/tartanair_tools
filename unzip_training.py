import argparse
import os
from os import system
from os.path import isdir, isfile
from glob import glob
from multiprocessing import Pool


def get_args():
    parser = argparse.ArgumentParser(description='TartanAir')

    parser.add_argument('--dataset-dir', default='./',
                        help='root directory for downloaded files')

    args = parser.parse_args()

    return args


def unzip_wrapper(path):
    datadir = '/'.join(path.split('/')[:-3])
    envname, difflevel, filename = path.split('/')[-3:]

    tmpdir = '/'.join([datadir, envname, difflevel, filename.replace('.zip', '')])
    logfile = path.replace('zip', 'log')
    cmd = f'unzip -o {path} -d {tmpdir} > {logfile}'
    print(cmd)
    system(cmd)


if __name__ == '__main__':
    args = get_args()

    # dataset directory
    datadir = args.dataset_dir
    if not isdir(datadir):
        print(f'dataset dir {dataset} does not exists!')
        exit()

    # unzip
    zipfiles = glob(datadir + '/**/*.zip', recursive=True)
    zipfiles.sort()
    with Pool(len(zipfiles)) as p:
        p.map(unzip_wrapper, zipfiles)

    # move
    tmpdirs = [path.replace('.zip', '') for path in zipfiles]
    for path in tmpdirs:
        envname, difflevel, dataname = path.split('/')[-3:]
        for t in glob('/'.join([path, envname, envname, difflevel, '*'])):
            trajectory = t.split('/')[-1]
            destination = '/'.join([datadir, envname, difflevel, trajectory])

            if not isdir(destination):
                system(f'mkdir -p {destination}')

            cmd = f'mv -f {trajectory}/* {destination}'
            print(cmd)
            system(cmd)

        system(f'rm -rf {path}')
