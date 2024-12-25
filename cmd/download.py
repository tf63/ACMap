import os

import click
from gdown import download


# These datasets are from https://github.com/zhoudw-zdw/RevisitingCIL
def download_dataset(name, out_dir):
    if name == 'CUB200':
        url = 'https://drive.google.com/uc?id=1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb'
        out_path = os.path.join(out_dir, 'cub.zip')
    elif name == 'ImageNet-R':
        url = 'https://drive.google.com/uc?id=1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR'
        out_path = os.path.join(out_dir, 'imagenet-r.zip')
    elif name == 'ImageNet-A':
        url = 'https://drive.google.com/uc?id=19l52ua_vvTtttgVRziCZJjal0TPE9f2p'
        out_path = os.path.join(out_dir, 'imagenet-a.zip')
    elif name == 'VTAB':
        url = 'https://drive.google.com/uc?id=1xUiwlnx4k0oDhYi26KL5KwrCAya-mvJ_'
        out_path = os.path.join(out_dir, 'vtab.zip')
    else:
        raise ValueError('Invalid Argment')

    download(url, out_path, quiet=False)


@click.command()
@click.option('--name', required=True, help='Dataset name (CUB200|ImageNet-R|ImageNet-A|VTAB)')
@click.option('--out_dir', required=True, help='Download destination')
def main(name, out_dir):
    download_dataset(name, out_dir)


if __name__ == '__main__':
    main()
