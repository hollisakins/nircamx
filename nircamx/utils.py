import os
os.environ['OPENBLAS_NUM_THREADS'] = '32'

import numpy as np
from glob import glob
import logging, toml
from dotmap import DotMap

import multiprocessing
total_n_procs = multiprocessing.cpu_count()
n_procs       = int(multiprocessing.cpu_count()/2)

sw_filters = ['f070w','f090w','f115w','f140m','f150w','f162m','f164n','f150w2','f182m','f187n','f200w','f210m','f212n']
lw_filters = ['f250m','f277w','f300m','f322w2','f323n','f335m','f356w','f360m','f405n','f410m','f430m','f444w','f460m','f466n','f470n','f480m']

def setup_logger():
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    return log

def parse_config_file(config_file):
    
    # Load the toml config file
    cfg = toml.load(config_file)
    
    cfg['files'] = list(np.unique(cfg['files']))

    # Check that the config file has necessary keywords
    if not 'stage1' in cfg:
        cfg['stage1'] = {'run': False}
        
    if not 'stage2' in cfg:
        cfg['stage2'] = {'run': False}
    else:
        if not 'files_to_skip' in cfg['stage2']:
            cfg['stage2']['files_to_skip'] = []
    
    if not 'stage3' in cfg:
        cfg['stage3'] = {'run': False}
    else:
        if not 'files_to_skip' in cfg['stage3']:
            cfg['stage3']['files_to_skip'] = []
    # in the future, add some logic to handle defaults, etc. 
    

    cfg = DotMap(cfg)

    for key in cfg.environment:
        os.environ[key] = cfg.environment[key]
    del cfg.environment

    cfg.working_dir = os.getcwd()

    cfg.base_dir = cfg.directory_structure.base_dir
    cfg.data_dir = cfg.directory_structure.data_dir
    cfg.data_path = os.path.join(cfg.base_dir, cfg.data_dir)

    cfg.product_dir = cfg.directory_structure.product_dir
    cfg.product_path = os.path.join(cfg.base_dir, cfg.product_dir)
    cfg.stage1_product_dir = cfg.directory_structure.stage1_product_dir
    cfg.stage1_product_path = os.path.join(cfg.product_path, cfg.stage1_product_dir)
    cfg.stage2_product_dir = cfg.directory_structure.stage2_product_dir
    cfg.stage2_product_path = os.path.join(cfg.product_path, cfg.stage2_product_dir)
    cfg.stage3_product_dir = cfg.directory_structure.stage3_product_dir
    cfg.stage3_product_path = os.path.join(cfg.product_path, cfg.stage3_product_dir)

    cfg.mosaic_dir = cfg.directory_structure.mosaic_dir
    cfg.mosaic_path = os.path.join(cfg.base_dir, cfg.mosaic_dir)

    cfg.reference_dir = cfg.directory_structure.reference_dir
    cfg.reference_path = os.path.join(cfg.base_dir, cfg.reference_dir)
    cfg.bad_pixel_dir = cfg.directory_structure.bad_pixel_dir
    cfg.bad_pixel_path = os.path.join(cfg.reference_path, cfg.bad_pixel_dir)
    cfg.refcat_dir = cfg.directory_structure.refcat_dir
    cfg.refcat_path = os.path.join(cfg.reference_path, cfg.refcat_dir)
    cfg.wisp_dir = cfg.directory_structure.wisp_dir
    cfg.wisp_path = os.path.join(cfg.reference_path, cfg.wisp_dir)
    cfg.mask_dir = cfg.directory_structure.mask_dir
    cfg.mask_path = os.path.join(cfg.reference_path, cfg.mask_dir)
    cfg.flats_dir = cfg.directory_structure.flats_dir
    cfg.flats_path = os.path.join(cfg.reference_path, cfg.flats_dir)
    
    del cfg.directory_structure

    return cfg


def get_files(files, path, filtname, prefix, suffix, skip=None):
    assert type(files) in [list, np.ndarray]
    if len(files) == 1:
        file = files[0]
        result = glob(os.path.join(path, filtname, prefix+file+suffix))[:]
    else:
        result = []
        for file in files:
            result += glob(os.path.join(path, filtname, prefix+file+suffix))[:]
    

    if skip is not None:
        files_to_skip = get_files(skip, path, filtname, prefix, suffix, skip=None)
        result = [f for f in result if f not in files_to_skip]

    result = np.sort(result)
    return result

def get_uncal_files(filtname, skip=None):
    return get_files(config.files, path=config.data_path, filtname=filtname, prefix='', suffix='*_uncal.fits', skip=skip)
    
def get_rate_files(filtname, skip=None):
    return get_files(config.files, path=config.stage1_product_path, filtname=filtname, prefix='', suffix='*_rate.fits', skip=skip)

def get_cal_files(filtname, skip=None):
    return get_files(config.files, path=config.stage2_product_path, filtname=filtname, prefix='', suffix='*_cal.fits', skip=skip)

def get_jhat_files(filtname, skip=None):
    return get_files(config.files, path=config.stage3_product_path, filtname=filtname, prefix='', suffix='*_jhat.fits', skip=skip)

def get_outlier_asn_files(filtname, skip=None):
    return get_files(config.files, path=config.stage3_product_path, filtname=filtname, prefix='outlier_detection_', suffix='*_asn.json', skip=skip)

def get_crf_files(filtname, skip=None):
    return get_files(config.files, path=config.stage3_product_path, filtname=filtname, prefix='', suffix='*_crf.fits', skip=skip)
    

################################################################################################################################

def Gaussian(x, a, mu, sig):
    return a * np.exp(-(x-mu)**2/(2*sig**2))


def check_files_exist(file_paths):
    """
    Checks if multiple files exist at the given paths.

    Args:
        file_paths: A list of file paths.

    Returns:
        True if all files exist, False otherwise.
    """
    for path in file_paths:
        if not os.path.exists(path):
        #if not os.path.isfile(path): # Use this line to specifically check for files
            return False
    return True

##### plotting
from astropy.io import fits
from astropy.visualization import ImageNormalize, ZScaleInterval
import matplotlib.pyplot as plt
from matplotlib import rcParams

def plot_two(image1, image2, group=0, title1=None, title2=None, save_file=None, scaling=None):
    """Display two images side-by-side for comparison. 
    As a quick-look function, this is not robust or versatile. For
    example, it uses ZScaleInterval for all image normalization. It
    shows the first integration for all images, appropriate for CEERS
    exposures that only have 1 integration.

    Args:
        image1 (str): filename of first image for comparison
        image2 (str): filename of second image for comparison
        group (Optional [int]): index of group number to plot for the
            4D uncal images
        title1 (Optional [str]): title for image1 plot
        title2 (Optional [str]): title for image2 plot
    """

    if isinstance(image1, str) or isinstance(image2, str):
        im1 = fits.getdata(image1, 'SCI')
        im2 = fits.getdata(image2, 'SCI')
    else:
        im1 = image1
        im2 = image2
    # If images are 4D, pick slices to plot
    if len(im1.shape) == 4:
        # take first integration - there's only 1 for CEERS images
        im1 = im1[0,group,:,:]
    if len(im2.shape) == 4:
        im2 = im2[0,group,:,:]
    # normalize with zscale
    
    if scaling == None:
        norm1 = ImageNormalize(im1, interval=ZScaleInterval())
        norm2 = ImageNormalize(im2, interval=ZScaleInterval())
    elif scaling == 1:
        norm1 = ImageNormalize(im1, interval=ZScaleInterval())
        norm2 = norm1
    elif scaling == 2:
        norm2 = ImageNormalize(im2, interval=ZScaleInterval())
        norm1 = norm2
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(8,4), tight_layout=True)
    ax1.imshow(im1, origin='lower', interpolation='none', cmap='Greys', norm=norm1)
    ax2.imshow(im2, origin='lower', interpolation='none', cmap='Greys', norm=norm2)
    ax1.axis('off')
    ax2.axis('off')
    if title1:
        ax1.set_title(title1)
    if title2:
        ax2.set_title(title2)
    if save_file != None:
        fig.savefig(save_file)

def plot_three(image1, image2, image3, group=0, title1=None, title2=None, title3=None, save_file=None, scaling=None):
    """Display three images side-by-side for comparison. 
    As a quick-look function, this is not robust or versatile. For
    example, it uses ZScaleInterval for all image normalization. It
    shows the first integration for all images, appropriate for CEERS
    exposures that only have 1 integration.

    Args:
        image1 (str): filename of first image for comparison
        image2 (str): filename of second image for comparison
        group (Optional [int]): index of group number to plot for the
            4D uncal images
        title1 (Optional [str]): title for image1 plot
        title2 (Optional [str]): title for image2 plot
    """

    if isinstance(image1, str):
        im1 = fits.getdata(image1, 'SCI')
    else:
        im1 = image1
    
    if isinstance(image2, str):
        im2 = fits.getdata(image2, 'SCI')
    else:
        im2 = image2

    if isinstance(image3, str):
        im3 = fits.getdata(image3, 'SCI')
    else:
        im3 = image3

    # If images are 4D, pick slices to plot
    if len(im1.shape) == 4:
        # take first integration - there's only 1 for CEERS images
        im1 = im1[0,group,:,:]
    if len(im2.shape) == 4:
        im2 = im2[0,group,:,:]
    if len(im3.shape) == 4:
        im3 = im3[0,group,:,:]
    # normalize with zscale
    
    if scaling == None:
        norm1 = ImageNormalize(im1, interval=ZScaleInterval())
        norm2 = ImageNormalize(im2, interval=ZScaleInterval())
        norm3 = ImageNormalize(im3, interval=ZScaleInterval())
    elif scaling == 1:
        norm1 = ImageNormalize(im1, interval=ZScaleInterval())
        norm2 = norm1
        norm3 = norm1
    elif scaling == 2:
        norm2 = ImageNormalize(im2, interval=ZScaleInterval())
        norm1 = norm2
        norm3 = norm2
    elif scaling == 3:
        norm3 = ImageNormalize(im3, interval=ZScaleInterval())
        norm1 = norm3
        norm2 = norm3
    fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(12,4), tight_layout=True)
    ax1.imshow(im1, origin='lower', interpolation='none', cmap='Greys', norm=norm1)
    ax2.imshow(im2, origin='lower', interpolation='none', cmap='Greys', norm=norm2)
    ax3.imshow(im3, origin='lower', interpolation='none', cmap='Greys', norm=norm3)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    if title1:
        ax1.set_title(title1)
    if title2:
        ax2.set_title(title2)
    if title3:
        ax3.set_title(title3)
    if save_file != None:
        fig.savefig(save_file)


def plot_hist(image, ax, bins, color, label):
    from jwst.datamodels import ImageModel
    with ImageModel(image) as im:
        data = im.data
        # consider only non-zero and unflagged pixels
        data = data[(im.data != 0) & (im.dq == 0)]
    ax.hist(data, bins=bins, color=color, label=label, alpha=0.5)

root_path = os.path.dirname(os.path.abspath(__file__))
def get_tile_corners(tile, field='cosmos'):
    field_wcs = toml.load(os.path.join(root_path, 'wcs.toml'))[field]
    tile_wcs = field_wcs[tile]
    return tile_wcs['corners']

def get_tile_wcs(tile, ps='30mas', field='cosmos'):
    field_wcs = toml.load(os.path.join(root_path, 'wcs.toml'))[field]
    tile_wcs = field_wcs[tile]
    if 'tangent_point' in tile_wcs: # tile-specific tangent point
        crval = tile_wcs['tangent_point']
    else: # otherwise fallback to field tangent point
        crval = field_wcs['tangent_point']
    
    crpix = tile_wcs[ps]['crpix']
    shape = tile_wcs[ps]['naxis']
    rotation = tile_wcs['rotation']
    
    return crpix, crval, shape, rotation

###############################################################################################
