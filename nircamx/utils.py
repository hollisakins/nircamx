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
    
    cfg = toml.load(config_file)
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
    
    del cfg.directory_structure
    # in the future, add some logic to handle defaults, etc. 

    return cfg


def get_uncal_files(filtname):
    if len(config.files) == 1:
        file = config.files[0]
        uncal_files = glob(os.path.join(config.data_path, filtname, file + '*_uncal.fits'))[:]
    else:
        uncal_files = []
        for file in config.files:
            uncal_files += glob(os.path.join(config.data_path, filtname, file + '*_uncal.fits'))[:]
            
    uncal_files = np.sort(uncal_files)
    return uncal_files

def get_rate_files(filtname):
    if len(config.files) == 1:
        file = config.files[0]
        rate_files = glob(os.path.join(config.stage1_product_path, filtname, file + '*_rate.fits'))[:]
    else:
        rate_files = []
        for file in config.files:
            rate_files += glob(os.path.join(config.stage1_product_path, filtname, file + '*_rate.fits'))[:]
    
    rate_files = np.sort(rate_files)
    return rate_files

def get_cal_files(filtname):
    if len(config.files) == 1:
        file = config.files[0]
        cal_files = glob(os.path.join(config.stage2_product_path, filtname, file + '*_cal.fits'))[:]
    else:
        cal_files = []
        for file in config.files:
            cal_files += glob(os.path.join(config.stage2_product_path, filtname, file + '*_cal.fits'))[:]
    
    cal_files = np.sort(cal_files)
    return cal_files

def get_jhat_files(filtname):
    if len(config.files) == 1:
        file = config.files[0]
        jhat_files = glob(os.path.join(config.stage3_product_path, filtname, file + '*_jhat.fits'))[:]
    else:
        jhat_files = []
        for file in config.files:
            jhat_files += glob(os.path.join(config.stage3_product_path, filtname, file + '*_jhat.fits'))[:]
    
    jhat_files = np.sort(jhat_files)
    return jhat_files

def get_outlier_asn_files(filtname):
    if len(config.files) == 1:
        file = config.files[0]
        outlier_asn_files = glob(os.path.join(config.stage3_product_path, filtname, f'outlier_detection_{file}*_asn.json'))
    else:
        outlier_asn_files = []
        for file in config.files:
            outlier_asn_files += glob(os.path.join(config.stage3_product_path, filtname, f'outlier_detection_{file}*_asn.json'))
    
    outlier_asn_files = np.sort(outlier_asn_files)
    return outlier_asn_files

def get_crf_files(filtname):
    if len(config.files) == 1:
        file = config.files[0]
        crf_files = glob(os.path.join(config.stage3_product_path, filtname, file + '*_crf.fits'))[:]
    else:
        crf_files = []
        for file in config.files:
            crf_files += glob(os.path.join(config.stage3_product_path, filtname, file + '*_crf.fits'))[:]
    
    crf_files = np.sort(crf_files)
    return crf_files

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


def plot_hist(image, ax, bins, color, label):
    from jwst.datamodels import ImageModel
    with ImageModel(image) as im:
        data = im.data
        # consider only non-zero and unflagged pixels
        data = data[(im.data != 0) & (im.dq == 0)]
    ax.hist(data, bins=bins, color=color, label=label, alpha=0.5)


###############################################################################################
