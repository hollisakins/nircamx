from . import utils
import os, glob
import shutil
from copy import deepcopy
from .ref import NIR_amps
import numpy as np
from datetime import datetime
from time import sleep

# Individual steps that make up calwebb_detector1
from scipy.ndimage import median_filter, binary_dilation, gaussian_filter
from scipy.optimize import curve_fit

from photutils.segmentation import SegmentationImage
from photutils.segmentation import detect_threshold, detect_sources
from photutils.background import (Background2D, BiweightLocationBackground,
                                  BkgIDWInterpolator, BkgZoomInterpolator,
                                  MedianBackground, SExtractorBackground)

from astropy.io import fits
from astropy.stats import (gaussian_fwhm_to_sigma, 
                           sigma_clipped_stats, 
                           SigmaClip, 
                           biweight_location, 
                           median_absolute_deviation)

from astropy.convolution import (Tophat2DKernel, 
                                 Gaussian2DKernel, 
                                 Ring2DKernel, 
                                 convolve, 
                                 convolve_fft)


logger = utils.setup_logger()

config = None 

def detector1_step(uncal_file):
    from jwst.pipeline import calwebb_detector1
    overwrite = config.stage1.detector1_step.overwrite

    filtname = uncal_file.split('/')[-2]
    assert (filtname in utils.sw_filters) or (filtname in utils.lw_filters)
    uncal_file_name = os.path.basename(uncal_file)
    rate_file_name = uncal_file_name.replace('_uncal.fits', '_rate.fits')
    output_dir = os.path.join(config.stage1_product_path, filtname)
    rate_file = os.path.join(output_dir, rate_file_name)
    
    if os.path.exists(rate_file) and not overwrite:
        logger.info(f"Skipping detector1_step on {uncal_file_name}, rate file already exists")
        return 
    
    if os.path.exists(rate_file) and overwrite:
        pattern = rate_file.replace('_rate.fits', '*')
        files = glob.glob(pattern)
        for file in files:
            os.remove(file)
            print(f"Removed file: {file}")
    
    logger.info(f"Running detector1_step on {uncal_file_name}")

    kwargs = {'output_dir': output_dir, 
              'save_results': True,
              'steps': {
                'group_scale': {'skip': False},
                'dq_init': {'skip': False},
                'emicorr': {'skip': True},
                'saturation': {
                    'skip': False,
                    'n_pix_grow_sat': 1,
                    'use_readpatt': True,
                },
                'ipc': {'skip': False},
                'superbias': {'skip': False},
                'refpix':{
                    'skip': False,
                    'odd_even_columns': True,
                    'odd_even_rows': True,
                    'gaussmooth': 1.0,
                    'halfwidth': 30,
                    'side_gain': 1.0,
                    'side_smoothing_length': 11,
                    'sigreject': 4.0,
                    'use_side_ref_pixels': True,
                    'irs2_mean_subtraction': False,
                    'ovr_corr_mitigation_ftr': 3.0,
                    'preserve_irs2_refpix': False,
                    'refpix_algorithm': 'median',
                    'use_side_ref_pixels': True,
                },
                'rscd': {'skip': False, 'type': 'baseline'},
                'firstframe': {'skip': False, 'bright_use_group1': False},
                'lastframe': {'skip': False},
                'linearity': {'skip': False},
                'dark_current': {
                    'skip': False, 
                    'average_dark_current': None,
                    'dark_output': None,
                },
                'reset': {'skip': False},
                'persistence': {
                    'skip': False,
                    'flag_pers_cutoff': 40.0,
                    'save_persistence': True,
                    'save_results': True,
                    'save_trapsfilled': False,
                },
                'charge_migration': {'skip': True},
                'jump': {
                    'skip': False,
                    'after_jump_flag_dn1': 0.0,
                    'after_jump_flag_dn2': 0.0,
                    'after_jump_flag_time1': 0.0,
                    'after_jump_flag_time2': 0.0,
                    'edge_size': 25,
                    'expand_factor': 2.2,
                    'expand_large_events': True,
                    'extend_ellipse_expand_ratio': 1.1,
                    'extend_inner_radius': 1.0,
                    'extend_min_area': 90,
                    'extend_outer_radius': 2.6,
                    'extend_snr_threshold': 1.2,
                    'find_showers': False,
                    'flag_4_neighbors': True,
                    'four_group_rejection_threshold': 5.0,
                    'mask_snowball_core_next_int': True,
                    'max_extended_radius': 200,
                    'max_jump_to_flag_neighbors': 300.0,
                    'max_shower_amplitude': 4.0,
                    'maximum_cores': 'none',
                    'min_diffs_single_pass': 10,
                    'min_jump_area': 15.0 ,
                    'min_jump_to_flag_neighbors': 15.0,
                    'min_sat_area': 1.0,
                    'min_sat_radius_extend': 2.0,
                    'minimum_groups': 3,
                    'minimum_sigclip_groups': 100,
                    'only_use_ints': True,
                    'rejection_threshold': 4.0,
                    'sat_expand': 2,
                    'sat_required_snowball': False,
                    'save_results': True,
                    'search_output_file': True,
                    'snowball_time_masked_next_int': 4000,
                    'three_group_rejection_threshold': 6.0,
                    'time_masked_after_shower': 15.0,
                    'use_ellipses': True,
                },
                'clean_flicker_noise': {'skip': not config.stage1.detector1_step.clean_flicker_noise, 'fit_by_channel': True},
                'ramp_fit': {
                    'skip': False, 
                    'algorithm': 'OLS_C',
                    'maximum_cores': 'none'
                },
                'gain_scale': {'skip': False},
              }  
            }



    calwebb_detector1.Detector1Pipeline.call(uncal_file, **kwargs)

    persitance_file1 = os.path.join(config.working_dir, rate_file_name.replace('_rate.fits','_persistence.fits'))
    persitance_file2 = os.path.join(config.working_dir, rate_file_name.replace('_rate.fits','_trapsfilled.fits'))
    persitance_file3 = os.path.join(config.working_dir, rate_file_name.replace('_rate.fits','_output_pers.fits'))

    if os.path.exists(persitance_file1): shutil.move(persitance_file1, os.path.join(config.stage1_product_path,filtname,os.path.basename(persitance_file1)))
    if os.path.exists(persitance_file2): shutil.move(persitance_file2, os.path.join(config.stage1_product_path,filtname,os.path.basename(persitance_file2)))
    if os.path.exists(persitance_file3): shutil.move(persitance_file3, os.path.join(config.stage1_product_path,filtname,os.path.basename(persitance_file3)))



def run_snowblind(rate_file):

    """
    Update JWST DQ mask using `snowblind`. 

    Parameters
    ----------
    rate_file : str
        Filename of a ``rate.fits`` exposure

    max_fraction : float
        Maximum allowed fraction of flagged pixels relative to the total

    new_jump_flag : int
        Integer DQ flag of identified snowballs

    min_radius : int
        Minimum radius of ``JUMP_DET`` flagged groups of pixels

    growth_factor : float
        Scale factor of the DQ mask

    Returns
    -------
    dq : array-like
        Image array with values ``new_jump_flag`` with identified snowballs

    mask_frac : float
        Fraction of masked pixels

    """
    max_fraction = config.stage1.remove_snowball_step.max_fraction
    new_jump_flag = config.stage1.remove_snowball_step.new_jump_flag
    min_radius = config.stage1.remove_snowball_step.min_radius
    growth_factor = config.stage1.remove_snowball_step.growth_factor

    from jwst.datamodels import ImageModel
    import snowblind

    find_snowballs = snowblind.SnowblindStep
    with ImageModel(rate_file) as dm:
        sb = find_snowballs.call(dm,
            save_results=False,
            new_jump_flag=new_jump_flag,
            min_radius=min_radius,
            growth_factor=growth_factor
        )

    mask_frac = ((sb.dq & new_jump_flag) > 0).sum() / sb.dq.size

    if mask_frac > max_fraction:
        logger.warning(f'snowblind: fraction of masked pixels {mask_frac*100:.2f}% > {max_fraction*100:.2f}% for {rate_file}, skipping...')
        return (sb.dq & 0)

    else:
        logger.info(f'snowblind: {rate_file} {mask_frac*100:.2f} masked with DQ={new_jump_flag}')
        return (sb.dq & new_jump_flag)


     
def remove_snowballs(image):
    logger.info(f'Running remove_snowballs on {os.path.basename(image)[:-10]}')
    from jwst.datamodels import ImageModel
    import snowblind

    with ImageModel(image) as immodel:
        snowblind_dq = run_snowblind(image)
        immodel.dq |= snowblind_dq
        immodel.save(image)


def calc_variance(data, template, coeff):
    """Calculates the absolute median deviation of wisp subtracted image.

    Determines the variance of the function: image - coefficient * template.
    Using the median absolute deviation squarred. This is not scaled to 
    represent the standard deviation of normally distributed data, as would 
    be appropriate for an error estimator. However, fit_wisp_feature() will 
    find the coefficient that minimizes this variance, and so the relative 
    values are what matter. 

    Args:
        data (float): image array of masked data values
        template (float): image array of wisp template
        coeff (float): coefficient for scaling wisp template

    Returns:
        var_mad (float): median absolute deviation squarred for given coeff
    """
    func = data - coeff * template
    sigma_mad = median_absolute_deviation(func, ignore_nan=True)
    var_mad = sigma_mad**2
    return var_mad

###########################################################################################
# TODO incorporate separate wisp template prep step into pipeline, unify file naming scheme
###########################################################################################
def remove_wisps(rate_file):
    plot = config.stage1.remove_wisp_step.plot
    apply_flat = config.stage1.remove_wisp_step.apply_flat

    try:
        crds_context = os.environ['CRDS_CONTEXT']
    except KeyError:
        import crds
        crds_context = crds.get_default_context()

    filtname = rate_file.split('/')[-2]
    rate_file_name = os.path.basename(rate_file)
    detector = rate_file_name.split('_')[3]
    rate_file_orig = rate_file.replace('_rate.fits', '_rate_without_wisps_sub.fits')
    
    if detector not in ['nrca3', 'nrca4', 'nrcb3', 'nrcb4']:
        logger.info(f'Skipping wisp correction for {rate_file_name}')
        return

    res = []
    wisp_template_names = []

    # check that image has not already been corrected
    from jwst.datamodels import ImageModel
    model = ImageModel(rate_file)
    for entry in model.history:
        if 'Removed wisps' in entry['description']:
            logger.info(f'{rate_file_name} already corrected for wisps, exiting')
            return

    logger.info(f'Removing wisps for {rate_file_name}')

    if apply_flat:
        logger.info('Applying flat to match wisp templates')
        import crds
        # pull flat from CRDS using the current context
        crds_dict = {'INSTRUME':'NIRCAM',
                     'DETECTOR':model.meta.instrument.detector,
                     'FILTER':model.meta.instrument.filter,
                     'PUPIL':model.meta.instrument.pupil,
                     'DATE-OBS':model.meta.observation.date,
                     'TIME-OBS':model.meta.observation.time}
        flats = crds.getreferences(crds_dict, reftypes=['flat'], context=crds_context)
        # if the CRDS loopup fails, should return a CrdsLookupError, but 
        # just in case:
        try:
            flatfile = flats['flat']
        except KeyError:
            logger.error(f'Flat was not found in CRDS with the parameters: {crds_dict}')
            sys.exit()

        logger.info(f'Using flat: {os.path.basename(flatfile)}')
        from jwst.datamodels import FlatModel
        from jwst.flatfield.flat_field import do_correction
        try:
            with FlatModel(flatfile) as flat:
                # use the JWST Calibration Pipeline flat fielding Step
                model, applied_flat = do_correction(model, flat)
        except:
            sleep(3)
            with FlatModel(flatfile) as flat:
                # use the JWST Calibration Pipeline flat fielding Step
                model, applied_flat = do_correction(model, flat)

    # construct mask for median calculation
    mask = np.zeros(model.data.shape, dtype=bool)
    mask[np.isnan(model.data)] = True
    # mask[model.dq > 0] = True

    # source detection
    threshold = detect_threshold(model.data, nsigma=5.5)
    segm = detect_sources(model.data, threshold, npixels=55)
    wobj = np.where(segm.data > 0)
    mask[wobj] = True

    masked_im = model.data.copy()
    masked_im[mask] = 0

    # consider subsets of image focused around wisps for variance scaling
    if detector == 'nrca3':
        x1, x2, y1, y2 = 100, 1300, 1100, 2046
    elif detector == 'nrca4': 
        x1, x2, y1, y2 = 300, 1450, 0, 900
    elif detector == 'nrcb3': 
        x1, x2, y1, y2 = 350, 1450, 0, 1000
    elif detector == 'nrcb4': 
        x1, x2, y1, y2 = 400, 1700, 850, 2046

    im_seg = masked_im[y1:y2,x1:x2]

    # read in template and mask nans
    wisp_file_names = [f'WISP_{detector.upper()}_{filtname.upper()}_CLEAR_masked.fits',
                       f'WISP_{detector.upper()}_{filtname.upper()}_CLEAR_masked_smoothed_1x1.fits',
                       f'WISP_{detector.upper()}_{filtname.upper()}_CLEAR_masked_smoothed_2x2.fits',
                       f'WISP_{detector.upper()}_{filtname.upper()}_CLEAR_masked_smoothed_3x3.fits']
    short_file_names = [f'Masked',
                       f'Masked + smoothed 1x1',
                       f'Masked + smoothed 3x3',
                       f'Masked + smoothed 5x5']

    # To be removed, was experimenting with custom wisp templates for PRIMER but it didn't work
    custom_wisp_rate_files = ['jw01837003021_08201_00002_nrca3_rate.fits',
                              'jw01837003021_08201_00001_nrca3_rate.fits',
                              'jw01837003021_08201_00001_nrcb3_rate.fits',
                              'jw01837003021_08201_00002_nrcb3_rate.fits',
                              'jw01837003021_08201_00001_nrcb4_rate.fits',
                              'jw01837003021_08201_00002_nrca4_rate.fits',
                              'jw01837003021_08201_00001_nrca4_rate.fits',
                              'jw01837003021_08201_00002_nrcb4_rate.fits',
                              'jw01837003022_08201_00001_nrca3_rate.fits',
                              'jw01837003022_08201_00001_nrcb3_rate.fits',
                              'jw01837003022_08201_00002_nrcb3_rate.fits',
                              'jw01837003022_08201_00002_nrca3_rate.fits',
                              'jw01837003022_08201_00001_nrcb4_rate.fits',
                              'jw01837003022_08201_00002_nrca4_rate.fits',
                              'jw01837003022_08201_00001_nrca4_rate.fits',
                              'jw01837003022_08201_00002_nrcb4_rate.fits',
                              'jw01837003023_08201_00001_nrcb3_rate.fits',
                              'jw01837003023_08201_00002_nrcb3_rate.fits',
                              'jw01837003023_08201_00002_nrca3_rate.fits',
                              'jw01837003023_08201_00001_nrca3_rate.fits',
                              'jw01837003023_08201_00002_nrcb4_rate.fits',
                              'jw01837003023_08201_00001_nrca4_rate.fits',
                              'jw01837003023_08201_00001_nrcb4_rate.fits',
                              'jw01837003023_08201_00002_nrca4_rate.fits',
                              'jw01837003024_08201_00001_nrcb3_rate.fits',
                              'jw01837003024_08201_00002_nrcb3_rate.fits',
                              'jw01837003024_08201_00001_nrca3_rate.fits',
                              'jw01837003024_08201_00002_nrca3_rate.fits',
                              'jw01837003024_08201_00002_nrcb4_rate.fits',
                              'jw01837003024_08201_00001_nrca4_rate.fits',
                              'jw01837003024_08201_00002_nrca4_rate.fits',
                              'jw01837003024_08201_00001_nrcb4_rate.fits']
    if os.path.basename(rate_file) in custom_wisp_rate_files:
        wisp_file_names = [f'../primer_{filtname}_{detector}_masked.fits',
                           f'../primer_{filtname}_{detector}_masked_smoothed_1x1.fits',
                           f'../primer_{filtname}_{detector}_masked_smoothed_2x2.fits',
                           f'../primer_{filtname}_{detector}_masked_smoothed_3x3.fits']


    if plot:
        import matplotlib.pyplot as plt
        # plt.style.use(os.path.join(config.reference_path, 'nircamx.mplstyle'))
        fig,(ax1,ax2) = plt.subplots(2, 1, figsize=(12,8), tight_layout=True)
    
    min_x, min_y = np.zeros(len(wisp_file_names)), np.zeros(len(wisp_file_names))
    for i in range(len(wisp_file_names)):
        wisp_file_name = wisp_file_names[i]
        short_file_name = short_file_names[i]
        logger.info(f'{wisp_file_name}')

        wisp_template = fits.getdata(os.path.join(config.wisp_path, wisp_file_name))
        wisp_template[np.isnan(wisp_template)] = 0
        wisp_template[model.data == 0] = 0

        # # smooth template before variance scaling, seems to help
        # wisptemp = gaussian_filter(wisp_template, 2.0, mode='constant', truncate=3.5)

        wisp_seg = wisp_template[y1:y2,x1:x2]

        logger.info('fitting coefficients')    
        coeffs = np.arange(0.01, 1.5, 0.01)
        # coeffs = np.linspace(0.01, 100, 1000)
        variance_mad = np.zeros(coeffs.shape[0])
        for j,c in enumerate(coeffs):
            variance_mad[j] = calc_variance(im_seg, wisp_seg, c)

        # fit with a curve to base scaling off of trend, rather than scatter
        fit_mad = np.polyfit(coeffs, variance_mad, deg=2)
        pfit_mad = np.poly1d(fit_mad)

        # show difference between curve and measured variances
        variance_mad_pred = pfit_mad(coeffs) 
        diff = variance_mad - variance_mad_pred
        
        m = np.argmin(variance_mad)
        minval = coeffs[m]
        min_x[i] = minval
        min_y[i] = variance_mad[m]

        # m = np.argmin(variance_mad_pred)
        # minval = coeffs[m]
        # min_x[i] = minval
        # min_y[i] = variance_mad_pred[m]

        logger.info(f'fit coefficient = {minval:.2f}')

        if plot:
            ax1.plot(coeffs, variance_mad_pred*1e4, f'C{i}', lw=1.5, label=short_file_name)
            ax1.plot(coeffs, variance_mad*1e4, f'C{i}o', lw=1.5)
            ax2.plot(coeffs, diff*1e6, f'C{i}', lw=1)

            for ax in [ax1,ax2]:
                ax.axvline(minval, color=f'C{i}', ls=':', lw=0.5)

        i += 1

    which_template = np.argmin(min_y)
    minval = min_x[which_template]
    wisp_file_name = wisp_file_names[which_template]
    logger.info(f'Using wisp template {wisp_file_name}')
    # which_template = 0
    # minval = 0.4

    ax2.set_xlabel('coefficient')
    ax2.set_ylabel(r'residuals (10$^{-6}$)')
    ax1.set_ylabel(r'var (from MAD, 10$^{-4}$)')

    outplot = rate_file.replace('_rate.fits', '_wisp_fit.pdf')
    logger.info(f'Saving fit diagnostic plot to {outplot}')
    fig.savefig(outplot)

    # close model and open a clean version to clear anything we've done
    # to it (ie, flat fielding)
    model.close()
    del wisp_template


    # copy original
    logger.info(f'Copying input to {rate_file_orig}')
    shutil.copy2(rate_file, rate_file_orig)

    wisp_template = fits.getdata(os.path.join(config.wisp_path, wisp_file_name))
    wisp_template[np.isnan(wisp_template)] = 0
    wisp_template[model.data == 0] = 0

    model = ImageModel(rate_file)
    # subtract out wisp
    corrected = model.data -  minval * wisp_template
    model.data = corrected
   
    # add history entry
    from stdatamodels import util as stutil
    time = datetime.now()
    stepdescription = f"Removed wisps ({wisp_file_name}, scale = {minval:.2f}) {time.strftime('%Y-%m-%d %H:%M:%S')}"
    substr = stutil.create_history_entry(stepdescription)
    model.history.append(substr)

    model.save(rate_file)
    logger.info(f'cleaned image saved to {rate_file_name}')
    model.close()

    if plot:
        from .utils import plot_two
        outplot = rate_file.replace('_rate.fits', '_wisp.pdf')
        logger.info(f'Saving wisp plot to {outplot}')
        plot_two(rate_file, rate_file_orig ,title1='Wisp removed',title2='Original Rate', save_file=outplot)




def fit_sky(data):
    """
    Measure 2D background using unmasked pixels

    data is the original rate file maskes using the mosaic tiermask

    Useful for chips with a large, low surface brightness light left 
    over from wisps. Model it, remove it, fit 1/f and then put the 
    background back in

    Returns 2D background map. 
    """
    # first mask any leftover bright wisps that were not removed
    # there are a few new wisps that are not included in the templates
    skystd = np.nanstd(data)
    # >2 sig works at least for F200W B4, check others!
    data[data > (2*skystd)] = 0
    mask = data == 0
    if config.performance.bottleneck: 
        import bottleneck
        data.byteswap(inplace=True)
        data = data.view(data.dtype.newbyteorder('='))
    try:
        bkg = Background2D(data, box_size=128, 
                       sigma_clip=SigmaClip(sigma=3), 
                       filter_size=5, 
                       bkg_estimator=BiweightLocationBackground(), 
                       exclude_percentile=90, mask=mask, 
                       interpolator=BkgZoomInterpolator())
    except:
        try: 
            bkg = Background2D(data, box_size=128,
                        sigma_clip=SigmaClip(sigma=3),
                        filter_size=5,
                        bkg_estimator=BiweightLocationBackground(),
                        exclude_percentile=95, mask=mask,
                        interpolator=BkgZoomInterpolator())
        except:
            bkg = Background2D(data, box_size=128,
                        sigma_clip=SigmaClip(sigma=3),
                        filter_size=5,
                        bkg_estimator=BiweightLocationBackground(),
                        exclude_percentile=97.5, mask=mask,
                        interpolator=BkgZoomInterpolator())
        #     except:
        #         try:
        #             bkg = Background2D(data, box_size=128,
        #                         sigma_clip=SigmaClip(sigma=3),
        #                         filter_size=5,
        #                         bkg_estimator=BiweightLocationBackground(),
        #                         exclude_percentile=99.0, mask=mask,
        #                         interpolator=BkgZoomInterpolator())
        #         except:
        #             try:
        #                 bkg = fit_sky_tot(data)
        #                 logger.warning('Failed to fit the sky background, returning simple mean of Gaussian')
        #                 return bkg
        #             except:
        #                 return 0
    return bkg.background

def fit_sky_tot(data):
    """Fit distribution of sky fluxes with a Gaussian. Returns simple mean of Gaussian distribution."""
    std = sigma_clipped_stats(data)[2]
    bins = np.linspace(-10*std, 10*std, 1000)
    h, b = np.histogram(data, bins=bins)
    h = h / np.max(h)
    bc = 0.5 * (b[1:] + b[:-1])
    binsize = b[1] - b[0]

    p0 = [1, bc[np.argmax(h)], std]
    popt,pcov = curve_fit(utils.Gaussian, bc, h, p0=p0)

    return popt[1]

def fit_pedestal(data):
    """Fit distribution of sky fluxes with a Gaussian"""
    bins = np.arange(-1, 1.5, 0.001)
    h,b = np.histogram(data, bins=bins)
    bc = 0.5 * (b[1:] + b[:-1])
    binsize = b[1] - b[0]

    p0 = [10, bc[np.argmax(h)], 0.01]
    popt,pcov = curve_fit(utils.Gaussian, bc, h, p0=p0)

    return popt[1]

def collapse_image(im, mask, maxiters, dimension='y', sig=2.):
    """collapse an image along one dimension to check for striping.

    By default, collapse columns to show horizontal striping (collapsing
    along columns). Switch to vertical striping (collapsing along rows)
    with dimension='x' 

    Striping is measured as a sigma-clipped median of all unmasked pixels 
    in the row or column.

    Args:
        im (float array): image data array
        mask (bool array): image mask array, True where pixels should be 
            masked from the fit (where DQ>0, source flux has been masked, etc.)
        dimension (Optional [str]): specifies which dimension along which 
            to collapse the image. If 'y', collapses along columns to 
            measure horizontal striping. If 'x', collapses along rows to 
            measure vertical striping. Default is 'y'
        sig (Optional [float]): sigma to use in sigma clipping
    """
    # axis=1 results in array along y
    # axis=0 results in array along x
    if dimension == 'y':
        res = sigma_clipped_stats(im, mask=mask, sigma=sig, 
                                  cenfunc=np.nanmedian,
                                  stdfunc=np.nanstd, axis=1, maxiters=maxiters)
    elif dimension == 'x':
        res = sigma_clipped_stats(im, mask=mask, sigma=sig, 
                                  cenfunc=np.nanmedian,
                                  stdfunc=np.nanstd, axis=0, maxiters=maxiters)

    return res[1]

class SourceMask:
    def __init__(self, img, nsigma=3., npixels=3):
        ''' Helper for making & dilating a source mask.
             See Photutils docs for make_source_mask.'''
        self.img = img
        self.nsigma = nsigma
        self.npixels = npixels

    def single(self, filter_fwhm=3., tophat_size=5., mask=None):
        '''Mask on a single scale'''
        if mask is None:
            image = self.img
        else:
            image = self.img*(1-mask)
        mask = make_source_mask2(image, nsigma=self.nsigma,
                                npixels=self.npixels,
                                dilate_size=1, filter_fwhm=filter_fwhm)
        return dilate_mask(mask, tophat_size)

    def multiple(self, filter_fwhm=[3.], tophat_size=[3.], mask=None):
        '''Mask repeatedly on different scales'''
        if mask is None:
            self.mask = np.zeros(self.img.shape, dtype=bool)
        for fwhm, tophat in zip(filter_fwhm, tophat_size):
            smask = self.single(filter_fwhm=fwhm, tophat_size=tophat)
            self.mask = self.mask | smask  # Or the masks at each iteration
        return self.mask


def produce_mask(data, bkg, sigma=3.0, maxiter=10 , nsigma=2.5, npixels=10, mask=None, radius=10):
    from photutils.utils import circular_footprint
    sigma_clip = SigmaClip(sigma, maxiter)
    threshold = detect_threshold(data-bkg, nsigma, sigma_clip=sigma_clip,mask=mask)
    segment_img = detect_sources(data-bkg, threshold, npixels)
    footprint = circular_footprint(radius)
    mask = segment_img.make_source_mask(footprint=footprint)
    return mask

def dilate_mask(mask, tophat_size):
    ''' Take a mask and make the masked regions bigger.'''
    area = np.pi * tophat_size**2.
    kernel = Tophat2DKernel(tophat_size)
    dilated_mask = convolve(mask, kernel) >= 1. / area
    return dilated_mask

def masksources(image):
    """ """
    from jwst.datamodels import ImageModel
    model = ImageModel(image)
    sci = model.data
    err = model.err
    wht = model.wht
    dq = model.dq

    # bad pixel mask for SegmentationImage.make_source_mask(
    from jwst.datamodels import dqflags
    bpflag = dqflags.pixel['DO_NOT_USE']
    bp = np.bitwise_and(dq, bpflag)
    bpmask = np.logical_not(bp == 0)
    logger.info('masksources: estimating background')
    # make a robust estimate of the mean background and replace blank areas
    # bad pixel handling has flipped in pipeline v1.9+
    sci_nan = np.choose(np.isnan(sci),(sci,err))
    # Use the biweight estimator as a robust estimate of the mean background
    robust_mean_background = biweight_location(sci_nan, c=6., ignore_nan=True)
    sci_filled = np.choose(np.isnan(sci),(sci,robust_mean_background))

    logger.info('masksources: initial source mask')
    # make an initial source mask
    ring = Ring2DKernel(40, 3)
    filtered = median_filter(sci_filled, footprint=ring.array)

    logger.info('masksources: mask tier 1')
    # mask out sources iteratively
    # Try a reasonably big filter for masking the bright stuff
    convolved_difference = convolve_fft(sci_filled-filtered,Gaussian2DKernel(25))
    threshold = detect_threshold(convolved_difference, nsigma=3.0)
    segment_img1 = detect_sources(convolved_difference, threshold, npixels=15, mask=bpmask)
    mask1 = SegmentationImage.make_source_mask(segment_img1)
    
    # grow the largest mask
    temp = np.zeros(sci.shape)
    temp[mask1] = 1
    sources = np.logical_not(temp == 0)
    dilation_sigma = 3
    dilation_window = 5
    dilation_kernel = Gaussian2DKernel(dilation_sigma) #, x_size=dilation_window, y_size=dilation_window)
    source_wings = binary_dilation(sources, dilation_kernel)
    temp[source_wings] = 1
    mask1 = np.logical_not(temp == 0)

    logger.info('masksources: mask tier 2')
    # A smaller smoothing for the next tier
    convolved_difference = convolve_fft(sci_filled-filtered,Gaussian2DKernel(10))
    threshold = detect_threshold(convolved_difference, nsigma=3.0)
    segment_img2 = detect_sources(convolved_difference, threshold, npixels=10, mask=mask1)
    mask2 = SegmentationImage.make_source_mask(segment_img2) | mask1 


    logger.info('masksources: mask tier 3')
    # Still smaller
    convolved_difference = convolve_fft(sci_filled-filtered,Gaussian2DKernel(5))
    threshold = detect_threshold(convolved_difference, nsigma=3.0)
    segment_img3 = detect_sources(convolved_difference, threshold, npixels=5, mask=mask2)
    mask3 = SegmentationImage.make_source_mask(segment_img3) | mask2
    
    logger.info('masksources: mask tier 4')
    # Smallest
    convolved_difference = convolve_fft(sci_filled-filtered,Gaussian2DKernel(2))
    threshold = detect_threshold(convolved_difference, nsigma=3.0)
    segment_img4 = detect_sources(convolved_difference, threshold, npixels=3, mask=mask3)
    mask4 = SegmentationImage.make_source_mask(segment_img4)
    dilated_mask4 = dilate_mask(mask4,3)
    finalmask = mask4 | mask3

    # save output mask
    maskname = image.replace('.fits', '_1fmask.fits')
    logger.info(f'masksources: saving mask to {maskname}')
    outmask = np.zeros(finalmask.shape, dtype=int)
    outmask[finalmask] = 1
    fits.writeto(maskname, outmask, overwrite=True)
    return outmask


def measure_fullimage_striping(fitdata, mask, maxiters):
    """Measures striping in countrate images using the full rows.

    Measures the horizontal & vertical striping present across the
    full image. The full image median will be used for amp-rows that
    are entirely or mostly masked out.

    Args:
        fitdata (float array): image data array for fitting
        mask (bool array): image mask array, True where pixels should be
            masked from the fit (where DQ>0, source flux has been masked, etc.)

    Returns:
        (horizontal_striping, vertical_striping):
    """

    # fit horizontal striping, collapsing along columns
    horizontal_striping = collapse_image(fitdata, mask, maxiters, dimension='y')
    # remove horizontal striping, requires taking transpose of image
    temp_image = fitdata.T - horizontal_striping
    # transpose back
    temp_image2 = temp_image.T

    # fit vertical striping, collapsing along rows
    vertical_striping = collapse_image(temp_image2, mask, maxiters, dimension='x')

    return horizontal_striping, vertical_striping


def find_optimal_threshold(model, mask, full_horizontal, maxiters):
    """ """
    maskparams = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.70, 0.75, 0.80])

    var_mad = np.zeros(len(maskparams))
    for m, maskparam in enumerate(maskparams):
        logger.info(f'trying maskparam = {maskparam}')
        hstriping = np.zeros(model.data.shape)
        for amp in ['A','B','C','D']:
            rowstart, rowstop, colstart, colstop = NIR_amps[amp]['data']
            ampdata = model.data[:, colstart:colstop]
            ampmask = mask[:, colstart:colstop]
            # fit horizontal striping in amp, collapsing along columns
            hstriping_amp = collapse_image(ampdata, ampmask, maxiters, dimension='y')
            # check that at least 1/4 of pixels in each row are unmasked
            nmask = np.sum(ampmask, axis=1)
            max_nmask = ampmask.shape[1]*maskparam
            hstriping[nmask > max_nmask, colstart:colstop] = full_horizontal[nmask > max_nmask][:,None]
            hstriping[nmask <= max_nmask, colstart:colstop] = hstriping_amp[nmask <= max_nmask][:,None]
            
        # remove horizontal striping    
        temp_sub = model.data - hstriping

        # fit vertical striping, collapsing along rows
        vstriping = collapse_image(temp_sub, mask, maxiters, dimension='x')

        temp_sci = model.data - hstriping
    
        # transpose back
        # outputs[:,:,m] = temp_sci - vstriping
        temp_sci = temp_sci - vstriping
        sigma_mad = sigma_clipped_stats(temp_sci[~mask], cenfunc=np.nanmedian, stdfunc=np.nanstd, sigma=5)[2]
        var_mad[m] = sigma_mad**2

        # sigma_mad = median_absolute_deviation(masked, ignore_nan=True) 

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.scatter(maskparams, var_mad)
    # plt.semilogy()
    # plt.savefig('/n23data2/hakins/jwst/scripts/thresh.pdf')
    # plt.close()

    minm = np.argmin(var_mad)
    return maskparams[minm]


def remove_striping(image):
    """Removes striping in rate.fits files before flat fielding.

    Measures and subtracts the horizontal & vertical striping present in 
    countrate images. The striping is most likely due to 1/f noise, and 
    the RefPixStep with odd_even_columns=True and use_side_ref_pixels=True
    does not fully remove the pattern, no matter what value is chosen for 
    side_smoothing_length. There is also residual vertical striping in NIRCam 
    images simulated with Mirage.

    Note: 
        The original rate image file is copied to *_rate_orig.fits, and 
        the rate image with the striping patterns removed is saved to 
        *_rate.fits, overwriting the input filename

    Args:
        image (str): image filename, including full relative path
        apply_flat (Optional [bool]): if True, identifies and applies the 
            corresponding flat field before measuring striping pattern. 
            Applying the flat first allows for a cleaner measure of the 
            striping, especially for the long wavelength detectors. 
            Default is True.
        mask_sources (Optional [bool]): If True, masks out sources in image
            before measuring the striping pattern so that source flux is 
            not included in the calculation of the sigma-clipped median.
            Sources are identified using the Mirage seed images.
            Default is True.
        maskparam (Optional [float]): fraction of masked amp-row pixels 
            above which full row fit is used. Default is 0.75
    """
    
    apply_flat = config.stage1.remove_striping_step.apply_flat
    mask_sources = config.stage1.remove_striping_step.mask_sources
    maskparam=config.stage1.remove_striping_step.maskparam
    subtract_background=config.stage1.remove_striping_step.subtract_background
    maxiters = config.stage1.remove_striping_step.maxiters
    if maskparam == 'none':
        maskparam = None

    try:
        crds_context = os.environ['CRDS_CONTEXT']
    except KeyError:
        import crds
        crds_context = crds.get_default_context()
    
    from jwst.datamodels import ImageModel
    model = ImageModel(image)
    # check that image has not already been corrected
    for entry in model.history:
        if 'Removed horizontal,vertical striping' in entry['description']:
            logger.info(f'{image} already corrected for 1/f noise, exiting')
            return

    logger.info('Measuring image striping')
    logger.info(f'Working on {image}')

    # apply the flat to get a cleaner meausurement of the striping
    if apply_flat:
        logger.info('Applying flat for cleaner measurement of striping patterns')
        import crds
        # pull flat from CRDS using the current context
        crds_dict = {'INSTRUME':'NIRCAM', 
                     'DETECTOR':model.meta.instrument.detector, 
                     'FILTER':model.meta.instrument.filter, 
                     'PUPIL':model.meta.instrument.pupil, 
                     'DATE-OBS':model.meta.observation.date,
                     'TIME-OBS':model.meta.observation.time}
        flats = crds.getreferences(crds_dict, reftypes=['flat'], context=crds_context)
        # if the CRDS loopup fails, should return a CrdsLookupError, but
        # just in case:
        try:
            flatfile = flats['flat']
        except KeyError:
            logger.error(f'Flat was not found in CRDS with the parameters: {crds_dict}')
            exit()

        logger.info(f'Using flat: {os.path.basename(flatfile)}')
        from jwst.flatfield.flat_field import do_correction
        from jwst.datamodels import FlatModel
        try:
            with FlatModel(flatfile) as flat:
                # use the JWST Calibration Pipeline flat fielding Step
                model, applied_flat = do_correction(model, flat)
        except:
            sleep(3)
            with FlatModel(flatfile) as flat:
                # use the JWST Calibration Pipeline flat fielding Step
                model, applied_flat = do_correction(model, flat)

    mask = np.zeros(model.data.shape, dtype=bool)
    mask[model.dq > 0] = True
    
    if mask_sources:
        # first look for a source mask that already exists
        srcmask = maskname = image.replace('.fits', '_1fmask.fits')
        if os.path.exists(srcmask):
            logger.info(f'Using existing source mask {srcmask}')
            seg = fits.getdata(srcmask)
        else:
            logger.info('Detecting sources to mask out source flux')
            seg = masksources(image)

        wobj = np.where(seg > 0)
        mask[wobj] = True


    # measure the pedestal in the unmasked parts of the image
    logger.info('Measuring the pedestal in the image')
    pedestal_data = model.data[~mask]
    pedestal_data = pedestal_data.flatten()
    median_image = np.median(pedestal_data)
    logger.info(f'Image median (unmasked and DQ==0): {median_image:.5e}')
    try:
        pedestal = fit_pedestal(pedestal_data)
    except RuntimeError as e:
        logger.error("Can't fit sky, using median value instead")
        pedestal = median_image
    else:
        logger.info(f'Fit pedestal: {pedestal:.5e}')
    # subtract off pedestal so it's not included in fit  
    model.data -= pedestal

    if subtract_background:
        try:
            logger.info('Further measuring and subtracting the 2D background')
            backgrounddata = deepcopy(model.data)
            backgrounddata[mask > 0] = 0
            bkgd = fit_sky(backgrounddata)
            # subtract off background so it's not included in fit  
            model.data -= bkgd
        except:
            logger.warning(f'2D background subtraction failed for {image}, only using pedestal subtraction')



    # measure full pattern across image
    full_horizontal, vertical_striping = measure_fullimage_striping(model.data, mask, maxiters)
    # if thresh is not defined by user, search array of possible values
    # from .utils import plot_two
    # plot_two(full_horizontal, vertical_striping ,title1='Horizontal',title2='Vertical', save_file=image.replace('_rate.fits','_bkg.pdf'))
    if maskparam is None:
        try:
            logger.info('maskparam=None, automatically determining optimal value (can be slow)')
            maskparam = find_optimal_threshold(model, mask, full_horizontal, maxiters)
        except:
            logger.error(f'find_optimal_threshold failed on {image}')

        logger.info(f'Using threshold: {maskparam:.2f}')

    horizontal_striping = np.zeros(model.data.shape)
    vertical_striping = np.zeros(model.data.shape)

    # keep track of number of number of times the number of masked pixels 
    # in an amp-row exceeds thersh and a full-row median is used instead
    ampcounts = []
    for amp in ['A','B','C','D']:
        ampcount = 0
        rowstart, rowstop, colstart, colstop = NIR_amps[amp]['data']
        ampdata = model.data[:, colstart:colstop]
        ampmask = mask[:, colstart:colstop]
        # fit horizontal striping in amp, collapsing along columns
        hstriping_amp = collapse_image(ampdata, ampmask, dimension='y', maxiters=maxiters)
        # check that at least maskparam of pixels in each row are unmasked
        nmask = np.sum(ampmask, axis=1)
        for i,row in enumerate(ampmask):
            if nmask[i] > (ampmask.shape[1]*maskparam):
                # use median from full row
                horizontal_striping[i,colstart:colstop] = full_horizontal[i]
                ampcount += 1
            # upper limit on total number of masked pixels
            elif nmask[i] > (0.95*ampmask.shape[1]):
                horizontal_striping[i,colstart:colstop] = full_horizontal[i]
                ampcount += 1
            else:
                # use the amp fit 
                horizontal_striping[i,colstart:colstop] = hstriping_amp[i]
        ampcounts.append('%s-%i'%(amp,ampcount))

    ampinfo = ', '.join(ampcounts)
    logger.info(f'{os.path.basename(image)}, full row medians used: {ampinfo}/{rowstop-rowstart}')

    # remove horizontal striping    
    temp_sub = model.data - horizontal_striping

    # fit vertical striping, collapsing along rows
    vstriping = collapse_image(temp_sub, mask, maxiters, dimension='x')
    vertical_striping[:,:] = vstriping

    # save fits
    fits.writeto(image.replace('.fits', '_horiz.fits'), horizontal_striping, overwrite=True)
    fits.writeto(image.replace('.fits', '_vert.fits'), vertical_striping, overwrite=True)

    model.close()
    
    # copy image 
    image_orig = image.replace('.fits', '_orig.fits')
    logger.info(f"Copying input to {image_orig}")
    shutil.copy2(image, image_orig)

    # remove striping from science image
    with ImageModel(image) as immodel:
        sci = immodel.data
        # to replace zeros
        wzero = np.where(sci == 0)
        temp_sci = sci - horizontal_striping
        # transpose back
        outsci = temp_sci - vertical_striping
        outsci[wzero] = 0
        # replace NaNs with zeros and update DQ array
        # the image has NaNs where an entire row/column has been masked out
        # so no median could be calculated.
        # All of the NaNs on LW detectors and most of them on SW detectors
        # are the reference pixels around the image edges. But there is one
        # additional row on some SW detectors
        wnan = np.isnan(outsci)
        from jwst.datamodels import dqflags
        bpflag = dqflags.pixel['DO_NOT_USE']
        outsci[wnan] = 0
        immodel.dq[wnan] = np.bitwise_or(immodel.dq[wnan], bpflag)

        # write output
        immodel.data = outsci
        # add history entry
        time = datetime.now()
        stepdescription = f"Removed horizontal,vertical striping; {time.strftime('%Y-%m-%d %H:%M:%S')}"
        # writing to file doesn't save the time stamp or software dictionary
        # with the History object, but left here for completeness
        software_dict = {'name':'remstriping.py',
                         'author':'Micaela Bagley',
                         'version':'1.0',
                         'homepage':'ceers.github.io'}
        from stdatamodels import util as stutil
        substr = stutil.create_history_entry(stepdescription, software=software_dict)
        immodel.history.append(substr)
        logger.info(f'Saving cleaned image to {image}')
        immodel.save(image)

    if config.stage1.remove_striping_step.plot:
        image_name = os.path.basename(image)
        logger.info(f'Making striping removal plot for {image_name}')
        image_orig  = image.replace('_rate.fits', '_rate_orig.fits')
        output_file = image.replace('_rate.fits','_striping.pdf')
        utils.plot_two(image, image_orig , title1='Striping removed', title2='Original Rate', save_file=output_file)





def persistence_step(rate_files):
    from jwst.datamodels import ImageModel, ModelContainer
    path = os.path.dirname(rate_files[0])
    images = ModelContainer()
    for rate_file in rate_files:
        images.append(ImageModel(rate_file))

    import snowblind
    output = snowblind.PersistenceFlagStep.call(images, 
        save_results=True, 
        suffix = "rate",
        input_dir = path, 
        output_dir = path)
