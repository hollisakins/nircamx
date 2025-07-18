from . import utils
import os 
import shutil
from datetime import datetime
import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, ZScaleInterval

from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.nddata import block_reduce
from astropy.stats import sigma_clip, biweight_location, biweight_midvariance, sigma_clipped_stats
from .stage1 import calc_variance
import warnings

from regions import Regions
from astropy.convolution import (Tophat2DKernel, 
                                 Gaussian2DKernel, 
                                 Ring2DKernel, 
                                 convolve, 
                                 convolve_fft)


logger = utils.setup_logger()

config = None

def image2_step(rate_file):
    from jwst.pipeline import calwebb_image2
    overwrite = config.stage2.image2_step.overwrite

    filtname = rate_file.split('/')[-2]
    assert (filtname in utils.sw_filters) or (filtname in utils.lw_filters)

    rate_file_name = os.path.basename(rate_file)
    cal_file_name = rate_file_name.replace('_rate.fits', '_cal.fits')
    output_dir = os.path.join(config.stage2_product_path, filtname)
    cal_file = os.path.join(output_dir, cal_file_name)
    
    if os.path.exists(cal_file) and not overwrite:
        logger.info(f"Skipping image2_step on {rate_file_name}, cal file already exists")
        return 

    logger.info(f"Running image2_step on {rate_file_name}")
    
    kwargs = {'output_dir': output_dir, 
              'save_results': True, 
              'steps': { 
                'bkg_subtract': {'skip': True},
                'assign_wcs': {
                    'skip': False,
                    'save_results': False,
                    'sip_approx': True,
                    'sip_degree': None,
                    'sip_inv_degree': None,
                    'sip_max_inv_pix_error': 0.25,
                    'sip_max_pix_error': 0.25,
                    'sip_npoints': 32,
                    'slit_y_high': 0.55,
                    'slit_y_low': -0.55,
                },
                'flat_field': {'skip': False},
                'photom': {'skip': False},
                'resample': {'skip': True}
              }}

    if config.stage2.image2_step.use_custom_flat:
        # jw01727028001_04101_00003_nrcalong_rate.fits
        detector = rate_file_name.split('_')[-2]
        flat_file = os.path.join(config.flats_path, f'flat_nircam_{filtname.upper()}_{detector.upper()}_CLEAR.fits')
        if os.path.exists(flat_file):
            kwargs['steps']['flat_field']['user_supplied_flat'] = flat_file
        else:
            logger.warning(f'Flat file {os.path.basename(flat_file)} was not found in {config.flats_path}')
            logger.warning(f'Falling back to CRDS flats')

    
    try:
        calwebb_image2.Image2Pipeline.call(rate_file, **kwargs)
    except ValueError as e:
        print(rate_file)
        raise e



def remove_edge(cal_file):
    from jwst.datamodels import ImageModel
    from stdatamodels import util as stutil

    cal_file_name = os.path.basename(cal_file)

    with ImageModel(cal_file) as model:
        # check that image has not already had edges removed
        for entry in model.history:
            if 'Removed edges' in entry['description']:
                logger.info(f'Edges already removed for {cal_file_name}, skipping...')
                return
        if os.path.exists(cal_file.replace('_cal.fits', '_before_removing_edge.fits')):
            logger.info(f'Edges already removed for {cal_file_name}, skipping...')
            return

        logger.info(f'Running edge removal for {cal_file_name}')

        size = model.data.shape[0]

        mean_ = []
        mean_h = []
        for ii in range(size):
            mean_.append(np.mean(model.data[:,ii]))
            mean_h.append(np.mean(model.data[ii,:]))

        index_beg = 0
        for ii in range(size):
            if index_beg == 0:
                if np.abs(np.mean(model.data[:,ii])) > np.std(mean_):
                    model.dq[:,ii] = 1
                else:
                    index_beg = 1

        index_end = 0
        for ii in range(size):
            if index_end ==0:
                if np.abs(np.mean(model.data[:,size-1-ii])) > np.std(mean_):
                    model.dq[:,size-1-ii] = 1
                else:
                    index_end = 1


        index_beg = 0
        for ii in range(size):
            if index_beg ==0:
                if np.abs(np.mean(model.data[ii,:])) > np.std(mean_h):
                    model.dq[:,ii] = 0
                else:
                    index_beg = 1

        index_end = 0
        for ii in range(size):
            if index_end ==0:
                if np.abs(np.mean(model.data[size-1-ii,:])) > np.std(mean_h):
                    model.dq[size-1-ii,:] = 0
                else:
                    index_end = 1

        # cal_file_without_edgeremoval = cal_file.replace('_cal.fits', '_before_removing_edge.fits')
        # shutil.copy2(cal_file, cal_file_without_edgeremoval)

        time = datetime.now()
        stepdescription = f"Removed edges; {time.strftime('%Y-%m-%d %H:%M:%S')}"
        substr = stutil.create_history_entry(stepdescription)
        model.history.append(substr)

        model.save(cal_file)


################################################################################################
# TBD, claw removal will be replaced with generic routine to apply masks defined by region files 
################################################################################################
def apply_masks(cal_file):
    cal_file_name = os.path.basename(cal_file)
    filtname = cal_file.split('/')[-2]
    reg_file = os.path.join(config.mask_path, filtname, cal_file_name.replace('_cal.fits', '.reg'))
    if not os.path.exists(reg_file):
        logger.info(f'No mask found for {cal_file_name}, skipping')
        return 

    flag = config.stage2.apply_mask_step.mask_flag
    set_to_nan = config.stage2.apply_mask_step.mask_set_nan
    logger.info(f'Applying mask to {cal_file_name}')
    from jwst.datamodels import ImageModel
    with ImageModel(cal_file) as model:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            wcs = model.get_fits_wcs()
        shape = np.shape(model.data)

        regs = Regions.read(reg_file)
        for reg in regs:
            reg = reg.to_pixel(wcs)
            mask = reg.to_mask(mode='center')
            mask = mask.to_image(shape)
            try:
                mask = mask.astype(bool)
            except:
                continue
            
            if set_to_nan:
                model.data[mask] = np.nan

            mask = (mask*flag).astype('uint32')
            model.dq |= mask
        
        model.save(cal_file)

    


# def remove_claws(cal_file):
#     plot = config.stage2.remove_claw_step.plot

#     cal_file_path = os.path.dirname(cal_file)
#     cal_file_name = os.path.basename(cal_file)
#     cal_file_orig = cal_file.replace('_cal.fits', '_cal_without_claw_sub.fits')

#     # check that image has not already been corrected
#     from jwst.datamodels import ImageModel
#     model = ImageModel(cal_file)
#     for entry in model.history:
#         if 'Removed claws' in entry['description']:
#             logger.info(f'{cal_file_name} already corrected for claws, exiting')
#             return

#     reg_file = os.path.join(config.claw_path, cal_file_name.replace('_cal.fits', '_cal.reg'))
#     if not os.path.exists(reg_file):
#         logger.warning(f'No claw region found for {cal_file_name}, skipping claw subtraction')
#         return 
    
#     detector = cal_file_name.split('_')[-2]
    
#     from .util_detect_source_and_create_seed_image_v3 import detect_source_and_background_for_image
#     detect_source_and_background_for_image(
#         cal_file, 
#         include_region_files = [reg_file], 
#         output_suffix = '_claws_seed_image', 
#         minpixarea = 0, 
#         ignore_background = True, 
#         sigma = 2, 
#         smooth_after = 1, 
#         overwrite = True
#     )
        
#     ################################################TODO
#     # first look for a source mask that already exists
#     rate_file = cal_file.replace(config.stage2_product_path, config.stage1_product_path).replace('_cal.fits','_rate.fits')
#     maskname = rate_file.replace('.fits', '_1fmask.fits')
#     logger.info(f'Using existing source mask {maskname}')
#     seg = fits.getdata(maskname)
#     srcmask = seg > 0 
#     dqmask = model.dq > 0
#     mask = np.logical_or(srcmask, dqmask)

#     logger.info('Measuring the pedestal in the image')
#     pedestal_data = model.data[~mask]
#     pedestal_data = pedestal_data.flatten()
#     median_image = np.median(pedestal_data)
#     logger.info(f'Image median (unmasked and DQ==0): {median_image:.5e}')
#     try:
#         from .stage1 import fit_pedestal
#         pedestal = fit_pedestal(pedestal_data)
#     except RuntimeError as e:
#         logger.error("Can't fit sky, using median value instead")
#         pedestal = median_image
#     else:
#         logger.info(f'Fit pedestal: {pedestal:.5e}')
#     # subtract off pedestal so it's not included in fit  
#     model.data -= pedestal

#     image_masked_without_nan = np.nan_to_num(model.data * np.logical_not(dqmask))
    


#     claw_file = cal_file.replace('_cal.fits', '_cal_claws_seed_image.fits')
#     claw_template = fits.getdata(claw_file, ext=0) * image_masked_without_nan
    
#     logger.info('Smoothing claw template with gaussian kernel')    
#     kernel = Gaussian2DKernel(x_stddev=6.0)
#     claw_template = convolve_fft(claw_template, kernel)
#     # claw_template = gaussian_filter(claw_template, 6.0, mode='constant', truncate=4.0)

#     # #mask_5_sigma              = make_source_mask(image, nsigma=3.5, npixels=5)
#     # threshold = detect_threshold(image, nsigma=3.5)

#     # segment_img2         = detect_sources(image, threshold, npixels=5)
#     # mask2 = SegmentationImage.make_source_mask(segment_img2) 

#     # image_masked_without_nan  = np.nan_to_num(image*np.logical_not(mask_5_sigma))
#     # for x in np.linspace(-0.1,2.,200):
#     #     if quadrant == 'b1':
#     #         print('B1')
#     #         res.append(np.var((image_masked_without_nan[5:750,5:750] -x*claws_template_conv[5:750,5:750])))
#     #     if quadrant == 'b2':
#     #         print('B2')
#     #         res.append(np.var((image_masked_without_nan[250:1250,1250:] -x*claws_template_conv[250:1250,1250:])))

#     # coef_claws = np.linspace(-0.1,2.,200)[np.argmin(res)]
#     # if ((coef_claws > 1.95) or  (coef_claws < 0)):
#     #     coef_claws = 0.
#     # plt.show()

#     # new_data = image - coef_claws*claws_template_conv


#     # consider subsets of image focused around claws for variance scaling
#     if detector == 'nrcb1':
#         x1, x2, y1, y2 = 5, 750, 5, 750
#     elif detector == 'nrcb2':
#         x1, x2, y1, y2 = 1250, 2048, 250, 1250
#     #TODO BLONG????????/????/????
        
#     im_seg = image_masked_without_nan[y1:y2,x1:x2]
#     claw_seg = claw_template[y1:y2,x1:x2]

#     logger.info('fitting coefficients')    
#     coeffs = np.linspace(-0.1,2.0,200)
    
#     variance_mad = np.zeros(coeffs.shape[0])
#     for i,c in enumerate(coeffs):
#         variance_mad[i] = calc_variance(im_seg, claw_seg, c)

#     # fit with a curve to base scaling off of trend, rather than scatter
#     fit_mad = np.polyfit(coeffs, variance_mad, deg=2)
#     pfit_mad = np.poly1d(fit_mad)
#     # show difference between curve and measured variances
#     # diff = variance_mad - pfit_mad(coeffs)
#     m = np.argmin(pfit_mad(coeffs))
#     # m = np.argmin(variance_mad)
#     minval = coeffs[m]

#     logger.info(f'{cal_file_name} - fit coefficient = {minval:.2f}')

#     if plot:
#         import matplotlib.pyplot as plt
#         # plt.style.use(os.path.join(config.reference_path, 'nircamx.mplstyle'))
#         fig, ax = plt.subplots(1, 1, figsize=(10,4), tight_layout=True)
#         ax.plot(coeffs, pfit_mad(coeffs)*1e4, 'C0', lw=1.5)
#         ax.plot(coeffs, variance_mad*1e4, 'C0o', lw=1.5)
#         ax.axvline(minval, color='k', ls='--', lw=1.5)
#         ax.set_xlabel('coefficient')
#         ax.set_ylabel(r'var (from MAD, 10$^{-4}$)')

#         outplot = cal_file.replace('_cal.fits', '_claw_fit.pdf')
#         fig.savefig(outplot)

#     # close model and open a clean version to clear anything we've done
#     model.close()

#     # copy original
#     logger.info(f'Copying input to {cal_file_orig}')
#     shutil.copy2(cal_file, cal_file_orig)

#     model = ImageModel(cal_file)
#     # subtract out claws
#     corr = model.data - minval * claw_template
#     model.data = corr
   
#     # add history entry
#     from stdatamodels import util as stutil
#     time = datetime.now()
#     stepdescription = f"Removed claws (scale = {minval:.2f}) {time.strftime('%Y-%m-%d %H:%M:%S')}"
#     substr = stutil.create_history_entry(stepdescription)
#     model.history.append(substr)

#     model.save(cal_file)
#     logger.info(f'cleaned image saved to {cal_file_name}')
#     model.close()

#     # clean up the results from util_detect_source_and_create_seed_image_v3.py
#     bkg2d = cal_file.replace('_cal.fits','_cal_claws_seed_image_bkg2d.fits')
#     maske = cal_file.replace('_cal.fits','_cal_claws_seed_image_masked.fits')
#     rms2d = cal_file.replace('_cal.fits','_cal_claws_seed_image_rms2d.fits')
#     unmas = cal_file.replace('_cal.fits','_cal_claws_seed_image_unmasked.fits')
#     valid = cal_file.replace('_cal.fits','_cal_claws_seed_image_valid_pixels.fits')
#     zeroo = cal_file.replace('_cal.fits','_cal_claws_seed_image_zeroonemask.fits')
#     image = cal_file.replace('_cal.fits','_cal_claws_seed_image.fits')
#     backu = cal_file.replace('_cal.fits','_cal_claws_seed_image.fits.backup')

#     if os.path.exists(bkg2d): os.remove(bkg2d)
#     if os.path.exists(maske): os.remove(maske)
#     if os.path.exists(rms2d): os.remove(rms2d)
#     if os.path.exists(unmas): os.remove(unmas)
#     if os.path.exists(valid): os.remove(valid)
#     if os.path.exists(zeroo): os.remove(zeroo)
#     if os.path.exists(image): os.remove(image)
#     if os.path.exists(backu): os.remove(backu)


#     if plot:
#         # outplot = cal_file.replace('_cal.fits', '_claw1.pdf')
#         # utils.plot_two(image_masked_without_nan, claw_template * minval, title1='Masked cal',title2='Claw template', save_file=outplot, scaling=1)

#         outplot = cal_file.replace('_cal.fits', '_claw.pdf')
#         utils.plot_two(cal_file_orig, cal_file ,title1='Original cal',title2='Claws removed', save_file=outplot)




def sky_subtraction(cal_file):
    '''Subtract a constant sky pedestal value from the cal file'''
    from stdatamodels import util as stutil
    from jwst.datamodels import ImageModel
    
    overwrite = config.stage2.skysub_step.overwrite
    with ImageModel(cal_file) as model:
        for entry in model.history:
            if 'Removed sky' in entry['description'] and not overwrite:
                logger.info(f'Sky subtraction already done for {os.path.basename(cal_file)}, skipping...')
                return

    logger.info(f'Running sky subtraction on {os.path.basename(cal_file)}')

    with ImageModel(cal_file) as model:
        sci = model.data
        dq = model.dq

        # Read in mask created during 1/f correction
        srcmask = cal_file.replace(config.stage2_product_path, config.stage1_product_path)
        srcmask = srcmask.replace('_cal.fits', '_rate_1fmask.fits')
        logger.info(f'Using existing source mask {os.path.basename(srcmask)}')
        seg = fits.getdata(srcmask)
        w = np.where((dq == 0) & (seg == 0))
        data = sci[w].flatten()
        
        # Apply a sigma clip to the data
        data = sigma_clip(data, sigma_upper=3, sigma_lower=10, maxiters=5, masked=False)
        data = data[~np.isinf(data) & ~np.isnan(data)]
        
        # Fit the pedestal
        from .stage1 import fit_sky_tot
        try:
            sky = fit_sky_tot(data)
        except:
            print(f'Failed on {cal_file}!!')
            raise

        # Subtract off sky    
        model.data -= sky

        # Update header
        model.meta.background.level = sky
        model.meta.background.subtracted = True
        model.meta.background.method = 'local'
        
        logger.info(f"Saving to {os.path.basename(cal_file)}")
        time = datetime.now()
        stepdescription = f"Removed sky {time.strftime('%Y-%m-%d %H:%M:%S')}"
        substr = stutil.create_history_entry(stepdescription)
        model.history.append(substr)
    
        model.save(cal_file)


def rescale_variance(cal_file):
    '''
    Perform variance map scaling 
    This routine models the 2D background of an individual exposure and rescales 
    the variance maps to match the measured variance of background pixels
    
     1. Run a background subtraction routine that creates a 2D model of 
        the background in an image. This will be used to calculate the 
        sky variance. 
        
     2. Rescale the variance maps. Determines a robust sky variance in the
        image and scales the VAR_RNOISE array to reproduce this value. The 
        VAR_RNOISE arrays are used for inverse variance weighting during 
        drizzling, so this step ensures that the resulting error arrays will 
        include the rms sky fluctuations
    '''
    from stdatamodels import util as stutil
    from jwst.datamodels import ImageModel
    
    overwrite = config.stage2.variance_step.overwrite
    with ImageModel(cal_file) as model:
        for entry in model.history:
            if 'Rescaled variance' in entry['description'] and not overwrite:
                logger.info(f'Variance rescaling already done for {os.path.basename(cal_file)}, skipping...')
                return

    logger.info(f'Rescaling variance for {os.path.basename(cal_file)}')

    # Run a full 2D background subtraction routine
    from .bkgsub import SubtractBackground
    bkg = SubtractBackground(
        ring_radius_in = config.stage2.variance_step.ring_radius_in,
        ring_width = config.stage2.variance_step.ring_width,
        ring_clip_max_sigma = config.stage2.variance_step.ring_clip_max_sigma,
        ring_clip_box_size = config.stage2.variance_step.ring_clip_box_size,
        ring_clip_filter_size = config.stage2.variance_step.ring_clip_filter_size,
        tier_kernel_size = config.stage2.variance_step.tier_kernel_size,
        tier_npixels = config.stage2.variance_step.tier_npixels,
        tier_nsigma = config.stage2.variance_step.tier_nsigma,
        tier_dilate_size = config.stage2.variance_step.tier_dilate_size,
        bg_box_size = config.stage2.variance_step.bg_box_size,
        bg_filter_size = config.stage2.variance_step.bg_filter_size,
        bg_exclude_percentile = config.stage2.variance_step.bg_exclude_percentile,
        bg_sigma = config.stage2.variance_step.bg_sigma,
        bg_interpolator = config.stage2.variance_step.bg_interpolator,
        suffix = 'bkgsub',
        replace_sci = True,
    )
    try:
        bkg.call(cal_file)
    except:
        print(f"!!! failed on {cal_file}")
        raise

    # rescale variance maps
    block_size = config.stage2.variance_step.block_size
    with ImageModel(cal_file) as model:
        sci = model.data
        var_rnoise = model.var_rnoise

        block_mask = block_reduce(bkg.mask_final, block_size) 
        unmasked_frac = np.sum(block_mask == 0)/np.sum(block_mask >= 0)

        block_sci = block_reduce(sci, block_size)
        block_mask = block_mask != 0
        unmasked_bins = block_sci[block_mask == 0]
        variance = biweight_midvariance(unmasked_bins)
        skyvar = variance / block_size**2 
        
        block_var_rnoise = block_reduce(var_rnoise, block_size)
        unmasked_bins = block_var_rnoise[block_mask == 0]
        mean = biweight_location(unmasked_bins)
        masked_mean_var_rnoise = mean / block_size**2 # because block_reduce sums by default

        correction_factor = skyvar / masked_mean_var_rnoise

        predicted_skyvar = correction_factor * var_rnoise
        
        model.var_rnoise = predicted_skyvar

        logger.info(f"Robust masked mean VAR_RDNOISE: {masked_mean_var_rnoise:.3e}")
        logger.info(f"Robust masked mean SKY_VARIANCE: {skyvar:.3e}")
        logger.info(f"Correction factor: {correction_factor:.2f}")
        logger.info(f"Fraction of pixels unmasked: {unmasked_frac*100:.1f}%")
                

        ### fix holes in variance maps, not sure if this is still necessary
        rnoise = model.var_rnoise
        poisson = model.var_poisson
        flat = model.var_flat

        w = np.where(rnoise == 0)
        rnoise[w] = np.inf

        w = np.where(poisson == 0)
        poisson[w] = np.inf

        w = np.where(flat == 0)
        flat[w] = np.inf

        model.var_rnoise = rnoise
        model.var_poisson = poisson
        model.flat = flat


        logger.info(f"Saving to {os.path.basename(cal_file)}")
        time = datetime.now()
        stepdescription = f"Rescaled variance {time.strftime('%Y-%m-%d %H:%M:%S')}"
        substr = stutil.create_history_entry(stepdescription)
        model.history.append(substr)
    
        model.save(cal_file)

    try:
        os.remove(cal_file.replace('_cal.fits', '_cal_bkgsub.fits'))
    except OSError:
        pass


def plot_cal_rate(cal_file):
    # cal_file_name = os.path.basename(cal_file)
    # logger.info(f'Making cal_rate plot for {cal_file_name}')
    # rate_file = cal_file.replace(config.stage2_product_path, config.stage1_product_path).replace('_cal.fits','_rate.fits')
    # output_file = cal_file.replace('_cal.fits','_cal_rate_fig.pdf')
    # utils.plot_two(rate_file, cal_file ,title1='Rate',title2='Cal (Background-subtracted)', save_file=output_file)
    
    outfile = cal_file.replace('_cal.fits','_cal.png')
    logger.info(outfile)
    im = fits.getdata(cal_file, 'SCI')
    dq = fits.getdata(cal_file, 'DQ') 
    mask = dq > 0 
    
    norm = ImageNormalize(im, interval=ZScaleInterval())

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True)
    ax.imshow(im, origin='lower', interpolation='none', cmap='Greys', norm=norm, zorder=-10)
    ax.imshow(np.ma.masked_where(~mask, im), interpolation='none', cmap='pink_r', norm=norm, zorder=0)
    ax.axis('off')

    fig.savefig(outfile, dpi=300)
    plt.close()

        


def create_diagonal_bins(image_shape, bin_width, theta):
    """
    Create diagonal bins across an image at angle theta.
    
    Parameters:
    -----------
    image_shape : tuple
        Shape of the image (height, width)
    bin_width : float
        Width of each diagonal bin in pixels
    theta : float
        Angle in degrees relative to the x-axis
    
    Returns:
    --------
    bin_indices : numpy.ndarray
        2D array where each pixel contains its bin index
    """
    theta = np.radians(theta)  # Convert angle to radians

    height, width = image_shape
    
    # Create coordinate grids
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Rotate coordinates: project onto axis perpendicular to diagonal direction
    # The perpendicular direction is at angle (theta + π/2)
    perpendicular_angle = theta + np.pi/2
    
    # Project coordinates onto the perpendicular axis
    # This gives us the distance from each pixel to the diagonal lines
    projected_coords = x * np.cos(perpendicular_angle) + y * np.sin(perpendicular_angle)
    
    # Find the range of projected coordinates to handle negative values
    min_proj = np.min(projected_coords)
    
    # Shift coordinates to make them all positive, then divide by bin_width
    bin_indices = ((projected_coords - min_proj) / bin_width).astype(int)
    
    return bin_indices

def get_pixels_in_bin(image, bin_indices, bin_number):
    """Get all pixels that belong to a specific bin"""
    mask = (bin_indices == bin_number)
    return image[mask]

def create_median_bin_image(image, bin_indices, sigma=3, maxiters=5, num_pixel_threshold=0):
    """
    Create an image where each pixel is replaced with the median value from its bin.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Original image array
    bin_indices : numpy.ndarray
        2D array of bin indices from create_diagonal_bins()
    
    Returns:
    --------
    median_image : numpy.ndarray
        Image where each pixel contains the median value from its diagonal bin
    """
    # Initialize output image with same shape and dtype as input
    median_image = np.zeros_like(image)
    
    # Get unique bin numbers
    unique_bins = np.unique(bin_indices)
    
    # Calculate median for each bin and assign to all pixels in that bin
    for bin_num in unique_bins:
        mask = (bin_indices == bin_num)
        bin_pixels = image[mask]
        if len(bin_pixels) < num_pixel_threshold:
            # print(f"Bin {bin_num} has fewer than {num_pixel_threshold} pixels, skipping median calculation.")
            continue
        median_value = sigma_clipped_stats(bin_pixels, mask=~np.isfinite(bin_pixels), maxiters=maxiters, sigma=sigma)[1]

        median_image[mask] = median_value
    
    return median_image


def fast_variance_objective(theta, image, bin_width):
    """
    Fast computation of variance for optimization.
    """

    theta = np.radians(theta)  # Convert angle to radians

    height, width = image.shape
    
    # Create coordinate grids
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Project coordinates
    perpendicular_angle = theta + np.pi/2
    projected_coords = x * np.cos(perpendicular_angle) + y * np.sin(perpendicular_angle)
    
    # Create bins
    min_proj = np.min(projected_coords)
    bin_indices = ((projected_coords - min_proj) / bin_width).astype(int)
    
    # Calculate variance efficiently
    total_variance = 0.0
    unique_bins = np.unique(bin_indices)
    
    for bin_num in unique_bins:
        mask = (bin_indices == bin_num)
        bin_pixels = image[mask]
        if len(bin_pixels) > 1:
            median_val = sigma_clipped_stats(bin_pixels, mask=~np.isfinite(bin_pixels), maxiters=5, sigma=3)[1]
            variance = calc_variance(bin_pixels, median_val, 1)
            total_variance += variance
    
    return total_variance

def remove_diagonal_striping(image):
    """    
    Subtract diagonal parallel striping features present in PRIMER UDS observations
    Implements a similar algorithm to the subtraction of 1/f noise, but uses diagonal 
    apertures rather than row-by-row subtraction

    """
    
    plot = config.stage2.remove_diagonal_striping_step.plot
    theta_min = config.stage2.remove_diagonal_striping_step.theta_min
    theta_max = config.stage2.remove_diagonal_striping_step.theta_max
    theta_step = config.stage2.remove_diagonal_striping_step.theta_step
    bin_width = config.stage2.remove_diagonal_striping_step.bin_width

    from jwst.datamodels import ImageModel
    model = ImageModel(image)
    # check that image has not already been corrected
    for entry in model.history:
        if 'Removed diagonal striping' in entry['description']:
            logger.info(f'{image} already corrected for diagonal striping patterns, exiting')
            return

    logger.info('Measuring diagonal striping')
    logger.info(f'Working on {image}')

    mask = np.zeros(model.data.shape, dtype=bool)
    mask[model.dq > 0] = True

    thetas = np.arange(theta_min, theta_max+theta_step, theta_step)
    variance = np.zeros_like(thetas)
    for i, theta_i in enumerate(thetas):
        logger.info(f'Testing {i+1}/{len(thetas)}: {theta_i:.2f} degrees')
        variance[i] = fast_variance_objective(theta_i, model.data, bin_width)

    min_variance = np.min(variance)
    theta = thetas[np.argmin(variance)]
    logger.info(f"Optimized angle: {theta:.2f} degrees, Variance: {min_variance:.2e}")

    bins = create_diagonal_bins(np.shape(model.data), bin_width, theta)
    med = create_median_bin_image(model.data, bins)

    model.close()
    
    image_orig = image.replace('.fits', '_before_diag_sub.fits')
    logger.info(f"Copying input to {image_orig}")
    shutil.copy2(image, image_orig)

    # remove striping from science image
    with ImageModel(image) as immodel:
        sci = immodel.data
        # to replace zeros
        wzero = np.where(sci == 0)
        outsci = sci - med
        outsci[wzero] = 0
        
        # write output
        immodel.data = outsci
        # add history entry
        time = datetime.now()
        stepdescription = f"Removed diagonal striping; {time.strftime('%Y-%m-%d %H:%M:%S')}"
        from stdatamodels import util as stutil
        substr = stutil.create_history_entry(stepdescription)
        immodel.history.append(substr)
        logger.info(f'Saving cleaned image to {image}')
        immodel.save(image)

    if plot:
        logger.info(f'Making diagonal striping removal plots')

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(thetas, variance, marker='o')
        plt.gca().axvline(theta, color='r', linestyle='--')
        plt.xlabel('Theta (degrees)')
        plt.ylabel('Variance')
        output_file = image.replace('_cal.fits', '_diag_striping_variance.pdf')
        plt.savefig(output_file)
        plt.close()
        logger.info(f'Saved plot to {output_file}')

        image_name = os.path.basename(image)
        image_orig  = image.replace('_cal.fits', '_cal_before_diag_sub.fits')
        output_file = image.replace('_cal.fits','_diag_striping.pdf')
        utils.plot_three(image_orig, med, image , title1='Original', title2='Stripes', title3='Stripes removed', scaling=3, save_file=output_file)
        logger.info(f'Saved plot to {output_file}')

