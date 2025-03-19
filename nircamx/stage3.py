from . import utils
import sys, os, glob, tqdm
from datetime import datetime

import numpy as np
import numpy.ma as ma
import shutil
from scipy.stats.mstats import trim
from scipy.optimize import curve_fit
from shapely.geometry import Polygon

from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from jhat import align_wcs_batch # jwst_photclass


logger = utils.setup_logger()
config = None


def fit_pedestal(data):
    """Fit distribution of sky fluxes with a Gaussian. Returns simple mean of Gaussian distribution."""
    std = sigma_clipped_stats(data)[2]
    bins = np.linspace(-10*std, 10*std, 500)
    h, b = np.histogram(data, bins=bins)
    h = h / np.max(h)
    bc = 0.5 * (b[1:] + b[:-1])
    binsize = b[1] - b[0]

    p0 = [1, bc[np.argmax(h)], std]
    popt,pcov = curve_fit(utils.Gaussian, bc, h, p0=p0)

    return popt[1]

def jhat_step(cal_file, filtname=None):
    assert filtname is not None
    
    verbose = config.stage3.jhat_step.verbose
    debug = config.stage3.jhat_step.debug

    input_dir = os.path.dirname(cal_file)
    cal_file_name = os.path.basename(cal_file)

    # if config.stage3.files_to_skip is not None:
    #     for file_to_skip in config.stage3.files_to_skip:
    #         f = [os.path.basename(p) for p in glob.glob(os.path.join(input_dir, file_to_skip))]
    #         if cal_file_name in f:
    #             logger.warning(f'Skipping stage3 for {cal_file_name}')
    #             return 

    input_files = [cal_file_name.replace('_cal.fits', '*_cal.fits')]

    outrootdir = config.stage3_product_path
    outsubdir = filtname

    refcat = os.path.join(config.refcat_path, config.stage3.jhat_step.refcat_dict[filtname])

    align_batch = align_wcs_batch()
    align_batch.verbose = config.stage3.jhat_step.verbose
    align_batch.debug = config.stage3.jhat_step.debug
    align_batch.sip_err = config.stage3.jhat_step.sip_err
    align_batch.replace_sip = True
    align_batch.sip_degree = 3
    align_batch.sip_points = 128
    align_batch.rough_cut_px_min = config.stage3.jhat_step.rough_cut_px_min
    align_batch.rough_cut_px_max = config.stage3.jhat_step.rough_cut_px_max
    align_batch.d_rotated_Nsigma = config.stage3.jhat_step.d_rotated_Nsigma

    # get the input files
    align_batch.get_input_files(input_files, directory=input_dir, detectors=None, filters=None, pupils=None)


    ixs_all = align_batch.getindices()

    if len(ixs_all)==0:
        logger.error('JHAT: No images found! exiting...')
        sys.exit(0)
        
    # get the output filenames
    ixs_exists,ixs_notexists = align_batch.get_output_filenames(ixs=ixs_all,
                                                                outrootdir=outrootdir,
                                                                outsubdir=outsubdir,
                                                                addfilter2outsubdir=False)    

    ixs_todo = ixs_notexists[:]
    
    if len(ixs_exists)>0:
        if config.stage3.jhat_step.overwrite:
            ixs_todo.extend(ixs_exists) 
            logger.info(f'{len(ixs_exists)} output images already exist, overwriting them since overwrite=True')
        else:
            logger.info(f'{len(ixs_exists)} output images already exist, skipping since overwrite=False')
        
    if len(ixs_todo)==0:
        # logger.error(f'There are {len(ixs_all)} images, but none of them need to be done!')
        return

        
    logger.info(f'Output directory:{os.path.dirname(align_batch.t.loc[ixs_todo[0],"outfilename"])}')
    try:
        align_batch.align_wcs(ixs_todo,
                            overwrite = config.stage3.jhat_step.overwrite,
                            outrootdir= outrootdir,
                            outsubdir = outsubdir,
                            addfilter2outsubdir = False,
                            photometry_method = 'aperture',
                            find_stars_threshold    = 3.0,
                            sci_xy_catalog          = None,
                            use_dq                  = False,
                            refcatname              = refcat,
                            refcat_racol            = 'RA',
                            refcat_deccol           = 'DEC',
                            refcat_magcol           = 'mag',
                            refcat_magerrcol        = 'mag_err',
                            refcat_colorcol         = None,
                            pmflag                  = False,
                            pm_median               = False,
                            load_photcat_if_exists  = False,
                            rematch_refcat          = False,
                            SNR_min                 = 10.0,
                            d2d_max                 = config.stage3.jhat_step.d2d_max, # maximum distance refcat to source in image
                            dmag_max                = 0.1, # maximum uncertainty of source 
                            sharpness_lim           = (None, None), # sharpness limits
                            roundness1_lim          = (None, None), # roundness1 limits 
                            delta_mag_lim           = config.stage3.jhat_step.delta_mag_lim, # limits on mag-refcat_mainfilter
                            objmag_lim              = config.stage3.jhat_step.objmag_lim, # limits on mag, the magnitude of the objects in the image
                            refmag_lim              = (None, None), # limits on refcat_mainfilter, the magnitude of the reference catalog
                            slope_min               = -10/2048.0,
                            Nbright4match           = None, # Use only the the brightest  Nbright sources from image for the matching with the ref catalog
                            Nbright                 = None, # U/se only the brightest Nbright sources from image
                            histocut_order          = config.stage3.jhat_step.histocut_order, # histocut_order defines whether the histogram cut is first done for dx or first for dy
                            xshift                  = 0.0, # added to the x coordinate before calculating ra,dec. This can be used to correct for large shifts before matching!
                            yshift                  = 0.0, # added to the y coordinate before calculating ra,dec. This can be used to correct for large shifts before matching!
                            iterate_with_xyshifts  = config.stage3.jhat_step.iterate_with_xyshifts, # After the first histogram fit, redo the match with refcat with x/yshift=median(dx/dy) and redo histofit. 
                                                                            # Use this if the offsets are big, since the second iteration will give you better matching with the refcat 
                            showplots = 0,
                            saveplots = config.stage3.jhat_step.saveplots, 
                            savephottable = config.stage3.jhat_step.savephottable
                            )
    except:
        logger.error(f'failed on {cal_file}')
        raise

    align_batch.write()


from astropy.stats import SigmaClip
from photutils.background import (Background2D, BiweightLocationBackground,
                                  BkgIDWInterpolator, BkgZoomInterpolator,
                                  MedianBackground, SExtractorBackground)
from .stage1 import produce_mask, dilate_mask, SourceMask


def stack_dq_by_detector(filtname):
    from jwst.datamodels import ImageModel
    threshold = config.stage3.bad_pixel_step.threshold
    overwrite = config.stage3.bad_pixel_step.overwrite

    if filtname in utils.sw_filters:
        files = [f'fl_pixels_{filtname}_nrca1.fits', f'fl_pixels_{filtname}_nrca2.fits',
                 f'fl_pixels_{filtname}_nrca3.fits', f'fl_pixels_{filtname}_nrca4.fits',
                 f'fl_pixels_{filtname}_nrcb1.fits', f'fl_pixels_{filtname}_nrcb2.fits',
                 f'fl_pixels_{filtname}_nrcb3.fits', f'fl_pixels_{filtname}_nrcb4.fits']
        files = [os.path.join(config.bad_pixel_path, f) for f in files]
        if utils.check_files_exist(files) and not overwrite:
            logger.info(f'Bad pixel masks for {filtname} already exist at {config.bad_pixel_path}/, skipping...')
            return
    elif filtname in utils.lw_filters:
        files = [f'fl_pixels_{filtname}_nrcalong.fits', f'fl_pixels_{filtname}_nrcblong.fits']
        files = [os.path.join(config.bad_pixel_path, f) for f in files]
        if utils.check_files_exist(files) and not overwrite:
            logger.info(f'Bad pixel masks for {filtname} already exist at {config.bad_pixel_path}/, skipping...')
            return
    
    logger.info(f'Building bad pixel masks for {filtname}...')

    cal_files = glob.glob(os.path.join(config.stage2_product_path, filtname, '*_cal.fits'))

    if filtname in utils.sw_filters:
        a1 = np.zeros((2048,2048))
        a2 = np.zeros((2048,2048))
        a3 = np.zeros((2048,2048))
        a4 = np.zeros((2048,2048))
        b1 = np.zeros((2048,2048))
        b2 = np.zeros((2048,2048))
        b3 = np.zeros((2048,2048))
        b4 = np.zeros((2048,2048))

        with tqdm.tqdm(total=len(cal_files)) as pbar:
            for cal_file in cal_files:
                flag = fits.getdata(cal_file, extname='DQ')
                flag[flag >= 1] = 1
                
                if 'nrca1' in cal_file:
                    a1 += flag
                if 'nrca2' in cal_file:
                    a2 += flag
                if 'nrca3' in cal_file:
                    a3 += flag
                if 'nrca4' in cal_file:
                    a4 += flag
                if 'nrcb1' in cal_file:
                    b1 += flag
                if 'nrcb2' in cal_file:
                    b2 += flag
                if 'nrcb3' in cal_file:
                    b3 += flag
                if 'nrcb4' in cal_file:
                    b4 += flag

                pbar.update(1)

        fits.writeto(os.path.join(config.bad_pixel_path, f'stack_dq_{filtname}_nrca1.fits'), a1, overwrite=True)
        fits.writeto(os.path.join(config.bad_pixel_path, f'stack_dq_{filtname}_nrca2.fits'), a2, overwrite=True)
        fits.writeto(os.path.join(config.bad_pixel_path, f'stack_dq_{filtname}_nrca3.fits'), a3, overwrite=True)
        fits.writeto(os.path.join(config.bad_pixel_path, f'stack_dq_{filtname}_nrca4.fits'), a4, overwrite=True)
        fits.writeto(os.path.join(config.bad_pixel_path, f'stack_dq_{filtname}_nrcb1.fits'), b1, overwrite=True)
        fits.writeto(os.path.join(config.bad_pixel_path, f'stack_dq_{filtname}_nrcb2.fits'), b2, overwrite=True)
        fits.writeto(os.path.join(config.bad_pixel_path, f'stack_dq_{filtname}_nrcb3.fits'), b3, overwrite=True)
        fits.writeto(os.path.join(config.bad_pixel_path, f'stack_dq_{filtname}_nrcb4.fits'), b4, overwrite=True)

        a1 = a1/np.max(a1)
        a2 = a2/np.max(a2)
        a3 = a3/np.max(a3)
        a4 = a4/np.max(a4)
        b1 = b1/np.max(b1)
        b2 = b2/np.max(b2)
        b3 = b3/np.max(b3)
        b4 = b4/np.max(b4)

        
        a1[a1 > threshold]  = 1
        a2[a2 > threshold]  = 1
        a3[a3 > threshold]  = 1
        a4[a4 > threshold]  = 1
        b1[b1 > threshold]  = 1
        b2[b2 > threshold]  = 1
        b3[b3 > threshold]  = 1
        b4[b4 > threshold]  = 1

        a1[a1 <= threshold] = 0
        a2[a2 <= threshold] = 0
        a3[a3 <= threshold] = 0
        a4[a4 <= threshold] = 0
        b1[b1 <= threshold] = 0
        b2[b2 <= threshold] = 0
        b3[b3 <= threshold] = 0
        b4[b4 <= threshold] = 0

        fits.writeto(os.path.join(config.bad_pixel_path, f'fl_pixels_{filtname}_nrca1.fits'), a1, overwrite=True)
        fits.writeto(os.path.join(config.bad_pixel_path, f'fl_pixels_{filtname}_nrca2.fits'), a2, overwrite=True)
        fits.writeto(os.path.join(config.bad_pixel_path, f'fl_pixels_{filtname}_nrca3.fits'), a3, overwrite=True)
        fits.writeto(os.path.join(config.bad_pixel_path, f'fl_pixels_{filtname}_nrca4.fits'), a4, overwrite=True)
        fits.writeto(os.path.join(config.bad_pixel_path, f'fl_pixels_{filtname}_nrcb1.fits'), b1, overwrite=True)
        fits.writeto(os.path.join(config.bad_pixel_path, f'fl_pixels_{filtname}_nrcb2.fits'), b2, overwrite=True)
        fits.writeto(os.path.join(config.bad_pixel_path, f'fl_pixels_{filtname}_nrcb3.fits'), b3, overwrite=True)
        fits.writeto(os.path.join(config.bad_pixel_path, f'fl_pixels_{filtname}_nrcb4.fits'), b4, overwrite=True)



    if filtname in utils.lw_filters:
        a = np.zeros((2048,2048))
        b = np.zeros((2048,2048))

        with tqdm.tqdm(total=len(cal_files)) as pbar:
            for cal_file in cal_files:
                flag = fits.getdata(cal_file, extname='DQ')
                flag[flag >= 1] = 1

                if 'nrcalong' in cal_file:
                    a += flag
                if 'nrcblong' in cal_file:
                    b += flag

                pbar.update(1)

        fits.writeto(os.path.join(config.bad_pixel_path, f'stack_dq_{filtname}_nrcalong.fits'), a, overwrite=True)
        fits.writeto(os.path.join(config.bad_pixel_path, f'stack_dq_{filtname}_nrcblong.fits'), b, overwrite=True)

        a = a/np.max(a)
        b = b/np.max(b)

        a[a > threshold] = 1
        b[b > threshold] = 1

        a[a <= threshold] = 0
        b[b <= threshold] = 0

        fits.writeto(os.path.join(config.bad_pixel_path, f'fl_pixels_{filtname}_nrcalong.fits'), a, overwrite=True)
        fits.writeto(os.path.join(config.bad_pixel_path, f'fl_pixels_{filtname}_nrcblong.fits'), b, overwrite=True)




def remove_bad_pixels(jhat_file, filtname=None):
    from stdatamodels import util as stutil
    from jwst.datamodels import ImageModel

    model = ImageModel(jhat_file)
    # check that image has not already been flagged
    for entry in model.history:
        if 'Masked bad pixels' in entry['description']:
            logger.info(f'DQ mask already udpated for {os.path.basename(jhat_file)}, skipping...')
            return

    logger.info(f'Updating DQ mask given bad pixel mask for {os.path.basename(jhat_file)}...')
    
    if 'nrca1' in jhat_file: detector = 'nrca1'
    if 'nrca2' in jhat_file: detector = 'nrca2'
    if 'nrca3' in jhat_file: detector = 'nrca3'
    if 'nrca4' in jhat_file: detector = 'nrca4'
    if 'nrcb1' in jhat_file: detector = 'nrcb1'
    if 'nrcb2' in jhat_file: detector = 'nrcb2'
    if 'nrcb3' in jhat_file: detector = 'nrcb3'
    if 'nrcb4' in jhat_file: detector = 'nrcb4'
    if 'nrcalong' in jhat_file: detector = 'nrcalong'
    if 'nrcblong' in jhat_file: detector = 'nrcblong'

    fl_file = os.path.join(config.bad_pixel_path, f'fl_pixels_{filtname}_{detector}.fits')
    fl = fits.getdata(fl_file).astype(bool)

    model.dq[fl] |= 1

    # add history entry
    time = datetime.now()
    stepdescription = f"Masked bad pixels; {time.strftime('%Y-%m-%d %H:%M:%S')}"
    substr = stutil.create_history_entry(stepdescription)
    model.history.append(substr)

    model.save(jhat_file)





def skymatch_step(jhat_files, filtname):
    from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
    from jwst.associations import asn_from_list

    visit_list = []
    for jhat_file in jhat_files:
        visit = os.path.basename(jhat_file).split('_')[0]
        if visit not in visit_list:
            visit_list.append(visit)

    for i,visit in enumerate(visit_list):
        logger.info(f'Running skymatch_step on visit {visit} ({i+1}/{len(visit_list)})...')
        visit_imgfile_list = sorted(glob.glob(os.path.join(config.stage3_product_path,filtname,f'{visit}*_jhat.fits')))
        asn_file = os.path.join(config.stage3_product_path, filtname, f'sky_{visit}_asn.json')
        asn = asn_from_list.asn_from_list(visit_imgfile_list, rule=DMS_Level3_Base, product_name='skymatch_files')
        with open(asn_file, 'w') as outfile:
            name, serialized = asn.dump(format='json')
            outfile.write(serialized)

        params = {'assign_mtwcs':      {'skip': True},
                  'tweakreg':          {'skip': True},
                  'skymatch':          {'skymethod'  : config.stage3.skymatch_step.skymethod,
                                        'match_down' : config.stage3.skymatch_step.match_down,
                                        'subtract'   : config.stage3.skymatch_step.subtract,
                                        'stepsize'   : config.stage3.skymatch_step.stepsize,
                                        'skystat'    : config.stage3.skymatch_step.skystat,
                                        'dqbits'     : config.stage3.skymatch_step.dqbits,
                                        'lower'      : config.stage3.skymatch_step.lower,
                                        'upper'      : config.stage3.skymatch_step.upper,
                                        'nclip'      : config.stage3.skymatch_step.nclip,
                                        'binwidth'   : config.stage3.skymatch_step.binwidth},
                  'outlier_detection': {'skip': True},
                  'resample':          {'skip': True},
                  'source_catalog':    {'skip': True}}
        if params['skymatch']['stepsize'] == 'none':
            params['skymatch']['stepsize'] = None
        if params['skymatch']['lower'] == 'none':
            params['skymatch']['lower'] = None
        if params['skymatch']['upper'] == 'none':
            params['skymatch']['upper'] = None
            
        from jwst.pipeline import calwebb_image3
        output = calwebb_image3.Image3Pipeline.call(
            asn_file, 
            output_dir = os.path.join(config.stage3_product_path,filtname), 
            steps=params, 
            save_results=True)


# max_radius = 20
def outlier_step_prep(jhat_files):
    '''Identifies groups of visits that overlap on the sky, and should be used 
       in conjunction for outlier detection'''

    from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
    from jwst.associations import asn_from_list
    
    max_radius = config.stage3.outlier_step.max_radius
    overwrite = config.stage3.outlier_step.overwrite

    visit_list = [] # list of all unique visit/sca (sensor chip assembly) combinations

    for jhat_file in jhat_files:
        visit = os.path.basename(jhat_file).split('_')[0]
        if visit not in visit_list:
            visit_list.append(visit)


    for i,visit in enumerate(visit_list):
        
        # list of files that correspond to this visit
        visit_imgfile_list = [f for f in jhat_files if visit in f]
        base_dir = os.path.dirname(visit_imgfile_list[0])
        asn_file = os.path.join(base_dir, f'outlier_detection_{visit}_asn.json')

        if os.path.exists(asn_file) and not overwrite:
            logger.info(f'Outlier asn file {os.path.basename(asn_file)} already exists, skipping generation')
            continue

        logger.info(f'Generating outlier asn file {os.path.basename(asn_file)} ({i+1}/{len(visit_list)})...')

        # additional files to include, might not be the same visit, but overlap on-sky
        addnl_visit_imgfile_list = [] 

        ra = []
        dec =  []
        for file in tqdm.tqdm(visit_imgfile_list):
            s_region = fits.getheader(file, extname='SCI')['S_REGION']
            ra += [float(s) for s in s_region.split()[2::2]]
            dec += [float(s) for s in s_region.split()[3::2]]

        coords  = SkyCoord(ra=ra, dec=dec, unit='deg')  
        

        for new_file in tqdm.tqdm(jhat_files):
            if new_file not in visit_imgfile_list:
                s_region = fits.getheader(new_file, extname='SCI')['S_REGION']
                ra_new = [float(s) for s in s_region.split()[2::2]]
                dec_new = [float(s) for s in s_region.split()[3::2]]

                coords_new  = SkyCoord(ra=ra_new, dec=dec_new, unit='deg')  
                    
                idx_1, idx_2, d2d, d3d  = coords_new.search_around_sky(coords, max_radius * u.arcsec)
                if np.size(idx_1)>0:
                    addnl_visit_imgfile_list.append(new_file)
    
        visit_imgfile_list += addnl_visit_imgfile_list

        asn = asn_from_list.asn_from_list(visit_imgfile_list, rule=DMS_Level3_Base, product_name='outlier_files')
        
        with open(asn_file, 'w') as outfile:
            name, serialized = asn.dump(format='json')
            outfile.write(serialized)




def outlier_step(asn_file, filtname):

    visit = os.path.basename(asn_file).split('_')[2]
    
    visit_path = os.path.join(config.stage3_product_path,filtname,visit)
    if not os.path.exists(visit_path):
        os.mkdir(visit_path)
    
    outlier_path = os.path.join(config.stage3_product_path,filtname,visit,'outliers')
    if not os.path.exists(outlier_path):
        os.mkdir(outlier_path)

    jhat_files = glob.glob(os.path.join(config.stage3_product_path,filtname,f'{visit}*_jhat.fits'))
    crf_files = glob.glob(os.path.join(config.stage3_product_path,filtname,f'{visit}*_crf.fits'))
    if (len(jhat_files) == len(crf_files)) and not config.stage3.outlier_step.overwrite:
        logger.info(f'All .crf files for visit {visit} already exist, skipping...')
        return
    
    params = {'assign_mtwcs':      {'skip': True},
              'tweakreg':          {'skip': True},
              'skymatch':          {'skip': True},
              'resample':          {'skip': True},
              'source_catalog':    {'skip': True},
              'outlier_detection': {'weight_type'              : config.stage3.outlier_step.weight_type,
                                    'pixfrac'                  : config.stage3.outlier_step.pixfrac,
                                    'kernel'                   : config.stage3.outlier_step.kernel,
                                    'fillval'                  : config.stage3.outlier_step.fillval,
                                    'maskpt'                   : config.stage3.outlier_step.maskpt,
                                    'snr'                      : config.stage3.outlier_step.snr,
                                    'scale'                    : config.stage3.outlier_step.scale,
                                    'backg'                    : config.stage3.outlier_step.backg,
                                    'resample_data'            : config.stage3.outlier_step.resample_data,
                                    'good_bits'                : config.stage3.outlier_step.good_bits,
                                    'save_intermediate_results': True, 'save_results': True}}
    
    from jwst.pipeline import calwebb_image3
    output = calwebb_image3.Image3Pipeline.call(asn_file, 
        output_dir = outlier_path, steps=params, save_results=True)

    crf_files = glob.glob(os.path.join(outlier_path, f'{visit}*_crf.fits'))
    for input_file in crf_files:
        output_file = os.path.join(config.stage3_product_path, filtname, os.path.basename(input_file))
        shutil.move(input_file, output_file)






def resample_step(filtname):

    from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
    from jwst.associations import asn_from_list

    
    imgfile_list = utils.get_crf_files(filtname)

    pixel_scale = config.stage3.resample_step.pixel_scale
    if isinstance(pixel_scale, str):
        assert pixel_scale.endswith('mas')
        pixel_scale_str = str(pixel_scale)
        pixel_scale = float(pixel_scale_str[:-3])/1000
    elif isinstance(pixel_scale, (float, int)):
        if pixel_scale > 1: # assumed given in mas
            pixel_scale_str = f'{str(int(pixel_scale))}mas'
            pixel_scale = float(pixel_scale)/1000
        else: # assumed given in arcsec
            pixel_scale_str = f'{str(int(pixel_scale*1000))}mas'
            pixel_scale = float(pixel_scale)

    mode = config.stage3.resample_step.mode # `tile` or `indiv`
    if mode == 'tile':
        tile = config.stage3.resample_step.tile
        version = config.stage3.resample_step.version
        logger.info(f'Running resample_step for tile {tile}, {filtname}, {pixel_scale_str}')

        mosaic_name = config.stage3.resample_step.mosaic_name
        mosaic_name = mosaic_name.replace('[filter]', filtname)
        mosaic_name = mosaic_name.replace('[field_name]', config.field_name)
        mosaic_name = mosaic_name.replace('[pixel_scale]', pixel_scale_str)
        mosaic_name = mosaic_name.replace('[version]', version)
        mosaic_name = mosaic_name.replace('[tile]', tile)
        mosaic_outdir = os.path.join(config.mosaic_path, filtname)
        logger.info(f'Output will go to {mosaic_outdir}/{mosaic_name}_i2d.fits')

        ### select the files that overlap the tile we want to drizzle
        from .ref import tile_corners
        tile_polygon = Polygon(tile_corners[tile])
        
        selected_files = []
        for file in imgfile_list:
            coords_rect = np.zeros((4,2))
            hdulist = fits.open(file, ignore_missing_simple=True)
            wcs = WCS(hdulist[1].header, naxis=2)
            pixcoords = np.array([[0., 0.], [2048., 0.], [2048., 2048.], [0., 2048.]])
            worldcoords = wcs.wcs_pix2world(pixcoords, 0)
            aa = 0
            for coords in worldcoords:
                coords_rect[aa,0] = coords[0]
                coords_rect[aa,1] = coords[1]
                aa += 1

            file_polygon = Polygon(coords_rect)

            if tile_polygon.intersects(file_polygon):
                selected_files.append(file)
                        
        logger.info(f'Preparing to drizzle+combine {len(selected_files)} images')

        asn_file = os.path.join(config.stage3_product_path,filtname,f'{mosaic_name}_asn.json')
        asn = asn_from_list.asn_from_list(selected_files, rule=DMS_Level3_Base, product_name=mosaic_name)
        with open(asn_file, 'w') as outfile:
            name, serialized = asn.dump(format='json')
            outfile.write(serialized)
        
        from .ref import cosmos_tangent_point
        if pixel_scale_str == '30mas':
            from .ref import tile_crpix_30mas
            crpix_0, crpix_1 = tile_crpix_30mas[tile]
            from .ref import tile_output_shape_30mas as output_shape
        elif pixel_scale_str == '60mas':
            from .ref import tile_crpix_60mas
            crpix_0, crpix_1 = tile_crpix_60mas[tile]
            from .ref import tile_output_shape_60mas as output_shape
        
        params = {'assign_mtwcs':      {'skip': True},
                  'tweakreg':          {'skip': True},
                  'skymatch':          {'skip': True},
                  'outlier_detection': {'skip': True},
                  'resample':          {'pixfrac'      : config.stage3.resample_step.pixfrac,
                                        'kernel'       : config.stage3.resample_step.kernel,
                                        'pixel_scale'  : pixel_scale,
                                        'rotation'     : 20,
                                        'output_shape' : output_shape,
                                        'crpix'        : [crpix_0,crpix_1],
                                        'crval'        : [cosmos_tangent_point[0],cosmos_tangent_point[1]],
                                        'fillval'      :'indef',
                                        'weight_type'  :'ivm',
                                        'single'       : False,
                                        'blendheaders' : True,
                                        'save_results' : True},
                  'source_catalog':    {'skip': True}}
        
        from jwst.pipeline import calwebb_image3
        output = calwebb_image3.Image3Pipeline.call(asn_file, 
            output_dir = mosaic_outdir, 
            steps = params, 
            save_results = True)
        
    else:
        raise Exception('only mode=tile supported atm')

    if config.stage3.resample_step.background_subtract:
        from .bkgsub import SubtractBackground
        
        mosaic_file = os.path.join(mosaic_outdir, f'{mosaic_name}_i2d.fits')

        bkg = SubtractBackground(
            ring_radius_in = config.stage3.resample_step.ring_radius_in,
            ring_width = config.stage3.resample_step.ring_width,
            ring_clip_max_sigma = config.stage3.resample_step.ring_clip_max_sigma,
            ring_clip_box_size = config.stage3.resample_step.ring_clip_box_size,
            ring_clip_filter_size = config.stage3.resample_step.ring_clip_filter_size,
            tier_kernel_size = config.stage3.resample_step.tier_kernel_size,
            tier_npixels = config.stage3.resample_step.tier_npixels,
            tier_nsigma = config.stage3.resample_step.tier_nsigma,
            tier_dilate_size = config.stage3.resample_step.tier_dilate_size,
            bg_box_size = config.stage3.resample_step.bg_box_size,
            bg_filter_size = config.stage3.resample_step.bg_filter_size,
            bg_exclude_percentile = config.stage2.bkgsub_var_step.bg_exclude_percentile,
            bg_sigma = config.stage3.resample_step.bg_sigma,
            bg_interpolator = config.stage3.resample_step.bg_interpolator,
            suffix = 'bkgsub',
            replace_sci = True,
        )

        bkg.call(mosaic_file)

        mosaic_file_orig = mosaic_file.replace('_i2d.fits', '_i2d_before_bkgsub.fits')
        logger.info(f"Copying input to {os.path.basename(mosaic_file_orig)}")
        shutil.copy2(mosaic_file, mosaic_file_orig)

        logger.info(f"Renaming {os.path.basename(bkg.outfile)} to {os.path.basename(mosaic_file)}")
        shutil.move(bkg.outfile, mosaic_file)
    
    if config.stage3.resample_step.split_extensions:
        logger.info('Splitting extensions')
        
        mosaic_file = os.path.join(mosaic_outdir, f'{mosaic_name}_i2d.fits')
        
        sci = fits.getdata(mosaic_file, extname='SCI')
        hdr = fits.getheader(mosaic_file, extname='SCI')
        err = fits.getdata(mosaic_file, extname='ERR')
        wht = fits.getdata(mosaic_file, extname='WHT')

        ext_outdir = os.path.join(mosaic_outdir, 'extensions')
        if not os.path.exists(ext_outdir):
            os.mkdir(ext_outdir)


        hdu = fits.PrimaryHDU(data=sci, header=hdr)
        hdu.writeto(os.path.join(ext_outdir, os.path.basename(mosaic_file).replace('_i2d.fits','_sci.fits')))
        
        hdr.update({'EXTNAME':'ERR'})
        hdu = fits.PrimaryHDU(data=err, header=hdr)
        hdu.writeto(os.path.join(ext_outdir, os.path.basename(mosaic_file).replace('_i2d.fits','_err.fits')))
        
        hdr.update({'EXTNAME':'WHT'})
        hdu = fits.PrimaryHDU(data=wht, header=hdr)
        hdu.writeto(os.path.join(ext_outdir, os.path.basename(mosaic_file).replace('_i2d.fits','_wht.fits')))

        has_srcmask = True
        try:
            srcmask = fits.getdata(mosaic_file, extname='SRCMASK')
        except:
            logger.info(f'{mosaic_name} has no extension SRCMASK')
            has_srcmask = False
        
        if has_srcmask:
            hdr.update({'EXTNAME':'SRCMASK'})
            hdu = fits.PrimaryHDU(data=srcmask, header=hdr)
            hdu.writeto(os.path.join(ext_outdir, os.path.basename(mosaic_file).replace('_i2d.fits','_srcmask.fits')))
