import os
import multiprocess as mp
from functools import partial
from . import utils
import argparse

###############################################################################################

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()
    # TODO implement arguments overrides to certain config options?
    
    config_file = args.config
    if not os.path.exists(config_file):
        raise Exception(f'config file {config_file} not found')

    config = utils.parse_config_file(config_file)
    utils.config = config
    
    if config.stage1.run:
        from . import stage1
        stage1.config = config

        if config.stage1.detector1_step.run:
            
            from .stage1 import detector1_step
            # import things from .stage1.py

            # if `products` does not exist, make it
            if not os.path.exists(config.product_path):
                os.mkdir(config.product_path)

            # if `products/pipeline_level1` does not exist, make it
            if not os.path.exists(config.stage1_product_path):
                os.mkdir(config.stage1_product_path)

            for filtname in config.filters:
                uncal_files = utils.get_uncal_files(filtname)
                
                # if `products/pipeline_level1/f444w` does not exist, make it
                if not os.path.exists(os.path.join(config.stage1_product_path,filtname)):
                    os.mkdir(os.path.join(config.stage1_product_path,filtname))

                with mp.Pool(utils.n_procs) as pool:
                    print(pool.map(detector1_step, uncal_files)) 

        if config.stage1.remove_snowball_step.run:
            from .stage1 import remove_snowballs
            
            for filtname in config.filters:
                rate_files = utils.get_rate_files(filtname)

                with mp.Pool(utils.n_procs) as pool:
                    print(pool.map(remove_snowballs, rate_files))
        
        # TODO updated wisp templates created by JADES team -- compare to Max's templates in F115W and F150W
        if config.stage1.remove_wisp_step.run:
            from .stage1 import remove_wisps

            for filtname in config.filters:
                rate_files = utils.get_rate_files(filtname)
                
                with mp.Pool(utils.n_procs) as pool:
                    print(pool.map(remove_wisps, rate_files))
        
        if config.stage1.remove_striping_step.run:
            from .stage1 import remove_striping
            for filtname in config.filters:
                rate_files = utils.get_rate_files(filtname)

                if config.stage1.remove_striping_step.pool:
                    with mp.Pool(utils.n_procs) as pool:
                        print(pool.map(remove_striping, rate_files))
                else:
                    for rate_file in rate_files:
                        remove_striping(rate_file)
                        
        if config.stage1.persitence_step.run:
            from .stage1 import persistence_step
            
            for filtname in config.filters:
                rate_files = utils.get_rate_files(filtname)
                persistence_step(rate_files)
                

    ############################################################################################################################################################
    ############################################################################################################################################################
    ############################################################################################################################################################
    if config.stage2.run:
        from . import stage2
        stage2.config = config

        if config.stage2.image2_step.run:
            from .stage2 import image2_step

            # if `products` does not exist, make it
            if not os.path.exists(config.product_path):
                os.mkdir(config.product_path)

            # if `products/pipeline_level2` does not exist, make it
            if not os.path.exists(config.stage2_product_path):
                os.mkdir(config.stage2_product_path)

            for filtname in config.filters:
                rate_files = utils.get_rate_files(filtname)

                # if `products/pipeline_level2/f444w` doesn't exist, make it
                if not os.path.exists(os.path.join(config.stage2_product_path,filtname)):
                    os.mkdir(os.path.join(config.stage2_product_path,filtname))

                if config.stage2.image2_step.pool:
                    with mp.Pool(utils.n_procs) as pool:
                        print(pool.map(image2_step, rate_files))                     

                else:
                    for rate_file in rate_files:
                        image2_step(rate_file)
                    
        if config.stage2.remove_edge_step.run:
            from .stage2 import remove_edge
            for filtname in config.filters:
                cal_files = utils.get_cal_files(filtname)
                with mp.Pool(utils.n_procs) as pool:
                    print(pool.map(remove_edge, cal_files))


        if config.stage2.remove_claw_step.run:
            from .stage2 import remove_claws

            for filtname in config.filters:
                cal_files = utils.get_cal_files(filtname)
                
                with mp.Pool(utils.n_procs) as pool:
                    print(pool.map(remove_claws, cal_files))

        
        # variance_step
        if config.stage2.bkgsub_var_step.run:
            from .stage2 import background_subtraction_variance_scaling
            for filtname in config.filters: 
                cal_files = utils.get_cal_files(filtname)

                if config.stage2.bkgsub_var_step.pool:
                    with mp.Pool(utils.n_procs) as pool:
                        print(pool.map(background_subtraction_variance_scaling, cal_files))
                else:
                    for cal_file in cal_files:
                        background_subtraction_variance_scaling(cal_file)


    if config.stage3.run:
        from . import stage3
        stage3.config = config

        if not os.path.exists(config.stage3_product_path):
            os.mkdir(config.stage3_product_path)

        if config.stage3.jhat_step.run:
            from .stage3 import jhat_step
            for filtname in config.filters:
                if not os.path.exists(os.path.join(config.stage3_product_path, filtname)):
                    os.mkdir(os.path.join(config.stage3_product_path, filtname))

                cal_files = utils.get_cal_files(filtname)
                if config.stage3.jhat_step.pool:
                    file_patterns = [cal_file.replace('_cal.fits', '*_cal.fits') for cal_file in cal_files]
                    with mp.Pool(utils.n_procs) as pool:
                        print(pool.map(partial(jhat_step, filtname=filtname), file_patterns))
                else:
                    for cal_file in cal_files:
                        jhat_step(cal_file, filtname=filtname)

        
        # remove_bad_pixels
        if config.stage3.bad_pixel_step.run:
            from .stage3 import remove_bad_pixels, stack_dq_by_detector
            for filtname in config.filters:
                stack_dq_by_detector(filtname) 

                jhat_files = utils.get_jhat_files(filtname)
                with mp.Pool(utils.n_procs) as pool:
                    print(pool.map(partial(remove_bad_pixels, filtname=filtname), jhat_files))

        # skymatch_step
        if config.stage3.skymatch_step.run:
            from .stage3 import skymatch_step
            for filtname in config.filters: 
                jhat_files = utils.get_jhat_files(filtname)
                skymatch_step(jhat_files, filtname)

        # outlier_step
        if config.stage3.outlier_step.run: 
            from .stage3 import outlier_step_prep, outlier_step
            for filtname in config.filters:
                jhat_files = utils.get_jhat_files(filtname)

                outlier_step_prep(jhat_files)

                asn_files = utils.get_outlier_asn_files(filtname)
                for asn_file in asn_files:
                    outlier_step(asn_file, filtname)


        # resample_step
        if config.stage3.resample_step.run:
            from .stage3 import resample_step

            if not os.path.exists(config.mosaic_path):
                os.mkdir(config.mosaic_path)

            for filtname in config.filters:
                if not os.path.exists(os.path.join(config.mosaic_path, filtname)):
                    os.mkdir(os.path.join(config.mosaic_path, filtname))
                
                resample_step(filtname)


        # # remove_background
        # if config.stage3.background_step.run:
        #     from .stage3 import remove_background

        #     for filtname in config.filters:
        #         jhat_files = utils.get_jhat_files(filtname)

        #         with mp.Pool(utils.n_procs) as pool:
        #             print(pool.map(remove_background, jhat_files))

        

if __name__ == '__main__':
    sys.exit(main())