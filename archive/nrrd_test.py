import ants
import numpy as np
#import napari
#from skimage import data, filters  # Just to generate test data (3D blobs).
from subprocess import call


def main():
    # 5 to 4
    #fix = ants.image_read("data/4_mapzebraintERKref.nrrd")
    #fix = ants.image_read("data/antiterk.nrrd")
    #fix = ants.image_read("data/antiERK12tERK_Gaussian_K4.nrrd")
    #fix = ants.image_read("data/3_mapZebrainH2B.tif")
    fix = ants.image_read("data/HuC_Gaussian_K4.nrrd")
    #fix = ants.image_read("data/ERK-ref.nrrd")
    mov = ants.image_read("data/5_stitched_10_rotate_downsampled.nrrd")
    #mov = ants.image_read("data/5n_concat_01_gray.tif")

    transform_types = [
        #"SyNBold", "SyNBoldAff",
        #"SyNabp", 
        #"TRSAA", 
        #"ElasticSyN", 
        #"Translation", "Rigid", "Similarity", "QuickRigid", "DenseRigid", 
        #"BOLDRigid", "Affine", "AffineFast", "BOLDAffine", 
        #"SyN", "SyNRA", "SyNOnly",
        #"SyNAggro", 
        #"SyNCC",
        #"TVMSQ", 
        "TVMSQC"
    ]
    
    for type in transform_types:
        print(f"\n\n>>>> {type}")
        reg = ants.registration(fixed=fix,
                                moving=mov,
                                type_of_transform=type,
                                verbose=True)
        reg_img = reg['warpedmovout']
        ants.image_write(reg_img, f"registered/5_to_h2bk4_{type}.nrrd")

    # 2r to 3
    """fix = ants.image_read("data/3_mapZebrainH2B.tif")
    mov = ants.image_read("data/2_Intermediate_rotated.tif")

    transform_types = ["TVMSQC"]
    for type in transform_types:
        print(f"\n\n>>>> {type}")
        reg = ants.registration(fixed=fix,
                                moving=mov,
                                type_of_transform=type,
                                verbose=True)
        reg_img = reg['warpedmovout']
        ants.image_write(reg_img, f"registered/2r_to_3_{type}.nrrd")"""


def josh():
    fixed_image = "data/3_mapZebrainH2B.tif"
    moved_image = "data/2_Intermediate_rotated.tif"
    registration_output_name = "2r_to_3_josh_fast"
    slow_registration = (
        "antsRegistration "
        + "-d 3 "
        + "--float 1 "
        +f"-o [{registration_output_name}, {registration_output_name}.nii] "
        + "-n WelchWindowedSinc "
        + "--winsorize-image-intensities [0.01,0.99] "
        + "--use-histogram-matching 1 "
        +f"-r [{fixed_image}, {moved_image}, 1] "
        + "-t rigid[0.1] "
        +f"-m MI[{fixed_image}, {moved_image},1,32, Regular,0.5] "
        + "-c [1000x500x500x500,1e-8,10] "
        + "--shrink-factors 8x4x2x1 "
        + "--smoothing-sigmas 2x1x1x0vox "
        + "-t Affine[0.1] "
        +f"-m MI[{fixed_image}, {moved_image},1,32, Regular,0.5] "
        + "-c [1000x500x500x500,1e-8,10] "
        + "--shrink-factors 8x4x2x1 "
        + "--smoothing-sigmas 2x1x1x0vox "
        + "-t SyN[0.05,6,0.5] "
        +f"-m CC[{fixed_image}, {moved_image},1,2] "
        + "-c [1000x500x500x500x500,1e-7,10] "
        + "--shrink-factors 12x8x4x2x1 "
        + "--smoothing-sigmas 4x3x2x1x0vox -v 1")
    fast_registration = (
        "antsRegistrationSyNQuick.sh "
        + "-d 3 "
        +f"-f {fixed_image} "
        +f"-m {moved_image} "
        +f"-o {registration_output_name} "
        + "-p f "
        + "-j 1")
    #call([slow_registration],shell=True)
    call(fast_registration, shell=True)


if __name__ == '__main__':
    main()
    #nrrd_vis()
    #josh()
