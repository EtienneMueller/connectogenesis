import logging
import math
from dacapo.experiments.architectures import CNNectomeUNetConfig
from dacapo.experiments.datasplits.datasets import RawGTDatasetConfig
from dacapo.experiments.datasplits.datasets.arrays import ZarrArrayConfig
from dacapo.experiments.datasplits.train_validate_datasplit_config import TrainValidateDataSplitConfig
from dacapo.experiments.run import Run
from dacapo.experiments.run_config import RunConfig
from dacapo.experiments.tasks import AffinitiesTaskConfig
from dacapo.experiments.trainers import GunpowderTrainerConfig
from dacapo.experiments.trainers.gp_augments import SimpleAugmentConfig, ElasticAugmentConfig, IntensityAugmentConfig
from funlib.geometry import Coordinate
from sys import stdout
from torchsummary import summary
from typing import List  #, Tuple


logging.basicConfig(level=logging.INFO)

TRAIN_HDF = "data/sample_A_padded_20160501.zarr"
VALIDATE_HDF = "data/sample_A+_padded_20160601.zarr"
#TRAIN_HDF = "data/sample_B+_20160601.zarr"
#VALIDATE_HDF = "data/sample_B_20160501.zarr"

#train_zarr = zarr.open(TRAIN_HDF, mode='r')
#validate_zarr = zarr.open(VALIDATE_HDF, mode='r')
#print(zarr.tree(train_zarr))
#print(zarr.tree(validate_zarr))

# TODO: create datasplit config
train_raw = ZarrArrayConfig(
    name="train_raw",
    file_name=TRAIN_HDF,
    dataset=f"volumes/raw",
)

train_gt = ZarrArrayConfig(
    name="train_gt",
    file_name=TRAIN_HDF,
    dataset=f"volumes/labels/neuron_ids",
)

validate_raw = ZarrArrayConfig(
    name="validate_raw",
    file_name=TRAIN_HDF,
    dataset=f"volumes/raw",
)

validate_gt = ZarrArrayConfig(
    name="validate_gt",
    file_name=TRAIN_HDF,
    dataset=f"volumes/labels/neuron_ids",
)

train_config = RawGTDatasetConfig(
    name="train_config",
    raw_config=train_raw,
    gt_config=train_gt
)

validate_config = RawGTDatasetConfig(
    name="test_config",
    raw_config=validate_raw,
    gt_config=validate_gt
)

datasplit_config = TrainValidateDataSplitConfig(
    name="DSConfig",
    train_configs=[train_config],
    validate_configs=[validate_config]
)

# train_source = h5py.File(PATH + TRAIN_HDF + ".hdf", 'r+')
# print(train_source['volumes']['raw'])
# # train_store = zarr.DirectoryStore(TRAIN_HDF + ".zarr")
# train_dest = zarr.group()
# #zarr.copy(train_source['volumes']['raw'], train_dest, log=stdout)
# zarr.copy_all(train_source['volumes'], train_dest, log=stdout)
# print(zarr.tree(train_dest))

# train_dest = zarr.group(store=train_store, overwrite=True)
# zarr_group = train_dest.create_group('volumes')
# hdf_group = train_source['volumes']['raw']
# zarr_array = zarr_group.create_dataset(
#                 'raw',
#                 shape=hdf_group.shape,
#                 dtype=hdf_group.dtype)

# test = zarr.open(PATH+TEST_HDF+".zarr")
# test2 = zarr.load(PATH+TEST_HDF+".zarr", path='volumes/raw')
# print(type(test2))
# print(test2.shape)
# print(type(test2[0][0]))
# test3 = zarr.load(PATH+TEST_HDF+".zarr", path='volumes/labels/neuron_ids')
# print(type(test3))
# print(test3.shape)

# %%
architecture_config = CNNectomeUNetConfig(
    name="small_unet",
    input_shape=Coordinate(212, 212, 212),
    eval_shape_increase=Coordinate(72, 72, 72),
    fmaps_in=1,
    num_fmaps=8,
    fmaps_out=32,
    fmap_inc_factor=4,
    downsample_factors=[(2, 2, 2), (2, 2, 2), (2, 2, 2)],
    constant_upsample=True
)

task_config = AffinitiesTaskConfig(
    name="AffinitiesPrediction",
    neighborhood=[(0,0,1),(0,1,0),(1,0,0)]
)

trainer_config = GunpowderTrainerConfig(
    name="gunpowder",
    batch_size=2,
    learning_rate=0.0001,
    augments=[
        SimpleAugmentConfig(),
        ElasticAugmentConfig(
            control_point_spacing=(100, 100, 100),
            control_point_displacement_sigma=(10.0, 10.0, 10.0),
            rotation_interval=(0, math.pi / 2.0),
            subsample=8,
            uniform_3d_rotation=True
        ),
        IntensityAugmentConfig(
            scale=(0.25, 1.75),
            shift=(-0.5, 0.35),
            clip=False
        )
    ],
    num_data_fetchers=20,
    snapshot_interval=10000,
    min_masked=0.15
    #min_labelled=0.1,  # unexpected kwarg
)

run_config = RunConfig(
    name="tutorial_run",
    task_config=task_config,
    architecture_config=architecture_config,
    trainer_config=trainer_config,
    datasplit_config=datasplit_config,
    repetition=0,
    num_iterations=100000,
    validation_interval=1000
)

run = Run(run_config)

# if you want a summary of the model you can print that here
print(summary(run.model, (1, 212, 212, 212)))