import h5py
import numcodecs
import zarr


TRAIN_HDF = "data/sample_A_20160501"
TEST_HDF = "data/sample_A+_20160601"
#TRAIN_HDF = "data/sample_B_20160501"
#TEST_HDF = "data/sample_B+_20160601"
#TRAIN_HDF = "data/sample_C_20160501"
#TEST_HDF = "data/sample_C+_20160601"

#TRAIN_HDF = "data/sample_A_padded_20160501"
#TEST_HDF = "data/sample_A+_padded_20160601"
#TRAIN_HDF = "data/sample_B_padded_20160501"
#TEST_HDF = "data/sample_B+_padded_20160601"
#TRAIN_HDF = "data/sample_C_padded_20160501"
#TEST_HDF = "data/sample_C+_padded_20160601"

def recursive(hdf_group, zarr_dest):
    for i in list(hdf_group.keys()):
        if isinstance(hdf_group[i], h5py._hl.group.Group):
            if i != '__DATA_TYPES__':
                zarr_group = zarr_dest.create_group(i)
                recursive(hdf_group[i], zarr_group)  
        else:
            dtype = hdf_group[i].dtype
            zarr_array = zarr_dest.create_dataset(
                i,
                shape=hdf_group[i].shape,
                dtype=dtype,
                object_codec=numcodecs.JSON())
            hdf_array = hdf_group.get(i)[()]
            if dtype == 'object':
                hdf_array = hdf_array.astype(str)
            for i, id in enumerate(hdf_array):
                zarr_array[i] = id


train_source = h5py.File(TRAIN_HDF + ".hdf", 'r+')
print(zarr.tree(train_source))
# train_store = zarr.DirectoryStore(TRAIN_HDF + ".zarr")
# train_dest = zarr.group(store=train_store, overwrite=True)
# recursive(train_source, train_dest)
# print(zarr.tree(train_dest))

test_source = h5py.File(TEST_HDF + ".hdf", 'r+')
print(zarr.tree(test_source))
# test_store = zarr.DirectoryStore(TEST_HDF + ".zarr")
# test_dest = zarr.group(store=test_store, overwrite=True)
# recursive(test_source, test_dest)
# print(zarr.tree(test_dest))
