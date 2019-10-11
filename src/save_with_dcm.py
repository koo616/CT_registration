from scipy import io
import os
from pydicom.dataset import Dataset, FileDataset
import pydicom
import numpy as np
import datetime


data_dir = "/data/dataset/urinary"
vol = io.loadmat(os.path.join(data_dir, "vol_9.mat"))

pre = vol['imgV_pre']
pre = pre.astype(np.int16)

file_meta = Dataset()
file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
file_meta.MediaStorageSOPInstanceUID = "1.2.3"
file_meta.ImplementationClassUID = "1.2.3.4"


ds = FileDataset("/data/proj/registration/result/1.dcm", {},
                 file_meta=file_meta, preamble=b"\0" * 128)

ds.PatientName = "Jung"
ds.PatientID = "00570730"

ds.is_little_endian = True
ds.is_implicit_VR = True

dt = datetime.datetime.now()
ds.ContentDate = dt.strftime('%Y%m%d')
timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
ds.ContentTime = timeStr

# ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRBigEndian

ds.PixelData = pre[:, :, 0].tobytes()
ds.save_as("/data/proj/registration/result/1.dcm")
