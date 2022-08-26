from .localization_dataset_types.storm_class import *
from .localization_dataset_types.Minflux_class import *
from .localization_dataset_types.base_class import *
from .localization_dataset_types.data_formats import *
from .CustomErrors import *


def custom_import_function(filepath):
    """This function is called when you press the custom import button, load your own data here with code you
    write yourself. Also make a backup copy of this code in case you update napari-storm this might get lost.
    Just follow the instructions and replace the print and error. I already took care of the most important imports (s. above)"""

    """
    I) 
    after you managed to load your data decide which dataset class fits best:
    1) LocalizationDataBaseClass
        most basic data class only requires that you provide : 
        a) x_pos_nm [float]
        b) y_pos_nm [float]
       (c) z_pos_nm [float]) <-- in case of 3D data 
       additionally you should provide information if the dataset is 3D (zdim_present=True) or 2D (zdim_present=False)
    2) StormDataClass
        specialized data class for STORM/PALM/... microscopy data. More requirements,
         but also more options compared to 1). In this data class the positions (x,y,z) are stored in pixel units 
         not nm. Therefore you also need to provide the pixselsize in nm. The dataset allows you to store 
         photon counts / intensity values for each localization, as well as uncertainty values, which of course 
         can then be displayed. If you cannot provide these values, just set them to ones. 
     3) MinfluxDataBaseClass
        specialized data class for Minflux data. Requires you to provide trace_ids for each localization compared to 1)
        
    ##### Additional Classes #######
    There are also classes for dataset collections, e.g. if you're having multicolor datasets. For more information 
    on this look into src/napari_storm/localization_dataset_types/ ... 
    Also feel free to create your own class that fits your purpose. Just check the files 
    in src/napari_storm/localization_dataset_types/ ... and implement it in a similar manner.
    ##################################
    
    II)
    After you decided on a dataset class and got everthing you need:
    1) create a data_rec_array=numpy.rec.array(<your_data_here>,dtype=<dataset_class_dtype>)
    2) initialize the datasetclass and return the result return <datasetclass>(data=data_rec_array, ...), 
    
    example: 
        file = D:/really_good_measurements/best_example_I_could_think_of.npy
        raw_data = np.load(file)
        data_rec_array=numpy.rec.array(raw_data, dtype=lm_base_data_dtype)
        zdim_present = True 
        name = file.split(/)[-1]
        return LocalizationDataBaseClass(data=data_rec_array, name=name, zdim_present=zdim)
    """
    print("nothing here yet")
    raise FileImportAbortedError("Custom Import Function has not been defined yet")
