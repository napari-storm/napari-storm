import numpy as np


class DatasetInterface:
    """This is the only communication route between napari-storm and the localization datasets, except for
    the list of datasets nothing is saved in this class and all the data is just passed back and forth"""

    def __init__(self, data_to_layer_itf, dock_widget, file_to_dataset_itf, data_filter_itf, data_adjustment_itf):
        self.list_of_datasets = []
        self.data_to_layer_itf = data_to_layer_itf
        self.dock_widget = dock_widget
        self.file_to_dataset_itf = file_to_dataset_itf
        self.data_filter_itf = data_filter_itf
        self.data_adjustment_itf = data_adjustment_itf

    def soft_refresh(self, list_of_datasets=None):
        """this is used to update all the napari-storm parts, depending on what changed"""

        self.data_to_layer_itf.save_current_camera()
        if not list_of_datasets:
            list_of_datasets = self.list_of_datasets
        for dataset in list_of_datasets:
            dataset_idx = list_of_datasets.index(dataset)
            self.data_to_layer_itf.update_layer(dataset, dataset_idx=dataset_idx)
        self.data_to_layer_itf.restore_camera()

    def hard_refresh(self, list_of_datasets=None, update_data_range=False):
        """this is used to update all napari-storm parts as well as the datasets"""
        self.data_to_layer_itf.save_current_camera()
        if not list_of_datasets:
            list_of_datasets = self.list_of_datasets
        for dataset in list_of_datasets:
            dataset_idx = list_of_datasets.index(dataset)
            if update_data_range:
                self.data_to_layer_itf.update_data_range(dataset, dataset_idx=dataset_idx)
            self.data_to_layer_itf.update_layer(dataset, dataset_idx=dataset_idx)
        self.dock_widget.color_scale_bar.update_numbers()
        self.data_to_layer_itf.restore_camera()


    def refresh_dataset_lists(self, datasets=None, clear=False):
        if clear:
            self.list_of_datasets = []
            self.data_filter_itf.clear_entries()
            self.data_adjustment_itf.clear_entries()
        if not(not datasets):
            for dataset in datasets:
                self.list_of_datasets.append(dataset)
                self.data_filter_itf.add_dataset_entry(dataset_name=dataset.name)
                self.data_adjustment_itf.add_dataset_entry(dataset_name=dataset.name)

    def pipe_locs(self, dataset=None):
        """Get active localization rec array from all datasets or a specific one if given"""
        if not dataset:
            list_of_locs = []
            for dataset in self.list_of_datasets:
                list_of_locs.append(dataset.locs)
            return list_of_locs
        else:
            return dataset.locs

    def pipe_all_locs(self, dataset):
        """Get all localization rec array from all datasets or a specific one if given"""
        if not dataset:
            list_of_locs = []
            for dataset in self.list_of_datasets:
                list_of_locs.append(dataset.locs_all)
            return list_of_locs
        else:
            return dataset.locs_all

    def pipe_property_bandpass(self, prop, dataset=None, l_val=-np.inf, u_val=np.inf):
        if not dataset:
            for dataset in self.list_of_datasets:
                dataset.bandpass_locs_filter_by_property(prop=prop, l_val=l_val, u_val=u_val)
        else:
            dataset.bandpass_locs_filter_by_property(prop=prop, l_val=l_val, u_val=u_val)

    def pipe_locs_restrict_percent(self, x_range_pc, y_range_pc, z_range_pc=None, dataset=None, reset=False):
        if not dataset:
            for dataset in self.list_of_datasets:
                dataset.restrict_locs_by_percent(x_range_pc=x_range_pc,
                                                 y_range_pc=y_range_pc,
                                                 z_range_pc=z_range_pc,
                                                 reset=reset)
        else:
            dataset.restrict_locs_by_percent(x_range_pc=x_range_pc,
                                             y_range_pc=y_range_pc,
                                             z_range_pc=z_range_pc,
                                             reset=reset)

    def pipe_locs_restrict_value(self, x_range_nm, y_range_nm, z_range_nm=None, dataset=None, reset=False):
        if not dataset:
            for dataset in self.list_of_datasets:
                dataset.restrict_locs_by_absolute(x_range_nm=x_range_nm,
                                                  y_range_nm=y_range_nm,
                                                  z_range_nm=z_range_nm,
                                                  reset=reset)
        else:
            dataset.restrict_locs_by_absolute(x_range_nm=x_range_nm,
                                              y_range_nm=y_range_nm,
                                              z_range_nm=z_range_nm,
                                              reset=reset)

    def pipe_indices_to_be_removed(self, indices, dataset=None, reset=False):
        """removes all the locs from locs active specified by their indices for one specified dataset or else goes
        through the list of datasets. If it goes through all, indices should be a list of the indices for each
        dataset"""
        if not dataset:
            for i in range(len(self.list_of_datasets)):
                self.list_of_datasets[i].remove_locs_by_index(indices[i], reset=reset)
        else:
            dataset.remove_locs_by_index(indices, reset=reset)

    def get_indices_by_prob_values(self, prop, dataset=None, l_val=-np.inf, u_val=np.inf):
        """finds out the indices of all locs that fulfill a condition specified by the lower and upper values of a
        property, works for one specified dataset or else goes through the list of datasets"""
        if not dataset:
            list_of_dataset_indices = []
            for dataset in self.list_of_datasets:
                list_of_dataset_indices.append(
                    dataset.get_idx_of_specified_prop_all(prop=prop, l_val=l_val, u_val=u_val))
            return list_of_dataset_indices
        else:
            return dataset.get_idx_of_specified_prop_all(prop=prop, l_val=l_val, u_val=u_val)

    def reset_filters(self, dataset=None):
        if not dataset:
            for dataset in self.list_of_datasets:
                dataset.reset_filters()
        else:
            dataset.reset_filters()

