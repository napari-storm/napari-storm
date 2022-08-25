


lm_base_data_dtype = [
    ('x_pos_nm', 'f4'),
    ('y_pos_nm', 'f4'),
    ('z_pos_nm', 'f4')]

storm_data_dtype = [('frame_number', 'i4'),
                     ('x_pos_pixels', 'f4'),
                     ('y_pos_pixels', 'f4'),
                     ('z_pos_pixels', 'f4'),
                     ('sigma_x_pixels', 'f4'),
                     ('sigma_y_pixels', 'f4'),
                     ('sigma_z_pixels', 'f4'),
                     ('photon_count', 'f4')]

minflux_AI_data_dtype = [('x_pos_nm', 'f4'),
                         ('y_pos_nm', 'f4'),
                         ('z_pos_nm', 'f4'),
                         ('eco', 'i4'),
                         ('ecc', 'i4'),
                         ('efo', 'f4'),
                         ('efc', 'f4'),
                         ('sta', 'i4'),
                         ('cfr', 'f4'),
                         ('dcr', 'f4'),
                         ('gvy', 'f4'),
                         ('gvx', 'f4'),
                         ('eoy', 'f4'),
                         ('eox', 'f4'),
                         ('dmz', 'f4'),
                         ('lcx', 'f4'),
                         ('lcy', 'f4'),
                         ('lcz', 'f4'),
                         ('fbg', 'f4'),
                         ('tic', 'i4'),
                         ('lnc_x', 'f4'),
                         ('lnc_y', 'f4'),
                         ('lnc_z', 'f4'),
                         ('ext_x', 'f4'),
                         ('ext_y', 'f4'),
                         ('ext_z', 'f4'),
                         ('time_s', 'f4'),
                         ('activation', 'i4'),
                         ('trace_id', 'i4')]

list_of_dataset_classes = [lm_base_data_dtype, storm_data_dtype, minflux_AI_data_dtype]
