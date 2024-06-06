import h5py
import numpy as np
from scipy.interpolate import interp1d
import json

def interpolate_g_factor(temperature_g_factor, target_temperature):
    temperatures = temperature_g_factor[...,0]
    g_factors = temperature_g_factor[...,1]
    # Use linear interpolation
    interpolator = interp1d(temperatures, g_factors, kind='linear', fill_value='extrapolate')
    return interpolator(target_temperature-273.15)

def read_h5_database(file_path,chain_key,temperature):
    with h5py.File(file_path, 'r') as f:
        # 反应链
        chain = f['chain'][chain_key]
        nuclei = chain['nuclei']
        reaction = chain['reaction']
        nuclei_len = len(nuclei)
        decay_matrix = np.zeros((nuclei_len,nuclei_len),dtype=np.float64)
        absorb_unit_matrix = np.zeros((nuclei_len,nuclei_len),dtype=np.float64)
        
        # 获取核素名称列表，并建立name到index的字典
        nuclei_names = [nuclei[i].decode('utf-8') for i in range(len(nuclei))]
        nuclei_name_to_index = {nuclei_names[i]: i for i in range(len(nuclei_names))}
        print(f"Chain {chain_key}:")
        print("Nuclei:", nuclei_names)
        print("Name to Index:", nuclei_name_to_index)

        # 获取反应信息
        for i in range(len(reaction)):
            reaction_entry = reaction[i]
            origin_nuclei_name = reaction_entry['origin_nuclei_name'].decode('utf-8')
            target_nuclei_name = reaction_entry['target_nuclei_name'].decode('utf-8')
            origin_nuclei_id = nuclei_name_to_index[origin_nuclei_name]
            target_nuclei_id = nuclei_name_to_index[target_nuclei_name]
            
            decay_status = bool(reaction_entry['decay_status'])
            absorb_status = bool(reaction_entry['absorb_status'])
            if decay_status:
                decay_constant = float(reaction_entry['decay_constant'])
                # 写入参数
                decay_matrix[origin_nuclei_id,origin_nuclei_id] = - decay_constant
                decay_matrix[target_nuclei_id ,origin_nuclei_id] = + decay_constant
                
            if absorb_status:
                absorb_gamma_cross_section = float(reaction_entry['absorb_gamma_cross_section'])
                absorb_fission_cross_section = float(reaction_entry['absorb_fission_cross_section'])
                # absorb_cross_section = absorb_gamma_cross_section + absorb_fission_cross_section
                
                gamma_non_linear_factor = json.loads(reaction_entry['gamma_non_linear_factor'])
                fission_non_linear_factor = json.loads(reaction_entry['fission_non_linear_factor'])
                
                gamma_temperature_g_factor = np.array(gamma_non_linear_factor,dtype=np.float64)
                fission_temperature_g_factor = np.array(fission_non_linear_factor,dtype=np.float64)
                
                gamma_g_factor = interpolate_g_factor(gamma_temperature_g_factor, temperature)
                fission_g_factor = interpolate_g_factor(fission_temperature_g_factor, temperature)
                # g_factor = 1
                gamma_cross_section_factor = gamma_g_factor*np.sqrt((293)/(temperature))/np.sqrt((4)/(np.pi))
                fission_cross_section_factor = fission_g_factor*np.sqrt((293)/(temperature))/np.sqrt((4)/(np.pi))
                # 写入参数
                absorb_unit_matrix[origin_nuclei_id,origin_nuclei_id] = - absorb_gamma_cross_section * gamma_cross_section_factor - absorb_fission_cross_section * fission_cross_section_factor
                absorb_unit_matrix[target_nuclei_id ,origin_nuclei_id] = + absorb_gamma_cross_section * gamma_cross_section_factor
                
    return decay_matrix , absorb_unit_matrix , nuclei_name_to_index , nuclei_names 
