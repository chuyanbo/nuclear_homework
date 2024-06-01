import h5py
import numpy as np
import json

# 创建一个复合数据类型
reaction_dtype = np.dtype([
    ('origin_nuclei_name', h5py.string_dtype(encoding='utf-8')),
    ('decay_status', np.bool_),
    ('decay_constant', np.float64),
    ('absorb_status', np.bool_),
    ('absorb_gamma_cross_section', np.float64),
    ('absorb_fission_cross_section', np.float64),
    ('non_linear_factor', h5py.string_dtype(encoding='utf-8')),  # 将非线性因子存储为JSON字符串
    ('target_nuclei_name', h5py.string_dtype(encoding='utf-8'))
])

# 创建反应数据
chain1_reaction_data = np.array([
    (
        "U235",           # origin_nuclei_name
        1,             # decay_status
        3.12e-17,         # decay_constant
        0,            # absorb_status
        99.3843,          # absorb_gamma_cross_section
        586.691,          # absorb_fission_cross_section
        json.dumps([[100, 0.961], [200, 0.9457], [400, 0.9294], [600, 0.9229]]),  # non_linear_factor
        "U236"            # target_nuclei_name
    ),
    ("U236",0,9.38E-16,1,5.13396,0,json.dumps([[100, 1], [200, 1], [400, 1], [600, 1]]),"Np237"),
    ("Np237",0,1.03E-14,1,175.43,0,json.dumps([[100, 1], [200, 1], [400, 1], [600, 1]]),"Pu238"),
    ("Pu238",0,2.51E-10,1,412.855,0,json.dumps([[100, 1], [200, 1], [400, 1], [600, 1]]),"Pu239")
], dtype=reaction_dtype)
chain2_reaction_data = np.array([
    ("U238",0,4.92E-18,1,2.73,0,json.dumps([[100, 1.0031], [200, 1.0049], [400, 1.0085], [600, 1.0122]]),"Np239"),
    ("Np239",1,3.41E-06,0,0,0,json.dumps([[100, 1], [200, 1], [400, 1], [600, 1]]),"Pu239"),
    ("Pu239",0,9.12E-13,1,274,0,json.dumps([[100, 1.1611], [200, 1.3388], [400, 1.8905], [600, 2.5321]]),"Pu240"),
    ("Pu240",0,3.35E-12,1,286,0,json.dumps([[100, 1], [200, 1], [400, 1], [600, 1]]),"Pu241"),
    ("Pu241",0,1.53E-09,1,363.047,0,json.dumps([[100, 1], [200, 1], [400, 1], [600, 1]]),"Pu242"),
    ("Pu241",1,1.53E-09,0,363.047,0,json.dumps([[100, 1], [200, 1], [400, 1], [600, 1]]),"Am241")
], dtype=reaction_dtype)

with h5py.File("data.h5", "w") as opt:
    chain_group = opt.create_group("chain")
    chain_1_group = chain_group.create_group("chain_1")
    nuclei1 = np.array(["U235", "U236", "Np237", "Pu238", "Pu239"], dtype=h5py.string_dtype(encoding='utf-8'))
    chain_1_group.create_dataset("nuclei", data=nuclei1)
    chain_1_group.create_dataset("reaction", data=chain1_reaction_data)

    chain_2_group = chain_group.create_group("chain_2")
    nuclei2 = np.array(["U238", "Np239", "Pu239", "Pu240", "Pu241","Pu242","Am241"], dtype=h5py.string_dtype(encoding='utf-8'))
    chain_2_group.create_dataset("nuclei", data=nuclei2)
    chain_2_group.create_dataset("reaction", data=chain1_reaction_data)
    
    