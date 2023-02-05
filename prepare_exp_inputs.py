import os
import pickle
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE'] = "1" # conda fix for pyswmm
prj_path = r""
module_path = os.path.join(prj_path, "Code")
swmm_model_path = os.path.join(prj_path, "Model", "Cyber_NancyRun_1min.inp")
cali_data_path = os.path.join(prj_path, "Model", "Cali_data")
os.chdir(module_path)
import numpy as np
from numpy.linalg import pinv, solve, eig
import pandas as pd
import matplotlib.pyplot as plt
from pyswmm import Simulation
from util import (swmm_info, swmm_get_object, swmm_get_links,
                  swmm_get_inp_sections, cal_unctrl_outflow, cal_aw)
from lqg_controller import LQG, LQE, SS, SWMM2SSModel

##### Form SWMM inp file
swmm_setting = swmm_get_inp_sections(swmm_model_path)
###### Load SWMM setting
with Simulation(swmm_model_path) as sim:
    swmm_info(sim)
    storages, junctions, orifices, conduits, subcatchs, raingauges = swmm_get_object(sim)
    links, links_sb = swmm_get_links(sim)
##### SS model settings
m2cm = 100
r2 = list(swmm_setting["XSECTIONS"]["Geom2"])[-9:]
nodes = None # Storages + Junctions
y_max = list( (swmm_setting["STORAGE"]["MaxDepth"] * m2cm).round(2) )
areas = list(swmm_setting["STORAGE"]["CNPc"].round(2))  # [m2]
storage_areas = {"S"+str(k+1): v for k, v in enumerate(swmm_setting["STORAGE"]["CNPc"])}
sb_areas = list(swmm_setting["SUBCATCHMENTS"]["Area"].round(2))  # [m2]
Y0 = np.array([0]*9).reshape((-1,1))
nodes = ["S2", "J2", "S4", "J4", "S3", "J3", "S5", "J5", "S8", "J8", "S7",
          "J7", "S9", "J9", "S6", "J6", "S1"]

##### Load cali data
cali_data = {}
for dur_hr in [24]:
    for yr in [1, 2, 5, 10, 25, 50, 100, 200]:
        file_path = os.path.join(cali_data_path,
                                 "{}yr_{}hr_{}mins.csv".format(yr, dur_hr, 1))
        df_swmm = pd.read_csv(file_path)
        df_swmm = df_swmm[subcatchs + raingauges + storages + orifices]
        # Unit conversion cms to cm
        for i, sb in enumerate(subcatchs):
            print(60 / areas[i] * m2cm )
            df_swmm[sb] = df_swmm[sb] * 60 / areas[i] * m2cm                    # Here is where we can change scale
                                                                                # => Change Uhmax too!
        cali_data[(yr, dur_hr)] = df_swmm

delay_n = [1, 3, 4, 1, 1, 1, 3, 1]
sols = [0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.495, 1.0]
unctrl_gates = {}
for i, o in enumerate(orifices):
    unctrl_gates[o] = [sols[i], r2[i], 0.65, r2[i]]

ss_model = SWMM2SSModel()
ss_model.create_ss_model(storages, junctions, orifices, conduits, subcatchs,
                         delay_n, links, links_sb, storage_areas, nodes=nodes, ratio=True)
with open(os.path.join(prj_path, "Model", "Inputs.pickle"), "wb") as f:
    pickle.dump([ss_model, cali_data], f)
    