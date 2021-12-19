import numpy as np
import pandas as pd
import prosail
import time
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS
import os
from pathlib import Path
from tgess.src.data_science.helper_functions import *

# Read config file
config = read_config()
tqdm.pandas()

# Get root folder of project
root_path = (Path(__file__).parent / ".." / ".." ).resolve()

types = ['Mixed Forest', 'Croplands', 'Grasslands', 'Evergreen Broadleaf',
       'Woody Savannas', 'Open Shrublands', 'Grassland',
       'Evergreen Needleleaf', 'Closed Shrublands', 'Deciduous Broadleaf']

nlcdClass = ['deciduousForest', 'cultivatedCrops', 'evergreenForest',
       'mixedForest', 'shrubScrub', 'pastureHay', 'grasslandHerbaceous',
       'woodyWetlands', 'emergentHerbaceousWetlands']

prosail_param_setups = {
    
    "generic":
    # From https://step.esa.int/docs/extra/ATBD_S2ToolBox_L2B_V1.1.pdf
    # and  https://www.sciencedirect.com/science/article/pii/S0034425701002401
    {
        "n":(1,2.5),
        "cab":(20, 80.0),
        # CAB:CAR = 1:5
        "car":(1,16.0),
        "cbrown":(0,2),
        "cw":(0.0, 0.2),
        "cm":(0.0, 0.2),
        "lai":(0.0,8.0),
        "lidfa":(5,85),
#         "lidfb":(-0.15,-0.15),
        "hspot":(0.01, 1),
        "tts":(15,60),
        "tto":(0,10),
        "psi":(0,360),
        "rsoil":(0,1),
        "psoil":(0.5, 2),
        "typelidf":(2,2),     
        "alpha":(40.0,40.0),
        # "ant":(0.0,0.0)
    },
    
    #nlcdClass: dictionary describing parameter ranges
    "deciduousForest":
    # From https://www.sciencedirect.com/science/article/pii/S0034425705003044?casa_token=CjTgNNj3u40AAAAA:BhOXLVDNun1dH9kqUdpb4cI2uiGlOPjHUhpaaa6RLIAhe63kRc5dBcUX9wBfSVOhB4c_zzvy3vU
    {
        # parameter: (min, max)
        "n":(1, 4.5),
        "cab":(0, 80.0),
        "car":(0, 20.0),
        "cbrown":(0.00001, 8),
        "cw":(0.001, 0.15),
        "cm":(0.001, 0.04),
        "lai":(1.0, 7.5),
        "lidfa":(10, 89),
#         "lidfb":(-0.15, -0.15),
        "hspot":(0.01, 1),
        "tts":(15,60),
        "tto":(0,10),
        "psi":(0,360),
        "rsoil":(0,1),
        "psoil":(0.5, 2),
        "typelidf":(2,2),     
        "alpha":(40.0,40.0),
        "ant":(0.0,0.0)
    },
    "cultivatedCrops":
    # From: https://www.mdpi.com/2072-4292/10/1/85/htm
    # Taking the min and max from all listed crops
    {
        # parameter: (min, max)
        "n":(1.2, 2.6),
        "cab":(0, 80.0),
        "car":(1, 24.0),
        "cbrown":(0, 1),
        "cw":(0.001, 0.08),
        "cm":(0.001, 0.02),
        "lai":(0.0, 10.0),
        "lidfa":(20, 90),
#         "lidfb":(-0.15, -0.15),
        "hspot":(0.01, 1),
        "tts":(15,60),
        "tto":(0,10),
        "psi":(0,360),
        "rsoil":(0,1),
        "psoil":(0.5, 2),
        "typelidf":(2,2),     
        "alpha":(40.0,40.0),
        "ant":(0.0,0.0)
    },
    "evergreenForest":{
        # From: 
        "n":(2, 2),
        "cab":(20, 60.0),
        "car":(0.6, 16.0),
        "cbrown":(0, 0.6),
        "cw":(0.0, 0.2),
        "cm":(0, 0.2),
        # In study, LAI was fixed to 4.2
        "lai":(0.0, 8.0),
        "lidfa":(20, 90),
#         "lidfb":(-0.15, -0.15),
        "hspot":(0.01, 1),
        "tts":(15,60),
        "tto":(0,10),
        "psi":(0,360),
        "rsoil":(0,1),
        "psoil":(0.5, 2),
        "typelidf":(2,2),     
        "alpha":(40.0,40.0),
        "ant":(0.0,0.0)
    },
    
}

def latin_hypercube_sampling(parameter_ranges, n_samples=1000):
    """
    Function to create combinations of parameters within specific ranges using latin hypercube sampling.
    
    Args:
       parameter_ranges (dict): Minimum and maximum value per parameter.
       n_samples: How many combinations in total should be created.

    Returns:
        param_configurations (list of dicts)
    """
    
    param_ranges = np.array([parameter_ranges[x] for x in parameter_ranges.keys()])
    param_names = list(parameter_ranges.keys())

    sampling = LHS(xlimits=param_ranges)
    
    samples = sampling(n_samples)
    
    param_configurations = [dict(zip(param_names,sample)) for sample in samples]
    
    return param_configurations

def create_S2_table(path="data/raw/", filename="S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0.xls"):
    """
    Function to get average spectral response from Sentinel-2 satellites A and B
    
    Args:
       path (str): Path where excel file is located
       filename (str): Filename of spectral response excel file

    Returns:
        output (np array): Array of reflectance values for Sentinel-2 bands
    """

    df_s2a = pd.read_excel(io=os.path.join(root_path, path, filename), sheet_name="Spectral Responses (S2A)")
    df_s2b = pd.read_excel(io=os.path.join(root_path, path, filename), sheet_name="Spectral Responses (S2B)")
    
    df_s2b.columns = [col.replace("S2B", "S2A") for col in list(df_s2b.columns)]
    
    df = pd.concat([df_s2a, df_s2b])
    df = df.groupby(df.index).mean()
    df = df[df['SR_WL'].between(400, 2500)]
    df = df.set_index("SR_WL")
    
    return df

def prosail_to_S2(spectra_input, s2_table):
    """
    Function to convert 1nm precision reflectance values into Sentinel-2 band values.
    Modified from: https://github.com/nunocesarsa/RTM_Inversion/blob/main/Jupyter/RTM%20Inversion/RTM_Pure_Inversion_FunctionCallCorrected_Optimized.ipynb
    
    Args:
       spectra_input (np array): Array of 1nm precision reflectance values
       s2_table (pd DataFrame): Table describing spectral response curves of Sentinel-2 bands

    Returns:
        output (np array): Array of reflectance values for Sentinel-2 bands
    """

    rho_s2 = s2_table.multiply(spectra_input, axis="index") #calculates the numerator

    w_band_sum = s2_table.sum(axis=0, skipna = True) #calculates the denominator

    output = (rho_s2.sum(axis=0)/w_band_sum).rename_axis("ID").values #runs the weighted mean and converts the output to a numpy array

    return output


def create_LUT(parameter_ranges, nlcd_class, n_samples=10000, s2_table_path="data/raw/", \
               s2_table_filename="S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0.xls", out_file="PROSAIL_LUT", verbosity=0):
    """
    Function to create a lookup table using PROSAIL
    
    Args:
        parameter_ranges (dict): Dictionary defining ranges of parameters to simulate
        nlcd_class (str): For which cover type should the simulation be made
        n_samples (int): How many samples to create
        s2_table_path (str): Filepath to spectral response table
        s2_table_filename (str): Filename of spectral response table
        out_file (str): Filename for saved output

    Returns:
        LUT (pd DataFrame): Pandas dataframe containing the created lookup table
        
    PROSAIL args:
        n: float
            The number of leaf layers. Unitless [-].
        cab: float
            The chlorophyll a+b concentration. [g cm^{-2}].
        car: float
            Carotenoid concentration.  [g cm^{-2}].
        cbrown: float
            The brown/senescent pigment. Unitless [-], often between 0 and 1
            but the literature on it is wide ranging!
        cw: float
            Equivalent leaf water. [cm]
        cm: float
            Dry matter [g cm^{-2}]
        lai: float
            leaf area index
        lidfa: float
            a parameter for leaf angle distribution. If ``typliedf``=2, average
            leaf inclination angle.
        tts: float
            Solar zenith angle
        tto: float
            Sensor zenith angle
        psi: float
            Relative sensor-solar azimuth angle ( saa - vaa )
        ant: float, optional
            Anthocyanins content. Used in Prospect-D and Prospect-PRO [g cm^{-2}]
        prot: float, optional
            Protein content. Used in Prospect-PRO. [g cm^{-2}]
        cbc: float, optional
            Carbon based constituents. Used in Prospect-PRO. [g cm^{-2}]
        alpha: float
            The alpha angle (in degrees) used in the surface scattering
            calculations. By default it's set to 40 degrees.
        prospect_version: str
            Which PROSPECT version to use. We have "5", "D" and "PRO"
        typelidf: int, optional
            The type of leaf angle distribution function to use. By default, is set
            to 2.
        lidfb: float, optional
            b parameter for leaf angle distribution. If ``typelidf``=2, ignored
        factor: str, optional
            What reflectance factor to return:
            * "SDR": directional reflectance factor (default)
            * "BHR": bi-hemispherical r. f.
            * "DHR": Directional-Hemispherical r. f. (directional illumination)
            * "HDR": Hemispherical-Directional r. f. (directional view)
            * "ALL": All of them
            * "ALLALL": All of the terms calculated by SAIL, including the above
        rsoil0: float, optional
            The soil reflectance spectrum
        rsoil: float, optional
            Soil scalar 1 (brightness)
        psoil: float, optional
            Soil scalar 2 (moisture)
        soil_spectrum1: 2101-element array
            First component of the soil spectrum
        soil_spectrum2: 2101-element array
            Second component of the soil spectrum
    """
    
    # Create Sentinel-2 Spectral Response Functions table
    s2_table = create_S2_table(path=s2_table_path, filename=s2_table_filename)
    
    samples = latin_hypercube_sampling(parameter_ranges, n_samples)
    
    parameters_LUT = pd.DataFrame(samples)
    reflectances_LUT = []
    for sample_params in tqdm(samples, \
                              desc="Running PROSAIL with configuration {}".format(nlcd_class)\
                              .format(n_samples, nlcd_class)) if verbosity>0 else samples:
                        
        rho_canopy = prosail.run_prosail(**sample_params, prospect_version='5', \
                            factor='SDR', rsoil0=None, \
                            soil_spectrum1=None, soil_spectrum2=None)
        
        rho_canopy = prosail_to_S2(rho_canopy, s2_table)
        
        reflectances_LUT.append(rho_canopy)
    
    bands = [col.replace("S2A_SR_AV_", "") for col in s2_table.columns]
    
    t = time.time()
    
#     LUT = parameters_LUT.join(pd.DataFrame(columns=range(400,2501), data=reflectances_LUT), how="outer")

    LUT = parameters_LUT.join(pd.DataFrame(columns=bands, data=reflectances_LUT), how="outer")
    if verbosity>0:
        print("Joined parameters and reflectances into LUT in {} seconds".format(time.time()-t))


    rename_cols = {"tts":"solar_zenith", 
                   "tto":"observer_zenith",
                   "psi":"relative_azimuth"
                  }

    LUT = LUT.rename(columns = rename_cols)
    LUT = LUT.dropna()
    
    t = time.time()
    out_file = out_file+"_{}_{}.csv".format(n_samples, nlcd_class)
    if verbosity>0:
        LUT.to_csv(out_file)
        print("Successfully created file {} in {} seconds".format(out_file, time.time()-t))
    return LUT

def create_prosail_datasets(prosail_param_setups, n_samples=[5000, 10000, 50000, 100000], path="../data/processed/", filename="PROSAIL_LUT_S2"):
    """
    Function to create several lookup tables with different setups. Saves to .csv files.
    
    Args:
        n_samples (list of int): Several values of number of samples to create
        prosail_param_setups(dict of dicts): Possible parameter setups per cover type
    """
    for n_samples in tqdm([5000, 10000, 50000, 100000], desc="Different number of samples"):
        for nlcd_class in tqdm(prosail_param_setups.keys(), desc="Different configurations"):
            df = create_LUT(prosail_param_setups[nlcd_class], nlcd_class, n_samples=n_samples, \
                       out_file=os.path.join(path, filename))





def auto_tune_prosail_parameters(n_samples=1000):
    import logging

    import numpy as np
    from ConfigSpace.hyperparameters import UniformFloatHyperparameter, Constant

    # Import ConfigSpace and different types of parameters
    from smac.configspace import ConfigurationSpace
    from smac.facade.smac_hpo_facade import SMAC4HPO
    # Import SMAC-utilities
    from smac.scenario.scenario import Scenario
    from sklearn.metrics import mean_squared_error
    data_path = config["experiment"]["data"]["path"]

    in_situ_train_file = config["experiment"]["data"]["in_situ_dataset"][:-4]+"_train.csv"
    in_situ_train_file = (root_path / data_path / in_situ_train_file).resolve()
    in_situ_df = pd.read_csv(in_situ_train_file)

    def spectral_similarity_rmse(cfg, target_cols=("LAI_Warren", "lai"), margin=0.1, \
                             bands=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']):
    
        rsoil_min = np.min([cfg["rsoil_min"], cfg["rsoil_max"]])
        rsoil_max = np.max([cfg["rsoil_min"], cfg["rsoil_max"]])
        psoil_min = np.min([cfg["psoil_min"], cfg["psoil_max"]])
        psoil_max = np.max([cfg["psoil_min"], cfg["psoil_max"]])

        parameter_ranges = {
            "n":(1,2.5),
            "cab":(20, 80.0),
            # CAB:CAR = 1:5
            "car":(1,16.0),
            "cbrown":(0,2),
            "cw":(0.0, 0.2),
            "cm":(0.0, 0.2),
            "lai":(0.0,8.0),
            "lidfa":(5,85),
        #         "lidfb":(-0.15,-0.15),
            "hspot":(0.01, 1),
            "tts":(15,60),
            "tto":(0,10),
            "psi":(0,360),
            "rsoil":(rsoil_min, rsoil_max),
            "psoil":(psoil_min, psoil_max),
            "typelidf":(2,2),     
            "alpha":(40.0,40.0),
        }    


        simulation_df = create_LUT(parameter_ranges, "generic", n_samples=n_samples, \
                           out_file="../data/processed/temp", verbosity=0)

        rmse = []
        for _, row in tqdm(in_situ_df.iterrows(), total=len(in_situ_df)):
            errors = []
            y = row[target_cols[0]]
            # select rows in simulation_df with target value close to y
            simulated_spectra = simulation_df[(simulation_df[target_cols[1]] >= (y*(1-margin))) & \
                                              (simulation_df[target_cols[1]] <= (y*(1+margin)))]

            if len(simulated_spectra) > 0:
                for band in bands:
                    y_pred = simulated_spectra[band]
                    
                    y_true = pd.concat([row]*len(y_pred))[band]
                    
                    if len(y_pred) == 1:
                        y_true = [y_true]
                        y_pred = [y_pred]
                    
                    error = np.sqrt(mean_squared_error(y_true, y_pred))
                    errors.append(error)

                rmse.extend(errors)
        return np.mean(rmse)


    

    logger = logging.getLogger("PROSAIL soil parameters")
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)  # Enable to show debug-output
    # logger.info("Running random forest example for SMAC. If you experience "
    #             "difficulties, try to decrease the memory-limit.")

    # Build Configuration Space which defines all parameters and their ranges.
    # To illustrate different parameter types,
    # we use continuous, integer and categorical parameters.
    cs = ConfigurationSpace()

    rsoil_min = UniformFloatHyperparameter("rsoil_min", 0.0, 1.0, default_value=0.5)
    rsoil_max = UniformFloatHyperparameter("rsoil_max", 0.0, 1.0, default_value=0.5)
    psoil_min = UniformFloatHyperparameter("psoil_min", 0.0, 1.5, default_value=0.75)
    psoil_max = UniformFloatHyperparameter("psoil_max", 0.0, 1.5, default_value=0.75)


    cs.add_hyperparameters([rsoil_min, rsoil_max, psoil_min,
                            psoil_max])

    # SMAC scenario object
    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternative runtime)
                         "runcount-limit": 100,  # max. number of function evaluations; for this example set to a low number
                         "cs": cs,  # configuration space
                         "deterministic": "true",
                         })

    # To optimize, we pass the function to the SMAC-object
    smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=spectral_similarity_rmse)

    # Example call of the function with default values
    # It returns: Status, Cost, Runtime, Additional Infos
    def_value = smac.get_tae_runner().run(cs.get_default_configuration(), 1)[1]
    print("Value for default configuration: %.2f" % def_value)

    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    inc_value = smac.get_tae_runner().run(incumbent, 1)[1]
    print("Optimized Value: %.2f" % inc_value)
    print(incumbent)

if __name__ == "__main__":
    auto_tune_prosail_parameters(n_samples=1000)
