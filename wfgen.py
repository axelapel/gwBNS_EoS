import pycbc.waveform.generator
import bilby
import numpy as np
import h5py
import argparse


def save(list_of_concatenated_strains, list_of_associated_params, path):
    """
    Saves all the concatenated strains required to feed the normalizing
    flow in an hdf file.
    """
    with h5py.File(path, 'a') as file:
        n_samples = len(list_of_concatenated_strains)
        for i in range(n_samples):
            concatenated_strains = list_of_concatenated_strains[i]
            params = list_of_associated_params[i]
            dataset = file.create_dataset(f"merger_BNS_{i}",
                                          data=concatenated_strains)
            dataset.attrs["mass_1"] = float(params["mass_1"])
            dataset.attrs["mass_2"] = float(params["mass_2"])
            dataset.attrs["lambda_1"] = float(params["lambda_1"])
            dataset.attrs["lambda_2"] = float(params["lambda_2"])
            print("\rSaving files {:.0f}%".format(i/n_samples*100),
                  end="", flush=True)
        print("\rAll files saved.\n")


# Paths
local_path = "data/"
cluster_path = "/scratch/alapel/data/"

train = "trainset_freq_projected_nonoise.hdf"
evaluation = "validationset_freq_projected_nonoise.hdf"
test = "testset_freq_projected_nonoise.hdf"

# Get arguments (if file is too large and requires segmentation)
"""
description = "Generate waveforms"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--index", type=int, default=None, help="Index to indentify the waveform")
args = parser.parse_args()
"""

# Adaptive step for compression
# /!\ Not possible with pycbc waveform
# generation which requires a constant step.
"""
low_freq = np.linspace(40, 200, 320)
mid_freq = np.linspace(200, 500, 1001)
high_freq = np.linspace(500, 1024, 2681)
freq_array = np.concatenate((low_freq, mid_freq[1:], high_freq[1:]))
"""

# Compression of the waveform
freq_array = np.linspace(30, 1024, 1500)
sample_points = pycbc.types.array.Array(freq_array)
df = sample_points[1] - sample_points[0]

# Projection parameters
time_event = bilby.gw.utils.get_event_time("GW170817")
ra = 197.45*np.pi/180
dec = -23.3814*np.pi/180
polarization = 0

list_of_concatenated_strains = []
list_of_params = []
n_samples = 76000

for i in range(n_samples):

    # Uniform priors
    upper_values = [1.7, 1.36, 600]
    mass1 = np.random.uniform(1.36, upper_values[0])
    mass2 = np.random.uniform(1.17, upper_values[1])
    lambda1 = np.random.uniform(0, upper_values[2])
    lambda2 = np.random.uniform(0, upper_values[2])

    # Intrinsic parameters for the waveform
    dict_params = dict(approximant="IMRPhenomPv2_NRTidal",
                       tc=time_event,
                       distance=40.,
                       spin1z=0.02,
                       spin2z=0.02,
                       f_ref=40.,
                       sample_points=sample_points,
                       mass1=mass1,
                       mass2=mass2,
                       lambda1=lambda1,
                       lambda2=lambda2)

    # Waveform simulation in radiation frame
    hp, hc = pycbc.waveform.waveform.get_fd_waveform_sequence(**dict_params)
    hp = pycbc.types.frequencyseries.FrequencySeries(
        hp, delta_f=df).to_timeseries()
    hc = pycbc.types.frequencyseries.FrequencySeries(
        hc, delta_f=df).to_timeseries()

    # Detectors
    det_h1 = pycbc.detector.Detector('H1')
    det_l1 = pycbc.detector.Detector('L1')
    det_v1 = pycbc.detector.Detector('V1')

    # Projection
    signal_h1 = det_h1.project_wave(
        hp, hc, ra, dec, polarization).to_frequencyseries()
    signal_l1 = det_l1.project_wave(
        hp, hc, ra, dec, polarization).to_frequencyseries()
    signal_v1 = det_v1.project_wave(
        hp, hc, ra, dec, polarization).to_frequencyseries()

    # Splitting the complex FT to real
    # and imag parts to input the net
    real_h1 = signal_h1.real()
    real_l1 = signal_l1.real()
    real_v1 = signal_v1.real()

    imag_h1 = signal_h1.imag()
    imag_l1 = signal_l1.imag()
    imag_v1 = signal_v1.imag()

    # Storing + normalization
    list_of_params.append({"mass_1": mass1/upper_values[0],
                           "mass_2": mass2/upper_values[1],
                           "lambda_1": lambda1/upper_values[2],
                           "lambda_2": lambda2/upper_values[2]})
    max = np.max((real_h1, real_l1, real_v1, imag_h1, imag_l1, imag_v1))
    list_of_concatenated_strains.append(
        np.concatenate((real_h1/max, real_l1/max, real_v1/max,
                        imag_h1/max, imag_l1/max, imag_v1/max), axis=0))

    print("\rProgress: {:.0f}%".format(i/n_samples*100),
          end="", flush=True)
print("\rStrains generated.")

# Saving
save(list_of_concatenated_strains, list_of_params, cluster_path + train)
