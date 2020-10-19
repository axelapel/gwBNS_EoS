import numpy as np

import bilby
from bilby.gw.eos import TabularEOS, EOSFamily

outdir = "/scratch/alapel/mcmc_out"
label = "bns_eos_mcmc_sampling"
bilby.core.utils.setup_logger(outdir=outdir, label=label)

np.random.seed(16)

# Using a consistent EoS to inject params
sly_eos = TabularEOS("SLY")
sly_fam = EOSFamily(sly_eos)

mass_1 = 1.5
mass_2 = 1.3
lambda_1 = sly_fam.lambda_from_mass(mass_1)
lambda_2 = sly_fam.lambda_from_mass(mass_2)

deg2rad = np.pi / 180.
event = "GW170817"
time_event = bilby.gw.utils.get_event_time(event)
injection_parameters = dict(mass_1=mass_1, mass_2=mass_2,
                            chi_1=0.02, chi_2=0.02,
                            luminosity_distance=40.,
                            theta_jn=146*deg2rad,
                            psi=151*deg2rad,
                            phase=1.3, geocent_time=time_event,
                            ra=197.45*deg2rad, dec=-23.3814*deg2rad,
                            lambda_1=lambda_1, lambda_2=lambda_2)

duration = 90
sampling_frequency = 2 * 1024
start_time = time_event + 2 - duration

waveform_arguments = dict(waveform_approximant='IMRPhenomPv2_NRTidal',
                          reference_frequency=50., minimum_frequency=40.0)

# Create the waveform_generator
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=waveform_arguments)

# Set up interferometers
interferometers = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
for interferometer in interferometers:
    interferometer.minimum_frequency = 40
interferometers.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=start_time)
interferometers.inject_signal(parameters=injection_parameters,
                              waveform_generator=waveform_generator)

# Priors
priors = bilby.gw.prior.BNSPriorDict()
for key in ['psi', 'geocent_time', 'ra', 'dec', 'chi_1', 'chi_2',
            'theta_jn', 'luminosity_distance', 'phase']:
    priors[key] = injection_parameters[key]

priors.pop('mass_1')
priors.pop('mass_2')
priors.pop('lambda_1')
priors.pop('lambda_2')
priors.pop("mass_ratio")
priors.pop("chirp_mass")
priors['mass_1'] = bilby.core.prior.Uniform(
    1.36, 1.7, name='mass_1')
priors['mass_2'] = bilby.core.prior.Uniform(
    1.17, 1.36, name='mass_2')
priors['lambda_1'] = bilby.core.prior.Uniform(
    0, 600, name='lambda_1')
priors['lambda_2'] = bilby.core.prior.Uniform(
    0, 600, name='lambda_2')

# For spectral decomposition
"""
priors['chirp_mass'] = bilby.core.prior.Gaussian(
    1.215, 0.1, name='chirp_mass', unit='$M_{\\odot}$')
priors['symmetric_mass_ratio'] = bilby.core.prior.Uniform(
    0.1, 0.25, name='symmetric_mass_ratio')
priors['eos_spectral_gamma_0'] = bilby.core.prior.Uniform(
    0.2, 2.0, name='gamma0', latex_label='$\\gamma_0')
priors['eos_spectral_gamma_1'] = bilby.core.prior.Uniform(
    -1.6, 1.7, name='gamma1', latex_label='$\\gamma_1')
priors['eos_spectral_gamma_2'] = bilby.core.prior.Uniform(
    -0.6, 0.6, name='gamma2', latex_label='$\\gamma_2')
priors['eos_spectral_gamma_3'] = bilby.core.prior.Uniform(
    -0.02, 0.02, name='gamma3', latex_label='$\\gamma_3')
"""

# Check causality and monotonicity
priors['eos_check'] = bilby.gw.prior.EOSCheck()

# Initialise the likelihood
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=interferometers, waveform_generator=waveform_generator,
    time_marginalization=False, phase_marginalization=False,
    distance_marginalization=False, priors=priors)

# Run sampler
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='emcee', npoints=1000,
    injection_parameters=injection_parameters, outdir=outdir, label=label,
    conversion_function=bilby.gw.conversion.generate_all_bns_parameters,
    resume=True)

result.plot_corner(save=True)
