import bilby
import numpy as np
import matplotlib.pyplot as plt

sly = bilby.gw.eos.TabularEOS("SLY4")
sly_family = bilby.gw.eos.EOSFamily(sly, 100)

mpa1 = bilby.gw.eos.TabularEOS("MPA1")
mpa1_family = bilby.gw.eos.EOSFamily(mpa1, 100)

total_mass = 2.73 * np.ones(10)
mass1 = np.linspace(1.36, 1.7, 10)
mass2 = total_mass - mass1

lambda1_sly = sly_family.lambda_from_mass(mass1)
lambda2_sly = sly_family.lambda_from_mass(mass2)
lambda1_mpa1 = mpa1_family.lambda_from_mass(mass1)
lambda2_mpa1 = mpa1_family.lambda_from_mass(mass2)

fig, ax = plt.subplots()
ax.plot(lambda1_sly, lambda2_sly)
ax.plot(lambda1_mpa1, lambda2_mpa1)
ax.set(xlim=(0, 1000), ylim=(0, 1000))
