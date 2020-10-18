import bilby

sly = bilby.gw.eos.TabularEOS("SLY4")
sly_family = bilby.gw.eos.EOSFamily(sly, 100)
