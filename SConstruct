#!/usr/bin/env scons

import os
from os.path import join

import SCons.Script as sc
from nestly import Nest
from nestly.scons import SConsWrap

# Command line options

sc.AddOption("--output", type="string", help="output folder", default="_output")
sc.AddOption("--slurm", action="store_true", help="run do_e_test.py via SLURM srun", default=False)

env = sc.Environment(
    ENV=os.environ,
    output=sc.GetOption("output"),
    use_slurm=sc.GetOption("slurm"),
)

sc.Export("env")

env.SConsignFile()

# EXPERIMENT 1a
flag = "exp_sim_compare_withMixture"
sc.SConscript(flag + "/sconscript", exports=["flag"])

# EXPERIMENT 1b
flag = "exp_sim_generate_withMixture"
sc.SConscript(flag + "/sconscript", exports=["flag"])

# EXPERIMENT 2: SDoH pipeline audit
flag = "exp_sdoh"
sc.SConscript(flag + "/sconscript", exports=["flag"])

# EXPERIMENT 3: CUB birds simulation
flag = "exp_sim_cub"
sc.SConscript(flag + "/sconscript", exports=["flag"])

