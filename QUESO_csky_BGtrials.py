'''
Author: Berit Schlüter
Inspired by: wg-nu-sources/2022_ESTES_Analyses/NSScripts_csky/csky_allsky_bgts.py by Sarah Mancina 

Purpose of this script: Calculate the background trials for given declination angles for QUESO in csky
'''

import argparse 

parser = argparse.ArgumentParser(description = 'Run BG trials for QUESO')
parser.add_argument('--dec_s', dest = 'dec_start',default = -70,  type = float,
                    help = 'declination angle starting point')
parser.add_argument('--dec_e', dest = 'dec_end',default = 70,  type = float,
                    help = 'declination angle end point')
parser.add_argument('--step', dest = 'dec_step', default = 10,  type = float,
                    help = 'declination angle step size')
parser.add_argument('-n', dest = 'n_trials', default = 1000, type = int,
                    help = 'numbor of trials for the bg estimation')
parser.add_argument('--cpu', dest = 'n_cpu', default = 20, type = int,
                    help = 'number of cpus')
parser.add_argument('-o', dest = 'save_path',
                    help = 'saving path of the output plots',
                    default = '/home/bschlueter/csky_queso_greco/QUESO_KDE/sigmacut10')
parser.add_argument('-o2', dest = 'save_path2',
                    help = 'saving path of the output files',
                    default = '/data/user/bschlueter/analysis/QUESO_KDE/sigmacut10/trials')
parser.add_argument('-a', dest = 'approach', type=int, help = "approach type: 1 = ESTES like, 2 = Greco like ")
parser.add_argument('--mcbg', dest = 'mcbg', action = 'store_true',
                    help = 'use MC background draws instead of RA randomization of data')
parser.add_argument('-s', dest = 'seed', default=1,
                    help = 'seed for the trial runner')


args      = parser.parse_args()
dec_start = args.dec_start
dec_end   = args.dec_end
dec_step  = args.dec_step
n_trials  = args.n_trials
n_cpu     = args.n_cpu
save_path = args.save_path
save_path2 = args.save_path2
approach  = args.approach
mcbg      = args.mcbg
seed      = args.seed


# Reading in all important packages

import csky as cy
import numpy as np
import pickle as pkl
import histlite as hl
import matplotlib.pyplot as plt


ana_dir = cy.utils.ensure_dir(save_path)
trials_dir = cy.utils.ensure_dir(save_path2)
bg_dir = cy.utils.ensure_dir('{}/bg'.format(trials_dir))
sig_dir = cy.utils.ensure_dir('{}/sig'.format(trials_dir))

repo = cy.selections.repo

timer = cy.timing.Timer()
time = timer.time

dec_degs = np.r_[dec_start:dec_end+1:dec_step]
n_sigs = np.r_[2:10:2, 10:30.1:4]

if approach == 1: # Approach to use Monte Carlo as BG

    if mcbg:
        with time('ana setup'):
            ana = cy.get_analysis(repo, 'MCBG', cy.selections.QUESODataSpecs.QUESO, dir=ana_dir, space_bg_kw={'bg_mc_weight':'atmo_weight'}, energy_kw ={'bg_mc_weight':'atmo_weight'}, load_sig = True)
            
        cy.CONF['ana'] = ana
        cy.CONF['mp_cpus'] = n_cpu  

    
        print('Approach: MC to model BG')
        #use mc to model background
        #scramble declination using reconstructed ang error, sigma, for bg trials
        inj_conf =  {'bg_weight_names':['atmo_weight'],
                     'randomize'      :['ra', 'dec']}
        key = 'MCBG'
        ana.save(ana_dir)
        
    
    else:
        with time('ana setup'):
            ana = cy.get_analysis(repo, 'ScrambleDecRa', cy.selections.QUESODataSpecs.QUESO, dir=ana_dir, space_bg_kw={'bg_mc_weight':'atmo_weight'}, energy_kw ={'bg_mc_weight':'atmo_weight'}, load_sig = True)
        print('Approach: scamble ra and dec')

        cy.CONF['ana'] = ana
        cy.CONF['mp_cpus'] = n_cpu  
        #use data to model background
        #scramble declination using perscribed randomization width for bg trials
        
        inj_conf = {'randomize': ['ra', cy.inj.DecRandomizer],
                    'sindec_bandwidth': np.radians(5),
                    'dec_rand_method': 'gaussian_fixed',
                    'dec_rand_kwargs': dict(randomization_width=np.radians(3)),
                    'dec_rand_pole_exlusion': np.radians(8)}
        
        key = 'ScrambleDecRa'
        ana.save(ana_dir)
        


else: #GRECO like approach
    print('Approach: MC to model BG, scramble Ra')
    ana = cy.get_analysis(cy.selections.repo,'ScrambleRa',cy.selections.QUESODataSpecs.QUESO)

    #set up config 
    cy.CONF['ana']     = ana
    cy.CONF['mp_cpus'] = n_cpu

    inj_conf = {}

    key = "ScrambleRa"
    ana.save(ana_dir)

def do_background_trials(dec_deg,key, N=n_trials, seed=seed,inj_conf = inj_conf):
    # get trial runner
    dec = np.radians(dec_deg)
    src = cy.sources(0, dec)
    tr = cy.get_trial_runner(inj_conf = inj_conf,src=src, seed=seed)
    # run trials
    trials = tr.get_many_fits(N, seed=seed, logging=False)
    # save to disk
    dir = cy.utils.ensure_dir('{}/dec/{:+04.0f}'.format(bg_dir, dec_deg))
    filename = '{}/trials__{}_N_{:06d}_seed_{:04d}.npy'.format(dir, key, N, seed)
    print('->', filename)
    # notice: trials.as_array is a numpy structured array, not a cy.utils.Arrays
    np.save(filename, trials.as_array)

def do_signal_trials(dec_deg,key, n_sig, N=n_trials, seed=seed,inj_conf = inj_conf):
    # get trial runner
    dec = np.radians(dec_deg)
    src = cy.sources(0, dec)
    tr = cy.get_trial_runner(inj_conf = inj_conf,src=src, seed=seed)
    # run trials
    trials = tr.get_many_fits(N, n_sig, poisson=True, seed=seed, logging=False)
    # save to disk
    dir = cy.utils.ensure_dir('{}/dec/{:+04.0f}/nsig/{:05.1f}'.format(sig_dir, dec_deg, n_sig))
    filename = '{}/trials__{}_N_{:06d}_seed_{:04d}.npy'.format(dir, key, N, seed)
    print('->', filename)
    # notice: trials.as_array is a numpy structured array, not a cy.utils.Arrays
    np.save(filename, trials.as_array)


# in real life you should keep track of compute time within your jobs
with time('bg trials'):
    for dec_deg in dec_degs:
        do_background_trials(dec_deg,key, N=n_trials, seed=seed)

with time('signal trials'):
    for dec_deg in dec_degs:
        for n_sig in n_sigs:
            do_signal_trials(dec_deg,key, n_sig, N=n_trials, seed=seed)


        


    

