#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
#import yellowbrick.style

# On rend silencieux certains avertissements
#import warnings
#warnings.filterwarnings("ignore", category=mpl.MatplotlibDeprecationWarning)
#yellowbrick.style.reset_orig()

# On place ici de quoi paramétrer le style de nos graphiques
font = {
    'family': 'sans-serif',
    'sans-serif': 'arial',
    'weight': 'normal',
    'size': 16
    }
title_font = {
    'font': 'Yanone Kaffeesatz',
    'fontweight': 'bold',
    'fontsize': 32
    }
subtitle_font = {
    'font': 'Yanone Kaffeesatz',
    'fontweight': 'normal',
    'fontsize': 20
    }

plt.rc('font', **font)
sns.set_context('notebook', font_scale=1.6)
sns.set_style('whitegrid')

chart_path = '04_presentation/charts/'
prez_path = '04_presentation/illustrations/'
savefig = {'facecolor': 'white', 'dpi': 96}

# Fonction qui permet d'exporter automatiquement un graphique en fichier image
def figsave(fig, filename):
    fig.tight_layout()
    try:
        fig.savefig(prez_path + filename, facecolor='none')
        fig.savefig(chart_path + filename, **savefig)
    except FileNotFoundError:
        fig.savefig('../' + prez_path + filename, facecolor='none')
        fig.savefig('../' + chart_path + filename, **savefig)

line_decor = 8 * '-'
