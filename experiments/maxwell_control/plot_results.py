from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'results.json').read_text())
d=np.load(ROOT/'synthetic_control.npz')
s=d['singular_values'];fraction=np.array(r['real_recording_diagnostic']['position_variance_fractions'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'text.color':'#192d40','axes.labelcolor':'#334b5c','xtick.color':'#526575','ytick.color':'#526575'})
fig,axs=plt.subplots(1,3,figsize=(12.5,6.1),gridspec_kw={'width_ratios':[1,1.25,1]})
fig.subplots_adjust(left=.075,right=.975,bottom=.24,top=.69,wspace=.49)
fig.text(.075,.92,'Can a blind learner recover field equations?',fontsize=22,weight='bold')
fig.text(.075,.86,'Synthetic positive control: yes. This real phone recording: insufficient observables.',fontsize=12)
fig.text(.075,.805,'The learner receives 106 generic candidate terms, with no divergence, curl, or Maxwell constraints.',fontsize=10.3,color='#526575')
axs[0].bar([1,2,3],fraction,color=['#175a9c','#168b7c','#bf7e24'],width=.6)
axs[0].set_yscale('log');axs[0].set_ylim(1e-5,2)
axs[0].set_xticks([1,2,3],['Direction 1','Direction 2','Direction 3'],rotation=20)
axs[0].set_ylabel('Fraction of position variance')
for x,v in zip([1,2,3],fraction):axs[0].text(x,v*1.25,f'{100*v:.3f}%',ha='center',fontsize=9)
axs[0].set_title('A · Real phone route',loc='left',fontsize=12,weight='bold',pad=18)
axs[0].grid(axis='y',alpha=.15)
xx=np.arange(1,len(s)+1)
axs[1].semilogy(xx,s/s[0],lw=2,color='#175a9c')
axs[1].axvspan(len(s)-8.5,len(s)+.5,color='#168b7c',alpha=.13)
axs[1].set(xlim=(1,len(s)+1),xlabel='Singular-value index',ylabel='Normalized singular value')
axs[1].annotate('9 approximate\nlinear identities',xy=(102,s[101]/s[0]),xytext=(20,.009),
                fontsize=10,arrowprops={'arrowstyle':'->','color':'#526575'})
axs[1].set_title('B · Synthetic data structure',loc='left',fontsize=12,weight='bold',pad=18)
axs[1].grid(alpha=.15)
labels=['All 10\nchannels','Magnetic\nchannels only','Shuffled\ncolumns']
counts=[len(r['discovered_equations']),len(r['ablation_and_negative_controls']['B_only_volume_control']['equations']),
        len(r['ablation_and_negative_controls']['independently_shuffled_columns_control']['equations'])]
axs[2].bar(range(3),counts,color=['#175a9c','#168b7c','#a9b6c2'],width=.6)
axs[2].set_xticks(range(3),labels);axs[2].set_ylim(0,10.7);axs[2].set_ylabel('Validated sparse identities')
for i,n in enumerate(counts):axs[2].text(i,n+.25,str(n),ha='center',fontsize=13,weight='bold')
axs[2].set_title('C · Synthetic controls',loc='left',fontsize=12,weight='bold',pad=18)
axs[2].grid(axis='y',alpha=.15)
fig.text(.075,.14,'Recovered: 8 scalar Maxwell relations plus charge conservation, in dimensionless units.',fontsize=10.8,weight='bold')
fig.text(.075,.101,'Held-out relative residuals: 0.10–0.20%. Synthetic pointwise noise: 0.01%; derivatives estimated by finite differences.',fontsize=9.3,color='#526575')
fig.text(.075,.065,'The synthetic generator encodes Maxwell physics. Its recovery validates the algorithm, not a new experimental discovery.',fontsize=9.3,color='#526575')
fig.text(.075,.031,'Real route: ADVIO-12 (Cortés et al., ECCV 2018). No electric field, charge density, or current-density observations are supplied by the phone.',fontsize=8.6,color='#526575')
fig.savefig(ROOT/'blind_field_discovery.png',dpi=180,facecolor='white')
