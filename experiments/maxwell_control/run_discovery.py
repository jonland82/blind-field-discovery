"""Generic sparse relation discovery, with a separately labeled synthetic control.

The learner sees scalar channels, coordinates, raw quadratic products, and
numerically estimated first derivatives. No divergence/curl features, Maxwell
penalties, dipole kernels, known equation supports, or physical coefficients
are passed to the learner. The control generator does encode Maxwell physics;
success on it is algorithm validation, not experimental rediscovery.
"""
from pathlib import Path
import json
import re
import numpy as np
from scipy.optimize import linprog
from scipy.linalg import svd

ROOT=Path(__file__).resolve().parent
OUT=ROOT
OUT.mkdir(exist_ok=True)

def make_control(seed,n=512,modes=24,h=.02,noise_fraction=1e-4):
    """Generate ten channels in dimensionless units, then finite-difference them.

    Generator-only identities: E=-grad(phi)-dA/dt; B=curl(A);
    rho=div(E); J=curl(B)-dE/dt. Frequencies are arbitrary (not a
    source-free dispersion relation), so sources need not vanish.
    Channel semantic names are not returned to the learner.
    """
    rng=np.random.default_rng(seed)
    k=rng.integers(-3,4,size=(modes,3)).astype(float)
    zero=np.linalg.norm(k,axis=1)==0;k[zero,0]=1
    omega=rng.uniform(-3,3,size=modes)
    amp=rng.normal(size=(modes,4))+1j*rng.normal(size=(modes,4))
    A=amp[:,:3];phi=amp[:,3]
    E=-1j*k*phi[:,None]+1j*omega[:,None]*A
    B=1j*np.cross(k,A)
    rho=1j*np.sum(k*E,axis=1)
    J=1j*np.cross(k,B)+1j*omega[:,None]*E
    channels=np.column_stack([E,B,rho,J])
    wave=np.column_stack([k,-omega])
    points=rng.uniform(-np.pi,np.pi,size=(n,4))
    def observe(p):return np.real(np.exp(1j*(p@wave.T))@channels)/np.sqrt(modes)
    exact=observe(points);scale=np.sqrt(np.mean(exact**2,axis=0))
    value=exact+rng.normal(size=exact.shape)*noise_fraction*scale
    derivatives=[]
    for axis in range(4):
        d=np.zeros(4);d[axis]=h
        plus=observe(points+d);minus=observe(points-d)
        plus+=rng.normal(size=plus.shape)*noise_fraction*scale
        minus+=rng.normal(size=minus.shape)*noise_fraction*scale
        derivatives.append((plus-minus)/(2*h))
    return points,value,np.stack(derivatives,axis=2)

def library(value,derivative):
    N,M=value.shape
    columns=[np.ones(N)];names=['1']
    for j in range(M):columns.append(value[:,j]);names.append(f'u{j}')
    for j in range(M):
        for axis in range(4):
            columns.append(derivative[:,j,axis]);names.append(f'd{axis}(u{j})')
    for j in range(M):
        for k in range(j,M):
            columns.append(value[:,j]*value[:,k]);names.append(f'u{j}*u{k}')
    return np.column_stack(columns),names

def dataset(seeds):
    blocks=[];points=[];values=[];derivs=[];experiments=[]
    for seed in seeds:
        p,v,d=make_control(seed)
        x,names=library(v,d)
        blocks.append(x);points.append(p);values.append(v);derivs.append(d)
        experiments.extend([seed]*len(x))
    return np.concatenate(blocks),names,dict(points=np.concatenate(points),values=np.concatenate(values),
            derivatives=np.concatenate(derivs),experiment=np.array(experiments))

def discover(train,valid,names):
    """Standard SVD plus L1 optimization finds sparse approximate identities."""
    scale=np.sqrt(np.mean(train**2,axis=0));scale=np.maximum(scale,1e-10)
    X=train/scale
    _,s,Vt=svd(X,full_matrices=False)
    ratios=s[:-1]/s[1:];gap=int(np.argmax(ratios))
    if ratios[gap]<10:
        return [],s,{'nullity':0,'gap_ratio':float(ratios[gap])}
    null=Vt[gap+1:].T;d=null.shape[1];p=X.shape[1]
    # Min ||N alpha||_1 subject to one coefficient being 1. Anchor every
    # candidate term equally; no field component or derivative is privileged.
    A=np.block([[null,-np.eye(p)],[-null,-np.eye(p)]])
    objective=np.r_[np.zeros(d),np.ones(p)]
    found={}
    for anchor in range(p):
        if np.linalg.norm(null[anchor])<1e-3:continue
        eq=np.r_[null[anchor],np.zeros(p)][None,:]
        fit=linprog(objective,A_ub=A,b_ub=np.zeros(2*p),A_eq=eq,b_eq=[1.],
                    bounds=[(None,None)]*d+[(0,None)]*p,method='highs')
        if not fit.success:continue
        c=null@fit.x[:d]
        support=np.flatnonzero(abs(c)>.025*np.max(abs(c)))
        if not 2<=len(support)<=8:continue
        _,_,vv=svd(X[:,support],full_matrices=False)
        cn=vv[-1];coef=cn/scale[support]
        coef/=coef[np.argmax(abs(coef))]
        # Canonical sign and scale: first listed coefficient is +1.
        coef/=coef[0]
        def error(data):
            terms=data[:,support]*coef
            return float(np.sqrt(np.mean(terms.sum(1)**2))/np.sqrt(np.mean(np.sum(terms**2,axis=1))))
        ev=error(valid)
        if ev>.02:continue
        key=tuple(support)
        row={'terms':[names[j] for j in support],'indices':support.tolist(),
             'coefficients':coef.tolist(),'train_relative_residual':error(train),
             'validation_relative_residual':ev}
        if key not in found or ev<found[key]['validation_relative_residual']:found[key]=row
    return list(found.values()),s,{'nullity':int(d),'gap_ratio':float(ratios[gap]),'gap_index':gap}

def evaluate(eq,data):
    terms=data[:,eq['indices']]*np.array(eq['coefficients'])
    return float(np.sqrt(np.mean(terms.sum(1)**2))/np.sqrt(np.mean(np.sum(terms**2,axis=1))))

def interpret(eq):
    # Evaluation-only naming. The learner has already selected coefficients.
    mapping={**{f'u{i}':f'E{a}' for i,a in enumerate('xyz')},
             **{f'u{i+3}':f'B{a}' for i,a in enumerate('xyz')},'u6':'rho',
             **{f'u{i+7}':f'J{a}' for i,a in enumerate('xyz')}}
    def rename(term):
        if term.startswith('d') and '(u' in term:
            axis='xyzt'[int(term[1])];inner=term[term.index('(')+1:-1]
            return f'd_{axis}({mapping[inner]})'
        return mapping.get(term,term)
    return ' '.join(f'{c:+.5f} {rename(term)}' for c,term in zip(eq['coefficients'],eq['terms']))+' = 0'

if __name__=='__main__':
    train,names,tr=dataset(range(100,112))
    valid,_,va=dataset(range(200,204))
    test,_,te=dataset(range(300,304))
    equations,s,info=discover(train,valid,names)
    for eq in equations:
        eq['test_relative_residual']=evaluate(eq,test)
        eq['interpreted_equation']=interpret(eq)
        print(eq['interpreted_equation'],'test residual',eq['test_relative_residual'],flush=True)
    # Measurement ablation: the same learner gets only the three B channels.
    bonly=[i for i,name in enumerate(names) if set(re.findall(r'u\d+',name)) <= {'u3','u4','u5'}]
    beq,bs,binfo=discover(train[:,bonly],valid[:,bonly],[names[i] for i in bonly])
    for eq in beq:
        eq['test_relative_residual']=evaluate(eq,test[:,bonly])
        eq['interpreted_equation']=interpret(eq)
    # A negative control destroys between-column relations while retaining marginals.
    shuffle_rng=np.random.default_rng(901)
    shuffled_train=np.column_stack([shuffle_rng.permutation(train[:,j]) for j in range(train.shape[1])])
    shuffled_valid=np.column_stack([shuffle_rng.permutation(valid[:,j]) for j in range(valid.shape[1])])
    neq,ns,ninfo=discover(shuffled_train,shuffled_valid,names)
    controls={'B_only_volume_control':{'equations':beq,'diagnostics':binfo,'candidate_terms':len(bonly)},
              'independently_shuffled_columns_control':{'equations':neq,'diagnostics':ninfo}}
    print('CONTROLS',json.dumps(controls),flush=True)
    # Geometry diagnostic on real data; no physical coefficients imported.
    pose=np.loadtxt(ROOT.parents[1]/'data/advio-12/iphone/arkit.csv',delimiter=',')
    xyz=pose[:,1:4];_,sp,V=svd(xyz-xyz.mean(0),full_matrices=False)
    fractions=sp**2/np.sum(sp**2)
    geometry={'position_variance_fractions':fractions.tolist(),
              'variance_in_best_fit_plane':float(fractions[:2].sum()),
              'normal_rms_metres':float(sp[-1]/np.sqrt(len(xyz))),
              'largest_rms_metres':float(sp[0]/np.sqrt(len(xyz))),
              'best_fit_plane_normal':V[-1].tolist(),
              'missing_measurements':['electric field','charge density','current density'],
              'conclusion':'The phone recording cannot identify the full electromagnetic equations.'}
    summary={'control_only':True,'generator_encodes_Maxwell':True,
             'learner_inputs':'unnamed scalar channels; generic raw terms, quadratic products, and first coordinate derivatives',
             'measurement_noise_fraction':1e-4,'finite_difference_step':.02,
             'train_experiments':12,'validation_experiments':4,'test_experiments':4,
             'train_samples':len(train),'test_samples':len(test),'candidate_terms':len(names),
             'learner_diagnostics':info,'discovered_equations':equations,
             'ablation_and_negative_controls':controls,
             'real_recording_diagnostic':geometry,
             'limitations':['Synthetic recovery validates the algorithm, not real-world discovery.',
                            'Coordinates, derivatives, polynomial library and sparsity are supplied mathematical priors.',
                            'Dimensionless generator units set electromagnetic constants to one.',
                            'Observations include all E, B, rho and J channels, unlike the phone recording.',
                            'Generic finite differences use dense local sampling unavailable on the phone route.']}
    (OUT/'results.json').write_text(json.dumps(summary,indent=2))
    np.savez_compressed(OUT/'synthetic_control.npz',train=train,validation=valid,test=test,
                       feature_names=np.array(names),singular_values=s)
    np.savez_compressed(OUT/'real_route_geometry.npz',xyz=xyz,plane_basis=V,variance_fractions=fractions)
    print('DIAGNOSTICS',json.dumps(info),'equations',len(equations))
    print('REAL_GEOMETRY',json.dumps(geometry))
