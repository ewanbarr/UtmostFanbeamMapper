import coords
import numpy as np
import ephem
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter,MultipleLocator

size = 1000

def gaussian(x,mu,sig):
    return np.e**(-((x-mu)**2)/(2*sig**2))

def fanbeam_model(ns_mean,ew_mean,ns_offsets,ew_offsets):
    ns_fwhm = np.radians(2.0)
    ns_sigma = ns_fwhm/2.355
    ew_fwhm = np.radians(41/3600.0)
    ew_sigma = ew_fwhm/2.355
    return (gaussian(ns_offsets,ns_mean,ns_sigma)*gaussian(ew_offsets,ew_mean,ew_sigma))

def sigprob(sigma):
    return 1-(2*(1-norm.cdf(sigma,0.0,1.0)))

def get_beam_loc(ra,dec,utc,beam_a,beam_b,sn_a,sn_b):
    sn_a = float(sn_a)
    sn_b = float(sn_b)
    boresight = ephem.Equatorial(ra,dec)

    ra_a,dec_a,info = coords.radec_of_offset_fanbeam(boresight.ra,boresight.dec,beam_a,utc)
    ns_a,ew_a = info['fanbeam_nsew']
    lst = info['mol_curr'].sidereal_time()
    
    ra_b,dec_b,info = coords.radec_of_offset_fanbeam(boresight.ra,boresight.dec,beam_b,utc)
    ns_b,ew_b =info['fanbeam_nsew']
    
    ns = np.linspace(np.radians(-3.0),np.radians(3.0),size)
    ew = np.linspace(np.radians(-100/3600.0),np.radians(100/3600.0),size)
    ns,ew = np.meshgrid(ns,ew)
    
    beam_a_map = fanbeam_model(ns_a,ew_a,ns+ns_a,ew+ew_a)
    beam_b_map = fanbeam_model(ns_b,ew_b,ns+ns_a,ew+ew_a)

    ratio = sn_a/sn_b
    ratio_sigma = ratio * np.sqrt(1/sn_a**2 + 1/sn_b**2)
    
    ratio_map = beam_a_map/beam_b_map
    ns_fwhm = np.radians(2.0)
    prob_map = norm.pdf(ratio_map,ratio,ratio_sigma) * norm.pdf(ns,0.0,ns_fwhm/2.355)
    prob_map/=prob_map.max()
    
    ns = ns+ns_a
    ew = ew+ew_a
    best_ns = ns[(prob_map.max(axis=1).argmax(),prob_map.max(axis=0).argmax())]
    best_ew = ew[(prob_map.max(axis=1).argmax(),prob_map.max(axis=0).argmax())]
    ha,dec = coords.nsew_to_hadec(best_ns,best_ew)
    ra,dec = coords.radec_to_J2000(lst-ha,dec,utc)
    return ra,dec,ns,ew,prob_map,lst

def deg_to_hhmmss(x,pos):
    return ephem.hours(np.radians(x))

def deg_to_ddmmss(x,pos):
    return ephem.degrees(np.radians(x))
    
def make_plots(ns,ew,prob,utc,lst,title):
    cont_fig = plt.figure(1)
    ax0 = cont_fig.add_subplot(111)
    cs = ax0.contour(ew,ns,prob,[1-sigprob(1),1-sigprob(2),1-sigprob(3)])
    
    ### switch to other plot

    fig = plt.figure(2)
    sigma1 = cs.collections[0].get_paths()[0].vertices
    sigma2 = cs.collections[1].get_paths()[0].vertices
    sigma3 = cs.collections[2].get_paths()[0].vertices
    
    ax3 = plt.subplot2grid([3,3],[1,0],rowspan=2,colspan=2)
    ax1 = plt.subplot2grid([3,3],[0,0],rowspan=1,colspan=2,sharex=ax3)
    ax2 = plt.subplot2grid([3,3],[1,2],rowspan=2,colspan=1,sharey=ax3)
    
    a = prob.max(axis=1).argmax()
    prob_best = prob[(a,np.arange(size))]
    ns_best = ns[(a,np.arange(size))]
    ew_best = ew[(a,np.arange(size))]

    hadec_best = np.array([coords.nsew_to_hadec(i,j) for i,j in zip(ns_best,ew_best)])
    ra_best,dec_best = np.array([coords.radec_to_J2000(lst-i,j,utc) for i,j in hadec_best]).T
        
    hadec = np.array([coords.nsew_to_hadec(ns_,ew_) for ew_,ns_ in sigma3])
    radec = np.array([coords.radec_to_J2000(lst-i,j,utc) for i,j in hadec])
    
    radec = np.degrees(radec)
    ra_best = np.degrees(ra_best)
    dec_best = np.degrees(dec_best)

    ax3.xaxis.set_major_formatter(FuncFormatter(deg_to_hhmmss))
    ax3.yaxis.set_major_formatter(FuncFormatter(deg_to_ddmmss))
    ax3.xaxis.set_major_locator(MultipleLocator(base=10*60/3600.0))
    ax3.yaxis.set_major_locator(MultipleLocator(base=1.0))

    ax1.plot(ra_best,prob_best)
    ax2.plot(prob_best,dec_best)
    ax3.plot(radec.T[0],radec.T[1])

    plt.sca(ax3)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=60)
    ax3.grid()
    ax2.grid()
    ax1.grid()
    ax3.set_xlabel("Right Ascension (J2000)")
    ax3.set_ylabel("Declination (J2000)")

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_xlabel("Prob.")
    ax1.set_ylabel("Prob.")
    plt.sca(ax1)
    plt.title(title)
    fig.tight_layout()
    plt.draw()

    # back to cont_fig
    cont_fig = plt.figure(1)
    ew = np.degrees(ew)
    ns = np.degrees(ns)
    
    ax0.cla()
    ax0.yaxis.set_major_formatter(FuncFormatter(deg_to_ddmmss))
    ax0.xaxis.set_major_formatter(FuncFormatter(deg_to_ddmmss))
    ax0.xaxis.set_major_locator(MultipleLocator(base=5/3600.0))
    ax0.yaxis.set_major_locator(MultipleLocator(base=1.0))
    cs = ax0.contour(ew,ns,prob,[1-sigprob(1),1-sigprob(2),1-sigprob(3)])
    fmt = {}
    strs = ['$1\sigma$', '$2\sigma$', '$3\sigma$']
    for l, s in zip(cs.levels, strs):
        fmt[l] = s
    plt.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=15,use_clabeltext=True)
    extent = [ew.min(),ew.max(),ns.min(),ns.max()]
    # limits
    ew_,ns_ = np.degrees(sigma3).T
    plt.xlim(ew_.min()-10/3600.0,ew_.max()+10/3600.0)
    plt.ylim(ns_.min()-1,ns_.max()+1)
    im = plt.imshow(np.sqrt(prob.T),aspect='auto',cmap='Greys',extent=extent)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=60) 
    plt.grid()
    plt.xlabel("Meridian angle (deg)")
    plt.ylabel("North-South angle (deg)")
    plt.title("%s (Telescope coordinates)"%title)
    cont_fig.tight_layout()


def test(a,b):    
    ns,ew,prob,lst = get_beam_loc("07:55:42.48","-29:33:49.2",'2016/3/17 09:00:36',212,213,a,b)
    make_plots(ns,ew,prob,'2016/3/17 09:00:36',lst,"FRB160317 Location")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='FRB localisation map arguments.')
    parser.add_argument('--coords', type=str, 
                      help='J2000 RA Dec of boresight beam in "HH:MM:SS.SS DD:MM:SS.SS" format',
                        required=True)
    parser.add_argument('--utc', type=str, 
                        help='UTC time in "yy/mm/ss HH:MM:SS" format (note the space!)',
                        required=True)
    parser.add_argument('--beams', type=int, nargs=2, 
                        help='The beams the the FRB falls in (brightest first)',required=True)
    parser.add_argument('--sn', type=float, nargs=2, 
                        help='The S/N in each beam (highest first)',required=True)
    parser.add_argument('--title', type=str,  help='Plot title base', 
                        default="FRB Location")
    args = parser.parse_args()
    ra,dec = args.coords.split()
    beam_a,beam_b = args.beams
    sn_a,sn_b = args.sn
    best_ra,best_dec,ns,ew,prob,lst = get_beam_loc(ra,dec,args.utc,beam_a,beam_b,sn_a,sn_b)
    print "Best coordinates for FRB:"
    print "R.A.  (J2000):",ephem.hours(best_ra)
    print "Decl. (J2000):",ephem.degrees(best_dec)
    make_plots(ns,ew,prob,args.utc,lst,args.title)
    plt.show()




