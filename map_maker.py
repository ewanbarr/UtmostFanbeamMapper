from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

class RADecMap(object):
    def __init__(self,extent,nra,ndec):
        """
        Container for equatorial maps.

        extent: extent of map [ra0,ra1,dec0,dec1]
        nra: sampling in ra
        ndec: sampling in dec
        """
        ra0,ra1,dec0,dec1 = extent
        min_ra = min(ra0,ra1)
        max_ra = max(ra0,ra1)
        min_dec = min(dec0,dec1)
        max_dec = max(dec0,dec1)
        self.ras = np.linspace(max_ra,min_ra,nra)
        self.decs = np.linspace(max_dec,min_dec,ndec)
        self.delta_ra = self.ras[1] - self.ras[0]
        self.delta_dec = self.decs[0] - self.decs[1]
        self.map = np.zeros([ndec,nra],dtype='float32')
        
    def to_fits(self,fname):
        """
        Write map to fits file (hopefully in right order)
        
        fname: name of output file        
        """
        deg = np.degrees
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [self.map.shape[1]/2, self.map.shape[0]/2]
        wcs.wcs.cdelt = np.array([deg(self.delta_ra), deg(self.delta_dec)])
        wcs.wcs.crval = [deg(self.ras[self.map.shape[1]/2]), deg(self.decs[self.map.shape[0]/2])]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        header = wcs.to_header()
        hdu = fits.PrimaryHDU(self.map)
        hdu.header.update(header)
        hdu.writeto(fname,clobber=True)

    def max_pixel(self):
        """
        Return location and coordinate of max pixel.
        """
        max_dec_idx = self.map.max(axis=1).argmax()
        max_ra_idx = self.map.max(axis=0).argmax()
        return (max_ra_idx,max_dec_idx),(self.ras[max_ra_idx],self.decs[max_dec_idx])


def gaussian(x,mu,sig):
    """
    Get the height of a normalised Gaussian
    x: x-axis coordinate
    mu: mean
    sig: standard deviation
    """
    return np.e**(-((x-mu)**2)/(2*sig**2))

def default_beam_model(ns_offsets):
    """
    Return pixel weight based on offset from beam centre in NS axis.
    ns_offsets: scalar or array of offsets
    """
    fwhm = np.radians(2.0)
    sigma = fwhm/2.355
    return 1/gaussian(ns_offsets,0.0,sigma)

def fanbeam_model(ns_offsets,ew_offsets):
    ns_fwhm = np.radians(2.0)
    ns_sigma = ns_fwhm/2.355
    ew_fwhm = np.radians(41/3600.0)
    ew_sigma = ew_fwhm/2.355
    return 1/(gaussian(ns_offsets,0.0,ns_sigma)*gaussian(ew_offsets,0.0,ew_sigma))

    
def make_map(fanbeams,
             extent,
             nra,
             ndec,
             beam_model=default_beam_model,
             op=np.median,
             cache = True
             ):
    """ Make an equatorial map from a set of fanbeams.
    
    fanbeams: An object of basetype FanBeamTimeMap
    extent: the shape of the map [min RA, max RA, min Dec, max Dec]
    nra: the sampling in RA
    ndec: the sampling in Dec
    beam_model: a function that takes an array of NS offsets and returns an array of weights
    op: an operation to apply to the array of values extracted from the fanbeams
    """
    #create output map
    output = RADecMap(extent,nra,ndec)
    
    #get all tracks and offsets
    if cache:
        tracks,offsets = fanbeams.radecs_to_tracks(output.ras,output.decs)
    
    #loop over ra and dec values to populate map
    for ii,ra in enumerate(output.ras):
        print ii,"/",output.ras.size,"\r",
        for jj,dec in enumerate(output.decs):
            #extract a trail through fanbeam space
            if cache:
                trail,xcoords = fanbeams.extract(tracks[jj,ii],fanbeams._xcoords)
                ns_offsets = offsets[jj,ii][xcoords]
            else:
                track,ns_offsets = fanbeams.radec_to_track(ra,dec)
                trail,xcoords = fanbeams.extract(track,fanbeams._xcoords)
                ns_offsets = ns_offsets[xcoords]
            
            #inline background subtraction would happen here

            if beam_model is not None:
                #apply a weighting to the trail
                trail *= beam_model(ns_offsets)
            
            #apply op to the trail
            output.map[jj,ii] = op(trail)
    print
    return output

