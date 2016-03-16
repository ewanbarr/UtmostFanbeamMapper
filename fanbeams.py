from astropy.io import fits
from coords import nsew_of_constant_dec,Molonglo,hadec_to_nsew
from scipy.ndimage import filters
import ephem as e
import numpy as np
import copy

SEC_TO_SID = 0.9972685185185185 # convert seconds to siderial "seconds"
EW_R = np.radians(2.0) # EW beam HWHM
NS_R = np.radians(1.0) # NS beam HWHM

class FanBeamTimeMap(object):
    def __init__(self,data,lsts):
        """
        Base class for fanbeams.
        
        data: numpy array containing data with axes [nbeams,nsamps]
        lsts: the lsts for each set of beams
        """
        self.data = data
        self.nsamps = data.shape[1]
        self.nbeams = data.shape[0]
        self.data[0] = self.data[1]
        self.lsts = lsts
        self.mask = np.ones(data.shape,dtype='bool')
        self._cache = {}
        self._xcoords = np.arange(self.nsamps)

    def radec_to_track(self,ra,dec):
        """
        Convert a single ra and dec to a track through the fanbeams.
        
        ra: ra in radians
        dec: dec in radians
        
        Note: does not cache track

        Returns fanbeam indexes and NS offsets
        """
        ns,ew = nsew_of_constant_dec(self.lsts-ra,dec)
        ew_offset = ew-self.ew
        ns_offset = ns-self.ns
        idxs = (self.nbeams * (ew_offset/self.ew_r + 1)/2).astype("int")
        return idxs, ns_offset

    def radecs_to_tracks(self,ras,decs):
        """
        Convert an array of ra and dec coords to a set of tracks and offsets.
        
        ras: ra array in radians
        decs: dec array in radians
        
        Note: Caches based on hash of inputs
        
        Returns fanbeam indexes and NS offsets
        """
        key = (hash(ras.tobytes()),hash(decs.tobytes()))
        if key in self._cache.keys():
            return self._cache[key]
        tracks = np.empty([decs.size,ras.size,self.nsamps],dtype='int32')
        offsets = np.empty([decs.size,ras.size,self.nsamps],dtype='float32')
        for ii,ra in enumerate(ras):
            print ii,"/",ras.size,"\r",
            for jj,dec in enumerate(decs):
                idxs,ns_offset = self.radec_to_track(ra,dec)
                tracks[jj,ii] = idxs
                offsets[jj,ii] = ns_offset
        print
        self._cache[key] = (tracks,offsets)
        return tracks,offsets

    def extract(self,idxs):
        """
        Extract a trail through fanbeam space.

        idxs: array of fambeam indexes (can be any value)

        Note: only valid data are returned
        """
        mask = (idxs > 0) & (idxs < self.nbeams)
        pixel_mask = self.mask[(idxs[mask],self._xcoords[mask])]
        x = self._xcoords[mask][pixel_mask]
        return self.data[(idxs[mask][pixel_mask],x)],x
    

class TransitFanBeamTimeMap(FanBeamTimeMap):
    def __init__(self,data,ns,ew,lsts):
        """
        Fanbeam container for transit observations
        
        data: numpy array containing data with axes [nbeams,nsamps]
        lsts: the lsts for each set of beams
        ns: ns coordinate of central fanbeam (radians)
        ew: ew coordinate of central fanbeam (radians)
        """
        super(TransitFanBeamTimeMap,self).__init__(data,lsts)
        self.ns = ns
        self.ew = ew
        self.ew_r = EW_R/np.cos(self.ew)


class TrackedFanBeamTimeMap(FanBeamTimeMap):
    def __init__(self,data,ra,dec,lsts):
        """
        Fanbeam container for transit observations

        data: numpy array containing data with axes [nbeams,nsamps]
        lsts: the lsts for each set of beams
        ra: ra coordinate of central fanbeam (radians)
        dec: dec coordinate of central fanbeam (radians)
        """
        super(TrackedFanBeamTimeMap,self).__init__(data,lsts)
        self.ra = float(ra)
        self.dec = float(dec)
        self.hour_angles = self.lsts-self.ra
        ns,ew = nsew_of_constant_dec(self.hour_angles,self.dec)
        self.ns = ns
        self.ew = ew
        self.ew_r = EW_R/np.cos(self.ew)


def filter_fanbeams(fanbeams,window,thresh=None,mode='nearest'):
    """
    Apply a median filter to a set of fanbeams.

    fanbeams: An object of basetype FanBeamTimeMap
    window: size of the filter window.
    thresh: sigma threshold for clipping

    Note: filter is applied along the fanbeam axis, not the time axis
    """
    clean = copy.deepcopy(fanbeams)
    background = copy.deepcopy(fanbeams)
    data = np.copy(fanbeams.data)
    med = filters.median_filter(data,size=[window,1],mode=mode)
    subtracted = data-med
    mad = 1.4826 * np.median(abs(subtracted),axis=0)
    subtracted/=mad
    if thresh is not None:
        subtracted = subtracted.clip(max=thresh)
    clean.data = subtracted
    background.data = med
    return clean,background

def _load_fanbeams(fname,utc_start,tsamp):
    """Helper function"""
    obs = Molonglo(date=utc_start)
    lst = obs.sidereal_time()
    hdu = fits.open(fname)[0]
    dlst = 2*np.pi*tsamp/86400.0 * SEC_TO_SID
    lsts = np.arange(hdu.data.shape[1])*dlst + lst
    return hdu.data,lsts,obs

def load_tracked_fanbeams(fname,utc_start,ra,dec,tsamp):
    """Load a tracked fanbeam observation"""
    data,lsts,obs = _load_fanbeams(fname,utc_start,tsamp)
    body = e.FixedBody()
    body._ra = ra
    body._dec = dec
    body.compute(obs)
    return TrackedFanBeamTimeMap(data,body.ra,body.dec,lsts)

def load_transit_fanbeams(fname,utc_start,ha,dec,tsamp):
    """Load a transit fanbeam observation"""
    data,lsts,obs = _load_fanbeams(fname,utc_start,tsamp)
    ns,ew = hadec_to_nsew(ha,dec)
    return TransitFanBeamTimeMap(data,ns,ew,lsts)


##### testing: IGNORE #####

def test():
    ra,dec = "07:16:35","-19:00:40"
    eq = e.Equatorial(ra,dec)
    ra = eq.ra
    dec = eq.dec
    lst = float(e.hours("03:56"))
    dlst = float(e.hours("00:00:20.0001"))*SEC_TO_SID
    data = read("molonglo_fb_map.fits")
    lsts = np.arange(data.shape[1])*dlst + lst
    return FanBeamTimeMap(data,ra,dec,lst,dlst)

def test_track_map(fname="molonglo_fb_map.fits"):
    ra,dec = "07:16:35","-19:00:40"
    eq = e.Equatorial(ra,dec)
    ra = eq.ra
    dec = eq.dec
    lst = float(e.hours("03:56"))
    dlst = float(e.hours("00:00:20.0016"))*SEC_TO_SID
    data = read(fname)
    lsts = np.arange(data.shape[1])*dlst + lst
    return TrackedFanBeamTimeMap(data,ra,dec,lsts)

def track_map(fname,utc_start,ra,dec):
    obs = Molonglo(date=utc_start)
    body = e.FixedBody()
    body._ra = ra
    body._dec = dec
    body.compute(obs)
    data = read(fname)
    lst = obs.sidereal_time()
    dlst = float(e.hours("00:00:20.0016"))*SEC_TO_SID
    lsts = np.arange(data.shape[1])*dlst + lst
    return TrackedFanBeamTimeMap(data,body.ra,body.dec,lsts)

def test_transit_map():
    obs = Molonglo("2015/07/28 22:34:23")
    lst = obs.sidereal_time()
    dlst = float(e.hours("00:00:01"))*SEC_TO_SID
    data = read("SN1987A.2015-07-28-22_34_23.fixed.fits")
    lsts = np.arange(data.shape[1])*dlst + lst
    ns,ew = hadec_to_nsew(0.0,np.radians(-70.0))
    return TransitFanBeamTimeMap(data,ns,ew,lsts)
            
    
    
