import numpy
from dadi import Numerics, PhiManip, Integration
from dadi.Spectrum_mod import Spectrum

'''
Models for testing two population scenarios.
'''

def SI(params, ns, pts):
    """
    Split into two populations, no migration.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    T: Time in the past of split (in units of 2*Na generations).
    """
    nu1, nu2, T = params

    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    phi = Integration.two_pops(phi, xx, T, nu1, nu2, m12=0, m21=0)

    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs


def IM(params, ns, pts):
    """
    Split into two populations, with different migration rates.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    T: Time in the past of split (in units of 2*Na generations).
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2 (2*Na*m21).
    """
    nu1, nu2, m12, m21, T = params
    xx = Numerics.default_grid(pts)
    
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    
    phi = Integration.two_pops(phi, xx, T, nu1, nu2, m12=m12, m21=m21)
    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    
    return fs    


def SC(params, ns, pts):
    """
    Split with no gene flow, followed by period of asymmetrical gene flow.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    T1: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    T2: The scaled time between the secondary contact and present.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2 (2*Na*m21).
    """
    nu1, nu2, m12, m21, T1, T2 = params

    xx = Numerics.default_grid(pts)
    
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    phi = Integration.two_pops(phi, xx, T1, nu1, nu2, m12=0, m21=0)

    phi = Integration.two_pops(phi, xx, T2, nu1, nu2, m12=m12, m21=m21)

    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs


def AM(params, ns, pts):
    """
    Split with asymmetric migration followed by isolation.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2 (2*Na*m21).
    T1: The scaled time between the split and the ancient migration (in units of 2*Na generations).
    T2: The scaled time between the ancient migration and present (in units of 2*Na generations).
    """
    nu1, nu2, m12, m21, T1, T2 = params

    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    
    phi = Integration.two_pops(phi, xx, T1, nu1, nu2, m12=m12, m21=m21)
    
    phi = Integration.two_pops(phi, xx, T2, nu1, nu2, m12=0, m21=0)

    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs


def SI_size(params, ns, pts):
    """
    Split with no migration, then size change with no migration.

    nu1a: Size of population 1 after split.
    nu2a: Size of population 2 after split.
    nu1b: Size of population 1 after time interval.
    nu2b: Size of population 2 after time interval.
    T1: Time in the past of split (in units of 2*Na generations)
    T2: Time of population size change (in units of 2*Na generations)
    """
    nu1a, nu2a, nu1b, nu2b, T1, T2 = params

    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    phi = Integration.two_pops(phi, xx, T1, nu1a, nu2a, m12=0, m21=0)
    
    phi = Integration.two_pops(phi, xx, T2, nu1b, nu2b, m12=0, m21=0)

    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs


def SC_size(params, ns, pts):
    """
    Split with no gene flow, followed by size change with asymmetrical gene flow.

    nu1a: Size of population 1 after split.
    nu2a: Size of population 2 after split.
    nu1b: Size of population 1 after time interval.
    nu2b: Size of population 2 after time interval.
    T1: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    T2: The scale time between the secondary contact and present (in units of 2*Na generations).
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2 (2*Na*m21).
    """
    nu1a, nu2a, nu1b, nu2b, m12, m21, T1, T2 = params

    xx = Numerics.default_grid(pts)
    
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    
    phi = Integration.two_pops(phi, xx, T1, nu1a, nu2a, m12=0, m21=0)

    phi = Integration.two_pops(phi, xx, T2, nu1b, nu2b, m12=m12, m21=m21)
    
    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs


def AM_size(params, ns, pts):
    """
    Split with asymmetrical gene flow, followed by size change with no gene flow.

    nu1a: Size of population 1 after split.
    nu2a: Size of population 2 after split.
    nu1b: Size of population 1 after time interval.
    nu2b: Size of population 2 after time interval.
    T1: Time in the past of split (in units of 2*Na generations)
    T2: The scale time between the ancient migration and present (in units of 2*Na generations).
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2 (2*Na*m21).
    """
    nu1a, nu2a, nu1b, nu2b, m12, m21, T1, T2 = params

    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    
    phi = Integration.two_pops(phi, xx, T1, nu1a, nu2a, m12=m12, m21=m21)
    
    phi = Integration.two_pops(phi, xx, T2, nu1b, nu2b, m12=0, m21=0)

    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs


def IM_size(params, ns, pts):
    """
    Split with asymmetrical gene flow, followed by size change with asymmetrical gene flow. Levels of gene flow are stable over T1 and T2.

    nu1a: Size of population 1 after split.
    nu2a: Size of population 2 after split.
    nu1b: Size of population 1 after time interval.
    nu2b: Size of population 2 after time interval.
    T1: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    T2: The scale time between the secondary contact and present (in units of 2*Na generations).
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2 (2*Na*m21).
    """
    nu1a, nu2a, nu1b, nu2b, m12, m21, T1, T2 = params

    xx = Numerics.default_grid(pts)
    
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    
    phi = Integration.two_pops(phi, xx, T1, nu1a, nu2a, m12=m12, m21=m21)

    phi = Integration.two_pops(phi, xx, T2, nu1b, nu2b, m12=m12, m21=m21)
    
    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs


def IM_size_asym(params, ns, pts):
    """
    Split with asymmetrical gene flow, followed by size change with asymmetrical gene flow. Levels of gene flow changed over T1 and T2.

    nu1a: Size of population 1 after split.
    nu2a: Size of population 2 after split.
    nu1b: Size of population 1 after time interval.
    nu2b: Size of population 2 after time interval.
    T1: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    T2: The scale time between the secondary contact and present (in units of 2*Na generations).
    m12a: Migration from pop 2 to pop 1 after T1 (2*Na*m12a).
    m21a: Migration from pop 1 to pop 2 after T1 (2*Na*m21a).
    m12b: Migration from pop 2 to pop 1 after T2 (2*Na*m12b).
    m21b: Migration from pop 1 to pop 2 after T2 (2*Na*m21b).
    """
    nu1a, nu2a, nu1b, nu2b, m12a, m21a, m12b, m21b, T1, T2 = params

    xx = Numerics.default_grid(pts)
    
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    
    phi = Integration.two_pops(phi, xx, T1, nu1a, nu2a, m12=m12a, m21=m21a)

    phi = Integration.two_pops(phi, xx, T2, nu1b, nu2b, m12=m12b, m21=m21b)
    
    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs
