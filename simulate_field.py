#!/usr/bin/env python3
"""
Roman WFI mosaic field simulator with Gaia+2MASS source statistics.

Generates mosaic footprint plots, stellar density analysis, predicted Roman
filter magnitude distributions, and nearest-neighbor separation statistics
for Roman WFI observations.

Usage examples
--------------
Target resolution (Simbad):
  python simulate_field.py --target "NGC 6819"
  python simulate_field.py --target "M31" --pa 45 --filter F087
  python simulate_field.py --target "47 Tuc" --bright-limit 12

Direct coordinates (--ra accepts any astropy Angle format):
  python simulate_field.py --ra 295.3275 --dec 40.1878
  python simulate_field.py --ra "19:41:18" --dec "+40:11:16" --nickname "MyField"
  python simulate_field.py --ra 19h41m18s --dec +40d11m16s --nickname "MyField"
  python simulate_field.py --ra 10.68 --dec 41.27 --nickname "MyField"
  python simulate_field.py --ra 80.5 --dec -69.5 --nickname "LMC"

Coordinates with custom target name:
  python simulate_field.py --ra 295.3275 --dec 40.1878 --target NGC6819
  python simulate_field.py --ra "17:20:12.4" --dec "-26:29:21" --target "GC field"

Advanced options:
  python simulate_field.py --ra 290.7 --dec 44.5 --nickname Kepler --radius 0.8
  python simulate_field.py --ra 80.5 --dec -69.5 --nickname LMC --center-sca WFI01
  python simulate_field.py --target "NGC 362" --filter F106 --show --separate
  python simulate_field.py --ra 84.658 --dec -69.095 --nickname "30Dor" --full-query
  python simulate_field.py --target "NGC 6819" --star-prob 0.8 --bright-limit 11
  python simulate_field.py --target "NGC 6819" --gaia-only   # skip 2MASS join
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path as MplPath
import pysiaf
from pysiaf.utils import rotations

from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astroquery.gaia import Gaia


# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------

# AB - Vega offsets for Gaia DR3 (Riello+2021 Tab 5.6)
G_AB_OFFSET  =  0.131
BP_AB_OFFSET =  0.067
RP_AB_OFFSET =  0.354

# AB - Vega offsets for 2MASS (Blanton & Roweis 2007)
J_AB_OFFSET  =  0.91
H_AB_OFFSET  =  1.39
KS_AB_OFFSET =  1.85

# Filters for which Tier-1 (Gaia+2MASS) calibration applies
TIER1_FILTERS = frozenset({'F106', 'F129', 'F146', 'F158', 'F184', 'F213'})

# 2MASS is shallow (~Ks 14.3); only crossmatch bright Gaia sources
TMASS_G_LIMIT = 16.0

# Cumulative AB magnitude thresholds for per-SCA counts
THRESHOLDS = [14, 15, 16, 17, 18, 19]

# Default calibration directory (allsky_maps, sibling of roman-sims)
_DEFAULT_CALIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', 'allsky_maps')


# ---------------------------------------------------------------------------
# Calibration loading and application
# ---------------------------------------------------------------------------

def load_calibration(filt: str, calib_dir: str, tag: str = 'hybrid') -> dict:
    """Load per-filter calibration npz (Tier-1 and/or Tier-2)."""
    tagged   = os.path.join(calib_dir, f'gaia_{filt}_calibration_{tag}.npz') if tag else None
    untagged = os.path.join(calib_dir, f'gaia_{filt}_calibration.npz')
    path = tagged if (tagged and os.path.exists(tagged)) else untagged
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'No calibration file for {filt} (tried {tagged!r} and {untagged!r}). '
            f'Point --calib-dir at the allsky_maps directory.')
    d = np.load(path, allow_pickle=True)
    c = {
        'coef':        np.array(d['coef']),
        'color_basis': str(d['color_basis']),
        'color_min':   float(d['color_min']),
        'color_max':   float(d['color_max']),
        'rms':         float(d['rms']),
        'has_g2m':     False,
    }
    if 'coef_g2m' in d.files and filt in TIER1_FILTERS:
        c.update({
            'has_g2m':    True,
            'coef_g2m':   np.array(d['coef_g2m']),
            'gaia_basis': str(d['gaia_basis']),
            'tmass_basis':str(d['tmass_basis']),
            'rms_g2m':    float(d['rms_g2m']),
            'g2m_x_min':  float(d['g2m_x_min']), 'g2m_x_max': float(d['g2m_x_max']),
            'g2m_y_min':  float(d['g2m_y_min']), 'g2m_y_max': float(d['g2m_y_max']),
        })
    return c


def _color_from_bands(basis: str, bands: dict) -> np.ndarray:
    a, b = basis.split('-')
    return bands[a] - bands[b]


def compute_roman_mag(bands: dict, calib: dict) -> tuple:
    """
    Predict F_AB for all sources.

    Chooses Tier-1 (Gaia+2MASS) per source where 2MASS is available,
    falls back to Tier-2 (Gaia-only) otherwise.

    Returns (f_ab, no_color_mask, used_tier1_mask).
    """
    G  = bands['G']
    n  = G.shape[0]

    # Tier-2 (always computable)
    x2 = _color_from_bands(calib['color_basis'], bands)
    no_color = ~np.isfinite(x2)
    if no_color.any() and (~no_color).any():
        x2 = x2.copy(); x2[no_color] = float(np.nanmedian(x2[~no_color]))
    elif no_color.all():
        x2 = np.full_like(x2, 1.5)
    x2c    = np.clip(x2, calib['color_min'], calib['color_max'])
    f_t2   = G + np.polyval(calib['coef'], x2c)

    used_t1 = np.zeros(n, dtype=bool)
    if not calib.get('has_g2m', False):
        return f_t2, no_color, used_t1

    # Tier-1 (Gaia+2MASS) where available
    xg = _color_from_bands(calib['gaia_basis'],  bands)
    yt = _color_from_bands(calib['tmass_basis'], bands)
    good = np.isfinite(xg) & np.isfinite(yt)
    used_t1 = good

    f_ab = f_t2.copy()
    if good.any():
        xc = np.clip(xg[good], calib['g2m_x_min'], calib['g2m_x_max'])
        yc = np.clip(yt[good], calib['g2m_y_min'], calib['g2m_y_max'])
        b  = calib['coef_g2m']
        f_ab[good] = (bands['Ks'][good]
                      + b[0] + b[1]*xc + b[2]*xc**2 + b[3]*xc**3
                      + b[4]*yc + b[5]*yc**2)
    return f_ab, no_color, used_t1


# ---------------------------------------------------------------------------
# 2MASS crossmatch query
# ---------------------------------------------------------------------------

def query_2mass_xmatch(ra_deg: float, dec_deg: float, radius_deg: float,
                       tmass_glim: float = TMASS_G_LIMIT,
                       max_retries: int = 3) -> 'pd.DataFrame':
    """
    Query 2MASS J/H/Ks for bright Gaia sources in the field via the
    Gaia DR3 cross-match table. Returns a DataFrame with columns
    (source_id, tmass_j, tmass_h, tmass_ks).
    """
    import pandas as pd
    q = (
        "SELECT g.source_id, "
        "       tm.j_m  AS tmass_j, "
        "       tm.h_m  AS tmass_h, "
        "       tm.ks_m AS tmass_ks "
        "FROM ( "
        "    SELECT source_id "
        "    FROM gaiadr3.gaia_source "
        "    WHERE CONTAINS(POINT('ICRS', ra, dec), "
        f"                  CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg})) = 1 "
        f"      AND phot_g_mean_mag < {tmass_glim} "
        ") AS g "
        "LEFT JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xm "
        "       ON g.source_id = xm.source_id "
        "LEFT JOIN gaiadr1.tmass_original_valid AS tm "
        "       ON xm.original_ext_source_id = tm.designation"
    )
    import time
    last = None
    for _attempt in range(max_retries):
        try:
            job = Gaia.launch_job_async(q, dump_to_file=False)
            return job.get_results().to_pandas()
        except Exception as e:
            last = e
            time.sleep(5 * (2 ** _attempt))
    raise RuntimeError(f'2MASS crossmatch query failed: {last}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Roman WFI mosaic field simulator with Gaia+2MASS source statistics.",
        epilog="""
Examples:
  # Resolve target by name via Simbad
  python simulate_field.py --target "NGC 6819"

  # Direct RA/Dec coordinates (decimal degrees, HH:MM:SS, 19h41m18s, 295d19m39s, …)
  python simulate_field.py --ra 290.7 --dec 44.5 --nickname "Kepler"
  python simulate_field.py --ra "19:41:18" --dec "+40:11:16" --nickname "NGC6819"

  # Custom SCA centering and position angle
  python simulate_field.py --target "M31" --center-sca WFI10 --pa 45

  # Dense field with full Gaia query and separate output files
  python simulate_field.py --ra 80.5 --dec -69.5 --nickname "LMC" --full-query --separate

  # Strict star classification and interactive display
  python simulate_field.py --target "NGC 362" --star-prob 0.9 --bright-limit 11 --show

  # Skip 2MASS crossmatch (Gaia-only Tier-2 calibration)
  python simulate_field.py --target "NGC 6819" --gaia-only
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_argument_group("Target (--target OR --ra/--dec required)")
    g.add_argument("--target", metavar="NAME",
                   help="Object name resolved via Simbad (e.g. 'NGC 6819'). "
                        "If --ra/--dec are also given, the name is used only as a label.")
    g.add_argument("--ra",  type=str, metavar="RA",
                   help="Right ascension (J2000); any astropy Angle format accepted: "
                        "decimal degrees (295.3275), HH:MM:SS.s (19:41:18), "
                        "19h41m18s, 295d19m39s, etc.")
    g.add_argument("--dec", type=str, metavar="DEC",
                   help="Declination (J2000); any astropy Angle format accepted: "
                        "decimal degrees (40.1878), DD:MM:SS (+40:11:16), "
                        "+40d11m16s, etc.")
    g.add_argument("--nickname", metavar="NAME",
                   help="Custom label for the field when using --ra/--dec (default: auto-generated from coordinates)")

    p.add_argument("--pa", type=float, default=0.0,
                   help="WFI position angle (deg E of N)")
    p.add_argument("--center-sca", default=None, metavar="SCA",
                   help="SCA aperture to place on target, e.g. WFI01 or 1 (default: WFI_CEN)")
    p.add_argument("--filter", dest="optical_element", default="F106", metavar="FILTER",
                   help="Roman WFI filter for predicted magnitude analysis")
    p.add_argument("--radius", type=float, default=0.5,
                   help="Gaia query radius (deg)")
    p.add_argument("--bright-limit", type=float, default=13.5,
                   help="Gaia G-mag bright cutoff (exclude brighter; removes saturated sources)")
    p.add_argument("--star-prob", type=float, default=0.7,
                   help="Minimum Gaia DSC star-class probability")
    p.add_argument("--show", action="store_true",
                   help="Display plots interactively in addition to saving")
    p.add_argument("--separate", action="store_true",
                   help="Save individual PNG files for each product in addition to the combined output")
    p.add_argument("--outdir", default=".", metavar="DIR",
                   help="Directory for saved plots and cached catalog")
    p.add_argument("--full-query", action="store_true",
                   help="Retrieve all Gaia columns (default: only essential photometric columns)")
    p.add_argument("--gaia-only", action="store_true",
                   help="Skip 2MASS crossmatch; use Tier-2 (Gaia-only) calibration only")
    p.add_argument("--tmass-glim", type=float, default=TMASS_G_LIMIT, metavar="G",
                   help=f"Only crossmatch 2MASS for Gaia sources brighter than this G (Vega) "
                        f"magnitude. 2MASS is shallow, so faint sources rarely match. "
                        f"Default: {TMASS_G_LIMIT}.")
    p.add_argument("--calib-dir", default=_DEFAULT_CALIB_DIR, metavar="PATH",
                   help="Directory containing gaia_<F>_calibration_hybrid.npz files. "
                        f"Default: {_DEFAULT_CALIB_DIR}")
    p.add_argument("--calib-tag", default="hybrid", metavar="TAG",
                   help="Calibration file suffix (default: hybrid). "
                        "Loads gaia_{F}_calibration_{tag}.npz.")
    # argparse treats values starting with '-' (e.g. "-71:44:16") as flags;
    # merge --ra/--dec with their value using '=' so negative coords are accepted.
    argv = sys.argv[1:]
    fixed = []
    i = 0
    while i < len(argv):
        if argv[i] in ('--ra', '--dec') and i + 1 < len(argv) and argv[i + 1].startswith('-'):
            fixed.append(f'{argv[i]}={argv[i + 1]}')
            i += 2
        else:
            fixed.append(argv[i])
            i += 1
    return p.parse_args(fixed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_angle(value: str, is_ra: bool) -> float:
    """Parse any astropy-supported angle string and return decimal degrees.

    RA defaults to hourangle units when the string looks like HH:MM:SS
    (no degree symbol or 'd' suffix and contains ':') so that '19:41:18'
    is treated as hours, not degrees.  All other strings (including plain
    floats) default to degrees.
    """
    looks_like_hms = is_ra and ':' in value and 'd' not in value.lower() and 'h' not in value.lower()
    unit = u.hourangle if looks_like_hms else u.deg
    try:
        return float(Angle(value, unit=unit).deg)
    except Exception as e:
        kind = "RA" if is_ra else "Dec"
        sys.exit(f"ERROR: cannot parse {kind} '{value}': {e}")


def resolve_target(args):
    """Return (ra_deg, dec_deg, label)."""
    if args.ra is not None and args.dec is not None:
        ra  = _parse_angle(args.ra,  is_ra=True)
        dec = _parse_angle(args.dec, is_ra=False)
        if args.nickname:
            label = args.nickname.replace(" ", "")
        elif args.target:
            label = args.target.replace(" ", "")
        else:
            label = f"RA{ra:.4f}Dec{dec:+.4f}"
        return ra, dec, label
    if args.target:
        print(f"Resolving '{args.target}' via Simbad …")
        sc = SkyCoord.from_name(args.target)
        ra, dec = sc.ra.deg, sc.dec.deg
        print(f"  -> RA = {ra:.5f} deg, Dec = {dec:.5f} deg")
        return ra, dec, args.target.replace(" ", "")
    sys.exit("ERROR: provide --target NAME or both --ra DEG and --dec DEG.")


def poly_area_arcmin2(ra_corners, dec_corners, ra_center=None, dec_center=None):
    """Polygon area (arcmin²) via planar shoelace on a local tangent plane."""
    if ra_center is None:
        ra_c = np.mean(ra_corners)
    else:
        ra_c = ra_center
    if dec_center is None:
        dec_c = np.mean(dec_corners)
    else:
        dec_c = dec_center
    x = (ra_corners - ra_c) * np.cos(np.radians(dec_c))
    y = dec_corners - dec_c
    n = len(x)
    area_deg2 = 0.5 * abs(sum(x[i] * y[(i + 1) % n] - x[(i + 1) % n] * y[i]
                               for i in range(n)))
    return area_deg2 * 3600.0


def stars_in_polygon(ra_stars, dec_stars, ra_corners, dec_corners):
    """Boolean mask: True where a star lies inside the sky polygon."""
    ra_c  = np.mean(ra_corners)
    dec_c = np.mean(dec_corners)
    cos_d = np.cos(np.radians(dec_c))
    poly_xy = np.column_stack([
        (ra_corners - ra_c) * cos_d,
        dec_corners - dec_c,
    ])
    star_xy = np.column_stack([
        (ra_stars - ra_c) * cos_d,
        dec_stars - dec_c,
    ])
    return MplPath(poly_xy).contains_points(star_xy)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Log in to Gaia for higher row cap and async reliability
    _user = os.environ.get('GAIA_USER')
    _pwd  = os.environ.get('GAIA_PASS')
    if _user and _pwd:
        try:
            Gaia.login(user=_user, password=_pwd)
            print(f"Logged in to Gaia as {_user}.")
        except Exception as e:
            print(f"WARNING: Gaia login failed ({e}); continuing anonymously.")
    else:
        print("No GAIA_USER/GAIA_PASS env vars found; querying anonymously (2000-row cap).")

    RA, DEC, label = resolve_target(args)
    PA              = args.pa
    OPTICAL_ELEMENT = args.optical_element
    GAIA_RADIUS     = args.radius
    BRIGHT_LIMIT    = args.bright_limit
    STAR_PROB       = args.star_prob

    # -----------------------------------------------------------------------
    # 1. Load calibration
    # -----------------------------------------------------------------------
    print(f"Loading {OPTICAL_ELEMENT} calibration from {args.calib_dir} …")
    calib = load_calibration(OPTICAL_ELEMENT, args.calib_dir, tag=args.calib_tag)
    tier_desc = (f"Tier-1 ({calib['gaia_basis']}+{calib['tmass_basis']}, "
                 f"rms={calib['rms_g2m']:.3f} mag)"
                 if calib['has_g2m'] and not args.gaia_only
                 else f"Tier-2 only ({calib['color_basis']}, rms={calib['rms']:.3f} mag)")
    print(f"  Calibration: {tier_desc}")

    # -----------------------------------------------------------------------
    # 2. Gaia catalog
    # -----------------------------------------------------------------------
    local_gaia = os.path.join(args.outdir, f"catalog_gaia_{label}.ecsv")

    # Check if cache has 2MASS columns (or if we're in gaia-only mode)
    _need_2mass = not args.gaia_only and calib['has_g2m']
    _cache_has_2mass = False
    if os.path.exists(local_gaia):
        try:
            _probe = Table.read(local_gaia, format="ascii.ecsv")
            _cache_has_2mass = 'tmass_j' in _probe.colnames
        except Exception:
            pass

    if os.path.exists(local_gaia) and (not _need_2mass or _cache_has_2mass):
        print(f"Loading cached Gaia catalog: {local_gaia}")
        gaia_raw = Table.read(local_gaia, format="ascii.ecsv")
    else:
        print(f"Querying Gaia DR3 around {label} (r = {GAIA_RADIUS} deg) …")
        if args.full_query:
            select_clause = "*"
        else:
            select_clause = ("source_id, ra, dec, phot_g_mean_mag, "
                             "phot_bp_mean_mag, phot_rp_mean_mag, "
                             "classprob_dsc_combmod_star")
        query = f"""
            SELECT {select_clause}
            FROM gaiadr3.gaia_source
            WHERE 1 = CONTAINS(
                POINT('ICRS', {RA}, {DEC}),
                CIRCLE('ICRS', ra, dec, {GAIA_RADIUS})
            )
        """
        import time
        for _attempt in range(5):
            try:
                job = Gaia.launch_job_async(query, dump_to_file=False)
                gaia_raw = job.get_results()
                break
            except Exception as _e:
                print(f"  WARNING: Gaia query attempt {_attempt+1} failed: {_e}; retrying …")
                time.sleep(5 * (2 ** _attempt))
        else:
            sys.exit("ERROR: Gaia query failed after 5 attempts.")
        print(f"  {len(gaia_raw)} Gaia sources retrieved.")

        # 2MASS crossmatch for bright sources
        if _need_2mass:
            n_bright = int(np.sum(np.array(gaia_raw['phot_g_mean_mag']) < args.tmass_glim))
            if n_bright >= 5:
                print(f"  Querying 2MASS crossmatch for {n_bright} bright sources "
                      f"(G < {args.tmass_glim}) …")
                try:
                    import pandas as pd
                    tm = query_2mass_xmatch(RA, DEC, GAIA_RADIUS, args.tmass_glim)
                    tm = tm.dropna(subset=['source_id']).drop_duplicates('source_id')
                    n_match = int((~tm['tmass_j'].isna()).sum())
                    print(f"  {n_match} 2MASS matches found.")

                    # Merge 2MASS columns into gaia_raw
                    gaia_df = gaia_raw.to_pandas()
                    gaia_df = gaia_df.merge(
                        tm[['source_id', 'tmass_j', 'tmass_h', 'tmass_ks']],
                        on='source_id', how='left')
                    gaia_raw = Table.from_pandas(gaia_df)
                except Exception as e:
                    print(f"  WARNING: 2MASS query failed ({e}); using Tier-2 only.")
                    for col in ('tmass_j', 'tmass_h', 'tmass_ks'):
                        gaia_raw[col] = np.full(len(gaia_raw), np.nan)
            else:
                print(f"  Fewer than 5 bright sources (G < {args.tmass_glim}); "
                      f"skipping 2MASS crossmatch.")
                for col in ('tmass_j', 'tmass_h', 'tmass_ks'):
                    gaia_raw[col] = np.full(len(gaia_raw), np.nan)

        gaia_raw.write(local_gaia, format="ascii.ecsv", overwrite=True)
        print(f"  Saved to {local_gaia}")

    # -----------------------------------------------------------------------
    # 3. Filter to stars and apply calibration
    # -----------------------------------------------------------------------
    mask = np.ones(len(gaia_raw), dtype=bool)
    if "classprob_dsc_combmod_star" in gaia_raw.colnames:
        prob = np.nan_to_num(np.array(gaia_raw["classprob_dsc_combmod_star"]))
        mask &= prob >= STAR_PROB
    mag_all = np.array(gaia_raw["phot_g_mean_mag"])
    mask &= np.isfinite(mag_all) & (mag_all > BRIGHT_LIMIT)
    stars = gaia_raw[mask]
    print(f"{len(stars)} Gaia stars after filtering (of {len(gaia_raw)} raw sources).")

    ra_s  = np.array(stars["ra"])
    dec_s = np.array(stars["dec"])
    mag_g = np.array(stars["phot_g_mean_mag"])

    # Build band arrays and compute predicted Roman magnitudes
    G_ab  = mag_g                                        + G_AB_OFFSET
    BP_ab = np.array(stars["phot_bp_mean_mag"])          + BP_AB_OFFSET
    RP_ab = np.array(stars["phot_rp_mean_mag"])          + RP_AB_OFFSET
    Ks_ab = (np.array(stars["tmass_ks"]).astype(float)   + KS_AB_OFFSET
             if "tmass_ks" in stars.colnames else np.full(len(stars), np.nan))
    J_ab  = (np.array(stars["tmass_j"]).astype(float)    + J_AB_OFFSET
             if "tmass_j"  in stars.colnames else np.full(len(stars), np.nan))
    H_ab  = (np.array(stars["tmass_h"]).astype(float)    + H_AB_OFFSET
             if "tmass_h"  in stars.colnames else np.full(len(stars), np.nan))

    bands = {'G': G_ab, 'BP': BP_ab, 'RP': RP_ab,
             'J': J_ab, 'H': H_ab, 'Ks': Ks_ab}

    if args.gaia_only and calib['has_g2m']:
        # Force Tier-2 by temporarily disabling has_g2m
        calib_t2 = {**calib, 'has_g2m': False}
        roman_mag, no_color, used_t1 = compute_roman_mag(bands, calib_t2)
    else:
        roman_mag, no_color, used_t1 = compute_roman_mag(bands, calib)

    n_tier1 = int(used_t1.sum())
    n_tier2 = len(stars) - n_tier1
    print(f"Predicted {OPTICAL_ELEMENT}(AB) magnitudes: "
          f"{n_tier1} via Tier-1 (Gaia+2MASS), {n_tier2} via Tier-2 (Gaia-only).")

    # Cumulative counts at each threshold across the full queried field
    print(f"\n--- Field-wide {OPTICAL_ELEMENT}(AB) cumulative counts ---")
    for t in THRESHOLDS:
        n = int(np.sum(roman_mag < t))
        area_sqdeg = np.pi * GAIA_RADIUS**2
        print(f"  < {t} mag : {n:6d}  ({n/area_sqdeg:.1f} /sq deg)")

    # -----------------------------------------------------------------------
    # 4. WFI footprint via pysiaf
    # -----------------------------------------------------------------------
    siaf = pysiaf.Siaf("Roman")

    if args.center_sca is not None:
        raw = args.center_sca.strip().upper().lstrip("WFI").lstrip("0") or "0"
        try:
            sca_num = int(raw)
        except ValueError:
            sys.exit(f"ERROR: cannot parse --center-sca '{args.center_sca}'. "
                     "Use e.g. WFI01, WFI1, or 1.")
        if not (1 <= sca_num <= 18):
            sys.exit(f"ERROR: --center-sca must be between 1 and 18, got {sca_num}.")
        center_ap_name = f"WFI{sca_num:02d}_FULL"
        print(f"Centering on {center_ap_name} aperture.")
    else:
        center_ap_name = "WFI_CEN"

    center_ap = siaf[center_ap_name]
    attitude  = rotations.attitude(center_ap.V2Ref, center_ap.V3Ref, RA, DEC, PA)

    sca_sky = []
    for i in range(1, 19):
        name = f"WFI{i:02d}_FULL"
        ap   = siaf[name]
        ap.set_attitude_matrix(attitude)
        ra_corners, dec_corners = ap.corners("sky", rederive=False)
        ra_c  = np.append(ra_corners,  ra_corners[0])
        dec_c = np.append(dec_corners, dec_corners[0])
        sca_sky.append((ra_c, dec_c, ra_corners, dec_corners,
                        np.mean(ra_corners), np.mean(dec_corners), i))

    print(f"Projected {len(sca_sky)} SCAs at RA={RA:.4f}, Dec={DEC:.4f}, PA={PA} deg.")

    # -----------------------------------------------------------------------
    # 5. Per-SCA statistics (in predicted Roman magnitudes)
    # -----------------------------------------------------------------------
    print(f"\n--- Per-SCA {OPTICAL_ELEMENT}(AB) statistics ---")
    thresh_labels = [f"<{t}" for t in THRESHOLDS]
    col_w_sca  = 4
    col_w_area = 14
    col_w_cnt  = 8
    hdr = (f"{'SCA':>{col_w_sca}}  {'Area(arcmin2)':>{col_w_area}}"
           + "".join(f"  {lbl:>{col_w_cnt}}" for lbl in thresh_labels))
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    sca_stats = []
    for _, _, ra_corners, dec_corners, _, _, num in sca_sky:
        in_sca = stars_in_polygon(ra_s, dec_s, ra_corners, dec_corners)
        area   = poly_area_arcmin2(ra_corners, dec_corners, ra_center=RA, dec_center=DEC)
        rm_in  = roman_mag[in_sca]
        counts = [int(np.sum(rm_in < t)) for t in THRESHOLDS]
        sca_stats.append(dict(sca=num, area_arcmin2=area,
                              **{f'n_lt_{t}': c for t, c in zip(THRESHOLDS, counts)}))
        print(f"{num:>{col_w_sca}}  {area:>{col_w_area}.1f}"
              + "".join(f"  {c:>{col_w_cnt}}" for c in counts))

    # Summary rows
    for stat_label, fn in [('Min',  min), ('Max',  max),
                            ('Mean', lambda x: f"{np.mean(x):.1f}"),
                            ('Total', sum)]:
        row = [fn([s[f'n_lt_{t}'] for s in sca_stats]) for t in THRESHOLDS]
        print(f"{stat_label:>{col_w_sca}}  {'-':>{col_w_area}}"
              + "".join(f"  {v:>{col_w_cnt}}" for v in row))

    # -----------------------------------------------------------------------
    # 6. Field-wide NN separation stats (on all raw Gaia sources)
    # -----------------------------------------------------------------------
    print(f"\n--- Field statistics (all {len(gaia_raw)} Gaia sources, r = {GAIA_RADIUS} deg) ---")
    coords_all = SkyCoord(ra=np.array(gaia_raw["ra"]),
                          dec=np.array(gaia_raw["dec"]), unit="deg")
    _, sep_nn, _ = coords_all.match_to_catalog_sky(coords_all, nthneighbor=2)
    sep_arcsec = sep_nn.to(u.arcsec).value
    print(f"  NN separation — mean:   {np.mean(sep_arcsec):.3f} arcsec")
    print(f"  NN separation — median: {np.median(sep_arcsec):.3f} arcsec")
    print(f"  NN separation — std:    {np.std(sep_arcsec):.3f} arcsec")

    cos_dec = np.cos(np.radians(DEC))

    # Pre-compute star rendering properties for the mosaic plot
    flux     = 10 ** ((mag_g.max() - mag_g) / 2.5)
    flux_max = flux.max()
    gray     = np.clip(0.15 + 0.85 * flux / flux_max, 0, 1)
    sizes    = np.clip(0.5 * flux / flux_max * 20, 0.3, 10)

    theta = np.linspace(0, 2 * np.pi, 400)

    def _draw_mosaic(ax):
        ax.set_facecolor("black")
        for ra_c, dec_c, _, _, ra_cen, dec_cen, num in sca_sky:
            shade = 0.18 if (num - 1) % 2 == 0 else 0.09
            ax.fill(ra_c, dec_c, fc="cornflowerblue", alpha=shade, zorder=1)
            ax.plot(ra_c, dec_c, color="cornflowerblue", lw=0.7, alpha=0.6, zorder=2)
            ax.text(ra_cen, dec_cen, str(num), ha="center", va="center",
                    fontsize=6, color="cornflowerblue", alpha=0.7, zorder=3)
        ax.scatter(ra_s, dec_s, c=[[g, g, g] for g in gray], s=sizes,
                   linewidths=0, zorder=4, label=f"Gaia DR3  (n = {len(stars)})")
        ax.plot(RA, DEC, "+", color="cyan", ms=12, mew=1.5, zorder=6)
        ax.text(RA, DEC, f"  {label}", color="cyan", fontsize=9, va="center", zorder=6)
        ax.plot(RA + GAIA_RADIUS / cos_dec * np.cos(theta),
                DEC + GAIA_RADIUS * np.sin(theta),
                color="cyan", lw=0.6, alpha=0.4, ls="--",
                label=f"Gaia query  (r = {GAIA_RADIUS} deg)")
        ax.set_xlabel("RA (deg)", fontsize=11, color="white")
        ax.set_ylabel("Dec (deg)", fontsize=11, color="white")
        ax.set_title(
            f"Roman WFI {OPTICAL_ELEMENT} — {label}\n"
            f"pysiaf footprint, PA = {PA} deg",
            fontsize=12, color="white",
        )
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        ax.set_aspect(1.0 / cos_dec)
        ax.invert_xaxis()
        ax.legend(loc="upper right", fontsize=8,
                  facecolor="#111111", labelcolor="white", edgecolor="gray")
        ax.grid(True, ls=":", alpha=0.2, color="gray")

    def _dark_axes(ax):
        ax.set_facecolor("#111111")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    def _draw_nn_hist(ax):
        ax.hist(sep_arcsec, bins=100, color="steelblue", edgecolor="none", alpha=0.85)
        ax.axvline(np.mean(sep_arcsec), color="tomato", lw=1.5,
                   label=f'Mean = {np.mean(sep_arcsec):.2f}"')
        ax.axvline(np.median(sep_arcsec), color="gold", lw=1.5, ls="--",
                   label=f'Median = {np.median(sep_arcsec):.2f}"')
        ax.set_xlabel("Nearest-neighbour separation (arcsec)", fontsize=10)
        ax.set_ylabel("Number of sources", fontsize=10)
        ax.set_title(
            f"Gaia DR3 NN separations — {label} field (r = {GAIA_RADIUS} deg)",
            fontsize=10,
        )
        ax.legend(fontsize=9, facecolor="#222222", labelcolor="white", edgecolor="#444444")
        ax.grid(True, ls=":", alpha=0.3, color="gray")
        _dark_axes(ax)

    def _draw_mag_hist(ax):
        finite = roman_mag[np.isfinite(roman_mag)]
        if len(finite) == 0:
            ax.text(0.5, 0.5, "No finite magnitudes", transform=ax.transAxes,
                    ha='center', va='center', color='white')
            _dark_axes(ax)
            return
        bins = np.arange(np.floor(finite.min()), np.ceil(finite.max()) + 0.5, 0.5)
        ax.hist(finite, bins=bins, color="steelblue", edgecolor="#333333",
                linewidth=0.4, alpha=0.85, label=f"n = {len(finite)}")
        ax.axvline(np.mean(finite), color="tomato", lw=1.5,
                   label=f"Mean = {np.mean(finite):.2f}")
        ax.axvline(np.median(finite), color="gold", lw=1.5, ls="--",
                   label=f"Median = {np.median(finite):.2f}")
        tier_note = (f"Tier-1: {n_tier1}  Tier-2: {n_tier2}"
                     if calib['has_g2m'] and not args.gaia_only else "Tier-2 only")
        ax.set_xlabel(f"Predicted {OPTICAL_ELEMENT}(AB) magnitude", fontsize=10)
        ax.set_ylabel("Number of stars", fontsize=10)
        ax.set_title(
            f"{OPTICAL_ELEMENT}(AB) distribution — {label}\n"
            f"G > {BRIGHT_LIMIT}, star prob ≥ {STAR_PROB}  |  {tier_note}",
            fontsize=10,
        )
        ax.legend(fontsize=9, facecolor="#222222", labelcolor="white", edgecolor="#444444")
        ax.grid(True, ls=":", alpha=0.3, color="gray")
        _dark_axes(ax)

    def _draw_summary_text(ax):
        ax.set_facecolor("#111111")
        ax.axis("off")
        counts_18 = [s['n_lt_18'] for s in sca_stats]
        lines = [
            f"{OPTICAL_ELEMENT}(AB)<18 per SCA — min: {min(counts_18)}  "
            f"max: {max(counts_18)}  mean: {np.mean(counts_18):.1f}  "
            f"total: {sum(counts_18)}",
            f"Tier-1 ({calib['tmass_basis']}): {n_tier1}  "
            f"Tier-2 only: {n_tier2}"
            if calib['has_g2m'] and not args.gaia_only else
            f"Gaia-only (Tier-2, {calib['color_basis']})",
            f"NN separation — mean: {np.mean(sep_arcsec):.3f}\"   "
            f"median: {np.median(sep_arcsec):.3f}\"   "
            f"std: {np.std(sep_arcsec):.3f}\"",
        ]
        ax.set_title("Field summary", fontsize=10, pad=6, color="white")
        for i, line in enumerate(lines):
            ax.text(0.5, 0.65 - i * 0.3, line,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, color="white", family="monospace")

    # -----------------------------------------------------------------------
    # 7. Combined output
    # -----------------------------------------------------------------------
    fig_combined = plt.figure(figsize=(26, 14))
    fig_combined.patch.set_facecolor("black")

    gs = GridSpec(1, 2, figure=fig_combined,
                  width_ratios=[2.2, 1],
                  hspace=0.0, wspace=0.12)

    ax_mosaic = fig_combined.add_subplot(gs[0, 0])
    ax_mosaic.set_facecolor("black")

    gs_right = gs[0, 1].subgridspec(3, 1,
                                     height_ratios=[1, 1, 0.45],
                                     hspace=0.55)
    ax_nn  = fig_combined.add_subplot(gs_right[0])
    ax_mag = fig_combined.add_subplot(gs_right[1])
    ax_sum = fig_combined.add_subplot(gs_right[2])

    _draw_mosaic(ax_mosaic)
    _draw_nn_hist(ax_nn)
    _draw_mag_hist(ax_mag)
    _draw_summary_text(ax_sum)

    combined_png = os.path.join(args.outdir,
                                f"field_summary_{label}_{OPTICAL_ELEMENT}.png")
    fig_combined.savefig(combined_png, dpi=200, facecolor=fig_combined.get_facecolor())
    print(f"\nSaved combined: {combined_png}")
    if args.show:
        plt.show()
    plt.close(fig_combined)

    # -----------------------------------------------------------------------
    # 8. Separate outputs (optional)
    # -----------------------------------------------------------------------
    if args.separate:
        fig, ax = plt.subplots(figsize=(9, 11), facecolor="black")
        _draw_mosaic(ax)
        plt.tight_layout()
        out_png = os.path.join(args.outdir, f"wfi_mosaic_{label}_{OPTICAL_ELEMENT}.png")
        plt.savefig(out_png, dpi=300)
        print(f"Saved: {out_png}")
        if args.show:
            plt.show()
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        _draw_nn_hist(ax2)
        plt.tight_layout()
        nn_png = os.path.join(args.outdir, f"nn_sep_histogram_{label}.png")
        plt.savefig(nn_png, dpi=300)
        print(f"Saved: {nn_png}")
        if args.show:
            plt.show()
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        _draw_mag_hist(ax3)
        plt.tight_layout()
        mag_png = os.path.join(args.outdir, f"mag_histogram_{label}_{OPTICAL_ELEMENT}.png")
        plt.savefig(mag_png, dpi=300)
        print(f"Saved: {mag_png}")
        if args.show:
            plt.show()
        plt.close(fig3)

    print("\nDone.")


if __name__ == "__main__":
    main()
