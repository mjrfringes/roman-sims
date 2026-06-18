#!/usr/bin/env python3
"""
Roman WFI mosaic field simulator with Gaia source statistics.

Generates mosaic footprint plots, stellar density analysis, magnitude distributions,
and nearest-neighbor separation statistics for Roman WFI observations.

Usage examples
--------------
Target resolution (Simbad):
  python simulate_field.py --target "NGC 6819"
  python simulate_field.py --target "M31" --pa 45 --filter F087
  python simulate_field.py --target "47 Tuc" --bright-limit 12

Direct coordinates:
  python simulate_field.py --ra 295.3275 --dec 40.1878
  python simulate_field.py --ra 10.68 --dec 41.27 --nickname "MyField"
  python simulate_field.py --ra 80.5 --dec -69.5 --nickname "LMC"

Coordinates with custom target name:
  python simulate_field.py --ra 295.3275 --dec 40.1878 --target NGC6819
  python simulate_field.py --ra 260.051625 --dec 57.915361 --target "Draco I"

Advanced options:
  python simulate_field.py --ra 290.7 --dec 44.5 --nickname Kepler --radius 0.8
  python simulate_field.py --ra 80.5 --dec -69.5 --nickname LMC --center-sca WFI01
  python simulate_field.py --target "NGC 362" --filter F106 --show --separate
  python simulate_field.py --ra 84.658 --dec -69.095 --nickname "30Dor" --full-query
  python simulate_field.py --target "NGC 6819" --star-prob 0.8 --bright-limit 11
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
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.gaia import Gaia


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Roman WFI mosaic field simulator with Gaia source statistics.",
        epilog="""
Examples:
  # Resolve target by name via Simbad
  python simulate_field.py --target "NGC 6819"

  # Direct RA/Dec coordinates
  python simulate_field.py --ra 290.7 --dec 44.5 --nickname "Kepler"

  # Custom SCA centering and position angle
  python simulate_field.py --target "M31" --center-sca WFI10 --pa 45

  # Dense field with full Gaia query and separate output files
  python simulate_field.py --ra 80.5 --dec -69.5 --nickname "LMC" --full-query --separate

  # Strict star classification and interactive display
  python simulate_field.py --target "NGC 362" --star-prob 0.9 --bright-limit 11 --show
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_argument_group("Target (--target OR --ra/--dec required)")
    g.add_argument("--target", metavar="NAME",
                   help="Object name resolved via Simbad (e.g. 'NGC 6819'). "
                        "If --ra/--dec are also given, the name is used only as a label.")
    g.add_argument("--ra",  type=float, metavar="DEG", help="Right ascension (J2000, deg)")
    g.add_argument("--dec", type=float, metavar="DEG", help="Declination (J2000, deg)")
    g.add_argument("--nickname", metavar="NAME",
                   help="Custom label for the field when using --ra/--dec (default: auto-generated from coordinates)")

    p.add_argument("--pa", type=float, default=0.0,
                   help="WFI position angle (deg E of N)")
    p.add_argument("--center-sca", default=None, metavar="SCA",
                   help="SCA aperture to place on target, e.g. WFI01 or 1 (default: WFI_CEN)")
    p.add_argument("--filter", dest="optical_element", default="F106", metavar="FILTER",
                   help="Optical element / filter")
    p.add_argument("--radius", type=float, default=0.5,
                   help="Gaia query radius (deg)")
    p.add_argument("--bright-limit", type=float, default=13.5,
                   help="Gaia G-mag bright cutoff (exclude brighter)")
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
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_target(args):
    """Return (ra_deg, dec_deg, label)."""
    if args.ra is not None and args.dec is not None:
        if args.nickname:
            label = args.nickname.replace(" ", "")
        elif args.target:
            label = args.target.replace(" ", "")
        else:
            label = f"RA{args.ra:.4f}Dec{args.dec:+.4f}"
        return args.ra, args.dec, label
    if args.target:
        print(f"Resolving '{args.target}' via Simbad …")
        sc = SkyCoord.from_name(args.target)
        ra, dec = sc.ra.deg, sc.dec.deg
        print(f"  -> RA = {ra:.5f} deg, Dec = {dec:.5f} deg")
        return ra, dec, args.target.replace(" ", "")
    sys.exit("ERROR: provide --target NAME or both --ra DEG and --dec DEG.")


def poly_area_arcmin2(ra_corners, dec_corners, ra_center=None, dec_center=None):
    """Polygon area (arcmin²) via planar shoelace on a local tangent plane.

    If ra_center/dec_center are not provided, uses the mean of the corners.
    """
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

    RA, DEC, label = resolve_target(args)
    PA              = args.pa
    OPTICAL_ELEMENT = args.optical_element
    GAIA_RADIUS     = args.radius
    BRIGHT_LIMIT    = args.bright_limit
    STAR_PROB       = args.star_prob

    # -----------------------------------------------------------------------
    # 1. Gaia catalog
    # -----------------------------------------------------------------------
    local_gaia = os.path.join(args.outdir, f"catalog_gaia_{label}.ecsv")

    if os.path.exists(local_gaia):
        print(f"Loading cached Gaia catalog: {local_gaia}")
        gaia_raw = Table.read(local_gaia, format="ascii.ecsv")
    else:
        print(f"Querying Gaia DR3 around {label} (r = {GAIA_RADIUS} deg) …")
        if args.full_query:
            select_clause = "*"
        else:
            select_clause = "ra, dec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, classprob_dsc_combmod_star"
        query = f"""
            SELECT {select_clause}
            FROM gaiadr3.gaia_source
            WHERE 1 = CONTAINS(
                POINT('ICRS', {RA}, {DEC}),
                CIRCLE('ICRS', ra, dec, {GAIA_RADIUS})
            )
        """
        job = Gaia.launch_job_async(query)
        gaia_raw = job.get_results()
        gaia_raw.write(local_gaia, format="ascii.ecsv", overwrite=True)
        print(f"  {len(gaia_raw)} sources saved to {local_gaia}")

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
    mag_s = np.array(stars["phot_g_mean_mag"])

    # -----------------------------------------------------------------------
    # 2. WFI footprint via pysiaf
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

    sca_sky = []   # (ra_closed, dec_closed, ra_corners, dec_corners, ra_cen, dec_cen, num)
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
    # 3. Per-SCA statistics
    # -----------------------------------------------------------------------
    print("\n--- Per-SCA Gaia source statistics ---")
    col_w = (4, 11, 16, 10, 8, 8, 14)
    hdr = (f"{'SCA':>{col_w[0]}}  {'Stars/SCA':>{col_w[1]}}  "
           f"{'Stars/arcmin2':>{col_w[2]}}  {'Mag_mean':>{col_w[3]}}  "
           f"{'Mag_min':>{col_w[4]}}  {'Mag_max':>{col_w[5]}}  "
           f"{'Area(arcmin2)':>{col_w[6]}}")
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    sca_stats = []
    for _, _, ra_corners, dec_corners, _, _, num in sca_sky:
        in_sca   = stars_in_polygon(ra_s, dec_s, ra_corners, dec_corners)
        n_in     = int(np.sum(in_sca))
        area     = poly_area_arcmin2(ra_corners, dec_corners, ra_center=RA, dec_center=DEC)
        density  = n_in / area if area > 0 else 0.0
        m_in     = mag_s[in_sca]
        mag_mean = float(np.mean(m_in)) if n_in > 0 else float("nan")
        mag_min  = float(np.min(m_in))  if n_in > 0 else float("nan")
        mag_max  = float(np.max(m_in))  if n_in > 0 else float("nan")
        sca_stats.append(dict(
            sca=num, n_stars=n_in, density=density,
            mag_mean=mag_mean, mag_min=mag_min, mag_max=mag_max,
            area_arcmin2=area,
        ))
        print(f"{num:>{col_w[0]}}  {n_in:>{col_w[1]}}  "
              f"{density:>{col_w[2]}.4f}  {mag_mean:>{col_w[3]}.2f}  "
              f"{mag_min:>{col_w[4]}.2f}  {mag_max:>{col_w[5]}.2f}  "
              f"{area:>{col_w[6]}.1f}")

    n_per_sca   = [s["n_stars"]  for s in sca_stats]
    dens_per_sca = [s["density"] for s in sca_stats]
    total_in_fpa = sum(n_per_sca)
    print(sep)
    print(f"{'Min':>{col_w[0]}}  {min(n_per_sca):>{col_w[1]}}  {min(dens_per_sca):>{col_w[2]}.4f}")
    print(f"{'Max':>{col_w[0]}}  {max(n_per_sca):>{col_w[1]}}  {max(dens_per_sca):>{col_w[2]}.4f}")
    print(f"{'Mean':>{col_w[0]}}  {np.mean(n_per_sca):>{col_w[1]}.1f}  {np.mean(dens_per_sca):>{col_w[2]}.4f}")
    print(f"{'Total':>{col_w[0]}}  {total_in_fpa:>{col_w[1]}}")

    # -----------------------------------------------------------------------
    # 4. Field-wide NN separation stats
    # -----------------------------------------------------------------------
    print(f"\n--- Field statistics (all {len(gaia_raw)} Gaia sources, r = {GAIA_RADIUS} deg) ---")
    coords_all = SkyCoord(ra=np.array(gaia_raw["ra"]),
                          dec=np.array(gaia_raw["dec"]), unit="deg")
    _, sep_nn, _ = coords_all.match_to_catalog_sky(coords_all, nthneighbor=2)
    sep_arcsec = sep_nn.to(u.arcsec).value
    print(f"  NN separation — mean:   {np.mean(sep_arcsec):.3f} arcsec")
    print(f"  NN separation — median: {np.median(sep_arcsec):.3f} arcsec")
    print(f"  NN separation — std:    {np.std(sep_arcsec):.3f} arcsec")
    print(f"  Filtered stars in FPA:  {total_in_fpa}")

    cos_dec = np.cos(np.radians(DEC))

    # Pre-compute star flux/size/gray for mosaic rendering
    flux = 10 ** ((mag_s.max() - mag_s) / 2.5)
    flux_max = flux.max()
    gray  = np.clip(0.15 + 0.85 * flux / flux_max, 0, 1)
    sizes = np.clip(0.5 * flux / flux_max * 20, 0.3, 10)

    # Pre-compute histogram bins
    mag_bins  = np.arange(np.floor(mag_s.min()), np.ceil(mag_s.max()) + 0.5, 0.5)
    theta     = np.linspace(0, 2 * np.pi, 400)

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
        ax.hist(mag_s, bins=mag_bins, color="steelblue", edgecolor="#333333",
                linewidth=0.4, alpha=0.85, label=f"n = {len(mag_s)}")
        ax.axvline(np.mean(mag_s), color="tomato", lw=1.5,
                   label=f"Mean = {np.mean(mag_s):.2f}")
        ax.axvline(np.median(mag_s), color="gold", lw=1.5, ls="--",
                   label=f"Median = {np.median(mag_s):.2f}")
        ax.set_xlabel("Gaia G magnitude", fontsize=10)
        ax.set_ylabel("Number of stars", fontsize=10)
        ax.set_title(
            f"G-magnitude distribution — {label}\n"
            f"G > {BRIGHT_LIMIT}, star prob ≥ {STAR_PROB}",
            fontsize=10,
        )
        ax.legend(fontsize=9, facecolor="#222222", labelcolor="white", edgecolor="#444444")
        ax.grid(True, ls=":", alpha=0.3, color="gray")
        _dark_axes(ax)

    def _draw_summary_text(ax):
        """Render field summary stats as a compact dark text block."""
        ax.set_facecolor("#111111")
        ax.axis("off")
        n_per_sca    = [s["n_stars"]  for s in sca_stats]
        dens_per_sca = [s["density"]  for s in sca_stats]
        lines = [
            f"Stars/SCA — min: {min(n_per_sca)}   max: {max(n_per_sca)}   "
            f"mean: {np.mean(n_per_sca):.1f}   total: {sum(n_per_sca)}",
            f"Density (stars/arcmin²) — min: {min(dens_per_sca):.4f}   "
            f"max: {max(dens_per_sca):.4f}   mean: {np.mean(dens_per_sca):.4f}",
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
    # 5. Combined output
    # -----------------------------------------------------------------------
    fig_combined = plt.figure(figsize=(26, 14))
    fig_combined.patch.set_facecolor("black")

    # Left column: mosaic; right column: two histograms + summary
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
    # 6. Separate outputs (optional)
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
        mag_png = os.path.join(args.outdir, f"mag_histogram_{label}.png")
        plt.savefig(mag_png, dpi=300)
        print(f"Saved: {mag_png}")
        if args.show:
            plt.show()
        plt.close(fig3)

    print("\nDone.")


if __name__ == "__main__":
    main()
