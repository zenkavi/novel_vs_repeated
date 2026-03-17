"""
level2_helpers.py
Helper functions for group-level (second-level) analyses.

Computes a one-sample t-test across subjects on first-level contrast maps
(effect size maps) using nilearn's SecondLevelModel. Generates group stat
maps and an HTML report with glass brain and ROI-centered visualizations.

Reports are labeled as uncorrected (no multiple-comparison correction applied).
"""

import base64
import io
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_glass_brain, plot_stat_map
from nilearn.image import load_img
from nilearn.reporting import get_clusters_table


# =============================================================================
# ROI definitions (MNI coordinates)
# =============================================================================

# Common coordinates from the value-based decision-making literature.
# vmPFC: Bartra et al. (2013) meta-analysis, Clithero & Rangel (2014)
# lOFC: Kringelbach & Rolls (2004) meta-analysis, Noonan et al. (2010)

ROI_COORDS = {
    'vmPFC': {
        'coords': (0, 48, -10),
        'description': 'Ventromedial prefrontal cortex (MNI: 0, 48, -10)',
    },
    'lOFC': {
        'coords': (-32, 42, -18),
        'description': 'Left lateral orbitofrontal cortex (MNI: -32, 42, -18)',
    },
}


# =============================================================================
# Path helpers
# =============================================================================

def get_contrast_path(
    subnum, session, task, contrast_id, output_dir,
    mnum='value_parametric',
    model_variant='rt_in_duration',
    space='MNI152NLin2009cAsym_res-2',
    map_type='effect_size',
):
    """Build the path to a first-level contrast map for one subject."""
    prefix = (f'sub-{subnum}_ses-{session}_task-{task}'
              f'_space-{space}_{mnum}_{model_variant}')

    return os.path.join(
        output_dir, mnum, model_variant,
        f'sub-{subnum}', f'ses-{session}', 'contrasts',
        f'{prefix}_{contrast_id}_{map_type}.nii.gz'
    )


def collect_contrast_maps(
    subjects, session, task, contrast_id, output_dir,
    mnum='value_parametric',
    model_variant='rt_in_duration',
    space='MNI152NLin2009cAsym_res-2',
    map_type='effect_size',
):
    """
    Collect first-level contrast maps across subjects.

    Returns
    -------
    contrast_maps : list of str
        Paths to contrast maps (one per subject)
    missing : list of str
        Subject numbers whose maps were not found
    """
    contrast_maps = []
    missing = []

    for subnum in subjects:
        path = get_contrast_path(
            subnum, session, task, contrast_id, output_dir,
            mnum=mnum, model_variant=model_variant,
            space=space, map_type=map_type,
        )
        if os.path.exists(path):
            contrast_maps.append(path)
        else:
            missing.append(subnum)

    return contrast_maps, missing


# =============================================================================
# Group-level model
# =============================================================================

def fit_group_onesample(contrast_maps):
    """
    Fit a one-sample t-test (intercept-only second-level model) on a list
    of first-level contrast maps.

    Returns
    -------
    second_level_model : SecondLevelModel
    group_tmap : Nifti1Image
    group_effect : Nifti1Image
    """
    n_subjects = len(contrast_maps)
    design_matrix = pd.DataFrame({'intercept': np.ones(n_subjects)})

    second_level_model = SecondLevelModel(smoothing_fwhm=None)
    second_level_model = second_level_model.fit(
        contrast_maps, design_matrix=design_matrix
    )

    group_tmap = second_level_model.compute_contrast(output_type='stat')
    group_effect = second_level_model.compute_contrast(output_type='effect_size')

    return second_level_model, group_tmap, group_effect


# =============================================================================
# Plotting
# =============================================================================

def _fig_to_base64(fig):
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def plot_group_glass_brain(stat_map, title='', threshold=3.0):
    """Glass brain visualization of a group stat map."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, display_mode in zip(axes, ['x', 'y', 'z']):
        plot_glass_brain(
            stat_map, threshold=threshold,
            display_mode=display_mode, axes=ax,
            title=title if display_mode == 'y' else '',
            colorbar=True,
        )
    return fig


def plot_roi_view(stat_map, roi_name, coords, title='', threshold=3.0):
    """
    Three-panel (sagittal, coronal, axial) stat map centered on an ROI.

    Parameters
    ----------
    stat_map : Nifti1Image
        Group stat map
    roi_name : str
        ROI label for the title
    coords : tuple of 3 ints
        MNI (x, y, z) coordinates to center on
    title : str
        Additional title text
    threshold : float
        t-stat threshold for display

    Returns
    -------
    fig : matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, display_mode in zip(axes, ['x', 'y', 'z']):
        cut = coords[{'x': 0, 'y': 1, 'z': 2}[display_mode]]
        plot_stat_map(
            stat_map, threshold=threshold,
            display_mode=display_mode, cut_coords=[cut],
            axes=ax, colorbar=True,
            title=f'{roi_name} ({display_mode}={cut})' if not title else '',
        )
    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    return fig


# =============================================================================
# Report
# =============================================================================

def generate_group_report(
    subjects, session, task, contrast_id, output_dir,
    mnum='value_parametric',
    model_variant='rt_in_duration',
    space='MNI152NLin2009cAsym_res-2',
    threshold=3.0,
    cluster_threshold=10,
    roi_coords=None,
):
    """
    Run a group-level one-sample t-test and generate an HTML report.

    The report is labeled as uncorrected (no multiple-comparison correction).

    Saves to:
        {output_dir}/{mnum}/{model_variant}/group/ses-{session}/
            {prefix}_{contrast_id}_tmap.nii.gz
            {prefix}_{contrast_id}_effect_size.nii.gz
            {prefix}_{contrast_id}_uncorrected_report.html

    Parameters
    ----------
    subjects : list of str
        Subject numbers to include
    session : str
        Session identifier
    task : str
        'yesNo' or 'binaryChoice'
    contrast_id : str
        Contrast name (e.g. 'stim_value_par')
    output_dir : str
        Root output directory
    mnum, model_variant, space : str
        Model identifiers
    threshold : float
        t-statistic threshold for visualization (uncorrected)
    cluster_threshold : int
        Minimum cluster size (in voxels) for cluster table
    roi_coords : dict or None
        ROI definitions. If None, uses the default ROI_COORDS.
        Each entry: {name: {'coords': (x,y,z), 'description': str}}

    Returns
    -------
    report_path : str
        Path to the saved HTML report
    group_tmap : Nifti1Image
        Group t-statistic map
    """
    import nibabel as nib

    if roi_coords is None:
        roi_coords = ROI_COORDS

    # Collect subject maps
    contrast_maps, missing = collect_contrast_maps(
        subjects, session, task, contrast_id, output_dir,
        mnum=mnum, model_variant=model_variant,
        space=space, map_type='effect_size',
    )

    if missing:
        print(f"  WARNING: Missing maps for subjects: {missing}")
    if len(contrast_maps) < 2:
        raise ValueError(
            f"Need at least 2 subjects for group analysis, "
            f"found {len(contrast_maps)}"
        )

    n_subjects = len(contrast_maps)
    print(f"  Running group analysis: task-{task} {contrast_id} "
          f"[{model_variant}] (n={n_subjects})")

    # Fit group model
    model, group_tmap, group_effect = fit_group_onesample(contrast_maps)

    # Output directory
    group_dir = os.path.join(
        output_dir, mnum, model_variant,
        'group', f'ses-{session}'
    )
    os.makedirs(group_dir, exist_ok=True)

    prefix = f'group_task-{task}_space-{space}_{mnum}_{model_variant}'

    # Save maps
    tmap_fn = os.path.join(group_dir, f'{prefix}_{contrast_id}_tmap.nii.gz')
    nib.save(group_tmap, tmap_fn)

    effect_fn = os.path.join(
        group_dir, f'{prefix}_{contrast_id}_effect_size.nii.gz'
    )
    nib.save(group_effect, effect_fn)

    print(f"  Saved group maps to: {group_dir}")

    # -- Glass brain --
    title_base = (f'Group (n={n_subjects}): task-{task} {contrast_id}\n'
                  f'{model_variant} | t > {threshold} (uncorrected)')

    fig_glass = plot_group_glass_brain(
        group_tmap, title=title_base, threshold=threshold
    )
    img_glass = _fig_to_base64(fig_glass)

    # -- ROI views --
    roi_sections_html = ''
    for roi_name, roi_info in roi_coords.items():
        coords = roi_info['coords']
        description = roi_info['description']

        fig_roi = plot_roi_view(
            group_tmap, roi_name, coords,
            title=f'{roi_name}: {description}',
            threshold=threshold,
        )
        img_roi = _fig_to_base64(fig_roi)

        # Also make an unthresholded version for this ROI
        fig_roi_unthresh = plot_roi_view(
            group_tmap, roi_name, coords,
            title=f'{roi_name} (unthresholded): {description}',
            threshold=None,
        )
        img_roi_unthresh = _fig_to_base64(fig_roi_unthresh)

        roi_sections_html += f"""
<div class="section">
    <h2>{roi_name}: {description}</h2>
    <h3>Thresholded (t > {threshold})</h3>
    <div class="plot-container">
        <img src="data:image/png;base64,{img_roi}" />
    </div>
    <h3>Unthresholded</h3>
    <div class="plot-container">
        <img src="data:image/png;base64,{img_roi_unthresh}" />
    </div>
</div>
"""

    # -- Cluster table --
    try:
        cluster_table = get_clusters_table(
            group_tmap, stat_threshold=threshold,
            min_distance=8,
            cluster_threshold=cluster_threshold,
        )
        cluster_html = cluster_table.to_html(
            classes='cluster-table', index=False, float_format='%.2f'
        )
        n_clusters = len(cluster_table)
    except Exception:
        cluster_html = '<p>No suprathreshold clusters found.</p>'
        n_clusters = 0

    # Subject list
    subjects_used = [
        os.path.basename(p).split('_')[0]
        for p in contrast_maps
    ]

    # -- HTML report --
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Group Report (uncorrected): {contrast_id} | task-{task} | {model_variant}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           margin: 0; padding: 20px 40px; background: #f8f9fa; color: #333; }}
    h1 {{ border-bottom: 3px solid #2c3e50; padding-bottom: 10px; color: #2c3e50; }}
    h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 6px; margin-top: 30px; }}
    h3 {{ color: #555; margin-bottom: 8px; }}
    .section {{ background: #fff; border-radius: 8px; padding: 20px 30px;
                margin-bottom: 30px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
    .plot-container {{ text-align: center; margin-bottom: 16px; }}
    .plot-container img {{ max-width: 100%; height: auto;
                           border: 1px solid #e0e0e0; border-radius: 4px; }}
    table {{ border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 6px 10px; border: 1px solid #ddd; text-align: left; }}
    th {{ background: #f0f4f8; }}
    .cluster-table {{ width: auto; }}
    .meta {{ color: #888; font-size: 13px; }}
    .info-grid {{ display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 16px; }}
    .info-card {{ flex: 1; min-width: 200px; background: #f0f4f8;
                  border-radius: 6px; padding: 12px 16px; }}
    .info-card h3 {{ margin-top: 0; }}
    .warning {{ color: #856404; background: #fff3cd; border: 1px solid #ffc107;
                border-radius: 6px; padding: 10px 16px; margin-bottom: 16px; }}
</style>
</head>
<body>

<h1>Group-Level Analysis Report (Uncorrected)</h1>
<p class="meta">
    Contrast: <strong>{contrast_id}</strong> | Task: {task} |
    Model: {mnum} | Variant: {model_variant}<br>
    Session: ses-{session} | Space: {space}
</p>
<div class="warning">
    Results shown at t > {threshold} (uncorrected). No correction for
    multiple comparisons has been applied. Min cluster size: {cluster_threshold} voxels.
</div>

<div class="section">
    <h2>Summary</h2>
    <div class="info-grid">
        <div class="info-card">
            <h3>Subjects</h3>
            <p>n = {n_subjects}</p>
            <p style="font-size: 12px;">{', '.join(subjects_used)}</p>
            {'<p style="color: #d62728;">Missing: ' + ', '.join(missing) + '</p>' if missing else ''}
        </div>
        <div class="info-card">
            <h3>Results</h3>
            <p>{n_clusters} suprathreshold clusters (t > {threshold}, uncorrected)</p>
        </div>
    </div>
</div>

<div class="section">
    <h2>Glass Brain (t > {threshold}, uncorrected)</h2>
    <div class="plot-container">
        <img src="data:image/png;base64,{img_glass}" />
    </div>
</div>

{roi_sections_html}

<div class="section">
    <h2>Cluster Table (t > {threshold}, uncorrected, min {cluster_threshold} voxels)</h2>
    {cluster_html}
</div>

</body>
</html>"""

    report_fn = f'{prefix}_{contrast_id}_uncorrected_report.html'
    report_path = os.path.join(group_dir, report_fn)
    with open(report_path, 'w') as f:
        f.write(html)

    print(f"  Report saved: {report_path}")
    return report_path, group_tmap