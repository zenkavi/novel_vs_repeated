"""
level2_helpers.py
Helper functions for group-level (second-level) analyses.

The workflow is split into two steps:
  1. Fitting: run_group_onesample() collects subject maps, fits the model,
     and saves group tmap and effect_size niftis.
  2. Reporting: generate_group_report() loads saved group maps and builds
     an HTML report with glass brain, ROI, and cluster visualizations.

This separation allows additional inference methods (e.g., non-parametric
inference) to be slotted in between fitting and reporting, or to generate
reports from previously computed maps without re-fitting.
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
# ROI definitions (MNI coordinates, bilateral)
# =============================================================================

ROI_COORDS = {
    'vmPFC': {
        'coords': [(-6, 48, -10), (6, 48, -10)],
        'description': 'Ventromedial prefrontal cortex',
    },
    'lOFC': {
        'coords': [(-32, 42, -18), (32, 42, -18)],
        'description': 'Lateral orbitofrontal cortex',
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


def get_group_dir(output_dir, mnum, model_variant, session):
    """Build the group output directory path."""
    return os.path.join(
        output_dir, mnum, model_variant,
        'group', f'ses-{session}'
    )


def get_group_map_prefix(task, space, mnum, model_variant):
    """Build the filename prefix for group-level maps."""
    return f'group_task-{task}_space-{space}_{mnum}_{model_variant}'


def get_group_tmap_path(
    session, task, contrast_id, output_dir,
    mnum='value_parametric',
    model_variant='rt_in_duration',
    space='MNI152NLin2009cAsym_res-2',
):
    """Build the full path to a saved group tmap."""
    group_dir = get_group_dir(output_dir, mnum, model_variant, session)
    prefix = get_group_map_prefix(task, space, mnum, model_variant)
    return os.path.join(group_dir, f'{prefix}_{contrast_id}_tmap.nii.gz')


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
# Group-level model fitting
# =============================================================================

def fit_group_onesample(contrast_maps):
    """
    Fit a one-sample t-test (intercept-only second-level model).

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


def run_group_onesample(
    subjects, session, task, contrast_id, output_dir,
    mnum='value_parametric',
    model_variant='rt_in_duration',
    space='MNI152NLin2009cAsym_res-2',
):
    """
    Collect subject maps, fit the group one-sample t-test, and save
    the group tmap and effect_size niftis.

    Saves to:
        {output_dir}/{mnum}/{model_variant}/group/ses-{session}/
            {prefix}_{contrast_id}_tmap.nii.gz
            {prefix}_{contrast_id}_effect_size.nii.gz

    Parameters
    ----------
    subjects : list of str
    session, task, contrast_id : str
    output_dir : str
        Root output directory
    mnum, model_variant, space : str

    Returns
    -------
    group_tmap_path : str
        Path to saved group tmap
    group_effect_path : str
        Path to saved group effect_size map
    """
    import nibabel as nib

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
    print(f"  Fitting group model: task-{task} {contrast_id} "
          f"[{model_variant}] (n={n_subjects})")

    model, group_tmap, group_effect = fit_group_onesample(contrast_maps)

    group_dir = get_group_dir(output_dir, mnum, model_variant, session)
    os.makedirs(group_dir, exist_ok=True)

    prefix = get_group_map_prefix(task, space, mnum, model_variant)

    tmap_path = os.path.join(group_dir, f'{prefix}_{contrast_id}_tmap.nii.gz')
    nib.save(group_tmap, tmap_path)

    effect_path = os.path.join(
        group_dir, f'{prefix}_{contrast_id}_effect_size.nii.gz'
    )
    nib.save(group_effect, effect_path)

    print(f"  Saved group maps to: {group_dir}")
    return tmap_path, effect_path


def run_group_nonparametric(
    subjects, session, task, contrast_id, output_dir,
    mnum='value_parametric',
    model_variant='rt_in_duration',
    space='MNI152NLin2009cAsym_res-2',
    n_perm=10000,
    two_sided_test=False,
    n_jobs=1,
):
    """
    Run non-parametric inference with TFCE on a group contrast.

    Collects subject-level effect_size maps and runs
    nilearn.glm.second_level.non_parametric_inference with tfce=True.

    Saves to:
        {output_dir}/{mnum}/{model_variant}/group/ses-{session}/
            {prefix}_{contrast_id}_tfce_tmap.nii.gz
            {prefix}_{contrast_id}_tfce_logp.nii.gz

    Parameters
    ----------
    subjects : list of str
    session, task, contrast_id : str
    output_dir : str
    mnum, model_variant, space : str
    n_perm : int
        Number of permutations (default 10000)
    two_sided_test : bool
        If True, tests for both positive and negative effects
    n_jobs : int
        Number of parallel jobs for permutation testing

    Returns
    -------
    outputs : dict
        The full output dict from non_parametric_inference
    """
    import nibabel as nib
    from nilearn.glm.second_level import non_parametric_inference

    contrast_maps, missing = collect_contrast_maps(
        subjects, session, task, contrast_id, output_dir,
        mnum=mnum, model_variant=model_variant,
        space=space, map_type='effect_size',
    )

    if missing:
        print(f"  WARNING: Missing maps for subjects: {missing}")
    if len(contrast_maps) < 2:
        raise ValueError(
            f"Need at least 2 subjects, found {len(contrast_maps)}"
        )

    n_subjects = len(contrast_maps)
    design_matrix = pd.DataFrame({'intercept': np.ones(n_subjects)})

    print(f"  Running non-parametric inference (TFCE, {n_perm} perms): "
          f"task-{task} {contrast_id} [{model_variant}] (n={n_subjects})")

    outputs = non_parametric_inference(
        second_level_input=contrast_maps,
        design_matrix=design_matrix,
        n_perm=n_perm,
        two_sided_test=two_sided_test,
        tfce=True,
        n_jobs=n_jobs,
        verbose=1,
    )

    # Save maps
    group_dir = get_group_dir(output_dir, mnum, model_variant, session)
    os.makedirs(group_dir, exist_ok=True)
    prefix = get_group_map_prefix(task, space, mnum, model_variant)

    tmap_fn = os.path.join(
        group_dir, f'{prefix}_{contrast_id}_tfce_tmap.nii.gz'
    )
    nib.save(outputs['t'], tmap_fn)

    logp_fn = os.path.join(
        group_dir, f'{prefix}_{contrast_id}_tfce_logp.nii.gz'
    )
    nib.save(outputs['logp_max_tfce'], logp_fn)

    print(f"  Saved TFCE maps to: {group_dir}")
    return outputs


def get_group_tfce_logp_path(
    session, task, contrast_id, output_dir,
    mnum='value_parametric',
    model_variant='rt_in_duration',
    space='MNI152NLin2009cAsym_res-2',
):
    """Build the full path to a saved TFCE logp map."""
    group_dir = get_group_dir(output_dir, mnum, model_variant, session)
    prefix = get_group_map_prefix(task, space, mnum, model_variant)
    return os.path.join(group_dir, f'{prefix}_{contrast_id}_tfce_logp.nii.gz')


def generate_group_report_corrected(
    subjects, session, task, contrast_id, output_dir,
    mnum='value_parametric',
    model_variant='rt_in_duration',
    space='MNI152NLin2009cAsym_res-2',
    alpha=0.05,
    roi_coords=None,
):
    """
    Generate an HTML report from TFCE-corrected group maps.

    Reads the TFCE logp map and the uncorrected tmap from:
        {output_dir}/{mnum}/{model_variant}/group/ses-{session}/

    Thresholds the tmap using the logp map at -log10(alpha).
    Shows glass brain and ROI views of the corrected results.

    Saves report as:
        {prefix}_{contrast_id}_tfce_corrected_report.html

    Parameters
    ----------
    subjects : list of str
    session, task, contrast_id : str
    output_dir : str
    mnum, model_variant, space : str
    alpha : float
        Significance threshold (default 0.05, i.e. logp > 1.3)
    roi_coords : dict or None

    Returns
    -------
    report_path : str
    """
    import nibabel as nib
    from nilearn.image import math_img

    if roi_coords is None:
        roi_coords = ROI_COORDS

    # Load maps
    tmap_path = get_group_tmap_path(
        session, task, contrast_id, output_dir,
        mnum=mnum, model_variant=model_variant, space=space,
    )
    logp_path = get_group_tfce_logp_path(
        session, task, contrast_id, output_dir,
        mnum=mnum, model_variant=model_variant, space=space,
    )

    for path, label in [(tmap_path, 'tmap'), (logp_path, 'TFCE logp')]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Group {label} not found: {path}\n"
                f"Run the appropriate fitting function first."
            )

    group_tmap = nib.load(tmap_path)
    logp_img = nib.load(logp_path)

    # Threshold: mask tmap where logp > -log10(alpha)
    logp_thresh = -np.log10(alpha)
    thresholded_tmap = math_img(
        f'tmap * (logp > {logp_thresh})',
        tmap=group_tmap, logp=logp_img,
    )

    # Check if anything survives
    thresh_data = thresholded_tmap.get_fdata()
    n_sig_voxels = int(np.sum(thresh_data != 0))

    # Subject info
    _, missing = collect_contrast_maps(
        subjects, session, task, contrast_id, output_dir,
        mnum=mnum, model_variant=model_variant,
        space=space, map_type='effect_size',
    )
    n_subjects = len(subjects) - len(missing)
    subjects_used = [f'sub-{s}' for s in subjects if s not in missing]

    group_dir = get_group_dir(output_dir, mnum, model_variant, session)
    prefix = get_group_map_prefix(task, space, mnum, model_variant)

    print(f"  Generating TFCE report: task-{task} {contrast_id} "
          f"[{model_variant}] (n={n_subjects}, "
          f"{n_sig_voxels} significant voxels at alpha={alpha})")

    # -- Glass brain (corrected) --
    title_base = (f'Group (n={n_subjects}): task-{task} {contrast_id}\n'
                  f'{model_variant} | TFCE corrected, alpha={alpha}')

    fig_glass = plot_group_glass_brain(
        thresholded_tmap, title=title_base, threshold=0.01
    )
    img_glass = _fig_to_base64(fig_glass)

    # -- logp map glass brain --
    fig_logp = plot_group_glass_brain(
        logp_img,
        title=f'-log10(p) TFCE map (threshold at {logp_thresh:.2f})',
        threshold=logp_thresh,
    )
    img_logp = _fig_to_base64(fig_logp)

    # -- ROI views (corrected tmap) --
    roi_sections_html = ''
    for roi_name, roi_info in roi_coords.items():
        coord_list = roi_info['coords']
        description = roi_info['description']

        hemi_plots_html = ''
        for coords in coord_list:
            side = 'Left' if coords[0] < 0 else 'Right' if coords[0] > 0 else 'Midline'
            coord_str = f'({coords[0]}, {coords[1]}, {coords[2]})'

            fig_roi = plot_roi_view(
                thresholded_tmap, coords,
                title=f'{roi_name} {side} {coord_str} (TFCE corrected)',
                threshold=0.01,
            )
            img_roi = _fig_to_base64(fig_roi)

            hemi_plots_html += f"""
    <div class="plot-container">
        <h3>{side} {coord_str}</h3>
        <img src="data:image/png;base64,{img_roi}" />
    </div>
"""

        roi_sections_html += f"""
<div class="section">
    <h2>{roi_name}: {description}</h2>
    {hemi_plots_html}
</div>
"""

    # -- HTML --
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Group Report (TFCE corrected): {contrast_id} | task-{task} | {model_variant}</title>
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
    .meta {{ color: #888; font-size: 13px; }}
    .info-grid {{ display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 16px; }}
    .info-card {{ flex: 1; min-width: 200px; background: #f0f4f8;
                  border-radius: 6px; padding: 12px 16px; }}
    .info-card h3 {{ margin-top: 0; }}
    .corrected {{ color: #155724; background: #d4edda; border: 1px solid #28a745;
                  border-radius: 6px; padding: 10px 16px; margin-bottom: 16px; }}
</style>
</head>
<body>

<h1>Group-Level Analysis Report (TFCE Corrected)</h1>
<p class="meta">
    Contrast: <strong>{contrast_id}</strong> | Task: {task} |
    Model: {mnum} | Variant: {model_variant}<br>
    Session: ses-{session} | Space: {space}
</p>
<div class="corrected">
    Results corrected using Threshold-Free Cluster Enhancement (TFCE) with
    permutation testing. Significance threshold: alpha = {alpha}
    (-log10(p) > {logp_thresh:.2f}).
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
            <p>{n_sig_voxels} significant voxels (TFCE corrected, alpha={alpha})</p>
        </div>
    </div>
</div>

<div class="section">
    <h2>Glass Brain: t-values (TFCE corrected, alpha={alpha})</h2>
    <div class="plot-container">
        <img src="data:image/png;base64,{img_glass}" />
    </div>
</div>

<div class="section">
    <h2>Glass Brain: -log10(p) TFCE map</h2>
    <div class="plot-container">
        <img src="data:image/png;base64,{img_logp}" />
    </div>
</div>

{roi_sections_html}

</body>
</html>"""

    report_fn = f'{prefix}_{contrast_id}_tfce_corrected_report.html'
    report_path = os.path.join(group_dir, report_fn)
    with open(report_path, 'w') as f:
        f.write(html)

    print(f"  Report saved: {report_path}")
    return report_path


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


def plot_roi_view(stat_map, coords, title='', threshold=3.0):
    """
    Ortho (sagittal + coronal + axial) stat map centered on one coordinate
    with crosshairs at the ROI center.
    """
    fig = plt.figure(figsize=(12, 4))
    display = plot_stat_map(
        stat_map, threshold=threshold,
        display_mode='ortho', cut_coords=coords,
        draw_cross=True, colorbar=True,
        title=title, figure=fig,
    )
    return fig


# =============================================================================
# Report generation (reads saved maps, no fitting)
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
    Generate an HTML report from previously saved group maps.

    Reads the group tmap from:
        {output_dir}/{mnum}/{model_variant}/group/ses-{session}/
            {prefix}_{contrast_id}_tmap.nii.gz

    Saves report to the same directory as:
        {prefix}_{contrast_id}_uncorrected_report.html

    Parameters
    ----------
    subjects : list of str
        Subject numbers (used for display in the report)
    session, task, contrast_id : str
    output_dir : str
        Root output directory
    mnum, model_variant, space : str
    threshold : float
        t-stat threshold for visualization (uncorrected)
    cluster_threshold : int
        Minimum cluster size (voxels) for cluster table
    roi_coords : dict or None
        ROI definitions. If None, uses default ROI_COORDS.

    Returns
    -------
    report_path : str
    """
    import nibabel as nib

    if roi_coords is None:
        roi_coords = ROI_COORDS

    # Load saved group tmap
    tmap_path = get_group_tmap_path(
        session, task, contrast_id, output_dir,
        mnum=mnum, model_variant=model_variant, space=space,
    )
    if not os.path.exists(tmap_path):
        raise FileNotFoundError(
            f"Group tmap not found: {tmap_path}\n"
            f"Run run_group_onesample() first."
        )
    group_tmap = nib.load(tmap_path)

    # Determine n_subjects from the subject list
    _, missing = collect_contrast_maps(
        subjects, session, task, contrast_id, output_dir,
        mnum=mnum, model_variant=model_variant,
        space=space, map_type='effect_size',
    )
    n_subjects = len(subjects) - len(missing)
    subjects_used = [f'sub-{s}' for s in subjects if s not in missing]

    group_dir = get_group_dir(output_dir, mnum, model_variant, session)
    prefix = get_group_map_prefix(task, space, mnum, model_variant)

    print(f"  Generating report: task-{task} {contrast_id} "
          f"[{model_variant}] (n={n_subjects})")

    # -- Glass brain --
    title_base = (f'Group (n={n_subjects}): task-{task} {contrast_id}\n'
                  f'{model_variant} | t > {threshold} (uncorrected)')

    fig_glass = plot_group_glass_brain(
        group_tmap, title=title_base, threshold=threshold
    )
    img_glass = _fig_to_base64(fig_glass)

    # -- ROI views (bilateral) --
    roi_sections_html = ''
    for roi_name, roi_info in roi_coords.items():
        coord_list = roi_info['coords']
        description = roi_info['description']

        hemi_plots_html = ''
        for coords in coord_list:
            side = 'Left' if coords[0] < 0 else 'Right' if coords[0] > 0 else 'Midline'
            coord_str = f'({coords[0]}, {coords[1]}, {coords[2]})'

            fig_roi = plot_roi_view(
                group_tmap, coords,
                title=f'{roi_name} {side} {coord_str}',
                threshold=threshold,
            )
            img_roi = _fig_to_base64(fig_roi)

            hemi_plots_html += f"""
    <div class="plot-container">
        <h3>{side} {coord_str}</h3>
        <img src="data:image/png;base64,{img_roi}" />
    </div>
"""

        roi_sections_html += f"""
<div class="section">
    <h2>{roi_name}: {description}</h2>
    {hemi_plots_html}
</div>
"""

    # -- Cluster table + top 5 peak views --
    cluster_plots_html = ''
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

        top_n = min(5, len(cluster_table))
        for i in range(top_n):
            row = cluster_table.iloc[i]
            peak_coords = (float(row['X']), float(row['Y']), float(row['Z']))
            peak_stat = float(row['Peak Stat'])
            cluster_id = row['Cluster ID']
            cluster_size = row['Cluster Size (mm3)']
            coord_str = f'({peak_coords[0]:.0f}, {peak_coords[1]:.0f}, {peak_coords[2]:.0f})'

            fig_cluster = plot_roi_view(
                group_tmap, peak_coords,
                title=f'Cluster {cluster_id}: {coord_str}, '
                      f't = {peak_stat:.2f}, {cluster_size} mm3',
                threshold=threshold,
            )
            img_cluster = _fig_to_base64(fig_cluster)

            cluster_plots_html += f"""
    <div class="plot-container">
        <h3>Cluster {cluster_id}: peak at {coord_str} (t = {peak_stat:.2f}, {cluster_size} mm3)</h3>
        <img src="data:image/png;base64,{img_cluster}" />
    </div>
"""
    except Exception:
        cluster_html = '<p>No suprathreshold clusters found.</p>'
        n_clusters = 0

    # -- HTML --
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

<div class="section">
    <h2>Top Cluster Peaks</h2>
    {cluster_plots_html}
</div>

</body>
</html>"""

    report_fn = f'{prefix}_{contrast_id}_uncorrected_report.html'
    report_path = os.path.join(group_dir, report_fn)
    with open(report_path, 'w') as f:
        f.write(html)

    print(f"  Report saved: {report_path}")
    return report_path


# =============================================================================
# Comparison report across model variants and tasks
# =============================================================================

def generate_comparison_report(
    subjects, session, contrast_id, output_dir,
    tasks=('yesNo', 'binaryChoice'),
    model_variants=('rt_in_duration', 'rt_duration_plus_mod'),
    mnum='value_parametric',
    space='MNI152NLin2009cAsym_res-2',
    threshold=3.0,
    roi_coords=None,
    report_dir=None,
):
    """
    Generate an HTML report comparing a contrast across tasks and model
    variants. Reads previously saved group tmaps (no fitting).

    Parameters
    ----------
    subjects : list of str
    session, contrast_id : str
    output_dir : str
        Root output directory (where per-variant group maps live)
    tasks : tuple of str
    model_variants : tuple of str
    mnum, space : str
    threshold : float
    roi_coords : dict or None
    report_dir : str or None
        If None, defaults to
        {output_dir}/{mnum}/rt_effect_comparison/group/ses-{session}/

    Returns
    -------
    report_path : str
    """
    import nibabel as nib

    if roi_coords is None:
        roi_coords = ROI_COORDS

    if report_dir is None:
        report_dir = os.path.join(
            output_dir, mnum, 'rt_effect_comparison',
            'group', f'ses-{session}'
        )
    os.makedirs(report_dir, exist_ok=True)

    # Load all group tmaps
    group_tmaps = {}
    for variant in model_variants:
        for task in tasks:
            tmap_path = get_group_tmap_path(
                session, task, contrast_id, output_dir,
                mnum=mnum, model_variant=variant, space=space,
            )
            if os.path.exists(tmap_path):
                group_tmaps[(variant, task)] = nib.load(tmap_path)
                print(f"  Loaded: {variant} / {task}")
            else:
                print(f"  WARNING: Missing group tmap: {tmap_path}")

    if not group_tmaps:
        raise FileNotFoundError(
            "No group tmaps found. Run run_group_onesample() first."
        )

    variant_labels = {
        'rt_in_duration': 'RT in duration',
        'rt_duration_plus_mod': 'RT duration + modulator',
        'fixed_duration': 'Fixed duration',
        'fixed_duration_plus_mod': 'Fixed duration + modulator',
    }

    # -- ROI comparison sections --
    roi_sections_html = ''
    for roi_name, roi_info in roi_coords.items():
        coord_list = roi_info['coords']
        description = roi_info['description']

        for coords in coord_list:
            side = 'Left' if coords[0] < 0 else 'Right' if coords[0] > 0 else 'Midline'
            coord_str = f'({coords[0]}, {coords[1]}, {coords[2]})'

            grid_html = ''
            for variant in model_variants:
                for task in tasks:
                    tmap = group_tmaps.get((variant, task))
                    if tmap is None:
                        grid_html += f"""
        <div class="grid-cell">
            <h4>{variant_labels.get(variant, variant)} / {task}</h4>
            <p>Missing</p>
        </div>
"""
                        continue

                    fig = plot_roi_view(
                        tmap, coords,
                        title=f'{variant_labels.get(variant, variant)} / {task}',
                        threshold=threshold,
                    )
                    img_b64 = _fig_to_base64(fig)

                    grid_html += f"""
        <div class="grid-cell">
            <h4>{variant_labels.get(variant, variant)} / {task}</h4>
            <img src="data:image/png;base64,{img_b64}" />
        </div>
"""

            roi_sections_html += f"""
<div class="section">
    <h2>{roi_name} {side} {coord_str}</h2>
    <p>{description}</p>
    <div class="comparison-grid">
        {grid_html}
    </div>
</div>
"""

    # -- Glass brain comparison --
    glass_html = ''
    for variant in model_variants:
        for task in tasks:
            tmap = group_tmaps.get((variant, task))
            if tmap is None:
                continue
            label = f'{variant_labels.get(variant, variant)} / {task}'
            fig_glass = plot_group_glass_brain(
                tmap, title=label, threshold=threshold
            )
            img_glass = _fig_to_base64(fig_glass)
            glass_html += f"""
        <div class="grid-cell wide">
            <h4>{label}</h4>
            <img src="data:image/png;base64,{img_glass}" />
        </div>
"""

    # -- HTML --
    n_tasks = len(tasks)
    grid_cols = n_tasks

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Comparison Report: {contrast_id} | RT effect</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           margin: 0; padding: 20px 40px; background: #f8f9fa; color: #333; }}
    h1 {{ border-bottom: 3px solid #2c3e50; padding-bottom: 10px; color: #2c3e50; }}
    h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 6px; margin-top: 30px; }}
    h3 {{ color: #555; margin-bottom: 8px; }}
    h4 {{ color: #666; margin: 4px 0; font-size: 13px; }}
    .section {{ background: #fff; border-radius: 8px; padding: 20px 30px;
                margin-bottom: 30px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
    .meta {{ color: #888; font-size: 13px; }}
    .warning {{ color: #856404; background: #fff3cd; border: 1px solid #ffc107;
                border-radius: 6px; padding: 10px 16px; margin-bottom: 16px; }}
    .comparison-grid {{
        display: grid;
        grid-template-columns: repeat({grid_cols}, 1fr);
        gap: 12px;
    }}
    .grid-cell {{
        text-align: center;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 8px;
        background: #fafafa;
    }}
    .grid-cell.wide {{
        grid-column: span {grid_cols};
    }}
    .grid-cell img {{
        max-width: 100%; height: auto;
        border-radius: 4px;
    }}
</style>
</head>
<body>

<h1>Model Variant Comparison: {contrast_id}</h1>
<p class="meta">
    Contrast: <strong>{contrast_id}</strong> |
    Variants: {', '.join(variant_labels.get(v, v) for v in model_variants)} |
    Tasks: {', '.join(tasks)}<br>
    Session: ses-{session} | Space: {space} | Subjects: {', '.join(subjects)}
</p>
<div class="warning">
    All maps shown at t > {threshold} (uncorrected). Columns correspond to
    tasks, rows to model variants. Each panel shows the same ROI coordinate
    across conditions.
</div>

<div class="section">
    <h2>Glass Brain Overview</h2>
    <div class="comparison-grid">
        {glass_html}
    </div>
</div>

{roi_sections_html}

</body>
</html>"""

    report_fn = (f'comparison_{contrast_id}_'
                 f'{"_vs_".join(model_variants)}_uncorrected_report.html')
    report_path = os.path.join(report_dir, report_fn)
    with open(report_path, 'w') as f:
        f.write(html)

    print(f"  Comparison report saved: {report_path}")
    return report_path