"""
level2_report_paired.py
Generate an HTML report comparing two sessions (e.g., week 3 vs week 1).

Each ROI and each cluster peak gets a row with:
  - Bar plot on the left (paired betas per subject with within-ROI SE)
  - Stacked sagittal + coronal views on the right (ses-a, ses-b, difference)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nilearn.plotting import view_img
from nilearn.reporting import get_clusters_table

from level2_helpers import (
    ROI_COORDS,
    get_group_dir, get_group_map_prefix, get_group_tmap_path,
    get_contrast_path,
    collect_contrast_maps, add_atlas_labels_to_cluster_table,
    is_cerebellar,
    extract_roi_betas_paired, roi_paired_ttest_table,
    extract_roi_betas_with_variance,
    fig_to_base64, plot_group_glass_brain, plot_stacked_sessions,
)


def _plot_paired_betas(var_df_a, var_df_b, coord_label, ttest_row,
                       ses_a_label, ses_b_label):
    """
    Paired bar plot: one bar per subject per session, with within-ROI
    SE error bars. Shows the paired t-test result in the title.
    """
    sub_a = var_df_a[var_df_a['coord_label'] == coord_label].sort_values('subject')
    sub_b = var_df_b[var_df_b['coord_label'] == coord_label].sort_values('subject')

    subjects = sub_a['subject'].values
    betas_a = sub_a['mean_beta'].values
    betas_b = sub_b['mean_beta'].values
    se_a = sub_a['se_beta'].values
    se_b = sub_b['se_beta'].values
    n = len(subjects)

    fig, ax = plt.subplots(figsize=(5, 3.5))

    x = np.arange(n)
    width = 0.35

    ax.bar(x - width/2, betas_a, width, label=ses_a_label,
           color='#7fb3d8', alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, betas_b, width, label=ses_b_label,
           color='#2171b5', alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.errorbar(x - width/2, betas_a, yerr=se_a, fmt='none',
                ecolor='#444', elinewidth=1.2, capsize=3, capthick=1)
    ax.errorbar(x + width/2, betas_b, yerr=se_b, fmt='none',
                ecolor='#444', elinewidth=1.2, capsize=3, capthick=1)

    ax.axhline(0, color='#333', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'sub-{s}' for s in subjects], fontsize=8, rotation=45)
    ax.set_ylabel('Mean beta (effect size)', fontsize=9)
    ax.legend(fontsize=7, loc='best')

    t_val = ttest_row['t_stat']
    p_bonf = ttest_row['p_bonf']
    sig = ttest_row['significant']
    sig_marker = ' *' if sig else ''
    mean_diff = ttest_row['mean_diff']

    ax.set_title(
        f'diff = {mean_diff:.4f},  t = {t_val:.2f},  '
        f'p(Bonf) = {p_bonf:.4f}{sig_marker}',
        fontsize=9, fontweight='bold' if sig else 'normal',
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


def _build_panel(coord_label, coords, title, subtitle, ttest_row,
                 var_df_a, var_df_b, ses_a_label, ses_b_label,
                 tmap_a, tmap_b, diff_tmap, threshold):
    """Build one ROI/cluster panel with bar plot + stacked session views."""

    sig = ttest_row['significant']
    sig_class = 'sig-yes' if sig else 'sig-no'
    sig_text = 'Significant' if sig else 'Not significant'

    # Bar plot
    fig_bar = _plot_paired_betas(
        var_df_a, var_df_b, coord_label, ttest_row,
        ses_a_label, ses_b_label,
    )
    img_bar = fig_to_base64(fig_bar)

    # Stacked session views (sagittal + coronal for each)
    stat_maps = [tmap_a, tmap_b, diff_tmap]
    labels = [ses_a_label, ses_b_label, 'Difference']
    fig_stacked = plot_stacked_sessions(
        stat_maps, coords, labels, threshold=threshold,
    )
    img_stacked = fig_to_base64(fig_stacked)

    return f"""
<div class="section">
    <div class="panel-header">
        <h2>{title}</h2>
        <span class="{sig_class}">{sig_text} (Bonferroni)</span>
    </div>
    <p class="panel-meta">{subtitle}</p>
    <div class="side-by-side">
        <div class="grid-cell">
            <img src="data:image/png;base64,{img_bar}" />
        </div>
        <div class="grid-cell">
            <img src="data:image/png;base64,{img_stacked}" />
        </div>
    </div>
</div>
"""


def generate_paired_report(
    subjects, sessions, task, contrast_id, output_dir,
    mnum='value_parametric',
    model_variant='rt_in_duration',
    space='MNI152NLin2009cAsym_res-2',
    threshold=3.0,
    cluster_threshold=10,
    alpha=0.05,
    roi_coords=None,
    max_cluster_panels=20,
):
    """
    Generate an HTML report comparing two sessions.

    Each ROI coordinate and each cluster peak gets a panel with:
      - Paired bar plot (per subject, within-ROI SE error bars)
      - Stacked sagittal + coronal views for ses-a, ses-b, and difference

    Parameters
    ----------
    subjects : list of str
    sessions : tuple of (session_a, session_b)
    task, contrast_id : str
    output_dir : str
    mnum, model_variant, space : str
    threshold, cluster_threshold, alpha : float
    roi_coords : dict or None
    max_cluster_panels : int

    Returns
    -------
    report_path : str
    """
    import nibabel as nib

    if roi_coords is None:
        roi_coords = ROI_COORDS

    ses_a, ses_b = sessions
    ses_a_label = f'Week {int(ses_a)} (ses-{ses_a})'
    ses_b_label = f'Week {int(ses_b)} (ses-{ses_b})'

    # Load paired difference tmap
    group_dir = get_group_dir(output_dir, mnum, model_variant, ses_b)
    prefix = get_group_map_prefix(task, space, mnum, model_variant)

    diff_tmap_path = os.path.join(
        group_dir,
        f'{prefix}_{contrast_id}_ses{ses_b}_minus_ses{ses_a}_tmap.nii.gz'
    )
    if not os.path.exists(diff_tmap_path):
        raise FileNotFoundError(
            f"Paired difference tmap not found: {diff_tmap_path}\n"
            f"Run run_group_paired_ttest() first."
        )
    diff_tmap = nib.load(diff_tmap_path)

    # Load individual session tmaps
    tmap_a_path = get_group_tmap_path(
        ses_a, task, contrast_id, output_dir,
        mnum=mnum, model_variant=model_variant, space=space,
    )
    tmap_b_path = get_group_tmap_path(
        ses_b, task, contrast_id, output_dir,
        mnum=mnum, model_variant=model_variant, space=space,
    )
    has_session_tmaps = os.path.exists(tmap_a_path) and os.path.exists(tmap_b_path)
    if has_session_tmaps:
        tmap_a = nib.load(tmap_a_path)
        tmap_b = nib.load(tmap_b_path)
    else:
        tmap_a = diff_tmap
        tmap_b = diff_tmap
        print(f"  NOTE: Individual session tmaps not found. "
              f"Using difference map for all views.")

    # Count subjects with both sessions
    n_subjects = 0
    for subnum in subjects:
        pa = get_contrast_path(
            subnum, ses_a, task, contrast_id, output_dir,
            mnum=mnum, model_variant=model_variant,
            space=space, map_type='effect_size',
        )
        pb = get_contrast_path(
            subnum, ses_b, task, contrast_id, output_dir,
            mnum=mnum, model_variant=model_variant,
            space=space, map_type='effect_size',
        )
        if os.path.exists(pa) and os.path.exists(pb):
            n_subjects += 1

    print(f"  Generating paired report: ses-{ses_b} - ses-{ses_a}, "
          f"task-{task} {contrast_id} [{model_variant}] (n={n_subjects})")

    # -- Glass brain of difference --
    title_diff = (f'Group (n={n_subjects}): ses-{ses_b} minus ses-{ses_a}\n'
                  f'task-{task} {contrast_id} [{model_variant}] | '
                  f't > {threshold} (uncorrected)')
    fig_glass = plot_group_glass_brain(diff_tmap, title=title_diff, threshold=threshold)
    img_glass = fig_to_base64(fig_glass)

    # -- Interactive viewer --
    interactive_view = view_img(
        diff_tmap, threshold=threshold,
        title=f'ses-{ses_b} minus ses-{ses_a}: {contrast_id} [{model_variant}]',
    )
    interactive_html = interactive_view.get_iframe(width=900, height=500)

    # -- Extract voxelwise variance for both sessions --
    print(f"  Extracting ROI betas (both sessions)...")
    paired_betas = extract_roi_betas_paired(
        subjects, sessions, task, contrast_id, output_dir,
        mnum=mnum, model_variant=model_variant, space=space,
        roi_coords=roi_coords,
    )
    ttest_results = roi_paired_ttest_table(paired_betas, alpha=alpha)

    print(f"  Extracting voxelwise variance for error bars...")
    var_df_a = extract_roi_betas_with_variance(
        subjects, ses_a, task, contrast_id, output_dir,
        mnum=mnum, model_variant=model_variant, space=space,
        roi_coords=roi_coords,
    )
    var_df_b = extract_roi_betas_with_variance(
        subjects, ses_b, task, contrast_id, output_dir,
        mnum=mnum, model_variant=model_variant, space=space,
        roi_coords=roi_coords,
    )

    # -- ROI panels --
    n_roi_tests = len(ttest_results)
    n_roi_sig = int(ttest_results['significant'].sum())
    roi_panels_html = ''

    for roi_name, roi_info in roi_coords.items():
        for coords in roi_info['coords']:
            side = 'L' if coords[0] < 0 else 'R' if coords[0] > 0 else 'M'
            coord_label = f'{roi_name}_{side}'
            coord_str = f'({coords[0]}, {coords[1]}, {coords[2]})'

            row_mask = ttest_results['coord_label'] == coord_label
            if not row_mask.any():
                continue
            ttest_row = ttest_results[row_mask].iloc[0]

            roi_panels_html += _build_panel(
                coord_label, coords,
                title=f'{roi_name} ({side})',
                subtitle=f'{roi_info["description"]} | {coord_str} | '
                         f'Sphere radius: {roi_info.get("radius", 10)} mm',
                ttest_row=ttest_row,
                var_df_a=var_df_a, var_df_b=var_df_b,
                ses_a_label=ses_a_label, ses_b_label=ses_b_label,
                tmap_a=tmap_a, tmap_b=tmap_b, diff_tmap=diff_tmap,
                threshold=threshold,
            )

    # -- Cluster-based panels --
    cluster_panels_html = ''
    cluster_html = '<p>No suprathreshold clusters found.</p>'
    n_clusters = 0

    try:
        cluster_table = get_clusters_table(
            diff_tmap, stat_threshold=threshold,
            min_distance=8,
            cluster_threshold=cluster_threshold,
        )
        cluster_table = add_atlas_labels_to_cluster_table(cluster_table)
        cluster_html = cluster_table.to_html(
            classes='cluster-table', index=False, float_format='%.2f'
        )
        n_clusters = len(cluster_table)

        non_cb_rows = [
            i for i in range(len(cluster_table))
            if not is_cerebellar(cluster_table.iloc[i])
        ]
        n_panels = min(max_cluster_panels, len(non_cb_rows))
        if n_panels > 0:
            # Build ad-hoc ROI coords from non-cerebellar cluster peaks
            panel_indices = non_cb_rows[:n_panels]
            cluster_roi_coords = {}
            for i in panel_indices:
                row = cluster_table.iloc[i]
                peak_coords = (float(row['X']), float(row['Y']), float(row['Z']))
                cluster_id = str(row['Cluster ID'])
                label = f'cluster_{cluster_id}'
                cluster_roi_coords[label] = {
                    'coords': [peak_coords],
                    'description': '',
                    'radius': 10,
                }

            # Extract and run paired tests for cluster peaks
            print(f"  Extracting betas for {n_panels} non-cerebellar cluster peaks...")
            cluster_paired = extract_roi_betas_paired(
                subjects, sessions, task, contrast_id, output_dir,
                mnum=mnum, model_variant=model_variant, space=space,
                roi_coords=cluster_roi_coords,
            )
            # Bonferroni across cluster peaks only (separate from ROI tests)
            cluster_ttest = roi_paired_ttest_table(cluster_paired, alpha=alpha)

            cluster_var_a = extract_roi_betas_with_variance(
                subjects, ses_a, task, contrast_id, output_dir,
                mnum=mnum, model_variant=model_variant, space=space,
                roi_coords=cluster_roi_coords,
            )
            cluster_var_b = extract_roi_betas_with_variance(
                subjects, ses_b, task, contrast_id, output_dir,
                mnum=mnum, model_variant=model_variant, space=space,
                roi_coords=cluster_roi_coords,
            )

            for i in panel_indices:
                row = cluster_table.iloc[i]
                peak_coords = (float(row['X']), float(row['Y']), float(row['Z']))
                peak_stat = float(row['Peak Stat'])
                cluster_id = str(row['Cluster ID'])
                cluster_size = row['Cluster Size (mm3)']
                coord_str = f'({peak_coords[0]:.0f}, {peak_coords[1]:.0f}, {peak_coords[2]:.0f})'
                label = f'cluster_{cluster_id}'

                atlas_labels = []
                for col in ['Harvard-Oxford', 'AAL']:
                    if col in row.index:
                        val = str(row[col])
                        if val and val != 'Background' and val != 'nan':
                            atlas_labels.append(val)
                region_label = ' / '.join(atlas_labels) if atlas_labels else 'Unlabeled region'

                coord_label = f'{label}_M' if peak_coords[0] == 0 else (
                    f'{label}_L' if peak_coords[0] < 0 else f'{label}_R'
                )

                ct_mask = cluster_ttest['coord_label'] == coord_label
                if not ct_mask.any():
                    continue
                ct_row = cluster_ttest[ct_mask].iloc[0]

                cluster_panels_html += _build_panel(
                    coord_label, peak_coords,
                    title=f'Cluster {cluster_id}: {region_label}',
                    subtitle=f'Peak: {coord_str} | t = {peak_stat:.2f} | '
                             f'{cluster_size} mm3 | 10mm sphere extraction',
                    ttest_row=ct_row,
                    var_df_a=cluster_var_a, var_df_b=cluster_var_b,
                    ses_a_label=ses_a_label, ses_b_label=ses_b_label,
                    tmap_a=tmap_a, tmap_b=tmap_b, diff_tmap=diff_tmap,
                    threshold=threshold,
                )

    except Exception as e:
        print(f"  WARNING: Cluster extraction failed: {e}")
        import traceback
        traceback.print_exc()

    # -- T-test summary table (ROIs) --
    ttest_display = ttest_results[[
        'roi_name', 'coord_label', 'x', 'y', 'z', 'radius',
        'n_subjects', 'mean_beta_ses_a', 'mean_beta_ses_b',
        'mean_diff', 'se_diff', 't_stat', 'p_uncorr', 'p_bonf', 'significant',
    ]].copy()
    ttest_display.columns = [
        'ROI', 'Label', 'X', 'Y', 'Z', 'Radius',
        'N', f'Mean (ses-{ses_a})', f'Mean (ses-{ses_b})',
        'Mean diff', 'SE diff', 't', 'p (uncorr)', 'p (Bonf)', 'Significant',
    ]
    ttest_html = ttest_display.to_html(
        classes='ttest-table', index=False, float_format='%.4f'
    )

    # -- HTML --
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Paired Report: ses-{ses_b} vs ses-{ses_a} | {contrast_id} | task-{task} | {model_variant}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           margin: 0; padding: 20px 40px; background: #f8f9fa; color: #333; }}
    h1 {{ border-bottom: 3px solid #2c3e50; padding-bottom: 10px; color: #2c3e50; }}
    h2 {{ color: #34495e; margin: 0; }}
    h3 {{ color: #555; margin-bottom: 8px; }}
    .section {{ background: #fff; border-radius: 8px; padding: 20px 30px;
                margin-bottom: 30px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
    .meta {{ color: #888; font-size: 13px; }}
    .panel-header {{
        display: flex; justify-content: space-between; align-items: center;
        border-bottom: 1px solid #bdc3c7; padding-bottom: 6px; margin-bottom: 6px;
    }}
    .panel-meta {{ color: #666; font-size: 12px; font-style: italic; margin-bottom: 10px; }}
    .plot-container {{ text-align: center; margin-bottom: 16px; }}
    .plot-container img {{ max-width: 100%; height: auto;
                           border: 1px solid #e0e0e0; border-radius: 4px; }}
    .info-grid {{ display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 16px; }}
    .info-card {{ flex: 1; min-width: 180px; background: #f0f4f8;
                  border-radius: 6px; padding: 12px 16px; }}
    .info-card h3 {{ margin-top: 0; }}
    .side-by-side {{
        display: grid; grid-template-columns: 1fr 1.4fr;
        gap: 8px; margin-bottom: 16px;
    }}
    .grid-cell {{
        text-align: center; border: 1px solid #e0e0e0;
        border-radius: 6px; padding: 6px; background: #fafafa;
    }}
    .grid-cell img {{ max-width: 100%; height: auto; border-radius: 4px; }}
    .sig-yes {{ color: #155724; background: #d4edda; border-radius: 4px; padding: 2px 8px; font-weight: bold; font-size: 12px; }}
    .sig-no {{ color: #856404; background: #fff3cd; border-radius: 4px; padding: 2px 8px; font-size: 12px; }}
    table {{ border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 6px 10px; border: 1px solid #ddd; text-align: left; }}
    th {{ background: #f0f4f8; }}
    .ttest-table {{ width: 100%; }}
    .cluster-table {{ width: auto; }}
    .method-note {{ color: #555; font-size: 12px; background: #f0f4f8;
                    border-radius: 6px; padding: 12px 16px; margin-bottom: 16px;
                    line-height: 1.6; }}
    .divider {{ border-top: 3px solid #2c3e50; margin: 40px 0 20px 0; }}
    .divider-label {{ color: #2c3e50; font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
</style>
</head>
<body>

<h1>Paired Session Comparison: ses-{ses_b} vs ses-{ses_a}</h1>
<p class="meta">
    Contrast: <strong>{contrast_id}</strong> | Task: {task} |
    Model: {mnum} | Variant: {model_variant}<br>
    Comparison: {ses_b_label} minus {ses_a_label} | Space: {space}
</p>

<div class="method-note">
    <strong>Method:</strong> For each subject, difference maps were computed
    (ses-{ses_b} minus ses-{ses_a} effect size). A one-sample t-test on the
    difference maps tests whether the contrast changed from {ses_a_label} to
    {ses_b_label}. For ROI analyses, mean betas were extracted within
    spherical masks and paired t-tests were performed with Bonferroni correction
    across {n_roi_tests} ROI coordinates (corrected alpha = {alpha}/{n_roi_tests} = {alpha/n_roi_tests:.4f}).
    Cluster-based panels use data-driven coordinates from the whole-brain
    difference map (exploratory, Bonferroni corrected across cluster peaks separately).
</div>

<div class="section">
    <h2>Summary</h2>
    <div class="info-grid">
        <div class="info-card">
            <h3>Subjects</h3>
            <p>n = {n_subjects} (with both sessions)</p>
        </div>
        <div class="info-card">
            <h3>ROI Tests</h3>
            <p>{n_roi_sig} / {n_roi_tests} significant after Bonferroni (alpha = {alpha})</p>
        </div>
        <div class="info-card">
            <h3>Whole-brain</h3>
            <p>{n_clusters} difference clusters (t > {threshold}, uncorrected)</p>
        </div>
    </div>
</div>

<div class="section">
    <h2>Glass Brain: Paired Difference (ses-{ses_b} minus ses-{ses_a})</h2>
    <div class="plot-container">
        <img src="data:image/png;base64,{img_glass}" />
    </div>
</div>

<div class="section">
    <h2>Interactive Viewer: Paired Difference</h2>
    <div class="plot-container">
        {interactive_html}
    </div>
</div>

<div class="section">
    <h2>ROI Paired T-Test Summary</h2>
    {ttest_html}
</div>

<div class="divider"></div>
<p class="divider-label">Hypothesis-Driven ROIs</p>

{roi_panels_html}

<div class="divider"></div>
<p class="divider-label">Data-Driven Cluster Peaks (Exploratory, non-cerebellar clusters only)</p>

<div class="section">
    <h2>Difference Map Cluster Table (t > {threshold}, uncorrected, min {cluster_threshold} voxels)</h2>
    {cluster_html}
</div>

{cluster_panels_html}

</body>
</html>"""

    report_fn = (f'{prefix}_{contrast_id}_ses{ses_b}_minus_ses{ses_a}'
                 f'_paired_report.html')
    report_path = os.path.join(group_dir, report_fn)
    with open(report_path, 'w') as f:
        f.write(html)

    print(f"  Paired report saved: {report_path}")
    return report_path