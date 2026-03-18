"""
level2_report_paired.py
Generate an HTML report comparing two sessions (e.g., week 3 vs week 1).

Includes:
  - Whole-brain glass brain and interactive viewer of the paired difference tmap
  - ROI-focused paired bar plots (session A vs session B per subject)
  - Paired t-tests with Bonferroni correction
  - Whole-brain cluster table with atlas labels
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
    get_group_dir, get_group_map_prefix,
    collect_contrast_maps, add_atlas_labels_to_cluster_table,
    extract_roi_betas_paired, roi_paired_ttest_table,
    extract_roi_betas_with_variance,
    fig_to_base64, plot_group_glass_brain, plot_roi_view,
)


def _plot_paired_roi_betas(paired_betas_df, coord_label, ttest_row,
                           ses_a_label, ses_b_label,
                           var_df_a=None, var_df_b=None):
    """
    Create a paired bar plot showing session A and session B betas for
    each subject, with per-subject error bars (SE = SD/sqrt(n_voxels)
    across voxels within the ROI sphere).
    """
    sub_data = paired_betas_df[
        paired_betas_df['coord_label'] == coord_label
    ].sort_values('subject')

    subjects = sub_data['subject'].values
    betas_a = sub_data['beta_ses_a'].values
    betas_b = sub_data['beta_ses_b'].values
    n = len(subjects)

    # Get per-subject SE from variance dataframes if available
    se_a = np.zeros(n)
    se_b = np.zeros(n)
    if var_df_a is not None and var_df_b is not None:
        for i, subnum in enumerate(subjects):
            row_a = var_df_a[
                (var_df_a['coord_label'] == coord_label) &
                (var_df_a['subject'] == subnum)
            ]
            if len(row_a) > 0:
                se_a[i] = row_a.iloc[0]['se_beta']

            row_b = var_df_b[
                (var_df_b['coord_label'] == coord_label) &
                (var_df_b['subject'] == subnum)
            ]
            if len(row_b) > 0:
                se_b[i] = row_b.iloc[0]['se_beta']

    fig, ax = plt.subplots(figsize=(6, 3.5))

    x = np.arange(n)
    width = 0.35

    bars_a = ax.bar(x - width/2, betas_a, width, label=ses_a_label,
                    color='#7fb3d8', alpha=0.85, edgecolor='white', linewidth=0.5)
    bars_b = ax.bar(x + width/2, betas_b, width, label=ses_b_label,
                    color='#2171b5', alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.errorbar(x - width/2, betas_a, yerr=se_a, fmt='none',
                ecolor='#444', elinewidth=1.2, capsize=3, capthick=1)
    ax.errorbar(x + width/2, betas_b, yerr=se_b, fmt='none',
                ecolor='#444', elinewidth=1.2, capsize=3, capthick=1)

    ax.axhline(0, color='#333', linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'sub-{s}' for s in subjects], fontsize=8, rotation=45)
    ax.set_ylabel('Mean beta (effect size)', fontsize=9)
    ax.legend(fontsize=8, loc='best')

    t_val = ttest_row['t_stat']
    p_bonf = ttest_row['p_bonf']
    sig = ttest_row['significant']
    sig_marker = ' *' if sig else ''
    mean_diff = ttest_row['mean_diff']

    ax.set_title(
        f'{coord_label}:  diff = {mean_diff:.4f},  t = {t_val:.2f},  '
        f'p(Bonf) = {p_bonf:.4f}{sig_marker}',
        fontsize=9, fontweight='bold' if sig else 'normal',
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


def generate_paired_report(
    subjects, sessions, task, contrast_id, output_dir,
    mnum='value_parametric',
    model_variant='rt_in_duration',
    space='MNI152NLin2009cAsym_res-2',
    threshold=3.0,
    cluster_threshold=10,
    alpha=0.05,
    roi_coords=None,
):
    """
    Generate an HTML report comparing two sessions.

    Requires that:
      - Level 1 GLMs have been run for both sessions
      - run_group_paired_ttest() has been run to produce the difference tmap

    The report includes:
      - Summary of both sessions
      - Glass brain and interactive viewer of the difference tmap
      - ROI paired bar plots with Bonferroni-corrected paired t-tests
      - Whole-brain cluster table with atlas labels

    Parameters
    ----------
    subjects : list of str
    sessions : tuple of (session_a, session_b)
        e.g., ('01', '03') for ses-03 minus ses-01
    task, contrast_id : str
    output_dir : str
    mnum, model_variant, space : str
    threshold : float
    cluster_threshold : int
    alpha : float
    roi_coords : dict or None

    Saves to:
        {output_dir}/{mnum}/{model_variant}/group/ses-{session_b}/
            {prefix}_{contrast_id}_ses{b}_minus_ses{a}_paired_report.html

    Returns
    -------
    report_path : str
    """
    import nibabel as nib

    if roi_coords is None:
        roi_coords = ROI_COORDS

    ses_a, ses_b = sessions
    ses_a_label = f'Week {int(ses_a)}  (ses-{ses_a})'
    ses_b_label = f'Week {int(ses_b)}  (ses-{ses_b})'

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

    # Also load individual session tmaps if available (for side-by-side ROI views)
    from level2_helpers import get_group_tmap_path
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

    # Subject count
    n_subjects = 0
    for subnum in subjects:
        from level2_helpers import get_contrast_path
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

    # -- ROI paired analysis --
    print(f"  Extracting paired ROI betas...")
    paired_betas = extract_roi_betas_paired(
        subjects, sessions, task, contrast_id, output_dir,
        mnum=mnum, model_variant=model_variant, space=space,
        roi_coords=roi_coords,
    )
    ttest_results = roi_paired_ttest_table(paired_betas, alpha=alpha)
    n_tests = len(ttest_results)
    n_sig = int(ttest_results['significant'].sum())

    # Extract voxelwise variance for per-subject error bars
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

    # -- Build ROI sections --
    roi_sections_html = ''
    for roi_name, roi_info in roi_coords.items():
        coord_list = roi_info['coords']
        description = roi_info['description']

        coord_panels_html = ''
        for coords in coord_list:
            side = 'L' if coords[0] < 0 else 'R' if coords[0] > 0 else 'M'
            coord_label = f'{roi_name}_{side}'
            coord_str = f'({coords[0]}, {coords[1]}, {coords[2]})'

            row_mask = ttest_results['coord_label'] == coord_label
            if not row_mask.any():
                continue
            ttest_row = ttest_results[row_mask].iloc[0]

            # Paired bar plot
            fig_bar = _plot_paired_roi_betas(
                paired_betas, coord_label, ttest_row,
                ses_a_label, ses_b_label,
                var_df_a=var_df_a, var_df_b=var_df_b,
            )
            img_bar = fig_to_base64(fig_bar)

            # Ortho view of difference tmap
            fig_diff = plot_roi_view(
                diff_tmap, coords,
                title=f'Difference (ses-{ses_b} - ses-{ses_a})',
                threshold=threshold,
            )
            img_diff = fig_to_base64(fig_diff)

            # Session-specific ortho views if available
            session_views_html = ''
            if has_session_tmaps:
                fig_a = plot_roi_view(
                    tmap_a, coords,
                    title=f'{ses_a_label}',
                    threshold=threshold,
                )
                img_a = fig_to_base64(fig_a)

                fig_b = plot_roi_view(
                    tmap_b, coords,
                    title=f'{ses_b_label}',
                    threshold=threshold,
                )
                img_b = fig_to_base64(fig_b)

                session_views_html = f"""
            <div class="triple-grid">
                <div class="grid-cell">
                    <h4>{ses_a_label}</h4>
                    <img src="data:image/png;base64,{img_a}" />
                </div>
                <div class="grid-cell">
                    <h4>{ses_b_label}</h4>
                    <img src="data:image/png;base64,{img_b}" />
                </div>
                <div class="grid-cell">
                    <h4>Difference</h4>
                    <img src="data:image/png;base64,{img_diff}" />
                </div>
            </div>
"""
            else:
                session_views_html = f"""
            <div class="side-by-side">
                <div class="grid-cell">
                    <img src="data:image/png;base64,{img_bar}" />
                </div>
                <div class="grid-cell">
                    <img src="data:image/png;base64,{img_diff}" />
                </div>
            </div>
"""

            sig = ttest_row['significant']
            sig_class = 'sig-yes' if sig else 'sig-no'
            sig_text = 'Significant' if sig else 'Not significant'

            if has_session_tmaps:
                coord_panels_html += f"""
        <div class="coord-panel">
            <div class="coord-header">
                <strong>{'Left' if coords[0] < 0 else 'Right' if coords[0] > 0 else 'Midline'} {coord_str}</strong>
                <span class="{sig_class}">{sig_text} (Bonferroni)</span>
            </div>
            <div class="bar-row">
                <img src="data:image/png;base64,{img_bar}" />
            </div>
            {session_views_html}
        </div>
"""
            else:
                coord_panels_html += f"""
        <div class="coord-panel">
            <div class="coord-header">
                <strong>{'Left' if coords[0] < 0 else 'Right' if coords[0] > 0 else 'Midline'} {coord_str}</strong>
                <span class="{sig_class}">{sig_text} (Bonferroni)</span>
            </div>
            {session_views_html}
        </div>
"""

        roi_sections_html += f"""
<div class="section">
    <h2>{roi_name}</h2>
    <p class="roi-desc">{description} | Sphere radius: {roi_info.get('radius', 10)} mm</p>
    {coord_panels_html}
</div>
"""

    # -- T-test summary table --
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

    # -- Cluster table on difference map --
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
    except Exception:
        cluster_html = '<p>No suprathreshold clusters found.</p>'
        n_clusters = 0

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
    h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 6px; margin-top: 30px; }}
    h3 {{ color: #555; margin-bottom: 8px; }}
    h4 {{ color: #666; margin: 4px 0; font-size: 12px; }}
    .section {{ background: #fff; border-radius: 8px; padding: 20px 30px;
                margin-bottom: 30px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
    .meta {{ color: #888; font-size: 13px; }}
    .roi-desc {{ color: #666; font-size: 13px; font-style: italic; margin-bottom: 12px; }}
    .info-grid {{ display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 16px; }}
    .info-card {{ flex: 1; min-width: 180px; background: #f0f4f8;
                  border-radius: 6px; padding: 12px 16px; }}
    .info-card h3 {{ margin-top: 0; }}
    .plot-container {{ text-align: center; margin-bottom: 16px; }}
    .plot-container img {{ max-width: 100%; height: auto;
                           border: 1px solid #e0e0e0; border-radius: 4px; }}
    .side-by-side {{
        display: grid; grid-template-columns: 1fr 1.8fr;
        gap: 8px; margin-bottom: 16px;
    }}
    .triple-grid {{
        display: grid; grid-template-columns: 1fr 1fr 1fr;
        gap: 8px; margin-bottom: 12px;
    }}
    .grid-cell {{
        text-align: center; border: 1px solid #e0e0e0;
        border-radius: 6px; padding: 6px; background: #fafafa;
    }}
    .grid-cell img {{ max-width: 100%; height: auto; border-radius: 4px; }}
    .bar-row {{
        text-align: center; margin-bottom: 8px;
    }}
    .bar-row img {{ max-width: 550px; height: auto; }}
    .coord-panel {{ margin-bottom: 24px; }}
    .coord-header {{
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 6px; font-size: 13px;
    }}
    .sig-yes {{ color: #155724; background: #d4edda; border-radius: 4px; padding: 2px 8px; font-weight: bold; }}
    .sig-no {{ color: #856404; background: #fff3cd; border-radius: 4px; padding: 2px 8px; }}
    table {{ border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 6px 10px; border: 1px solid #ddd; text-align: left; }}
    th {{ background: #f0f4f8; }}
    .ttest-table {{ width: 100%; }}
    .cluster-table {{ width: auto; }}
    .method-note {{ color: #555; font-size: 12px; background: #f0f4f8;
                    border-radius: 6px; padding: 12px 16px; margin-bottom: 16px;
                    line-height: 1.6; }}
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
    difference maps tests whether the contrast increased from {ses_a_label} to
    {ses_b_label}. For the ROI analysis, mean betas were extracted within
    spherical masks and paired t-tests were performed with Bonferroni correction
    across {n_tests} ROI coordinates (corrected alpha = {alpha}/{n_tests} = {alpha/n_tests:.4f}).
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
            <p>{n_sig} / {n_tests} significant after Bonferroni (alpha = {alpha})</p>
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

{roi_sections_html}

<div class="section">
    <h2>Whole-Brain Cluster Table: Difference (t > {threshold}, uncorrected, min {cluster_threshold} voxels)</h2>
    {cluster_html}
</div>

</body>
</html>"""

    report_fn = (f'{prefix}_{contrast_id}_ses{ses_b}_minus_ses{ses_a}'
                 f'_paired_report.html')
    report_path = os.path.join(group_dir, report_fn)
    with open(report_path, 'w') as f:
        f.write(html)

    print(f"  Paired report saved: {report_path}")
    return report_path