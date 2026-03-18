"""
level2_report_roi.py
Generate an ROI-focused HTML report with:
  - Extracted mean betas per subject per ROI sphere
  - One-sample t-tests with Bonferroni correction
  - Bar plots showing individual subject betas
  - Ortho views centered on each ROI
  - Cluster summary table with atlas labels
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nilearn.reporting import get_clusters_table

from level2_helpers import (
    ROI_COORDS,
    get_group_tmap_path, get_group_dir, get_group_map_prefix,
    collect_contrast_maps, add_atlas_labels_to_cluster_table,
    extract_roi_betas, roi_ttest_table,
    extract_roi_betas_with_variance,
    fig_to_base64, plot_roi_view,
)


def _plot_roi_betas(roi_betas_df, coord_label, ttest_row, var_df=None):
    """
    Create a bar plot of individual subject betas for one ROI coordinate
    with per-subject error bars (SE = SD/sqrt(n_voxels) across voxels
    within the ROI sphere) and group mean overlay.
    """
    sub_data = roi_betas_df[
        roi_betas_df['coord_label'] == coord_label
    ].sort_values('subject')

    subjects = sub_data['subject'].values
    betas = sub_data['mean_beta'].values
    mean = ttest_row['mean_beta']
    se_group = ttest_row['se']

    # Get per-subject SE from variance dataframe if available
    n = len(subjects)
    se_per_sub = np.zeros(n)
    if var_df is not None:
        for i, subnum in enumerate(subjects):
            row = var_df[
                (var_df['coord_label'] == coord_label) &
                (var_df['subject'] == subnum)
            ]
            if len(row) > 0:
                se_per_sub[i] = row.iloc[0]['se_beta']

    fig, ax = plt.subplots(figsize=(5, 3.5))

    colors = ['#5b9bd5' if b >= 0 else '#ed7d31' for b in betas]
    bars = ax.bar(range(n), betas, color=colors, alpha=0.8,
                  edgecolor='white', linewidth=0.5)

    if var_df is not None:
        ax.errorbar(range(n), betas, yerr=se_per_sub, fmt='none',
                    ecolor='#444', elinewidth=1.2, capsize=3, capthick=1)

    ax.axhline(0, color='#333', linewidth=0.8)
    ax.axhline(mean, color='#c0392b', linewidth=2, linestyle='-', alpha=0.8)
    ax.axhspan(mean - se_group, mean + se_group, color='#c0392b', alpha=0.12)

    ax.set_xticks(range(n))
    ax.set_xticklabels([f'sub-{s}' for s in subjects], fontsize=8, rotation=45)
    ax.set_ylabel('Mean beta (effect size)', fontsize=9)

    t_val = ttest_row['t_stat']
    p_bonf = ttest_row['p_bonf']
    sig = ttest_row['significant']
    sig_marker = ' *' if sig else ''

    ax.set_title(
        f'{coord_label}:  t = {t_val:.2f},  p(Bonf) = {p_bonf:.4f}{sig_marker}',
        fontsize=10, fontweight='bold' if sig else 'normal',
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


def generate_roi_report(
    subjects, session, task, contrast_id, output_dir,
    mnum='value_parametric',
    model_variant='rt_in_duration',
    space='MNI152NLin2009cAsym_res-2',
    threshold=3.0,
    cluster_threshold=10,
    alpha=0.05,
    roi_coords=None,
):
    """
    Generate an ROI-focused HTML report with:
      - Spherical ROI extraction of mean betas per subject
      - One-sample t-tests with Bonferroni correction across ROIs
      - Bar plots of individual subject betas
      - Ortho views of the uncorrected group tmap centered on each ROI
      - Cluster summary table with atlas labels

    Saves to:
        {output_dir}/{mnum}/{model_variant}/group/ses-{session}/
            {prefix}_{contrast_id}_roi_report.html
    """
    import nibabel as nib

    if roi_coords is None:
        roi_coords = ROI_COORDS

    # Load uncorrected tmap for visualization
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

    # Extract ROI betas
    print(f"  Extracting ROI betas...")
    roi_betas = extract_roi_betas(
        subjects, session, task, contrast_id, output_dir,
        mnum=mnum, model_variant=model_variant, space=space,
        roi_coords=roi_coords,
    )

    # Extract voxelwise variance for per-subject error bars
    print(f"  Extracting voxelwise variance for error bars...")
    var_df = extract_roi_betas_with_variance(
        subjects, session, task, contrast_id, output_dir,
        mnum=mnum, model_variant=model_variant, space=space,
        roi_coords=roi_coords,
    )

    # Run t-tests with Bonferroni correction
    ttest_results = roi_ttest_table(roi_betas, alpha=alpha)
    n_tests = len(ttest_results)
    n_sig = int(ttest_results['significant'].sum())

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

    print(f"  Generating ROI report: task-{task} {contrast_id} "
          f"[{model_variant}] (n={n_subjects}, "
          f"{n_sig}/{n_tests} ROIs significant after Bonferroni)")

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

            # Get t-test row for this coordinate
            row_mask = ttest_results['coord_label'] == coord_label
            if not row_mask.any():
                continue
            ttest_row = ttest_results[row_mask].iloc[0]

            # Bar plot
            fig_bar = _plot_roi_betas(roi_betas, coord_label, ttest_row, var_df=var_df)
            img_bar = fig_to_base64(fig_bar)

            # Ortho view
            fig_ortho = plot_roi_view(
                group_tmap, coords,
                title=f'{roi_name} {coord_str} (uncorrected, t > {threshold})',
                threshold=threshold,
            )
            img_ortho = fig_to_base64(fig_ortho)

            # Significance styling
            sig = ttest_row['significant']
            sig_class = 'sig-yes' if sig else 'sig-no'
            sig_text = 'Significant' if sig else 'Not significant'

            coord_panels_html += f"""
        <div class="coord-panel">
            <div class="coord-header">
                <strong>{'Left' if coords[0] < 0 else 'Right' if coords[0] > 0 else 'Midline'} {coord_str}</strong>
                <span class="{sig_class}">{sig_text} (Bonferroni)</span>
            </div>
            <div class="side-by-side">
                <div class="grid-cell">
                    <img src="data:image/png;base64,{img_bar}" />
                </div>
                <div class="grid-cell">
                    <img src="data:image/png;base64,{img_ortho}" />
                </div>
            </div>
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
        'n_subjects', 'mean_beta', 'se', 't_stat', 'p_uncorr', 'p_bonf',
        'significant',
    ]].copy()
    ttest_display.columns = [
        'ROI', 'Label', 'X', 'Y', 'Z', 'Radius (mm)',
        'N', 'Mean beta', 'SE', 't', 'p (uncorr)', 'p (Bonf)',
        'Significant',
    ]
    ttest_html = ttest_display.to_html(
        classes='ttest-table', index=False, float_format='%.4f'
    )

    # -- Cluster table with atlas labels --
    try:
        cluster_table = get_clusters_table(
            group_tmap, stat_threshold=threshold,
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
<title>ROI Report: {contrast_id} | task-{task} | {model_variant}</title>
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
    .info-card {{ flex: 1; min-width: 200px; background: #f0f4f8;
                  border-radius: 6px; padding: 12px 16px; }}
    .info-card h3 {{ margin-top: 0; }}
    .side-by-side {{
        display: grid;
        grid-template-columns: 1fr 1.8fr;
        gap: 8px;
        margin-bottom: 16px;
    }}
    .grid-cell {{
        text-align: center;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 6px;
        background: #fafafa;
    }}
    .grid-cell img {{
        max-width: 100%; height: auto; border-radius: 4px;
    }}
    .coord-panel {{
        margin-bottom: 20px;
    }}
    .coord-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
        font-size: 13px;
    }}
    .sig-yes {{
        color: #155724; background: #d4edda;
        border-radius: 4px; padding: 2px 8px; font-weight: bold;
    }}
    .sig-no {{
        color: #856404; background: #fff3cd;
        border-radius: 4px; padding: 2px 8px;
    }}
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

<h1>ROI-Focused Analysis Report</h1>
<p class="meta">
    Contrast: <strong>{contrast_id}</strong> | Task: {task} |
    Model: {mnum} | Variant: {model_variant}<br>
    Session: ses-{session} | Space: {space}<br>
    ROI coordinates from Lakhani et al. (2026) meta-analysis Table 2
</p>

<div class="method-note">
    <strong>Method:</strong> For each ROI, mean effect size (beta) was extracted
    within a spherical mask centered on the coordinate listed below.
    One-sample t-tests were performed across {n_subjects} subjects, testing
    whether the mean beta differs from zero. Bonferroni correction was applied
    across {n_tests} ROI coordinates (corrected alpha = {alpha}/{n_tests} = {alpha/n_tests:.4f}).
</div>

<div class="section">
    <h2>Summary</h2>
    <div class="info-grid">
        <div class="info-card">
            <h3>Subjects</h3>
            <p>n = {n_subjects}</p>
            <p style="font-size: 12px;">{', '.join(subjects_used)}</p>
        </div>
        <div class="info-card">
            <h3>ROI Tests</h3>
            <p>{n_sig} / {n_tests} significant after Bonferroni (alpha = {alpha})</p>
        </div>
        <div class="info-card">
            <h3>Whole-brain</h3>
            <p>{n_clusters} clusters (t > {threshold}, uncorrected)</p>
        </div>
    </div>
</div>

<div class="section">
    <h2>ROI Statistical Summary</h2>
    {ttest_html}
</div>

{roi_sections_html}

<div class="section">
    <h2>Whole-Brain Cluster Table (t > {threshold}, uncorrected, min {cluster_threshold} voxels)</h2>
    {cluster_html}
</div>

</body>
</html>"""

    report_fn = f'{prefix}_{contrast_id}_roi_report.html'
    report_path = os.path.join(group_dir, report_fn)
    with open(report_path, 'w') as f:
        f.write(html)

    print(f"  ROI report saved: {report_path}")
    return report_path