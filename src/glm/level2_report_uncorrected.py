"""
level2_report_uncorrected.py
Generate an uncorrected group-level HTML report with interactive viewer
and per-cluster subject-level bar plots.
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
    get_group_tmap_path, get_group_dir, get_group_map_prefix,
    collect_contrast_maps, add_atlas_labels_to_cluster_table,
    extract_roi_betas_with_variance,
    fig_to_base64, plot_group_glass_brain, plot_roi_view,
)


def _plot_cluster_betas(var_df, coord_label):
    """
    Create a bar plot of individual subject betas for one cluster coordinate
    with per-subject error bars (SE = SD/sqrt(n_voxels)).
    """
    sub_data = var_df[
        var_df['coord_label'] == coord_label
    ].sort_values('subject')

    subjects = sub_data['subject'].values
    betas = sub_data['mean_beta'].values
    se_vals = sub_data['se_beta'].values
    n = len(subjects)

    mean = float(np.mean(betas))
    se_group = float(np.std(betas, ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    fig, ax = plt.subplots(figsize=(5, 3.5))

    colors = ['#5b9bd5' if b >= 0 else '#ed7d31' for b in betas]
    ax.bar(range(n), betas, color=colors, alpha=0.8,
           edgecolor='white', linewidth=0.5)
    ax.errorbar(range(n), betas, yerr=se_vals, fmt='none',
                ecolor='#444', elinewidth=1.2, capsize=3, capthick=1)

    ax.axhline(0, color='#333', linewidth=0.8)
    ax.axhline(mean, color='#c0392b', linewidth=2, linestyle='-', alpha=0.8)
    ax.axhspan(mean - se_group, mean + se_group, color='#c0392b', alpha=0.12)

    ax.set_xticks(range(n))
    ax.set_xticklabels([f'sub-{s}' for s in subjects], fontsize=8, rotation=45)
    ax.set_ylabel('Mean beta (effect size)', fontsize=9)

    from scipy import stats
    if n >= 2 and np.std(betas, ddof=1) > 0:
        t_stat, p_val = stats.ttest_1samp(betas, 0)
        ax.set_title(
            f'mean = {mean:.3f}, t = {t_stat:.2f}, p = {p_val:.4f}',
            fontsize=9,
        )
    else:
        ax.set_title(f'mean = {mean:.3f}', fontsize=9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


def generate_group_report(
    subjects, session, task, contrast_id, output_dir,
    mnum='value_parametric',
    model_variant='rt_in_duration',
    space='MNI152NLin2009cAsym_res-2',
    threshold=3.0,
    cluster_threshold=10,
    roi_coords=None,
    max_cluster_panels=20,
):
    """
    Generate an HTML report from previously saved group maps.

    Includes glass brain, interactive viewer, ROI ortho views,
    cluster table with atlas labels, and per-cluster bar plots showing
    individual subject betas extracted from 10mm spheres at each peak.

    Parameters
    ----------
    max_cluster_panels : int
        Maximum number of cluster peaks to show bar plots for (default 20)
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

    print(f"  Generating report: task-{task} {contrast_id} "
          f"[{model_variant}] (n={n_subjects})")

    # -- Glass brain --
    title_base = (f'Group (n={n_subjects}): task-{task} {contrast_id}\n'
                  f'{model_variant} | t > {threshold} (uncorrected)')

    fig_glass = plot_group_glass_brain(
        group_tmap, title=title_base, threshold=threshold
    )
    img_glass = fig_to_base64(fig_glass)

    # -- Interactive viewer --
    interactive_view = view_img(
        group_tmap, threshold=threshold,
        title=f'Group: task-{task} {contrast_id} [{model_variant}]',
    )
    interactive_html = interactive_view.get_iframe(width=900, height=500)

    # -- ROI views --
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
            img_roi = fig_to_base64(fig_roi)

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

    # -- Cluster table + per-cluster bar plots --
    cluster_panels_html = ''
    cluster_html = '<p>No suprathreshold clusters found.</p>'
    n_clusters = 0

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

        # Build ad-hoc ROI coords from cluster peaks for extraction
        n_panels = min(max_cluster_panels, len(cluster_table))
        if n_panels > 0:
            cluster_roi_coords = {}
            for i in range(n_panels):
                row = cluster_table.iloc[i]
                peak_coords = (float(row['X']), float(row['Y']), float(row['Z']))
                cluster_id = str(row['Cluster ID'])

                # Build a label from atlas columns if available
                atlas_labels = []
                for col in ['Harvard-Oxford', 'AAL']:
                    if col in row.index:
                        val = str(row[col])
                        if val and val != 'Background' and val != 'nan':
                            atlas_labels.append(val)
                region_label = atlas_labels[0] if atlas_labels else 'Unlabeled'

                label = f'cluster_{cluster_id}'
                cluster_roi_coords[label] = {
                    'coords': [peak_coords],
                    'description': region_label,
                    'radius': 10,
                }

            # Extract betas for all cluster peaks
            print(f"  Extracting betas for {n_panels} cluster peaks...")
            cluster_var_df = extract_roi_betas_with_variance(
                subjects, session, task, contrast_id, output_dir,
                mnum=mnum, model_variant=model_variant, space=space,
                roi_coords=cluster_roi_coords,
            )

            # Build panels
            for i in range(n_panels):
                row = cluster_table.iloc[i]
                peak_coords = (float(row['X']), float(row['Y']), float(row['Z']))
                peak_stat = float(row['Peak Stat'])
                cluster_id = str(row['Cluster ID'])
                cluster_size = row['Cluster Size (mm3)']
                coord_str = f'({peak_coords[0]:.0f}, {peak_coords[1]:.0f}, {peak_coords[2]:.0f})'
                label = f'cluster_{cluster_id}'

                # Get region name
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

                # Bar plot
                fig_bar = _plot_cluster_betas(cluster_var_df, coord_label)
                img_bar = fig_to_base64(fig_bar)

                # Ortho view
                fig_ortho = plot_roi_view(
                    group_tmap, peak_coords,
                    title=f'Cluster {cluster_id}: {coord_str}, '
                          f't = {peak_stat:.2f}, {cluster_size} mm3',
                    threshold=threshold,
                )
                img_ortho = fig_to_base64(fig_ortho)

                cluster_panels_html += f"""
<div class="section">
    <h2>Cluster {cluster_id}: {region_label}</h2>
    <p class="cluster-meta">Peak: {coord_str} | t = {peak_stat:.2f} | {cluster_size} mm3 | 10mm sphere extraction</p>
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

    except Exception as e:
        print(f"  WARNING: Cluster extraction failed: {e}")
        import traceback
        traceback.print_exc()

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
    .cluster-meta {{ color: #666; font-size: 12px; font-style: italic; margin-bottom: 10px; }}
    .info-grid {{ display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 16px; }}
    .info-card {{ flex: 1; min-width: 200px; background: #f0f4f8;
                  border-radius: 6px; padding: 12px 16px; }}
    .info-card h3 {{ margin-top: 0; }}
    .warning {{ color: #856404; background: #fff3cd; border: 1px solid #ffc107;
                border-radius: 6px; padding: 10px 16px; margin-bottom: 16px; }}
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

<div class="section">
    <h2>Interactive Viewer</h2>
    <div class="plot-container">
        {interactive_html}
    </div>
</div>

{roi_sections_html}

<div class="section">
    <h2>Cluster Table (t > {threshold}, uncorrected, min {cluster_threshold} voxels)</h2>
    {cluster_html}
</div>

{cluster_panels_html}

</body>
</html>"""

    report_fn = f'{prefix}_{contrast_id}_uncorrected_report.html'
    report_path = os.path.join(group_dir, report_fn)
    with open(report_path, 'w') as f:
        f.write(html)

    print(f"  Report saved: {report_path}")
    return report_path