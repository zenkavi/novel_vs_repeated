"""
level2_report_uncorrected.py
Generate an uncorrected group-level HTML report with interactive viewer.
"""

import os
import numpy as np
from nilearn.plotting import view_img
from nilearn.reporting import get_clusters_table

from level2_helpers import (
    ROI_COORDS,
    get_group_tmap_path, get_group_dir, get_group_map_prefix,
    collect_contrast_maps, add_atlas_labels_to_cluster_table,
    fig_to_base64, plot_group_glass_brain, plot_roi_view,
)


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

    Includes glass brain, interactive viewer, ROI ortho views,
    cluster table with atlas labels, and top 5 cluster peak views.

    Reads the group tmap from:
        {output_dir}/{mnum}/{model_variant}/group/ses-{session}/

    Saves report to the same directory as:
        {prefix}_{contrast_id}_uncorrected_report.html
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

    # -- Cluster table + top 5 peak views --
    cluster_plots_html = ''
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
            img_cluster = fig_to_base64(fig_cluster)

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