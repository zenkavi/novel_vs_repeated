"""
level2_report_roi.py
Generate an ROI-focused HTML report showing uncorrected and TFCE-corrected
maps side by side for each ROI coordinate, plus a cluster summary table.
"""

import os
import numpy as np
from nilearn.reporting import get_clusters_table

from level2_helpers import (
    ROI_COORDS,
    get_group_tmap_path, get_group_tfce_logp_path,
    get_group_dir, get_group_map_prefix,
    collect_contrast_maps, add_atlas_labels_to_cluster_table,
    fig_to_base64, plot_roi_view,
)


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
    Generate an ROI-focused HTML report showing uncorrected and TFCE-corrected
    maps side by side for each ROI coordinate, plus a cluster summary table
    with atlas labels.

    Saves to:
        {output_dir}/{mnum}/{model_variant}/group/ses-{session}/
            {prefix}_{contrast_id}_roi_report.html
    """
    import nibabel as nib
    from nilearn.image import math_img

    if roi_coords is None:
        roi_coords = ROI_COORDS

    # Load uncorrected tmap
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

    # Try to load TFCE logp map
    logp_path = get_group_tfce_logp_path(
        session, task, contrast_id, output_dir,
        mnum=mnum, model_variant=model_variant, space=space,
    )
    has_tfce = os.path.exists(logp_path)
    if has_tfce:
        logp_img = nib.load(logp_path)
        logp_thresh = -np.log10(alpha)
        corrected_tmap = math_img(
            f'tmap * (logp > {logp_thresh})',
            tmap=group_tmap, logp=logp_img,
        )
        n_sig_voxels = int(np.sum(corrected_tmap.get_fdata() != 0))
    else:
        corrected_tmap = None
        n_sig_voxels = 0
        print(f"  NOTE: TFCE logp map not found. "
              f"Report will show uncorrected maps only.")

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
          f"[{model_variant}] (n={n_subjects})")

    # -- ROI sections --
    roi_sections_html = ''
    for roi_name, roi_info in roi_coords.items():
        coord_list = roi_info['coords']
        description = roi_info['description']

        coord_rows_html = ''
        for coords in coord_list:
            side = 'Left' if coords[0] < 0 else 'Right' if coords[0] > 0 else 'Midline'
            coord_str = f'({coords[0]}, {coords[1]}, {coords[2]})'

            # Uncorrected view
            fig_uncorr = plot_roi_view(
                group_tmap, coords,
                title=f'Uncorrected (t > {threshold})',
                threshold=threshold,
            )
            img_uncorr = fig_to_base64(fig_uncorr)

            # Corrected view
            if has_tfce:
                fig_corr = plot_roi_view(
                    corrected_tmap, coords,
                    title=f'TFCE corrected (alpha={alpha})',
                    threshold=0.01,
                )
                img_corr = fig_to_base64(fig_corr)

                corr_cell = f"""
                <div class="grid-cell">
                    <h4>TFCE corrected (alpha={alpha})</h4>
                    <img src="data:image/png;base64,{img_corr}" />
                </div>"""
            else:
                corr_cell = """
                <div class="grid-cell">
                    <h4>TFCE corrected</h4>
                    <p>Not available</p>
                </div>"""

            coord_rows_html += f"""
        <div class="coord-label"><strong>{side} {coord_str}</strong></div>
        <div class="side-by-side">
            <div class="grid-cell">
                <h4>Uncorrected (t > {threshold})</h4>
                <img src="data:image/png;base64,{img_uncorr}" />
            </div>
            {corr_cell}
        </div>
"""

        roi_sections_html += f"""
<div class="section">
    <h2>{roi_name}</h2>
    <p class="roi-desc">{description}</p>
    {coord_rows_html}
</div>
"""

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
    tfce_status = (
        f'{n_sig_voxels} significant voxels (TFCE, alpha={alpha})'
        if has_tfce else 'TFCE not run'
    )

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
        grid-template-columns: 1fr 1fr;
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
    .coord-label {{
        margin: 10px 0 4px 0; font-size: 13px;
    }}
    table {{ border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 6px 10px; border: 1px solid #ddd; text-align: left; }}
    th {{ background: #f0f4f8; }}
    .cluster-table {{ width: auto; }}
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

<div class="section">
    <h2>Summary</h2>
    <div class="info-grid">
        <div class="info-card">
            <h3>Subjects</h3>
            <p>n = {n_subjects}</p>
            <p style="font-size: 12px;">{', '.join(subjects_used)}</p>
        </div>
        <div class="info-card">
            <h3>Uncorrected</h3>
            <p>{n_clusters} clusters (t > {threshold})</p>
        </div>
        <div class="info-card">
            <h3>TFCE Corrected</h3>
            <p>{tfce_status}</p>
        </div>
    </div>
</div>

{roi_sections_html}

<div class="section">
    <h2>Cluster Table (t > {threshold}, uncorrected, min {cluster_threshold} voxels)</h2>
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