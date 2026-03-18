"""
level2_report_corrected.py
Generate a TFCE-corrected group-level HTML report.
"""

import os
import numpy as np

from level2_helpers import (
    ROI_COORDS,
    get_group_tmap_path, get_group_tfce_logp_path,
    get_group_dir, get_group_map_prefix,
    collect_contrast_maps,
    fig_to_base64, plot_group_glass_brain, plot_roi_view,
)


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

    Reads the TFCE logp map and the uncorrected tmap, thresholds by
    -log10(alpha), and shows glass brain and ROI views of corrected results.

    Saves report as:
        {prefix}_{contrast_id}_tfce_corrected_report.html
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

    logp_thresh = -np.log10(alpha)
    thresholded_tmap = math_img(
        f'tmap * (logp > {logp_thresh})',
        tmap=group_tmap, logp=logp_img,
    )

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

    # -- Glass brains --
    title_base = (f'Group (n={n_subjects}): task-{task} {contrast_id}\n'
                  f'{model_variant} | TFCE corrected, alpha={alpha}')

    fig_glass = plot_group_glass_brain(
        thresholded_tmap, title=title_base, threshold=0.01
    )
    img_glass = fig_to_base64(fig_glass)

    fig_logp = plot_group_glass_brain(
        logp_img,
        title=f'-log10(p) TFCE map (threshold at {logp_thresh:.2f})',
        threshold=logp_thresh,
    )
    img_logp = fig_to_base64(fig_logp)

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
                thresholded_tmap, coords,
                title=f'{roi_name} {side} {coord_str} (TFCE corrected)',
                threshold=0.01,
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