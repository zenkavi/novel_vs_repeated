"""
level2_report_comparison.py
Generate an HTML report comparing a contrast across tasks and model variants.
"""

import os
import numpy as np

from level2_helpers import (
    ROI_COORDS,
    get_group_tmap_path, get_group_dir,
    fig_to_base64, plot_group_glass_brain, plot_roi_view,
)


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

    Saves to:
        {report_dir}/comparison_{contrast_id}_..._uncorrected_report.html
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
                    img_b64 = fig_to_base64(fig)

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
            img_glass = fig_to_base64(fig_glass)
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