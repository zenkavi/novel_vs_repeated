"""
level2_helpers.py
Core helper functions for group-level (second-level) analyses.

Contains:
  - ROI coordinate definitions (from Lakhani et al. 2026 meta-analysis)
  - Atlas lookup for anatomical labeling
  - Path helpers for locating saved maps
  - Model fitting (parametric and non-parametric)
  - Plotting utilities shared across report modules

Report generation functions live in separate modules:
  - level2_report_uncorrected.py
  - level2_report_corrected.py
  - level2_report_comparison.py
  - level2_report_roi.py
"""

import base64
import io
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_glass_brain, plot_stat_map, view_img
from nilearn.image import load_img
from nilearn.reporting import get_clusters_table


# =============================================================================
# ROI definitions (MNI coordinates from Lakhani et al. 2026 meta-analysis
# Table 2 cluster centers of mass, plus hippocampus from literature)
# =============================================================================

ROI_COORDS = {
    'vmPFC': {
        'coords': [(0, 46, -2)],
        'description': 'Ventromedial PFC',
        'radius': 12,
    },
    'dmPFC': {
        'coords': [(0, 22, 43)],
        'description': 'Dorsomedial PFC',
        'radius': 12,
    },
    'ACC': {
        'coords': [(0, 44, 12)],
        'description': 'Anterior cingulate cortex',
        'radius': 10,
    },
    'PCC': {
        'coords': [(-4, -40, 34), (5, -33, 34)],
        'description': 'Posterior cingulate cortex',
        'radius': 10,
    },
    'Striatum': {
        'coords': [(-2, 7, -3), (10, 9, -5)],
        'description': 'Striatum',
        'radius': 10,
    },
    'Insula': {
        'coords': [(-33, 21, 0), (36, 20, -2)],
        'description': 'Insular cortex',
        'radius': 10,
    },
    'MFG': {
        'coords': [(-46, 11, 31), (51, 15, 31)],
        'description': 'Middle frontal gyrus',
        'radius': 10,
    },
    'Frontal Pole': {
        'coords': [(45, 40, 15)],
        'description': 'Right frontal pole',
        'radius': 10,
    },
    'lOFC': {
        'coords': [(-32, 42, -18), (32, 42, -18)],
        'description': 'Lateral orbitofrontal cortex',
        'radius': 10,
    },
    'MTG': {
        'coords': [(-62, -36, -3)],
        'description': 'Left middle temporal gyrus',
        'radius': 10,
    },
    'Hippocampus': {
        'coords': [(-26, -18, -18), (26, -18, -18)],
        'description': 'Hippocampus',
        'radius': 8,
    },
}


# =============================================================================
# Atlas lookup for anatomical labeling of cluster peaks
# =============================================================================

_ATLASES = None


def _load_atlases():
    """Lazy-load atlas data. Called once on first use, then cached."""
    global _ATLASES
    if _ATLASES is not None:
        return _ATLASES

    from nilearn import datasets, image

    _ATLASES = {}

    try:
        ho = datasets.fetch_atlas_harvard_oxford(
            'cort-maxprob-thr25-2mm', verbose=0
        )
        ho_img = image.load_img(ho.maps)
        ho_labels = ho.labels

        def ho_lookup(idx):
            if 0 <= idx < len(ho_labels):
                return ho_labels[idx]
            return 'Unknown'

        _ATLASES['Harvard-Oxford'] = (ho_img, ho_lookup)
    except Exception as e:
        print(f"  WARNING: Could not load Harvard-Oxford atlas: {e}")

    try:
        aal = datasets.fetch_atlas_aal(version='3v2', verbose=0)
        aal_img = image.load_img(aal.maps)
        aal_index_to_label = {
            int(idx): label
            for idx, label in zip(aal.indices, aal.labels)
        }

        def aal_lookup(idx):
            return aal_index_to_label.get(int(idx), 'Unknown')

        _ATLASES['AAL'] = (aal_img, aal_lookup)
    except Exception as e:
        print(f"  WARNING: Could not load AAL atlas: {e}")

    return _ATLASES


def coords_to_labels(mni_coords):
    """Look up anatomical labels for MNI coordinates across loaded atlases."""
    atlases = _load_atlases()
    results = {}

    for name, (img, lookup_fn) in atlases.items():
        try:
            inv_aff = np.linalg.inv(img.affine)
            vox = np.round(
                inv_aff @ np.array([*mni_coords, 1])
            )[:3].astype(int)

            data = img.get_fdata()
            if all(0 <= vox[i] < data.shape[i] for i in range(3)):
                idx = int(data[vox[0], vox[1], vox[2]])
                results[name] = lookup_fn(idx)
            else:
                results[name] = 'Out of bounds'
        except Exception:
            results[name] = 'Error'

    return results


def add_atlas_labels_to_cluster_table(cluster_table):
    """Add atlas label columns to a cluster table DataFrame."""
    atlases = _load_atlases()
    if not atlases:
        return cluster_table

    for atlas_name in atlases:
        cluster_table[atlas_name] = cluster_table.apply(
            lambda row: coords_to_labels(
                (float(row['X']), float(row['Y']), float(row['Z']))
            ).get(atlas_name, ''),
            axis=1,
        )

    return cluster_table


def is_cerebellar(cluster_row):
    """
    Check whether a cluster table row is in the cerebellum based on
    atlas labels. Returns True if any atlas column contains 'cerebellum'
    (case-insensitive).
    """
    for col in ['Harvard-Oxford', 'AAL']:
        if col in cluster_row.index:
            val = str(cluster_row[col]).lower()
            if 'cerebell' in val:
                return True
    return False


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
    missing : list of str
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
    """Fit a one-sample t-test (intercept-only second-level model)."""
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
    """Collect subject maps, fit group one-sample t-test, save maps."""
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
            f"Need at least 2 subjects, found {len(contrast_maps)}"
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
    """Run non-parametric inference with TFCE on a group contrast."""
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


# =============================================================================
# Plotting utilities (shared across report modules)
# =============================================================================

def fig_to_base64(fig):
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
    """Ortho stat map centered on one coordinate with crosshairs."""
    fig = plt.figure(figsize=(12, 4))
    display = plot_stat_map(
        stat_map, threshold=threshold,
        display_mode='ortho', cut_coords=coords,
        draw_cross=True, colorbar=True,
        title=title, figure=fig,
    )
    return fig


def plot_stacked_sessions(stat_maps, coords, labels, threshold=3.0):
    """
    Plot multiple stat maps stacked vertically, each showing sagittal
    and coronal views centered on the same coordinate.

    Parameters
    ----------
    stat_maps : list of Niimg-like
    coords : tuple of 3 floats
    labels : list of str, one per map
    threshold : float

    Returns
    -------
    fig : matplotlib Figure
    """
    n_maps = len(stat_maps)
    fig, axes = plt.subplots(n_maps, 2, figsize=(8, 2.5 * n_maps))
    if n_maps == 1:
        axes = axes.reshape(1, 2)

    for i, (smap, label) in enumerate(zip(stat_maps, labels)):
        for j, mode in enumerate(['x', 'y']):
            cut = coords[0] if mode == 'x' else coords[1]
            plot_stat_map(
                smap, threshold=threshold,
                display_mode=mode, cut_coords=[cut],
                draw_cross=True, colorbar=(j == 1),
                title=label if j == 0 else '',
                axes=axes[i, j],
                annotate=False,
            )

    fig.tight_layout(h_pad=0.5)
    return fig


# =============================================================================
# ROI extraction and statistical testing
# =============================================================================

def extract_roi_betas(
    subjects, session, task, contrast_id, output_dir,
    mnum='value_parametric',
    model_variant='rt_in_duration',
    space='MNI152NLin2009cAsym_res-2',
    roi_coords=None,
):
    """
    Extract mean effect_size (beta) within each ROI sphere for each subject.

    Uses NiftiSpheresMasker to extract the mean signal within a sphere
    of the specified radius around each ROI coordinate.

    Parameters
    ----------
    subjects : list of str
    session, task, contrast_id : str
    output_dir : str
    mnum, model_variant, space : str
    roi_coords : dict or None
        If None, uses default ROI_COORDS

    Returns
    -------
    results : pandas.DataFrame
        Columns: subject, roi_name, coord_label, x, y, z, radius, mean_beta
        One row per subject per ROI coordinate.
    """
    from nilearn.maskers import NiftiSpheresMasker

    if roi_coords is None:
        roi_coords = ROI_COORDS

    # Build a flat list of (roi_name, coord_label, coords, radius)
    roi_list = []
    for roi_name, roi_info in roi_coords.items():
        radius = roi_info.get('radius', 10)
        for coords in roi_info['coords']:
            side = 'L' if coords[0] < 0 else 'R' if coords[0] > 0 else 'M'
            label = f'{roi_name}_{side}'
            roi_list.append((roi_name, label, coords, radius))

    # Group ROIs by radius for efficient batched extraction
    from collections import defaultdict
    radius_groups = defaultdict(list)
    for i, (roi_name, label, coords, radius) in enumerate(roi_list):
        radius_groups[radius].append((i, roi_name, label, coords))

    # Pre-build maskers (one per unique radius)
    maskers = {}
    index_maps = {}
    for radius, group in radius_groups.items():
        seeds = [item[3] for item in group]
        maskers[radius] = NiftiSpheresMasker(
            seeds=seeds, radius=radius,
            standardize=False, allow_overlap=True,
        )
        index_maps[radius] = group

    rows = []
    for subnum in subjects:
        map_path = get_contrast_path(
            subnum, session, task, contrast_id, output_dir,
            mnum=mnum, model_variant=model_variant,
            space=space, map_type='effect_size',
        )
        if not os.path.exists(map_path):
            print(f"  WARNING: Missing map for sub-{subnum}, skipping")
            continue

        for radius, group in index_maps.items():
            signal = maskers[radius].fit_transform(map_path)
            values = np.squeeze(signal)
            if values.ndim == 0:
                values = np.array([float(values)])

            for j, (orig_idx, roi_name, label, coords) in enumerate(group):
                rows.append({
                    'subject': subnum,
                    'roi_name': roi_name,
                    'coord_label': label,
                    'x': coords[0],
                    'y': coords[1],
                    'z': coords[2],
                    'radius': radius,
                    'mean_beta': float(values[j]),
                })

    return pd.DataFrame(rows)


def _make_sphere_mask(coords, radius, ref_img):
    """
    Create a binary sphere mask image in the space of ref_img.

    Parameters
    ----------
    coords : tuple of 3 floats
        Center of sphere in MNI mm
    radius : float
        Radius in mm
    ref_img : nibabel image
        Reference image defining the voxel grid and affine

    Returns
    -------
    mask_img : nibabel.Nifti1Image
        Binary mask with 1 inside sphere, 0 outside
    """
    import nibabel as nib

    affine = ref_img.affine
    shape = ref_img.shape[:3]

    i, j, k = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
        indexing='ij',
    )
    vox_coords = np.stack([i.ravel(), j.ravel(), k.ravel(), np.ones(i.size)])
    mm_coords = (affine @ vox_coords)[:3].T
    center_mm = np.array(coords)
    dist = np.sqrt(np.sum((mm_coords - center_mm) ** 2, axis=1))
    mask_data = (dist <= radius).reshape(shape).astype(np.int8)

    return nib.Nifti1Image(mask_data, affine)


def extract_roi_betas_with_variance(
    subjects, session, task, contrast_id, output_dir,
    mnum='value_parametric',
    model_variant='rt_in_duration',
    space='MNI152NLin2009cAsym_res-2',
    roi_coords=None,
):
    """
    Extract mean beta, SD, and voxel count within each ROI sphere
    for each subject. Uses voxelwise extraction (not just the mean).

    Returns
    -------
    results : pandas.DataFrame
        Columns: subject, roi_name, coord_label, x, y, z, radius,
                 mean_beta, sd_beta, n_voxels, se_beta
    """
    from nilearn.maskers import NiftiMasker

    if roi_coords is None:
        roi_coords = ROI_COORDS

    # Build flat ROI list
    roi_list = []
    for roi_name, roi_info in roi_coords.items():
        radius = roi_info.get('radius', 10)
        for coords in roi_info['coords']:
            side = 'L' if coords[0] < 0 else 'R' if coords[0] > 0 else 'M'
            label = f'{roi_name}_{side}'
            roi_list.append((roi_name, label, coords, radius))

    # Build sphere masks using first available subject as reference
    ref_img = None
    for subnum in subjects:
        map_path = get_contrast_path(
            subnum, session, task, contrast_id, output_dir,
            mnum=mnum, model_variant=model_variant,
            space=space, map_type='effect_size',
        )
        if os.path.exists(map_path):
            ref_img = load_img(map_path)
            break

    if ref_img is None:
        raise FileNotFoundError("No subject maps found to use as reference.")

    # Pre-build sphere masks and maskers
    maskers = {}
    for roi_name, label, coords, radius in roi_list:
        sphere_mask = _make_sphere_mask(coords, radius, ref_img)
        maskers[label] = NiftiMasker(mask_img=sphere_mask, standardize=False)

    rows = []
    for subnum in subjects:
        map_path = get_contrast_path(
            subnum, session, task, contrast_id, output_dir,
            mnum=mnum, model_variant=model_variant,
            space=space, map_type='effect_size',
        )
        if not os.path.exists(map_path):
            print(f"  WARNING: Missing map for sub-{subnum}, skipping")
            continue

        for roi_name, label, coords, radius in roi_list:
            voxel_values = maskers[label].fit_transform(map_path)
            voxel_values = np.squeeze(voxel_values)

            n_vox = len(voxel_values)
            mean_val = float(np.mean(voxel_values))
            sd_val = float(np.std(voxel_values, ddof=1)) if n_vox > 1 else 0.0
            se_val = sd_val / np.sqrt(n_vox) if n_vox > 0 else 0.0

            rows.append({
                'subject': subnum,
                'roi_name': roi_name,
                'coord_label': label,
                'x': coords[0],
                'y': coords[1],
                'z': coords[2],
                'radius': radius,
                'mean_beta': mean_val,
                'sd_beta': sd_val,
                'n_voxels': n_vox,
                'se_beta': float(se_val),
            })

    return pd.DataFrame(rows)


def roi_ttest_table(roi_betas_df, alpha=0.05):
    """
    Run one-sample t-tests on extracted ROI betas and apply Bonferroni
    correction across ROI coordinates.

    Parameters
    ----------
    roi_betas_df : pandas.DataFrame
        Output of extract_roi_betas
    alpha : float
        Nominal significance level before correction

    Returns
    -------
    results : pandas.DataFrame
        One row per ROI coordinate with columns:
        roi_name, coord_label, x, y, z, radius, n_subjects,
        mean_beta, se, t_stat, p_uncorr, p_bonf, significant
    """
    from scipy import stats

    groups = roi_betas_df.groupby('coord_label', sort=False)
    n_tests = len(groups)
    alpha_bonf = alpha / n_tests

    rows = []
    for label, group in groups:
        betas = group['mean_beta'].values
        n = len(betas)
        mean = float(np.mean(betas))
        se = float(np.std(betas, ddof=1) / np.sqrt(n))

        if n >= 2 and se > 0:
            t_stat, p_val = stats.ttest_1samp(betas, 0)
            t_stat = float(t_stat)
            p_val = float(p_val)
        else:
            t_stat = np.nan
            p_val = np.nan

        p_bonf = min(p_val * n_tests, 1.0) if not np.isnan(p_val) else np.nan

        first = group.iloc[0]
        rows.append({
            'roi_name': first['roi_name'],
            'coord_label': label,
            'x': first['x'],
            'y': first['y'],
            'z': first['z'],
            'radius': first['radius'],
            'n_subjects': n,
            'mean_beta': mean,
            'se': se,
            't_stat': t_stat,
            'p_uncorr': p_val,
            'p_bonf': p_bonf,
            'significant': p_bonf < alpha if not np.isnan(p_bonf) else False,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Paired session comparison (e.g., week 3 vs week 1)
# =============================================================================

def run_group_paired_ttest(
    subjects, sessions, task, contrast_id, output_dir,
    mnum='value_parametric',
    model_variant='rt_in_duration',
    space='MNI152NLin2009cAsym_res-2',
):
    """
    Run a paired t-test comparing two sessions at the group level.

    Computes (session_b - session_a) difference maps per subject, then
    fits a one-sample t-test on the differences.

    Parameters
    ----------
    subjects : list of str
    sessions : tuple of (session_a, session_b)
        e.g., ('01', '03') to test ses-03 > ses-01
    task, contrast_id : str
    output_dir : str
    mnum, model_variant, space : str

    Returns
    -------
    tmap_path : str
    effect_path : str
    """
    import nibabel as nib
    from nilearn.image import math_img

    ses_a, ses_b = sessions

    diff_maps = []
    missing = []
    for subnum in subjects:
        path_a = get_contrast_path(
            subnum, ses_a, task, contrast_id, output_dir,
            mnum=mnum, model_variant=model_variant,
            space=space, map_type='effect_size',
        )
        path_b = get_contrast_path(
            subnum, ses_b, task, contrast_id, output_dir,
            mnum=mnum, model_variant=model_variant,
            space=space, map_type='effect_size',
        )
        if not os.path.exists(path_a) or not os.path.exists(path_b):
            missing.append(subnum)
            continue

        diff = math_img('b - a', a=path_a, b=path_b)
        diff_maps.append(diff)

    if missing:
        print(f"  WARNING: Missing maps for subjects: {missing}")
    if len(diff_maps) < 2:
        raise ValueError(
            f"Need at least 2 subjects with both sessions, "
            f"found {len(diff_maps)}"
        )

    n_subjects = len(diff_maps)
    print(f"  Fitting paired t-test: ses-{ses_b} - ses-{ses_a}, "
          f"task-{task} {contrast_id} [{model_variant}] (n={n_subjects})")

    model, group_tmap, group_effect = fit_group_onesample(diff_maps)

    # Save with a paired-specific naming convention
    group_dir = get_group_dir(output_dir, mnum, model_variant, ses_b)
    os.makedirs(group_dir, exist_ok=True)
    prefix = get_group_map_prefix(task, space, mnum, model_variant)

    tmap_path = os.path.join(
        group_dir,
        f'{prefix}_{contrast_id}_ses{ses_b}_minus_ses{ses_a}_tmap.nii.gz'
    )
    nib.save(group_tmap, tmap_path)

    effect_path = os.path.join(
        group_dir,
        f'{prefix}_{contrast_id}_ses{ses_b}_minus_ses{ses_a}_effect_size.nii.gz'
    )
    nib.save(group_effect, effect_path)

    print(f"  Saved paired maps to: {group_dir}")
    return tmap_path, effect_path


def extract_roi_betas_paired(
    subjects, sessions, task, contrast_id, output_dir,
    mnum='value_parametric',
    model_variant='rt_in_duration',
    space='MNI152NLin2009cAsym_res-2',
    roi_coords=None,
):
    """
    Extract mean betas per subject per ROI for two sessions, returning
    both individual session betas and the difference.

    Returns
    -------
    results : pandas.DataFrame
        Columns: subject, roi_name, coord_label, x, y, z, radius,
                 beta_ses_a, beta_ses_b, beta_diff
    """
    if roi_coords is None:
        roi_coords = ROI_COORDS

    ses_a, ses_b = sessions

    df_a = extract_roi_betas(
        subjects, ses_a, task, contrast_id, output_dir,
        mnum=mnum, model_variant=model_variant, space=space,
        roi_coords=roi_coords,
    )
    df_b = extract_roi_betas(
        subjects, ses_b, task, contrast_id, output_dir,
        mnum=mnum, model_variant=model_variant, space=space,
        roi_coords=roi_coords,
    )

    merged = df_a.merge(
        df_b,
        on=['subject', 'roi_name', 'coord_label', 'x', 'y', 'z', 'radius'],
        suffixes=(f'_ses{ses_a}', f'_ses{ses_b}'),
    )

    merged = merged.rename(columns={
        f'mean_beta_ses{ses_a}': 'beta_ses_a',
        f'mean_beta_ses{ses_b}': 'beta_ses_b',
    })
    merged['beta_diff'] = merged['beta_ses_b'] - merged['beta_ses_a']

    return merged


def roi_paired_ttest_table(paired_betas_df, alpha=0.05):
    """
    Run paired t-tests (ses_b - ses_a) on extracted ROI betas with
    Bonferroni correction.
    """
    from scipy import stats

    groups = paired_betas_df.groupby('coord_label', sort=False)
    n_tests = len(groups)

    rows = []
    for label, group in groups:
        diffs = group['beta_diff'].values
        betas_a = group['beta_ses_a'].values
        betas_b = group['beta_ses_b'].values
        n = len(diffs)

        mean_a = float(np.mean(betas_a))
        mean_b = float(np.mean(betas_b))
        mean_diff = float(np.mean(diffs))
        se_diff = float(np.std(diffs, ddof=1) / np.sqrt(n))

        if n >= 2 and se_diff > 0:
            t_stat, p_val = stats.ttest_1samp(diffs, 0)
            t_stat = float(t_stat)
            p_val = float(p_val)
        else:
            t_stat = np.nan
            p_val = np.nan

        p_bonf = min(p_val * n_tests, 1.0) if not np.isnan(p_val) else np.nan

        first = group.iloc[0]
        rows.append({
            'roi_name': first['roi_name'],
            'coord_label': label,
            'x': first['x'],
            'y': first['y'],
            'z': first['z'],
            'radius': first['radius'],
            'n_subjects': n,
            'mean_beta_ses_a': mean_a,
            'mean_beta_ses_b': mean_b,
            'mean_diff': mean_diff,
            'se_diff': se_diff,
            't_stat': t_stat,
            'p_uncorr': p_val,
            'p_bonf': p_bonf,
            'significant': p_bonf < alpha if not np.isnan(p_bonf) else False,
        })

    return pd.DataFrame(rows)