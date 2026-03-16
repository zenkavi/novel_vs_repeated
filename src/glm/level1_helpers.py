"""
level1_helpers.py
Helper functions for the value_parametric level 1 GLM model.

Supports 4 model variants that differ in how reaction time is handled:

  rt_in_duration:         RT as duration of stim_ev and stim_value_par
  fixed_duration:         Fixed duration (0 = stick) for stimulus regressors
  rt_duration_plus_mod:   Variable duration + RT parametric modulator
  fixed_duration_plus_mod: Fixed duration + RT parametric modulator

Includes:
  - Event formatting (declarative regressor specs)
  - Confound loading
  - Design matrix construction
  - VIF computation
  - Contrast preparation
  - HTML report generation
"""

import base64
import glob
import io
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor


# =============================================================================
# Model variant definitions
# =============================================================================

# Each variant is defined by:
#   - regressor_specs: per-task list of (reg_name, event_type, modulation)
#   - task_regressors: columns to show in correlation/VIF plots
#   - stim_duration: None (use event file duration, i.e. RT) or 0 (stick)
#   - description: human-readable label for reports

MODEL_VARIANTS = {
    'rt_in_duration': {
        'description': 'RT as stimulus duration',
        'stim_duration': None,
        'regressor_specs': {
            'yesNo': [
                ('cross_ev',        'fixCross', 1),
                ('stim_ev',         'stim',     1),
                ('stim_value_par',  'stim',     'valStim_dmn'),
                ('feedback_ev',     'feedback', 1),
                ('reward_par',      'feedback', 'reward_dmn'),
            ],
            'binaryChoice': [
                ('cross_ev',        'fixCross', 1),
                ('stim_ev',         'stim',     1),
                ('stim_value_par',  'stim',     'valChosenMinusUnchosen_dmn'),
                ('feedback_ev',     'feedback', 1),
                ('reward_par',      'feedback', 'reward_dmn'),
            ],
        },
        'task_regressors': [
            'cross_ev', 'stim_ev', 'stim_value_par',
            'feedback_ev', 'reward_par',
        ],
    },
    'fixed_duration': {
        'description': 'Fixed stimulus duration (stick function)',
        'stim_duration': 0,
        'regressor_specs': {
            'yesNo': [
                ('cross_ev',        'fixCross', 1),
                ('stim_ev',         'stim',     1),
                ('stim_value_par',  'stim',     'valStim_dmn'),
                ('feedback_ev',     'feedback', 1),
                ('reward_par',      'feedback', 'reward_dmn'),
            ],
            'binaryChoice': [
                ('cross_ev',        'fixCross', 1),
                ('stim_ev',         'stim',     1),
                ('stim_value_par',  'stim',     'valChosenMinusUnchosen_dmn'),
                ('feedback_ev',     'feedback', 1),
                ('reward_par',      'feedback', 'reward_dmn'),
            ],
        },
        'task_regressors': [
            'cross_ev', 'stim_ev', 'stim_value_par',
            'feedback_ev', 'reward_par',
        ],
    },
    'rt_duration_plus_mod': {
        'description': 'RT as stimulus duration + RT parametric modulator',
        'stim_duration': None,
        'regressor_specs': {
            'yesNo': [
                ('cross_ev',        'fixCross', 1),
                ('stim_ev',         'stim',     1),
                ('stim_value_par',  'stim',     'valStim_dmn'),
                ('stim_rt_par',     'stim',     '_rt_dmn'),
                ('feedback_ev',     'feedback', 1),
                ('reward_par',      'feedback', 'reward_dmn'),
            ],
            'binaryChoice': [
                ('cross_ev',        'fixCross', 1),
                ('stim_ev',         'stim',     1),
                ('stim_value_par',  'stim',     'valChosenMinusUnchosen_dmn'),
                ('stim_rt_par',     'stim',     '_rt_dmn'),
                ('feedback_ev',     'feedback', 1),
                ('reward_par',      'feedback', 'reward_dmn'),
            ],
        },
        'task_regressors': [
            'cross_ev', 'stim_ev', 'stim_value_par', 'stim_rt_par',
            'feedback_ev', 'reward_par',
        ],
    },
    'fixed_duration_plus_mod': {
        'description': 'Fixed stimulus duration + RT parametric modulator',
        'stim_duration': 0,
        'regressor_specs': {
            'yesNo': [
                ('cross_ev',        'fixCross', 1),
                ('stim_ev',         'stim',     1),
                ('stim_value_par',  'stim',     'valStim_dmn'),
                ('stim_rt_par',     'stim',     '_rt_dmn'),
                ('feedback_ev',     'feedback', 1),
                ('reward_par',      'feedback', 'reward_dmn'),
            ],
            'binaryChoice': [
                ('cross_ev',        'fixCross', 1),
                ('stim_ev',         'stim',     1),
                ('stim_value_par',  'stim',     'valChosenMinusUnchosen_dmn'),
                ('stim_rt_par',     'stim',     '_rt_dmn'),
                ('feedback_ev',     'feedback', 1),
                ('reward_par',      'feedback', 'reward_dmn'),
            ],
        },
        'task_regressors': [
            'cross_ev', 'stim_ev', 'stim_value_par', 'stim_rt_par',
            'feedback_ev', 'reward_par',
        ],
    },
}


# =============================================================================
# Data loading
# =============================================================================

def get_from_sidecar(subnum, session, task, runnum, keyname, data_path):
    """Read metadata from BOLD sidecar JSON."""
    fn = os.path.join(
        data_path,
        f'sub-{subnum}/ses-{session}/func/'
        f'sub-{subnum}_ses-{session}_task-{task}_run-{runnum}_bold.json'
    )
    with open(fn) as f:
        bold_sidecar = json.load(f)
    if isinstance(keyname, list):
        return [bold_sidecar.get(k) for k in keyname]
    return bold_sidecar[keyname]


def get_confounds(subnum, session, task, runnum, data_path, scrub_thresh=0.5):
    """
    Load and format confound regressors from fmriprep output.

    Includes: 6 motion parameters (trans + rot), std_dvars,
    framewise_displacement, and a scrub regressor.
    """
    fn = os.path.join(
        data_path,
        f'derivatives/sub-{subnum}/ses-{session}/func/'
        f'sub-{subnum}_ses-{session}_task-{task}_run-{runnum}'
        f'_desc-confounds_timeseries.tsv'
    )
    confounds = pd.read_csv(fn, sep='\t')

    confound_cols = (
        [x for x in confounds.columns if 'trans' in x]
        + [x for x in confounds.columns if 'rot' in x]
        + ['std_dvars', 'framewise_displacement']
    )
    formatted_confounds = confounds[confound_cols].fillna(0).copy()
    formatted_confounds['scrub'] = (
        formatted_confounds['framewise_displacement'] > scrub_thresh
    ).astype(int)

    return formatted_confounds


def get_n_scans(subnum, session, task, runnum, data_path, space):
    """Get number of scans from preprocessed BOLD image."""
    import nibabel as nib
    fn = os.path.join(
        data_path,
        f'derivatives/sub-{subnum}/ses-{session}/func/'
        f'sub-{subnum}_ses-{session}_task-{task}_run-{runnum}'
        f'_space-{space}_desc-preproc_bold.nii.gz'
    )
    img = nib.load(fn)
    return img.shape[3]


# =============================================================================
# Event formatting
# =============================================================================

def _load_events_and_behavior(subnum, session, task, runnum, data_path):
    """
    Load the events and behavioral files for one run, and verify that
    the number of trials matches between them.

    Returns
    -------
    events : DataFrame
        BIDS events with onset, duration, trial_type columns.
    behavior : DataFrame
        Behavioral data with one row per trial.
    """
    events_fn = os.path.join(
        data_path,
        f'sub-{subnum}/ses-{session}/func/'
        f'sub-{subnum}_ses-{session}_task-{task}_run-{runnum}_events.tsv'
    )
    events = pd.read_csv(events_fn, sep='\t')

    beh_fn = os.path.join(
        data_path,
        f'sub-{subnum}/ses-{session}/beh/'
        f'sub-{subnum}_ses-{session}_task-{task}_run-{runnum}_beh.tsv'
    )
    behavior = pd.read_csv(beh_fn, sep='\t')

    # Verify alignment: behavior should have one row per trial (= per stim event)
    n_stim = (events['trial_type'] == 'stim').sum()
    n_beh = len(behavior)
    if n_stim != n_beh:
        raise ValueError(
            f"Mismatch for sub-{subnum} ses-{session} task-{task} run-{runnum}: "
            f"events has {n_stim} stim trials but behavior has {n_beh} rows."
        )

    return events, behavior


def _make_regressor(events, behavior, event_type, reg_name, modulation,
                    stim_duration=None):
    """
    Build a single regressor DataFrame from events and behavior.

    Parameters
    ----------
    events : DataFrame
        Full BIDS events table.
    behavior : DataFrame
        Behavioral data (one row per trial, aligned to stim events).
    event_type : str
        Value in the trial_type column to filter on (e.g. 'stim', 'feedback').
    reg_name : str
        Name for this regressor in the design matrix.
    modulation : int, float, or str
        If numeric, used as a constant modulation value (1 for unmodulated).
        If the string '_rt_dmn', computes demeaned RT from the original stim
        durations (before any duration override).
        Otherwise, treated as a column name in the behavior DataFrame.
    stim_duration : float or None
        If not None, override the duration for 'stim' event types with this
        fixed value. Does not affect 'fixCross' or 'feedback' events.

    Returns
    -------
    DataFrame with columns: onset, duration, trial_type, modulation
    """
    df = events.query(
        f'trial_type == "{event_type}"'
    )[['onset', 'duration']].reset_index(drop=True)

    # Compute RT modulator BEFORE overriding duration, so it always
    # reflects the actual reaction time regardless of the variant.
    if modulation == '_rt_dmn':
        rt = df['duration'].values.copy()
        mod_values = rt - rt.mean()
    elif isinstance(modulation, (int, float)):
        mod_values = modulation
    else:
        mod_values = behavior[modulation].values

    # Override stim duration if requested (only for stim events)
    if stim_duration is not None and event_type == 'stim':
        df['duration'] = stim_duration

    df['trial_type'] = reg_name
    df['modulation'] = mod_values

    return df


def get_events_value_parametric(subnum, session, task, runnum, data_path,
                                regressor_specs=None, stim_duration=None):
    """
    Build event regressors for the value parametric model.

    Parameters
    ----------
    subnum, session, task, runnum, data_path : str
        Standard BIDS identifiers and path.
    regressor_specs : list of tuples, optional
        Each tuple is (reg_name, event_type, modulation_source).
        If None, uses the 'rt_in_duration' variant specs for this task.
    stim_duration : float or None
        If not None, override stimulus event durations with this fixed value.

    Returns
    -------
    DataFrame with columns: onset, duration, trial_type, modulation
        Sorted by onset, ready for make_first_level_design_matrix.
    """
    events, behavior = _load_events_and_behavior(
        subnum, session, task, runnum, data_path
    )

    if regressor_specs is None:
        regressor_specs = MODEL_VARIANTS['rt_in_duration']['regressor_specs'][task]

    regressors = [
        _make_regressor(events, behavior, event_type, reg_name, modulation,
                        stim_duration=stim_duration)
        for reg_name, event_type, modulation in regressor_specs
    ]

    formatted_events = pd.concat(regressors, ignore_index=True)
    formatted_events = formatted_events.sort_values('onset').reset_index(drop=True)
    formatted_events = formatted_events[['onset', 'duration', 'trial_type', 'modulation']]

    return formatted_events


# =============================================================================
# Design matrix
# =============================================================================

def make_design_matrix_value_parametric(
    subnum, session, task, runnum, data_path,
    model_variant='rt_in_duration',
    space='MNI152NLin2009cAsym_res-2',
    hrf_model='spm', drift_model='cosine',
    scrub_thresh=0.5
):
    """
    Build the first-level design matrix for the value parametric model.

    Parameters
    ----------
    model_variant : str
        One of: 'rt_in_duration', 'fixed_duration',
        'rt_duration_plus_mod', 'fixed_duration_plus_mod'.
    """
    variant = MODEL_VARIANTS[model_variant]

    tr = get_from_sidecar(subnum, session, task, runnum, 'RepetitionTime', data_path)
    n_scans = get_n_scans(subnum, session, task, runnum, data_path, space)
    frame_times = np.arange(n_scans) * tr

    formatted_events = get_events_value_parametric(
        subnum, session, task, runnum, data_path,
        regressor_specs=variant['regressor_specs'][task],
        stim_duration=variant['stim_duration'],
    )
    formatted_confounds = get_confounds(
        subnum, session, task, runnum, data_path, scrub_thresh=scrub_thresh
    )

    design_matrix = make_first_level_design_matrix(
        frame_times,
        formatted_events,
        drift_model=drift_model,
        add_regs=formatted_confounds,
        hrf_model=hrf_model
    )

    return design_matrix


# =============================================================================
# VIF
# =============================================================================

def compute_vif(design_matrix, columns=None):
    """
    Compute Variance Inflation Factors for design matrix columns.

    VIF > 5: moderate multicollinearity
    VIF > 10: severe multicollinearity

    Parameters
    ----------
    design_matrix : DataFrame
        Full design matrix.
    columns : list or None
        Columns to compute VIF for. If None, auto-detects task regressors.
    """
    if columns is None:
        nuisance = ['trans', 'rot', 'drift', 'framewise', 'scrub',
                     'constant', 'dvars']
        columns = [c for c in design_matrix.columns
                   if all(n not in c for n in nuisance)]
    dm = design_matrix[columns].copy()

    nonzero_cols = dm.columns[dm.abs().sum() > 0]
    dm = dm[nonzero_cols]

    vif_data = pd.DataFrame({
        'regressor': dm.columns,
        'VIF': [variance_inflation_factor(dm.values, i)
                for i in range(dm.shape[1])]
    })
    return vif_data


# =============================================================================
# Contrasts
# =============================================================================

def make_contrasts(design_matrix):
    """
    Build contrast vectors for the value parametric model.

    Returns canonical contrasts for each task regressor (vs baseline).
    """
    contrast_matrix = np.eye(design_matrix.shape[1])
    contrasts = {
        col: contrast_matrix[i]
        for i, col in enumerate(design_matrix.columns)
    }

    to_filter = ['trans', 'rot', 'drift', 'framewise', 'scrub',
                 'constant', 'dvars']
    beh_contrasts = {
        k: v for k, v in contrasts.items()
        if all(filt not in k for filt in to_filter)
    }

    return beh_contrasts


def match_dm_cols(dm_list):
    """Align design matrix columns across runs (for fixed effects)."""
    col_nums = [len(dm.columns) for dm in dm_list]
    target_cols = dm_list[np.argmax(col_nums)].columns
    for i, df in enumerate(dm_list):
        dm_list[i] = df.reindex(target_cols, axis=1, fill_value=0)
    return dm_list


# =============================================================================
# Plotting helpers (return figure objects, never show)
# =============================================================================

def plot_dm(dm, title=''):
    """Plot design matrix, return figure."""
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_design_matrix(dm, axes=ax)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_correlation_matrix(dm, title='', columns=None):
    """Plot lower-triangle correlation heatmap for task regressors.

    Parameters
    ----------
    dm : DataFrame
        Full design matrix.
    title : str
        Plot title.
    columns : list or None
        If provided, restrict to these columns. If None, auto-detects
        task regressors (excludes nuisance/drift/constant).
    """
    if columns is None:
        nuisance = ['trans', 'rot', 'drift', 'framewise', 'scrub',
                     'constant', 'dvars']
        columns = [c for c in dm.columns
                   if all(n not in c for n in nuisance)]
    corr = dm[columns].corr()
    n = len(columns)
    figsize = (max(5, n * 1.2), max(4, n * 1.0))
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, cmap='RdBu_r', center=0,
        vmin=-1, vmax=1, annot=True, fmt='.2f', annot_kws={'size': 9},
        square=True, linewidths=0.5, ax=ax
    )
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_vif(vif_data, title=''):
    """Plot VIF bar chart with threshold lines, return figure."""
    fig, ax = plt.subplots(figsize=(10, max(4, len(vif_data) * 0.5)))
    colors = [
        '#d62728' if v > 10 else '#ff7f0e' if v > 5 else '#2ca02c'
        for v in vif_data['VIF']
    ]
    ax.barh(range(len(vif_data)), vif_data['VIF'], color=colors)
    ax.set_yticks(range(len(vif_data)))
    ax.set_yticklabels(vif_data['regressor'], fontsize=9)
    ax.set_xlabel('VIF')
    ax.set_title(title)
    ax.axvline(x=5, color='orange', linestyle='--', alpha=0.7, label='VIF = 5')
    ax.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='VIF = 10')
    ax.legend()
    plt.tight_layout()
    return fig


# =============================================================================
# HTML report
# =============================================================================

def _fig_to_base64(fig):
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_report(
    subnum, session, data_path,
    model_variant='rt_in_duration',
    mnum='value_parametric',
    space='MNI152NLin2009cAsym_res-2',
    hrf_model='spm', drift_model='cosine',
    scrub_thresh=0.5,
    output_dir='outputs'
):
    """
    Generate an HTML report with design matrix diagnostics for all runs
    of a given subject and session.

    For ses-01 this means: yesNo run-01, yesNo run-02, binaryChoice run-03.

    Parameters
    ----------
    subnum : str
        Subject number (e.g. '601')
    session : str
        Session number (e.g. '01')
    data_path : str
        Path to the BIDS root directory
    model_variant : str
        One of: 'rt_in_duration', 'fixed_duration',
        'rt_duration_plus_mod', 'fixed_duration_plus_mod'.
    mnum : str
        Model name (used in filenames)
    space : str
        Output space identifier
    hrf_model : str
        HRF model for nilearn
    drift_model : str
        Drift model for nilearn
    scrub_thresh : float
        Framewise displacement threshold for scrubbing
    output_dir : str
        Directory to save the report and design matrices

    Returns
    -------
    report_path : str
        Path to the saved HTML report
    design_matrices : dict
        Dictionary mapping (task, runnum) to design matrix DataFrames
    """
    variant = MODEL_VARIANTS[model_variant]
    task_regs = variant['task_regressors']

    os.makedirs(output_dir, exist_ok=True)

    runs = [
        ('yesNo', '01'),
        ('yesNo', '02'),
        ('binaryChoice', '03'),
    ]

    design_matrices = {}
    run_sections_html = []

    for task, runnum in runs:
        run_label = f'task-{task}_run-{runnum}'
        print(f"  Processing sub-{subnum} ses-{session} {run_label} "
              f"[{model_variant}]...")

        # -- Build design matrix --
        dm = make_design_matrix_value_parametric(
            subnum, session, task, runnum, data_path,
            model_variant=model_variant,
            space=space, hrf_model=hrf_model, drift_model=drift_model,
            scrub_thresh=scrub_thresh
        )
        design_matrices[(task, runnum)] = dm

        # -- Save design matrix CSV --
        dm_fn = (f'sub-{subnum}_ses-{session}_{run_label}_{mnum}'
                 f'_design_matrix.csv')
        dm.to_csv(os.path.join(output_dir, dm_fn), index=False)

        # -- Parametric modulator stats --
        formatted_events = get_events_value_parametric(
            subnum, session, task, runnum, data_path,
            regressor_specs=variant['regressor_specs'][task],
            stim_duration=variant['stim_duration'],
        )
        par_regs = [r for r in task_regs if '_par' in r]
        par_stats = {}
        for reg in par_regs:
            vals = formatted_events.query(
                f'trial_type == "{reg}"'
            )['modulation']
            par_stats[reg] = {
                'mean': f'{vals.mean():.3f}',
                'std': f'{vals.std():.3f}',
                'min': f'{vals.min():.3f}',
                'max': f'{vals.max():.3f}',
            }

        # -- Confound summary --
        confounds = get_confounds(
            subnum, session, task, runnum, data_path, scrub_thresh
        )
        n_scrubbed = int(confounds['scrub'].sum())
        n_volumes = len(confounds)

        # -- Plots (task regressors only for corr and VIF) --
        task_cols = [c for c in task_regs if c in dm.columns]

        fig_dm = plot_dm(
            dm, title=f'sub-{subnum} ses-{session} {run_label}')
        img_dm = _fig_to_base64(fig_dm)

        fig_corr = plot_correlation_matrix(
            dm,
            title=f'Correlations: sub-{subnum} ses-{session} {run_label}',
            columns=task_cols)
        img_corr = _fig_to_base64(fig_corr)

        vif_data = compute_vif(dm, columns=task_cols)
        fig_vif = plot_vif(
            vif_data,
            title=f'VIF: sub-{subnum} ses-{session} {run_label}')
        img_vif = _fig_to_base64(fig_vif)

        # -- Task regressor correlations --
        beh_corr = dm[task_cols].corr().round(3)

        # -- Build HTML section for this run --
        vif_rows = ''
        for _, row in vif_data.iterrows():
            css_class = ''
            if row['VIF'] > 10:
                css_class = ' class="vif-severe"'
            elif row['VIF'] > 5:
                css_class = ' class="vif-moderate"'
            vif_rows += (f'<tr{css_class}><td>{row["regressor"]}</td>'
                         f'<td>{row["VIF"]:.2f}</td></tr>\n')

        par_rows = ''
        for reg, stats in par_stats.items():
            par_rows += (
                f'<tr><td>{reg}</td>'
                f'<td>{stats["mean"]}</td><td>{stats["std"]}</td>'
                f'<td>{stats["min"]}</td><td>{stats["max"]}</td></tr>\n'
            )

        beh_corr_html = beh_corr.to_html(
            classes='corr-table', float_format='%.3f')

        section = f"""
        <div class="run-section">
            <h2>{run_label}</h2>

            <div class="stats-row">
                <div class="stat-card">
                    <h3>Design Matrix</h3>
                    <p>Shape: {dm.shape[0]} timepoints x {dm.shape[1]} regressors</p>
                    <p>Volumes: {n_volumes} | Scrubbed: {n_scrubbed}</p>
                </div>
                <div class="stat-card">
                    <h3>Parametric Modulators</h3>
                    <table class="stats-table">
                        <tr><th>Regressor</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
                        {par_rows}
                    </table>
                </div>
            </div>

            <div class="plot-row">
                <div class="plot-container">
                    <h3>Design Matrix</h3>
                    <img src="data:image/png;base64,{img_dm}" />
                </div>
            </div>

            <div class="plot-row">
                <div class="plot-container half">
                    <h3>Correlation Matrix (task regressors)</h3>
                    <img src="data:image/png;base64,{img_corr}" />
                </div>
                <div class="plot-container half">
                    <h3>Variance Inflation Factors (task regressors)</h3>
                    <img src="data:image/png;base64,{img_vif}" />
                </div>
            </div>

            <div class="tables-row">
                <div class="table-container">
                    <h3>Task Regressor Correlations</h3>
                    {beh_corr_html}
                </div>
                <div class="table-container">
                    <h3>VIF Values</h3>
                    <table class="stats-table">
                        <tr><th>Regressor</th><th>VIF</th></tr>
                        {vif_rows}
                    </table>
                </div>
            </div>
        </div>
        """
        run_sections_html.append(section)

    # -- Contrasts section --
    first_dm = list(design_matrices.values())[0]
    contrasts = make_contrasts(first_dm)
    contrasts_rows = ''
    for name, vec in contrasts.items():
        nonzero = [first_dm.columns[i] for i in np.where(vec != 0)[0]]
        contrasts_rows += (f'<tr><td>{name}</td>'
                           f'<td>{", ".join(nonzero)}</td></tr>\n')

    contrasts_json = {k: v.tolist() for k, v in contrasts.items()}
    contrasts_fn = f'sub-{subnum}_ses-{session}_{mnum}_contrasts.json'
    with open(os.path.join(output_dir, contrasts_fn), 'w') as f:
        json.dump(contrasts_json, f, indent=4)

    # -- Column consistency check --
    all_cols = [set(dm.columns) for dm in design_matrices.values()]
    cols_match = all(c == all_cols[0] for c in all_cols[1:])
    col_counts = {
        f'{t} run-{r}': len(dm.columns)
        for (t, r), dm in design_matrices.items()
    }

    consistency_html = '<ul>\n'
    for label, count in col_counts.items():
        consistency_html += f'<li>{label}: {count} columns</li>\n'
    consistency_html += (
        f'<li>Columns identical across runs: {cols_match}</li>\n</ul>')

    # -- Assemble full HTML --
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Level 1 Report: sub-{subnum} ses-{session} | {mnum} | {model_variant}</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        margin: 0; padding: 20px 40px;
        background: #f8f9fa; color: #333;
    }}
    h1 {{
        border-bottom: 3px solid #2c3e50; padding-bottom: 10px;
        color: #2c3e50;
    }}
    h2 {{
        color: #34495e; border-bottom: 1px solid #bdc3c7;
        padding-bottom: 6px; margin-top: 30px;
    }}
    h3 {{ color: #555; margin-bottom: 8px; }}
    .run-section {{
        background: #fff; border-radius: 8px;
        padding: 20px 30px; margin-bottom: 30px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }}
    .summary-section {{
        background: #fff; border-radius: 8px;
        padding: 20px 30px; margin-bottom: 30px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }}
    .stats-row {{
        display: flex; gap: 20px; margin-bottom: 16px; flex-wrap: wrap;
    }}
    .stat-card {{
        flex: 1; min-width: 280px;
        background: #f0f4f8; border-radius: 6px; padding: 12px 16px;
    }}
    .stat-card h3 {{ margin-top: 0; }}
    .plot-row {{
        display: flex; gap: 20px; margin-bottom: 16px; flex-wrap: wrap;
    }}
    .plot-container {{
        flex: 1; min-width: 300px; text-align: center;
    }}
    .plot-container.half {{ flex: 0 0 48%; }}
    .plot-container img {{
        max-width: 100%; height: auto;
        border: 1px solid #e0e0e0; border-radius: 4px;
    }}
    .tables-row {{
        display: flex; gap: 20px; margin-bottom: 16px; flex-wrap: wrap;
    }}
    .table-container {{
        flex: 1; min-width: 280px; overflow-x: auto;
    }}
    table {{ border-collapse: collapse; font-size: 13px; width: 100%; }}
    th, td {{ padding: 6px 10px; border: 1px solid #ddd; text-align: left; }}
    th {{ background: #f0f4f8; }}
    .stats-table {{ width: auto; }}
    .corr-table {{ width: auto; }}
    .vif-severe {{ background: #f8d7da; }}
    .vif-moderate {{ background: #fff3cd; }}
    .meta {{ color: #888; font-size: 13px; }}
</style>
</head>
<body>

<h1>Level 1 Design Matrix Report</h1>
<p class="meta">
    Subject: sub-{subnum} | Session: ses-{session} | Model: {mnum}<br>
    Variant: <strong>{model_variant}</strong> ({variant['description']})<br>
    Space: {space} | HRF: {hrf_model} | Drift: {drift_model} | Scrub threshold: {scrub_thresh}
</p>

{''.join(run_sections_html)}

<div class="summary-section">
    <h2>Cross-Run Summary</h2>

    <h3>Column Consistency</h3>
    {consistency_html}

    <h3>Contrasts</h3>
    <table class="stats-table">
        <tr><th>Contrast</th><th>Nonzero Columns</th></tr>
        {contrasts_rows}
    </table>
    <p class="meta">Contrasts saved to: {contrasts_fn}</p>
</div>

</body>
</html>"""

    report_fn = f'sub-{subnum}_ses-{session}_{mnum}_report.html'
    report_path = os.path.join(output_dir, report_fn)
    with open(report_path, 'w') as f:
        f.write(html)

    print(f"  Report saved: {report_path}")
    return report_path, design_matrices


# =============================================================================
# GLM fitting and contrast computation
# =============================================================================

def load_design_matrices(subnum, session, mnum, model_variant, dm_dir):
    """
    Load saved design matrix CSVs for all runs of a subject/session.

    Parameters
    ----------
    subnum : str
        Subject number (e.g. '601')
    session : str
        Session number (e.g. '01')
    mnum : str
        Model name (e.g. 'value_parametric')
    model_variant : str
        Variant name (e.g. 'rt_in_duration')
    dm_dir : str
        Directory where design matrix CSVs were saved by generate_report.
        Expected path: {dm_dir}/{mnum}/{model_variant}/

    Returns
    -------
    design_matrices : dict
        Maps (task, runnum) to design matrix DataFrames.
    """
    dm_path = os.path.join(dm_dir, mnum, model_variant)

    runs = [
        ('yesNo', '01'),
        ('yesNo', '02'),
        ('binaryChoice', '03'),
    ]

    design_matrices = {}
    for task, runnum in runs:
        fn = os.path.join(
            dm_path,
            f'sub-{subnum}_ses-{session}_task-{task}_run-{runnum}'
            f'_{mnum}_design_matrix.csv'
        )
        dm = pd.read_csv(fn)
        design_matrices[(task, runnum)] = dm

    return design_matrices


def get_bold_and_mask(subnum, session, task, runnum, data_path, space):
    """
    Get paths to the preprocessed BOLD image and brain mask for one run.

    Returns
    -------
    bold_path : str
    mask_path : str
    """
    bold_path = os.path.join(
        data_path,
        f'derivatives/sub-{subnum}/ses-{session}/func/'
        f'sub-{subnum}_ses-{session}_task-{task}_run-{runnum}'
        f'_space-{space}_desc-preproc_bold.nii.gz'
    )
    mask_path = os.path.join(
        data_path,
        f'derivatives/sub-{subnum}/ses-{session}/func/'
        f'sub-{subnum}_ses-{session}_task-{task}_run-{runnum}'
        f'_space-{space}_desc-brain_mask.nii.gz'
    )
    return bold_path, mask_path


def fit_level1(
    subnum, session, data_path, dm_dir,
    mnum='value_parametric',
    model_variant='rt_in_duration',
    space='MNI152NLin2009cAsym_res-2',
    noise_model='ar1',
    hrf_model='spm',
    drift_model='cosine',
    smoothing_fwhm=5,
):
    """
    Fit the first-level GLM for one subject/session using pre-saved
    design matrices.

    Fits a fixed-effects model across all runs (yesNo run-01, run-02,
    binaryChoice run-03) simultaneously.

    Parameters
    ----------
    subnum : str
        Subject number
    session : str
        Session number
    data_path : str
        BIDS root directory
    dm_dir : str
        Directory containing saved design matrix CSVs
        (organized as {dm_dir}/{mnum}/{model_variant}/)
    mnum : str
        Model name
    model_variant : str
        Variant name
    space : str
        Output space
    noise_model, hrf_model, drift_model : str
        GLM parameters (should match what was used to build the design matrices)
    smoothing_fwhm : float
        Smoothing kernel FWHM in mm

    Returns
    -------
    fmri_glm : FirstLevelModel
        Fitted GLM object
    design_matrices : list of DataFrames
        Design matrices used (one per run, column-aligned)
    """
    import nibabel as nib
    from nilearn.glm.first_level import FirstLevelModel

    runs = [
        ('yesNo', '01'),
        ('yesNo', '02'),
        ('binaryChoice', '03'),
    ]

    # Load saved design matrices
    saved_dms = load_design_matrices(
        subnum, session, mnum, model_variant, dm_dir
    )

    # Collect BOLD images and design matrices in run order
    fmri_imgs = []
    design_matrices = []

    for task, runnum in runs:
        bold_path, _ = get_bold_and_mask(
            subnum, session, task, runnum, data_path, space
        )
        if not os.path.exists(bold_path):
            raise FileNotFoundError(
                f"Preprocessed BOLD not found: {bold_path}"
            )
        fmri_imgs.append(bold_path)
        design_matrices.append(saved_dms[(task, runnum)])

    # Align columns across runs (drift regressors may differ)
    design_matrices = match_dm_cols(design_matrices)

    # Use the mask from the last run
    _, mask_path = get_bold_and_mask(
        subnum, session, runs[-1][0], runs[-1][1], data_path, space
    )
    mask_img = nib.load(mask_path)

    # Get TR
    tr = get_from_sidecar(
        subnum, session, runs[0][0], runs[0][1], 'RepetitionTime', data_path
    )

    # Define and fit GLM
    fmri_glm = FirstLevelModel(
        t_r=tr,
        noise_model=noise_model,
        hrf_model=hrf_model,
        drift_model=drift_model,
        smoothing_fwhm=smoothing_fwhm,
        mask_img=mask_img,
        subject_label=subnum,
        minimize_memory=True,
    )

    print(f"  Fitting GLM for sub-{subnum} ses-{session} [{model_variant}]...")
    fmri_glm = fmri_glm.fit(fmri_imgs, design_matrices=design_matrices)

    return fmri_glm, design_matrices


def save_glm_and_contrasts(
    fmri_glm, design_matrices,
    subnum, session, output_dir,
    mnum='value_parametric',
    model_variant='rt_in_duration',
    space='MNI152NLin2009cAsym_res-2',
):
    """
    Save the fitted GLM object and compute/save all contrast maps.

    Output directory structure
    -------------------------
    {output_dir}/{mnum}/{model_variant}/
        sub-{subnum}/ses-{session}/
            {prefix}_level1_glm.pkl
            {prefix}_contrasts.json
            contrasts/
                {prefix}_{contrast_id}_effect_size.nii.gz
                {prefix}_{contrast_id}_tmap.nii.gz

    Parameters
    ----------
    fmri_glm : FirstLevelModel
        Fitted GLM
    design_matrices : list of DataFrames
        Column-aligned design matrices
    subnum, session : str
        Subject and session identifiers
    output_dir : str
        Root output directory for all GLM results
    mnum : str
        Model name
    model_variant : str
        Variant name
    space : str
        Output space
    """
    import nibabel as nib
    import pickle

    # Build output paths
    deriv_base = os.path.join(
        output_dir, mnum, model_variant,
        f'sub-{subnum}', f'ses-{session}'
    )
    contrasts_path = os.path.join(deriv_base, 'contrasts')
    os.makedirs(contrasts_path, exist_ok=True)

    # Filename prefix
    prefix = (f'sub-{subnum}_ses-{session}'
              f'_space-{space}_{mnum}_{model_variant}')

    # Save fitted GLM
    glm_fn = os.path.join(deriv_base, f'{prefix}_level1_glm.pkl')
    with open(glm_fn, 'wb') as f:
        pickle.dump(fmri_glm, f)
    print(f"  Saved GLM: {glm_fn}")

    # Build contrasts from the first design matrix
    contrasts = make_contrasts(design_matrices[0])

    # Compute and save contrast maps
    dm_ncols = len(design_matrices[0].columns)

    for contrast_id, contrast_val in contrasts.items():
        # Zero-pad or trim to match design matrix width
        if len(contrast_val) < dm_ncols:
            contrast_val = np.pad(
                contrast_val, (0, dm_ncols - len(contrast_val))
            )
        elif len(contrast_val) > dm_ncols:
            contrast_val = contrast_val[:dm_ncols]

        # Effect size
        effect_map = fmri_glm.compute_contrast(
            contrast_val, output_type='effect_size'
        )
        effect_fn = os.path.join(
            contrasts_path,
            f'{prefix}_{contrast_id}_effect_size.nii.gz'
        )
        nib.save(effect_map, effect_fn)

        # T-statistic
        stat_map = fmri_glm.compute_contrast(
            contrast_val, output_type='stat'
        )
        stat_fn = os.path.join(
            contrasts_path,
            f'{prefix}_{contrast_id}_tmap.nii.gz'
        )
        nib.save(stat_map, stat_fn)

    print(f"  Saved {len(contrasts)} contrasts to: {contrasts_path}")

    # Save contrasts JSON for reference
    contrasts_json = {k: v.tolist() for k, v in contrasts.items()}
    json_fn = os.path.join(deriv_base, f'{prefix}_contrasts.json')
    with open(json_fn, 'w') as f:
        json.dump(contrasts_json, f, indent=4)


def run_level1_pipeline(
    subnum, session, data_path, dm_dir, output_dir,
    mnum='value_parametric',
    model_variant='rt_in_duration',
    space='MNI152NLin2009cAsym_res-2',
    noise_model='ar1',
    hrf_model='spm',
    drift_model='cosine',
    smoothing_fwhm=5,
):
    """
    Full pipeline: load design matrices, fit GLM, save contrasts.

    This is the main entry point for running a single subject/session.
    """
    fmri_glm, design_matrices = fit_level1(
        subnum, session, data_path, dm_dir,
        mnum=mnum,
        model_variant=model_variant,
        space=space,
        noise_model=noise_model,
        hrf_model=hrf_model,
        drift_model=drift_model,
        smoothing_fwhm=smoothing_fwhm,
    )

    save_glm_and_contrasts(
        fmri_glm, design_matrices,
        subnum, session, output_dir,
        mnum=mnum,
        model_variant=model_variant,
        space=space,
    )