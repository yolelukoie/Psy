import mne
import pandas as pd
import pickle
import numpy as np
from mne_bids import BIDSPath, read_raw_bids
from mne.preprocessing import ICA
from antropy import spectral_entropy
from scipy.stats import ttest_ind


def fix_mne_channel_metadata(raw):
    """
    Fixes channel types
    - Sets 'VEOG', 'HEOG', 'EKG' to proper types
    - Sets all other known scalp channels to type 'eeg'
    - Applies standard 10-20 montage
    - Marks 'CB1' and 'CB2' as bad (these are peripheral channels)
    """

    aux_types = {
        'VEOG': 'eog',
        'HEOG': 'eog',
        'EKG': 'ecg'
    }
    present_aux = {ch: typ for ch, typ in aux_types.items() if ch in raw.ch_names}
    raw.set_channel_types(present_aux)

    # Set all other channels to EEG
    for ch in raw.ch_names:
        if ch not in present_aux and ch not in ['CB1', 'CB2']:
            raw.set_channel_types({ch: 'eeg'})

    # Mark CB1 and CB2 as bad
    for bad_ch in ['CB1', 'CB2']:
        if bad_ch in raw.ch_names and bad_ch not in raw.info['bads']:
            raw.info['bads'].append(bad_ch)

    # Apply standard 10-20 montage (approximate spatial positions)
    raw.set_montage('standard_1020', match_case=False)

    return raw

def preprocess(raw, subj=None):
    """
    Preprocesses EEG data:
    - Sets average reference
    - Band-pass filter
    - ICA for artefact removal
    """

    raw.set_eeg_reference('average') # projection True/False?
    raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin') # 1-40 Hz
    ica = ICA(n_components=20, random_state=97, method = 'fastica')
    ica.fit(raw)

    eog_inds, scores = ica.find_bads_eog(raw) # eog has right names already
    ica.exclude = eog_inds

    """
    raw.info['bads'] = look thought raw data graphs to find bad channels
    raw.interpolate_bads
    """

    ica.apply(raw)

    return raw

def bandpower(psds, freqs, band):
    fmin, fmax = band
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return psds[:, :, idx].mean(axis=2)

def load_all_data():

    bids_root = "ds003478"

    # Load metadata
    df_meta = pd.read_excel("ds003478/Data_4_Import_REST.xlsx", sheet_name="Depression Rest")
    participants_df = pd.read_csv("ds003478/participants.tsv", sep="\t")

    # Load mapping from BIDS participant name to real subject ID
    participants_df = participants_df[["participant_id", "Original_ID"]]
    participants_df["Original_ID"] = participants_df["Original_ID"].astype(int)
    df_meta["id"] = df_meta["id"].astype(int)

    # Merge to get IDs
    merged = df_meta.merge(participants_df, left_on="id", right_on="Original_ID")
    merged["subject"] = merged["participant_id"].str.replace("sub-", "")
    merged = merged.drop(columns=["participant_id", "Original_ID"])

    bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30)
}

    runs = ['01', '02']
    all_data = [] # storage of preprocessed data
    logs = []

    for i, row in merged.iterrows():
        subj = row["subject"]
        if int(row["id"]) in [516, 544]:
            continue

        for run in runs:
            print(f"Processing subject {subj} run {run}")
            # Loading and preprocessing
            try:
                bids_path = BIDSPath(root=bids_root, subject=subj, task='Rest', run=run, datatype='eeg', extension='.set')
                raw = read_raw_bids(bids_path, verbose=False)
                raw.load_data()

                raw = fix_mne_channel_metadata(raw)
                raw = preprocess(raw, subj)

                fig = raw.plot(show=False, n_channels=64, scalings='auto') # save plot
                plot_path = f"Plots/preprocessed_plot_subject{subj}_run{run}.png"
                fig.savefig(plot_path, dpi=300)

                save_path = f"derivatives/preprocessed/sub-{subj}_task-Rest_run-{run}_cleaned_raw.fif" # save raw data
                raw.save(save_path, overwrite=True)

                logs.append({ #logging successful processing
                    "subject": subj,
                    "run": run,
                    "status": "success",
                    "plot_file": plot_path,
                    "saved_file": save_path
                })

                # Store raw and metadata together
                entry = row.to_dict()
                entry["preprocessed"] = raw
                all_data.append(entry)

            except Exception as e:
                print(f"❌ Could not load subject {subj} run {run}: {e}")
                logs.append({ # logging errors
                    "subject": subj,
                    "run": run,
                    "status": "error",
                    "error": str(e)
                })
                
            # feature extraction 
            try:
                max_time = raw.times[-1]
                safe_tmax = min(370, max_time)
                if safe_tmax > 10:
                    raw.crop(tmin=10, tmax=safe_tmax)  # remove first 10s and last seconds for cleaner segments
                else:
                    raise ValueError(f"Not enough data after cropping for subject {subj}, run {run}") 
                
                epochs = mne.make_fixed_length_epochs(raw, duration=2.0)
                epochs.load_data()
                psds, freqs = epochs.compute_psd(fmin=1, fmax=40, method='welch').get_data(return_freqs=True)
                features = extract_all_features(raw, epochs, psds, freqs, subj, run, bands)
                entry.update(features)
            except Exception as e:
                print(f"❌ Could not extract features for subject {subj} run {run}: {e}")
                logs.append({
                    "subject": subj,
                    "run": run,
                    "status": "feature_extraction_error",
                    "error": str(e)
                })
                continue

    pd.DataFrame(logs).to_csv("logs/preprocessing_log.csv", index=False)
    if all_data:
        pd.DataFrame(all_data).drop(columns=["preprocessed"]).to_csv("all_metadata.csv", index=False)
        features = export_features_only(all_data, "features.csv")
    else:
        print("No data processed.")
    

    return features

# region ------ feature extraction ------

def extract_bandpowers(psds, freqs, bands):
    """
    Extracts average power for each band across all channels.
    """
    def bandpower(psds, freqs, band):
        fmin, fmax = band
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        return psds[:, :, idx].mean(axis=2)

    return {band: np.mean(bandpower(psds, freqs, rng), axis=0) for band, rng in bands.items()}

def compute_faa(alpha_power, ch_names, left='F3', right='F4'):
    """
    Computes frontal alpha asymmetry.
    """
    try:
        f3_idx = ch_names.index(left)
        f4_idx = ch_names.index(right)
        faa = np.log(alpha_power[f4_idx]) - np.log(alpha_power[f3_idx])
        return faa
    except ValueError:
        print(f"Channels {left} or {right} not found in {ch_names}. Cannot compute FAA.")
        return np.nan

def compute_spectral_entropy(raw, picks=None):
    """
    Computes normalized spectral entropy using antropy.
    """
    if picks is None:
        picks = mne.pick_types(raw.info, eeg=True)
    
    data = raw.get_data(picks=picks)
    sfreq = raw.info['sfreq']
    
    entropies = [spectral_entropy(sig, sf=sfreq, normalize=True) for sig in data]
    return np.mean(entropies)

def extract_all_features(raw, epochs, psds, freqs, subj, run, bands):
    """
    Extracts all features for a subject-run:
    - Band powers
    - Frontal Alpha Asymmetry (FAA)
    - Spectral Entropy
    """
    # Average PSDs across epochs
    psds_avg = np.mean(psds, axis=0)  # shape (n_channels, n_freqs)
    band_powers = extract_bandpowers(psds, freqs, bands)
    
    # FAA
    ch_names = raw.info['ch_names']
    faa = compute_faa(band_powers['alpha'], ch_names)

    # Entropy
    entropy = compute_spectral_entropy(raw)

    # Flatten band powers
    flat_powers = {}
    for band, powers in band_powers.items():
        for i, ch_power in enumerate(powers):
            key = f'{band}_ch_{ch_names[i]}'
            flat_powers[key] = ch_power

    # Combine all features
    features = {
        "subject": subj,
        "run": run,
        "faa": faa,
        "spectral_entropy": entropy,
        **flat_powers
    }
    return features

def export_features_only(all_data):
    """
    Extracts EEG features and metadata and saves to csv.
    """
    df = pd.DataFrame(all_data)

    if 'preprocessed' in df.columns:
        df = df.drop(columns=['preprocessed'])

    # Select columns
    eeg_features = [col for col in df.columns if (
        col.startswith(('delta_', 'theta_', 'alpha_', 'beta_')) or
        col in ['faa', 'spectral_entropy']
    )]
    metadata = ['subject', 'run', 'BDI', 'MDD', 'sex', 'age', 'HamD', 'BDI_Anh', 'BDI_Mel', 'TAI']
    
    selected_cols = [col for col in metadata if col in df.columns] + eeg_features
    final_df = df[selected_cols]
    final_df.to_csv("features.csv", index=False)
    return final_df

# endregion