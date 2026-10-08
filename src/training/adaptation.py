"""Adapt the alphabet baseline; evaluate on complete held-out webcam recordings."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from config import settings
from src.dataset.dataset_utils import FEATURE_COLUMNS, feature_key
from src.training.trainer import load_training_data, save_training_run
from src.training.evaluator import evaluate_predictions
from src.recognition.hand_canonicalization import VERSION, canonicalize_batch


def adapt_model(imported, webcam, output, baseline, seed=42, canonical=False):
    external, external_info = load_training_data(Path(imported), practice=True)
    local, local_info = load_training_data(Path(webcam), practice=True, minimum_classes=1)
    if external_info['schema'].get('dataset_kind') != 'external_images' or local_info['schema'].get('dataset_kind') != 'alphabet_webcam':
        raise ValueError('Use imported alphabet and separate webcam alphabet datasets.')
    # Randomly choose one complete recording per targeted letter, reproducibly.
    rng=np.random.default_rng(seed)
    held_out=[]
    for label, rows in local.groupby('label', sort=True):
        sessions=sorted(rows.session_id.unique())
        if len(sessions)<2:
            raise ValueError(f'{label} needs at least two separate webcam recordings before adaptation.')
        held_out.append(str(rng.choice(sessions)))
    test=local[local.session_id.isin(held_out)].copy()
    raw_test = test[FEATURE_COLUMNS].to_numpy(dtype=float)
    local_train=local[~local.session_id.isin(held_out)].copy()
    train=pd.concat([external,local_train],ignore_index=True)
    if canonical:
        train[FEATURE_COLUMNS] = canonicalize_batch(train[FEATURE_COLUMNS].to_numpy(dtype=float))
        test[FEATURE_COLUMNS] = canonicalize_batch(raw_test)
    test_keys={feature_key(x) for x in test[FEATURE_COLUMNS].to_numpy(dtype=float)}
    # Withhold whole imported families matching a test vector, including variants.
    keys=[feature_key(x) for x in train[FEATURE_COLUMNS].to_numpy(dtype=float)]
    overlap=np.array([key in test_keys for key in keys])
    matching_groups=set(train.loc[overlap,'source_group_id'].dropna())
    remove=overlap | train.source_group_id.isin(matching_groups).to_numpy()
    removed=int(remove.sum())
    train=train.loc[~remove].copy()
    keys=[feature_key(x) for x in train[FEATURE_COLUMNS].to_numpy(dtype=float)]
    feature_labels={}
    for key,label in zip(keys,train.label):
        if key in feature_labels and feature_labels[key]!=label:
            raise ValueError('Imported/webcam features have conflicting labels; inspect the data before training.')
        feature_labels[key]=label
    unique=~pd.Series(keys).duplicated().to_numpy()
    train=train.loc[unique].copy()
    if set(train.label)!=set(settings.ALPHABET_LABELS):
        raise ValueError('Combined training must retain every static alphabet letter.')
    if not set(test.label)<=set(train.label):
        raise ValueError('A held-out letter has no training examples.')
    model=RandomForestClassifier(n_estimators=settings.RANDOM_FOREST_TREES,min_samples_leaf=settings.RANDOM_FOREST_MIN_SAMPLES_LEAF,random_state=seed,n_jobs=-1)
    model.fit(train[FEATURE_COLUMNS].to_numpy(dtype=np.float32),train.label.to_numpy())
    x=test[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    evaluation=evaluate_predictions(test.label.to_numpy(),model.predict(x),model.classes_.tolist())
    from src.recognition.predictor import LivePredictor
    old=LivePredictor(Path(baseline))
    old_x = canonicalize_batch(raw_test) if old.transform == VERSION else raw_test
    baseline_evaluation=evaluate_predictions(test.label.to_numpy(),old.model.predict(old_x),old.classes)
    metadata={'model_type':'RandomForestClassifier','practice':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),
              'schema':external_info['schema'],'feature_count':126,'labels':model.classes_.tolist(),
              'sklearn_version':sklearn.__version__,'split_mode':'webcam_session_holdout',
              'training_samples':len(train),'testing_samples':len(test),'test_accuracy':evaluation['accuracy'],
              'test_scope':'Only collected webcam letters, same signer unless separately assessed',
              'held_out_sessions':held_out,'held_out_labels':sorted(test.label.unique()),
              'excluded_training_rows_matching_test':removed,'training_duplicates_removed':int((~unique).sum()),
              'imported_dataset':str(Path(imported).resolve()),'webcam_dataset':str(Path(webcam).resolve()),
              'imported_sha256':external_info['dataset_sha256'],'webcam_sha256':local_info['dataset_sha256'],
              'baseline_model_sha256':old.metadata['model_sha256'],'baseline_accuracy_on_same_test':baseline_evaluation['accuracy'],
              'session_overlap':0,'random_state':seed,'model_parameters':model.get_params(),
              'baseline_evaluation_on_same_test':baseline_evaluation}
    if canonical:
        metadata['runtime_transform'] = VERSION
    metadata['held_out_hand_counts'] = {
        'left_only': int(((raw_test[:,:63].any(axis=1)) & (~raw_test[:,63:].any(axis=1))).sum()),
        'right_only': int(((~raw_test[:,:63].any(axis=1)) & (raw_test[:,63:].any(axis=1))).sum()),
        'both': int((raw_test[:,:63].any(axis=1) & raw_test[:,63:].any(axis=1)).sum())}
    save_training_run(model,metadata,evaluation,Path(output))
    print(f'Baseline on held-out webcam recordings: {baseline_evaluation["accuracy"]:.1%}')
    print(f'Adapted on SAME recordings: {evaluation["accuracy"]:.1%}')
    print(f'Test letters: {", ".join(metadata["held_out_labels"])} | test rows: {len(test)}')
    print('These scores do not establish full-alphabet webcam or unseen-signer accuracy.')
    print(f'Saved: {output}')
    return metadata
