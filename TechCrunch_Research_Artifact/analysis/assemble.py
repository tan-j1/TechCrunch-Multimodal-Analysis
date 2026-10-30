"""Join video measures, inclusive-language scores and archival metadata by video_id."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRIC_COLUMNS = {
    'humor_A_attempts': 'Joke_Attempt',
    'humor_B_attempts': 'Joke_Attempt_second',
    'smile_duration_ratio': 'Smile_Duration_Ratio',
    'pocket_hands_ratio': 'Pocket_Hands_Ratio',
    'hand_gesture_frequency_per_min': 'Hand Gesture Frequency',
    'body_openness_mean': 'Body_Openness',
    'eye_contact_ratio': 'Eye_Contact_Ratio',
    'vocal_pitch_var_semitone_std': 'Vocal_Pitch_Var',
    'speech_rate_wpm': 'Speech_Rate',
    'pause_per_100_words': 'Pause_Count',
    'laughter_duration_ratio': 'Laughter_Duration_Ratio',
    'lead_speaking_time_s': 'overall__lead_speaking_time_s',
}
METADATA_COLUMNS = ['video_id', 'Company Name', 'Video Title', 'Video Link', 'Alum & Event',
                    'Money_Usd', 'Speaker_Gender',
                    'Judge_Male_Rate', 'text_valid']


def load_scores(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.csv' and path.is_file():
        scores = pd.read_csv(path)
    else:
        files = sorted(path.glob('*.json')) if path.is_dir() else [path]
        rows = []
        for file in files:
            item = json.loads(file.read_text(encoding='utf-8'))
            if isinstance(item, dict) and 'inclusive_collective_language' in item:
                rows.append({k: item[k] for k in ('video_id', 'inclusive_collective_language')})
        scores = pd.DataFrame(rows)
    if not {'video_id','inclusive_collective_language'}.issubset(scores):
        raise ValueError('Scores must contain video_id and inclusive_collective_language.')
    scores = scores[['video_id','inclusive_collective_language']].copy()
    if scores.video_id.isna().any() or not scores.video_id.is_unique:
        raise ValueError('Scores must have unique nonmissing video_id values.')
    values = pd.to_numeric(scores.inclusive_collective_language, errors='raise')
    if not values.isin([0,1,2]).all():
        raise ValueError('Inclusive collective language scores must be 0, 1, or 2.')
    scores['inclusive_collective_language'] = values
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True,
                        help='metrics_per_video.csv from pipeline.run')
    parser.add_argument('--scores', type=Path, required=True,
                        help='ICL score CSV, score JSON, or directory of score JSON files')
    parser.add_argument('--metadata', type=Path, default=ROOT/'data/analysis/input_428.csv',
                        help='CSV with funding, video coding and transcript-quality fields')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/analysis_input.csv')
    args = parser.parse_args()
    metadata = pd.read_csv(args.metadata)[METADATA_COLUMNS].copy()
    metrics = pd.read_csv(args.metrics)[['video_id', *METRIC_COLUMNS]].rename(columns=METRIC_COLUMNS)
    scores = load_scores(args.scores)
    for label, frame in [('metadata',metadata),('metrics',metrics)]:
        if frame.video_id.isna().any() or not frame.video_id.is_unique:
            raise ValueError(f'{label} must have unique nonmissing video_id values.')
    for label, frame in [('metrics',metrics),('scores',scores)]:
        unknown = set(frame.video_id) - set(metadata.video_id)
        if unknown:
            raise ValueError(f'{label} video_id absent from metadata: {sorted(unknown)}')
    data = metadata.merge(metrics, on='video_id', how='left', validate='one_to_one')
    data = data.merge(scores, on='video_id', how='left', validate='one_to_one')
    data['api_status'] = data.inclusive_collective_language.notna().map({True:'success',False:'excluded'})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    data.to_csv(args.output,index=False,float_format='%.17g')
    print(json.dumps({'records':len(data),'video_measures':len(metrics),'language_scores':len(scores),
                      'output':str(args.output.resolve())}))


if __name__ == '__main__':
    main()
