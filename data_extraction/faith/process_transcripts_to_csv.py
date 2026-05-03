import json
import csv
from pathlib import Path

def convert_transcripts_to_csv(transcripts_json_path, output_csv_path):
    """
    Convert transcripts.json to training data CSV format.
    
    Expected columns: Utterance, Type, ID
    Where:
    - Utterance: The spoken text
    - Type: T (Therapist) or P (Patient)
    - ID: video_id_segment_number
    """
    
    # Load transcripts
    with open(transcripts_json_path, 'r', encoding='utf-8') as f:
        transcripts = json.load(f)
    
    rows = []
    
    # Process each video's transcripts
    for video in transcripts:
        if video.get('status') != 'success':
            continue
        
        video_id = video.get('video_id')
        diarization = video.get('diarization', [])
        
        # Process each segment (utterance)
        for idx, segment in enumerate(diarization, 1):
            speaker = segment.get('speaker', 'Unknown')
            text = segment.get('text', '')
            
            # Map speaker to type (handle typos and variations)
            speaker_lower = speaker.lower()
            
            if any(term in speaker_lower for term in ['therapist', 'ther', 'counselor', 'therapist 2', 'co-therapist', 'primary', 'clinician']):
                speaker_type = 'T'
            elif any(term in speaker_lower for term in ['patient', 'client', 'participant']):
                speaker_type = 'P'
            else:
                # Unknown speaker - try to infer or skip
                print(f"Warning: Unknown speaker type '{speaker}' in {video_id}")
                speaker_type = 'U'
            
            # Create conversation ID (video_id_segment_number)
            conversation_id = f"{video_id}_{idx}"
            
            rows.append({
                'Utterance': text,
                'Type': speaker_type,
                'ID': conversation_id
            })
    
    # Write CSV
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Utterance', 'Type', 'ID'])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Converted {len(rows)} utterances")
    print(f"✓ Saved to {output_path}")
    
    # Print summary
    therapist_count = sum(1 for r in rows if r['Type'] == 'T')
    patient_count = sum(1 for r in rows if r['Type'] == 'P')
    print(f"\nSpeaker breakdown:")
    print(f"  Therapist (T): {therapist_count}")
    print(f"  Patient (P): {patient_count}")

if __name__ == '__main__':
    import sys
    
    # Default paths
    transcripts_path = 'extracted_transcripts_chunked/transcripts.json'
    output_path = 'faith_training_data.csv'
    
    # Allow command line overrides
    if len(sys.argv) > 1:
        transcripts_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    if not Path(transcripts_path).exists():
        print(f"✗ Error: {transcripts_path} not found")
        sys.exit(1)
    
    convert_transcripts_to_csv(transcripts_path, output_path)
