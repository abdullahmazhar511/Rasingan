# YouTube Playlist Transcript Extractor with Gemini Diarization

Extract transcripts from YouTube video playlists with automatic speaker diarization using Google's Gemini AI. Gemini processes the audio directly for transcription and speaker identification.

## Features

- **Dual Processing Modes**: Choose between full video or audio-only processing
- **Playlist Processing**: Automatically fetch all videos from a YouTube playlist
- **Media Download**: Extract video or audio from each video (based on mode)
- **Gemini Diarization**: Use Gemini AI to transcribe and identify speakers
- **Speaker Identification**: Automatic speaker diarization with dialogue organization
- **Batch Processing**: Handle multiple videos efficiently with logging
- **Multiple Output Formats**: Save results as JSON and CSV
- **Error Handling**: Graceful handling of download failures or processing errors

## Features

- **Playlist Processing**: Automatically fetch all videos from a YouTube playlist
- **Transcript Extraction**: Download transcripts from each video (supports multiple languages)
- **Gemini AI Processing**: Analyze transcripts with Gemini for summaries, key points, and custom extraction
- **Batch Processing**: Handle multiple videos efficiently with logging
- **Multiple Output Formats**: Save results as JSON and CSV
- **Error Handling**: Graceful handling of disabled transcripts or unavailable videos

## Processing Modes

### Video Mode (Default: `use_video=True`)
**Pros:**
- Better diarization - Gemini can use visual cues and lip-reading hints
- Single speaker identification across scenes
- Better context understanding
- Can detect speaker changes by visual cues

**Cons:**
- Larger file sizes (requires more disk space)
- Longer download times
- Slower processing
- Higher bandwidth usage

**Best for:** Interviews, panel discussions, presentations where accurate speaker identification is critical

### Audio Mode (`use_video=False`)
**Pros:**
- Smaller file sizes
- Faster downloads
- Quicker processing
- Lower bandwidth usage
- Works well for single-speaker or clear audio content

**Cons:**
- Less accurate speaker diarization (audio-only)
- More challenging for complex multi-speaker scenarios
- May miss visual context cues

**Best for:** Podcasts, lectures, solo commentary, or when speed/bandwidth is priority

### How to Choose

**Use VIDEO mode (default) for:**
- Multi-speaker interviews or discussions
- High accuracy diarization needed
- Videos with visual speaker changes

```bash
# Process with video (default)
python youtube_transcript_extractor.py "https://www.youtube.com/playlist?list=PLxxxxxxxx"
```

**Use AUDIO mode for:**
- Single speaker or clear audio separation
- Need for faster processing
- Limited storage/bandwidth
- Podcasts or audio-focused content

```bash
# Process audio only
python youtube_transcript_extractor.py "https://www.youtube.com/playlist?list=PLxxxxxxxx" --audio-only
```

## Installation
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use `choco install ffmpeg`
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `apt-get install ffmpeg` (Ubuntu/Debian)

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Gemini API key:
   - Get your API key from [Google AI Studio](https://aistudio.google.com/apikey)
   - Set as environment variable:
   ```bash
   # Windows PowerShell
   $env:GEMINI_API_KEY = "your_api_key_here"
   
   # Windows Command Prompt
   set GEMINI_API_KEY=your_api_key_here
   
   # Linux/Mac
   export GEMINI_API_KEY="your_api_key_here"
   ```

## Quick Start

### Basic Usage
Process all videos from default playlist (video mode):

```bash
# Uses default playlist
python youtube_transcript_extractor.py

# Or specify custom playlist
python youtube_transcript_extractor.py "https://www.youtube.com/playlist?list=PLxxxxxxxx"
```

### Audio Mode Only
Process only audio (faster, lighter files):

```bash
python youtube_transcript_extractor.py --audio-only
```

### Limit Number of Videos
Process only first N videos (useful for testing or budget control):

```bash
python youtube_transcript_extractor.py --max-videos 5
```

### Custom Output Directory
Specify where to save results:

```bash
python youtube_transcript_extractor.py --output my_transcripts/
```

### Skip Gemini Processing
Extract transcripts only without additional Gemini analysis:

```bash
python youtube_transcript_extractor.py --no-gemini
```

### Custom Extraction Prompt
Provide custom instructions for Gemini processing:

```bash
python youtube_transcript_extractor.py --prompt "Extract all mentioned resources and links"
```

### Combine Options
```bash
python youtube_transcript_extractor.py \
  --audio-only \
  --max-videos 10 \
  --output results/ \
  --prompt "Find all action items mentioned"
```

### Use Custom Playlist
```bash
python youtube_transcript_extractor.py "https://www.youtube.com/playlist?list=PLxxxxxxxx"
```

```

## Usage Examples

### Extract Only Transcripts (No Gemini Processing)

```bash
python youtube_transcript_extractor.py "https://www.youtube.com/playlist?list=PLxxxxxxxx" --no-gemini
```

### Audio Mode with Custom Output Directory

```bash
python youtube_transcript_extractor.py "https://www.youtube.com/playlist?list=PLxxxxxxxx" \
  --audio-only \
  --output transcripts/my_project
```

### Video Mode (Full Processing)

```bash
python youtube_transcript_extractor.py "https://www.youtube.com/playlist?list=PLxxxxxxxx" \
  --output transcripts/my_project
```

### Test with 3 Videos Before Full Processing

```bash
# Quick test with 3 videos
python youtube_transcript_extractor.py "https://www.youtube.com/playlist?list=PLxxxxxxxx" --max-videos 3

# If successful, process all
python youtube_transcript_extractor.py "https://www.youtube.com/playlist?list=PLxxxxxxxx"
```

## Limiting Playlist Processing

Use the `--max-videos` flag to process only a subset of videos from a playlist:

```bash
# Process only first 5 videos
python youtube_transcript_extractor.py "https://www.youtube.com/playlist?list=PLxxxxxxxx" --max-videos 5

# Output will show:
# Extracting transcripts from 5 videos (of 50 available)...
```

**Use cases for `--max-videos`:**
- **Testing**: Process a few videos first to verify setup works
- **Budget Control**: Limit API costs by processing only N videos
- **Quick Preview**: Get a sample of insights from large playlists
- **Development**: Test workflow before processing entire playlist
- **Resource Management**: Distribute processing across multiple runs

## Output Files

The extractor creates the following files in the output directory:

1. **transcripts.json**: Diarized transcripts with speaker identification and metadata
   ```json
   {
     "video_id": "...",
     "url": "https://www.youtube.com/watch?v=...",
     "transcript": "Speaker 1: ...\nSpeaker 2: ...",
     "transcript_length": 5000,
     "diarized": true,
     "processing_mode": "video",
     "status": "success"
   }
   ```

2. **processed_gemini_analysis.json**: Additional Gemini analysis and insights (if enabled)
   ```json
   {
     "video_id": "...",
     "gemini_response": "Summary and analysis...",
     "status": "success"
   }
   ```

3. **summary.csv**: Quick reference table
   ```
   Video ID,URL,Status,Transcript Length,Extracted At
   dQw4w9WgXcQ,https://www.youtube.com/watch?v=dQw4w9WgXcQ,success,5000,...
   ```

## Command-Line Arguments

```bash
usage: youtube_transcript_extractor.py [-h] [--audio-only] [--max-videos MAX_VIDEOS] 
                                       [--output OUTPUT] [--no-gemini] 
                                       [--prompt PROMPT] [--api-key API_KEY]
                                       playlist_url

Extract transcripts from YouTube playlist videos using Gemini AI

positional arguments:
  playlist_url          YouTube playlist URL (optional, default: PLzTb3s1HQCQXL3EbHx3OaIG0G7XuXh9Da)

optional arguments:
  -h, --help           Show this help message and exit
  
  --audio-only         Process audio only (default: process full video for better 
                       diarization)
  
  --max-videos MAX_VIDEOS
                       Maximum number of videos to process from playlist (default: all)
  
  --output OUTPUT, -o OUTPUT
                       Output directory for results (default: extracted_transcripts)
  
  --no-gemini          Skip Gemini processing, only extract transcripts
  
  --prompt PROMPT, -p PROMPT
                       Custom extraction prompt for Gemini processing
  
  --api-key API_KEY    Gemini API key (default: GEMINI_API_KEY env variable)
```

## API Reference

### Python API

If you want to use the extractor in your own Python code:

```python
from youtube_transcript_extractor import YouTubePlaylistTranscriptExtractor

# Initialize with custom settings
extractor = YouTubePlaylistTranscriptExtractor(
    gemini_api_key="your_api_key",  # Or set GEMINI_API_KEY env variable
    output_dir="transcripts",
    use_video=True  # False for audio-only mode
)

# Extract from playlist
results = extractor.extract_from_playlist(
    playlist_url="https://www.youtube.com/playlist?list=PLxxxxxxxx",
    extraction_prompt="Custom prompt here",
    process_with_gemini=True,
    max_videos=10  # Process only first 10 videos
)
```

### Class Methods

#### `__init__(gemini_api_key: Optional[str] = None, output_dir: str = "extracted_transcripts", use_video: bool = True)`
Initialize the extractor.

**Parameters:**
- `gemini_api_key`: Gemini API key. If None, uses GEMINI_API_KEY env var.
- `output_dir`: Directory to save extracted data. Default: "extracted_transcripts"
- `use_video`: Processing mode. If True (default), downloads and processes full video. If False, extracts and processes audio only.

#### `get_playlist_video_ids(playlist_url: str) -> List[str]`
Extract video IDs from a YouTube playlist.

#### `download_media(video_id: str, media_dir: str = "temp_media") -> Optional[str]`
Download media (video or audio) from a YouTube video based on use_video setting.

#### `extract_transcript(video_id: str) -> Optional[Dict]`
Extract and diarize transcript from a video using Gemini.
- Downloads media (video or audio based on mode)
- Uploads to Gemini for processing
- Returns speaker-diarized transcript with processing mode info

#### `process_transcript_with_gemini(transcript: str, video_id: str, extraction_prompt: Optional[str] = None) -> Dict`
Perform additional analysis on the diarized transcript using Gemini AI.

#### `extract_from_playlist(playlist_url: str, extraction_prompt: Optional[str] = None, process_with_gemini: bool = True, max_videos: Optional[int] = None) -> Dict`
Main method to extract and process entire playlist (or subset).

**Parameters:**
- `playlist_url`: YouTube playlist URL
- `extraction_prompt`: Custom prompt for additional Gemini analysis
- `process_with_gemini`: Whether to process transcripts with Gemini
- `max_videos`: Maximum number of videos to process from playlist. If None (default), processes all videos.

## Logging

The extractor provides detailed logging output:

```
2024-05-03 10:30:45,123 - INFO - Starting YouTube Playlist Transcript Extraction
2024-05-03 10:30:46,456 - INFO - Found 25 videos in playlist
2024-05-03 10:30:47,789 - INFO - ✓ Extracted transcript from dQw4w9WgXcQ (5000 chars)
2024-05-03 10:30:52,345 - INFO - ✓ Processed dQw4w9WgXcQ with Gemini
```

## Error Handling

The extractor handles common errors gracefully:

- **Transcripts Disabled**: Videos with disabled transcripts are marked as failed with reason
- **No Transcript Available**: Auto-generated transcripts may not be available for all videos
- **API Errors**: Gemini API errors are caught and logged
- **Invalid Playlist**: Invalid playlist URLs are handled with clear error messages

## Environment Variables

Set these for production use:

```bash
GEMINI_API_KEY=your_api_key_here
```

## Troubleshooting

### "GEMINI_API_KEY not provided"
- Ensure your API key is set as an environment variable
- Or pass it via command line: `python youtube_transcript_extractor.py <url> --api-key "your_key"`

### "FFmpeg not found"
- Install FFmpeg on your system (see Installation section)
- Verify installation: `ffmpeg -version`
- Add FFmpeg to system PATH if necessary

### "Error downloading media for video"
- Check that the video is publicly available
- Ensure you have sufficient disk space for media files
- Try the specific video URL manually to verify access
- Video mode requires more space than audio-only mode

### "Diarization quality is poor"
- Videos with multiple speakers separated clearly work best
- Background noise can affect accuracy
- Very quiet or distorted audio may produce unclear results
- Video mode generally provides better diarization than audio-only
- Try switching between video and audio modes: `--audio-only` flag

### "Processing is too slow"
- Consider using audio-only mode for faster processing: `--audio-only`
- Video mode takes longer but provides better diarization
- Process fewer videos at once: `--max-videos 5`
- Check your internet speed for media downloads

### Rate limiting from Gemini
- Wait a moment and retry
- Consider processing fewer videos at once with `--max-videos`
- Check your Gemini API quota at [aistudio.google.com](https://aistudio.google.com)

## Limitations

- YouTube playlist URL must be valid and public
- FFmpeg must be installed on the system for media extraction
- Gemini API has rate limits and token limits
- Diarization quality depends on audio clarity and number of distinct speakers
- Very long videos may be truncated during processing due to token limits
- Video processing takes more time and bandwidth than audio-only mode
- Audio-only mode may have less accurate diarization but faster processing

## License

MIT
