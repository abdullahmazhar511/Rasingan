# YouTube Transcript Extractor - Two Approaches

This folder contains two scripts for extracting diarized transcripts from YouTube videos, each optimized for different use cases.

## Scripts Overview

### 1. `youtube_transcript_extractor.py` - Direct URL Processing
**Best for:** Short to medium-length videos (< 30 mins), simple API-based approach

**Features:**
- Sends YouTube URLs directly to Gemini (no downloads)
- No file upload/download overhead
- Faster for short videos
- Minimal dependencies
- Single-threaded processing
- JSON format diarization output with video metadata

**Output Structure:**
```json
{
  "video_id": "...",
  "url": "...",
  "video_details": {
    "title": "...",
    "duration_formatted": "HH:MM:SS",
    "uploader": "...",
    "upload_date": "..."
  },
  "diarization": [
    {"start": "00:00", "speaker": "Speaker 1", "text": "Welcome"},
    {"start": "00:05", "speaker": "Speaker 2", "text": "Thanks"}
  ],
  "num_speakers": 2,
  "status": "success"
}
```

**Usage:**
```bash
# Single video
python youtube_transcript_extractor.py "https://www.youtube.com/watch?v=..."

# Playlist with limit
python youtube_transcript_extractor.py "https://www.youtube.com/playlist?list=..." --max-videos 5

# Custom output directory
python youtube_transcript_extractor.py "https://..." --output my_transcripts
```

**Pros:**
- ✅ Simpler implementation
- ✅ No file downloads needed
- ✅ Faster for short videos
- ✅ Lower bandwidth usage
- ✅ Works with Gemini URL parameter

**Cons:**
- ❌ May struggle with very long videos (30+ mins)
- ❌ Single-threaded (processes one at a time)
- ❌ Gemini has token limits


### 2. `youtube_transcript_extractor_chunked.py` - Chunked Processing
**Best for:** Long videos (30+ mins), batch processing, concurrent operations

**Features:**
- Downloads audio from YouTube
- Splits audio into 15-minute chunks
- Processes chunks concurrently (configurable workers)
- Handles files larger than Gemini upload limit
- Combines results from multiple chunks
- Rate-limited Gemini API calls
- Multi-threaded with ThreadPoolExecutor
- Supports batch URL processing
- JSON format diarization with video metadata

**Output Structure:** Same as above

**Usage:**
```bash
# Single video
python youtube_transcript_extractor_chunked.py "https://www.youtube.com/watch?v=..."

# Batch processing from file
python youtube_transcript_extractor_chunked.py --urls urls.txt

# Specify workers
python youtube_transcript_extractor_chunked.py "https://..." --workers 4

# Keep audio files
python youtube_transcript_extractor_chunked.py "https://..." --keep-audio

# Custom output directory
python youtube_transcript_extractor_chunked.py "https://..." --output my_transcripts
```

**Pros:**
- ✅ Handles very long videos (1+ hour)
- ✅ Concurrent processing (batch multiple URLs)
- ✅ Better for production workflows
- ✅ Chunk handling for large files
- ✅ Rate limiting built-in
- ✅ Batch URL processing

**Cons:**
- ❌ Requires audio download (slower for short videos)
- ❌ Needs FFmpeg installed
- ❌ More dependencies (pydub)
- ❌ More complex code


## Comparison Table

| Feature | Direct URL | Chunked |
|---------|-----------|---------|
| Short videos (<15 min) | ⭐⭐⭐ | ⭐⭐ |
| Long videos (30+ min) | ⭐ | ⭐⭐⭐ |
| Multiple videos | ⭐⭐ | ⭐⭐⭐ |
| Speed for short videos | Fast | Slower |
| File size handling | Limited | Unlimited |
| Concurrent processing | No | Yes |
| Bandwidth usage | Low | High |
| Dependencies | Minimal | FFmpeg, pydub |
| Code complexity | Simple | Complex |
| Error recovery | Basic | Advanced |


## Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Additional for Chunked Version
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg

# Or use conda
conda install ffmpeg
```

## Environment Setup

Set your Gemini API key:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or pass via command line:
```bash
python youtube_transcript_extractor.py "https://..." --api-key "your-api-key"
```

## Output Files

Both scripts generate:
- `transcripts.json` - Complete transcript data with metadata
- `summary.csv` - Quick summary table
- Logs printed to console

## Performance Notes

**Direct URL Approach:**
- ~5-30 seconds per video
- Limited by Gemini token window
- Best for videos < 30 minutes

**Chunked Approach:**
- ~30-120 seconds per video (includes download + processing)
- Can handle any length video
- 15-min chunks processed concurrently
- Rate limited to ~2 calls/second


## Troubleshooting

### "GEMINI_API_KEY not found"
```bash
export GEMINI_API_KEY="your-key"
# Or
python youtube_transcript_extractor.py "https://..." --api-key "your-key"
```

### FFmpeg not found (Chunked version)
Install FFmpeg - see Installation section above

### Gemini API errors
Check API quota and rate limits - both scripts implement retry logic

### Large file upload fails (Chunked version)
Reduce CHUNK_DURATION_SECONDS in the script (currently 15 minutes)

## Example Output Structure

```json
{
  "video_id": "dQw4w9WgXcQ",
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "video_details": {
    "title": "Never Gonna Give You Up",
    "duration": 212,
    "duration_formatted": "00:03:32",
    "uploader": "Rick Astley",
    "upload_date": "20090101",
    "view_count": 1000000000,
    "description": "..."
  },
  "diarization": [
    {
      "start": "00:00",
      "speaker": "Speaker 1",
      "text": "Never gonna give you up"
    },
    {
      "start": "00:05",
      "speaker": "Speaker 1",
      "text": "Never gonna let you down"
    }
  ],
  "num_speakers": 1,
  "speakers": ["Speaker 1"],
  "num_segments": 15,
  "status": "success",
  "processed_at": "2026-05-03T10:30:45.123456"
}
```

## When to Use Each

**Use `youtube_transcript_extractor.py`:**
- YouTube Shorts or short clips (<15 mins)
- Quick API testing
- Minimal overhead needed
- Limited bandwidth
- Playlist extraction (default behavior)

**Use `youtube_transcript_extractor_chunked.py`:**
- Long-form content (podcasts, lectures, streams)
- Batch processing multiple videos
- Need concurrent processing
- Production workflows
- Video archives


## Additional Notes

- Both scripts use the new Google Gemini 2 Flash model
- JSON output is always pretty-printed for readability
- Video metadata is fetched via yt-dlp
- Diarization format is standardized across both
- All timestamps are ISO format for compatibility
- Error handling and logging included
