"""
YouTube Transcript Extractor with Chunking
Combines youtube2transcripts chunking approach with structured JSON diarization output.
Handles large videos by splitting into 15-minute chunks and processing concurrently.
"""

import os
import json
import logging
import argparse
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import warnings

import yt_dlp
from google import genai
from dotenv import load_dotenv
from pydub import AudioSegment
from tenacity import retry, stop_after_attempt, wait_exponential
from ratelimit import limits, sleep_and_retry

os.environ['GEMINI_API_KEY'] = "AIzaSyBlKnujodqvelyTn5zwtOOI8yrr6GsH5ec"
MODEL = "gemini-3-flash-preview"  # Use the latest Gemini 3 Flash model for best performance

# Suppress warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants for rate limiting and chunking
CALLS_PER_SECOND = 1.5  # Gemini rate limit buffer
PERIOD = 1
MAX_WORKERS = 5  # Concurrent jobs
CHUNK_DURATION_SECONDS = 15 * 60  # 15 minutes per chunk

# Gemini 3 Flash Batch pricing (per million tokens) - from https://ai.google.dev/gemini-api/docs/pricing
INPUT_COST_PER_MILLION = 1.00  # USD - Batch pricing for text/image/video
OUTPUT_COST_PER_MILLION = 3.00  # USD - Batch pricing
USD_TO_INR = 95.0  # Exchange rate

def calculate_cost(input_tokens: int, output_tokens: int, exchange_rate: float = USD_TO_INR) -> Dict:
    """Calculate cost in USD and INR."""
    input_cost_usd = (input_tokens / 1_000_000) * INPUT_COST_PER_MILLION
    output_cost_usd = (output_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION
    total_cost_usd = input_cost_usd + output_cost_usd
    total_cost_inr = total_cost_usd * exchange_rate
    
    return {
        'input_cost_usd': round(input_cost_usd, 6),
        'output_cost_usd': round(output_cost_usd, 6),
        'total_cost_usd': round(total_cost_usd, 6),
        'total_cost_inr': round(total_cost_inr, 2),
        'exchange_rate': exchange_rate
    }


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for filesystem and API compatibility (ASCII only, max 100 chars)."""
    import unicodedata
    
    # Remove or replace non-ASCII characters
    filename = unicodedata.normalize('NFKD', filename)
    filename = filename.encode('ascii', 'ignore').decode('ascii')
    
    # Remove invalid filesystem characters
    invalid_chars = '<>:"/\\|?*![]()\'&,'
    for char in invalid_chars:
        filename = filename.replace(char, '')
    
    # Replace spaces and dashes with underscores
    filename = filename.replace(' ', '_')
    filename = filename.replace('-', '_')
    
    # Remove any leading/trailing underscores
    filename = filename.strip('_')
    
    # Truncate to max 100 chars to avoid Windows path length issues
    if len(filename) > 100:
        filename = filename[:100]
    
    return filename


def download_audio(url: str, output_path: str = "/tmp/audio") -> Tuple[Optional[str], str]:
    """
    Download audio from YouTube URL.
    
    Args:
        url: YouTube video URL
        output_path: Directory to save audio file
        
    Returns:
        Tuple of (audio_file_path, video_title)
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    try:
        # Get video info
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Unknown')
            sanitized_title = sanitize_filename(title)
        
        # Check if file already exists
        expected_filepath = os.path.join(output_path, f"{sanitized_title}.mp3")
        if os.path.exists(expected_filepath):
            logger.info(f"✓ Audio already downloaded: {sanitized_title}.mp3")
            return expected_filepath, title
        
        # Download audio
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(output_path, sanitized_title + '.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"⏬ Downloading audio from: {url}")
            ydl.download([url])
        
        filename = f"{sanitized_title}.mp3"
        filepath = os.path.join(output_path, filename)
        logger.info(f"✓ Downloaded audio: {filename}")
        
        return filepath, title
        
    except Exception as e:
        logger.error(f"✗ Error downloading audio: {str(e)}")
        logger.debug(f"Exception details: {type(e).__name__}: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return None, "Unknown"


def split_audio(audio_path: str, chunk_duration: int) -> List[str]:
    """
    Split audio file into chunks of specified duration.
    
    Args:
        audio_path: Path to audio file
        chunk_duration: Duration of each chunk in seconds
        
    Returns:
        List of chunk file paths
    """
    try:
        logger.info(f"🔀 Splitting audio into {chunk_duration}s chunks...")
        audio = AudioSegment.from_mp3(audio_path)
        duration_ms = len(audio)
        chunk_duration_ms = chunk_duration * 1000
        
        chunks = []
        for i in range(0, duration_ms, chunk_duration_ms):
            chunk = audio[i:i + chunk_duration_ms]
            chunk_path = f"{audio_path}_chunk_{len(chunks)}.mp3"
            chunk.export(chunk_path, format="mp3")
            chunks.append(chunk_path)
        
        logger.info(f"✓ Created {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"✗ Error splitting audio: {str(e)}")
        logger.debug(f"Exception type: {type(e).__name__}")
        logger.debug(f"Audio file exists: {os.path.exists(audio_path)}, Size: {os.path.getsize(audio_path) if os.path.exists(audio_path) else 'N/A'} bytes")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return []





@sleep_and_retry
@limits(calls=CALLS_PER_SECOND, period=PERIOD)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def process_chunk_with_gemini(api_client: genai.Client, file_path: str, chunk_num: int, total_chunks: int) -> Tuple[str, Dict]:
    """Process audio chunk with Gemini for diarization. Returns (response_text, cost_data)."""
    try:
        logger.info(f"🤖 Processing chunk {chunk_num}/{total_chunks} with Gemini...")
        
        prompt = f"""Please transcribe this audio chunk and perform speaker diarization.
This is part {chunk_num} of {total_chunks}.
This appears to be a therapy or counseling session.

IMPORTANT: This video has exactly 2 speakers.
The speakers are either:
- A Patient and a Therapist, OR
- Two Therapists

Identify each speaker as follows:
- "Patient" - The person seeking therapy/advice (if present)
- "Therapist" - The professional providing therapy/guidance
- If there are 2 therapists, label them as "Therapist" and "Therapist 2" (or use context like "Primary Therapist" vs "Co-Therapist" if possible)

Return the diarization as a JSON array in this exact format:
[
  {{"start": "00:00", "speaker": "Therapist", "text": "text here"}},
  {{"start": "00:05", "speaker": "Patient", "text": "text here"}},
  {{"start": "00:10", "speaker": "Therapist", "text": "text here"}}
]

Rules:
- Use MM:SS format for timestamps (relative to this chunk)
- Identify speakers as either "Patient" or "Therapist" based on context
- If both speakers are therapists, distinguish them appropriately
- There are EXACTLY 2 speakers total
- Each entry should have: start, speaker, text
- Keep the text concise but complete
- Maintain chronological order

Return ONLY the JSON array, no other text."""
        
        # Upload the file
        logger.debug(f"📤 Uploading chunk: {os.path.basename(file_path)}")
        myfile = api_client.files.upload(file=file_path)
        logger.debug(f"✓ Upload complete: {os.path.basename(file_path)}")
        
        # Generate content with the uploaded file
        response = api_client.models.generate_content(
            model=MODEL,
            contents=[prompt, myfile]
        )
        
        # Extract usage data
        cost_data = {
            'input_tokens': getattr(response.usage_metadata, 'prompt_token_count', 0) if hasattr(response, 'usage_metadata') else 0,
            'output_tokens': getattr(response.usage_metadata, 'candidates_token_count', 0) if hasattr(response, 'usage_metadata') else 0,
        }
        
        return response.text, cost_data
        
    except Exception as e:
        logger.error(f"✗ Processing failed for chunk {chunk_num}: {str(e)}")
        raise


def get_video_metadata(video_id: str) -> Optional[Dict]:
    """Extract video metadata from YouTube."""
    try:
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            
            metadata = {
                'title': info.get('title', 'Unknown'),
                'duration': info.get('duration', 0),
                'duration_formatted': format_duration(info.get('duration', 0)),
                'uploader': info.get('uploader', 'Unknown'),
                'upload_date': info.get('upload_date', 'Unknown'),
                'view_count': info.get('view_count', 0),
                'description': info.get('description', '')[:500],
            }
            
            logger.info(f"✓ Retrieved metadata: {metadata['title']}")
            return metadata
            
    except Exception as e:
        logger.error(f"✗ Error getting metadata: {e}")
        return None


def format_duration(seconds: int) -> str:
    """Format duration in seconds to HH:MM:SS."""
    if not seconds:
        return "00:00:00"
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"


def parse_diarization_response(response_text: str) -> List[Dict]:
    """
    Parse Gemini response to extract diarization JSON.
    
    Args:
        response_text: Raw response from Gemini
        
    Returns:
        List of diarization segments
    """
    try:
        # Remove markdown code blocks if present
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        diarization = json.loads(text.strip())
        
        if isinstance(diarization, list):
            return diarization
        else:
            logger.warning("Response was not JSON array format")
            return []
            
    except json.JSONDecodeError as e:
        logger.error(f"✗ Failed to parse JSON response: {e}")
        return []


def process_audio_file_full(
    api_client: genai.Client,
    audio_path: str,
    video_id: str,
    original_title: str
) -> Tuple[Optional[List[Dict]], Dict]:
    """
    Process entire audio file without chunking.
    
    Args:
        api_client: Gemini client
        audio_path: Path to audio file
        video_id: YouTube video ID
        original_title: Original video title
        
    Returns:
        Tuple of (diarization, cost_data)
    """
    try:
        logger.info(f"Processing full audio without chunking...")
        
        # Create a prompt for the full audio (without chunk context)
        prompt = f"""Please transcribe this audio and perform speaker diarization.
This appears to be a therapy or counseling session.

IMPORTANT: This video has exactly 2 speakers.
The speakers are either:
- A Patient and a Therapist, OR
- Two Therapists

Identify each speaker as follows:
- "Patient" - The person seeking therapy/advice
- "Therapist" - The professional providing therapy/guidance (use Therapist for both if there are 2 therapists)

Return the diarization as a JSON array in this exact format:
[
  {{"start": "00:00", "speaker": "Therapist", "text": "text here"}},
  {{"start": "00:05", "speaker": "Patient", "text": "text here"}},
  {{"start": "00:10", "speaker": "Therapist", "text": "text here"}}
]

Rules:
- Use HH:MM:SS format for timestamps
- Identify speakers as either "Patient" or "Therapist" based on context
- There are EXACTLY 2 speakers total
- Each entry should have: start, speaker, text
- Keep the text concise but complete
- Maintain chronological order
- Do NOT create speaker numbers or labels like "Therapist 1" or "Therapist 2"

Return ONLY the JSON array, no other text."""
        
        # Upload the file
        logger.debug(f"📤 Uploading audio: {os.path.basename(audio_path)}")
        myfile = api_client.files.upload(file=audio_path)
        logger.debug(f"✓ Upload complete: {os.path.basename(audio_path)}")
        
        # Generate content with the uploaded file
        response = api_client.models.generate_content(
            model=MODEL,
            contents=[prompt, myfile]
        )
        
        # Extract usage data
        cost_data = {
            'input_tokens': getattr(response.usage_metadata, 'prompt_token_count', 0) if hasattr(response, 'usage_metadata') else 0,
            'output_tokens': getattr(response.usage_metadata, 'candidates_token_count', 0) if hasattr(response, 'usage_metadata') else 0,
        }
        
        # Parse response
        diarization = parse_diarization_response(response.text)
        logger.info(f"✓ Processed full audio: {len(diarization)} segments (tokens: in={cost_data.get('input_tokens', 0)}, out={cost_data.get('output_tokens', 0)})")
        
        return diarization, cost_data
        
    except Exception as e:
        logger.error(f"✗ Error processing full audio: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, {'input_tokens': 0, 'output_tokens': 0}


def process_audio_file_chunked(
    api_client: genai.Client,
    audio_path: str,
    video_id: str,
    original_title: str
) -> Tuple[Optional[List[Dict]], Dict]:
    """
    Process audio file by splitting into chunks and processing each.
    
    Args:
        api_client: Gemini client
        audio_path: Path to audio file
        video_id: YouTube video ID
        original_title: Original video title
        
    Returns:
        Tuple of (combined_diarization, cost_data)
    """
    try:
        # Split audio into chunks
        chunks = split_audio(audio_path, CHUNK_DURATION_SECONDS)
        if not chunks:
            logger.error("Failed to split audio")
            return None, {'input_tokens': 0, 'output_tokens': 0}
        
        all_diarizations = []
        total_cost = {'input_tokens': 0, 'output_tokens': 0}
        
        # Process each chunk
        for chunk_num, chunk_path in enumerate(chunks, 1):
            try:
                logger.info(f"Processing chunk {chunk_num}/{len(chunks)}...")
                
                # Process with Gemini (returns text and cost data)
                response_text, cost_data = process_chunk_with_gemini(api_client, chunk_path, chunk_num, len(chunks))
                
                # Parse diarization
                diarization = parse_diarization_response(response_text)
                all_diarizations.extend(diarization)
                
                # Accumulate costs
                total_cost['input_tokens'] += cost_data.get('input_tokens', 0)
                total_cost['output_tokens'] += cost_data.get('output_tokens', 0)
                
                logger.info(f"✓ Processed chunk {chunk_num}: {len(diarization)} segments (tokens: in={cost_data.get('input_tokens', 0)}, out={cost_data.get('output_tokens', 0)})")
                
                # Clean up chunk
                try:
                    os.remove(chunk_path)
                except:
                    pass
                    
            except Exception as e:
                logger.error(f"✗ Error processing chunk {chunk_num}: {e}")
                continue
        
        if all_diarizations:
            logger.info(f"✓ Total segments from all chunks: {len(all_diarizations)}")
            logger.info(f"💰 Total API usage: {total_cost['input_tokens']} input tokens, {total_cost['output_tokens']} output tokens")
            return all_diarizations, total_cost
        else:
            return None, total_cost
            
    except Exception as e:
        logger.error(f"✗ Error processing audio: {str(e)}")
        logger.debug(f"Exception type: {type(e).__name__}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return None, {'input_tokens': 0, 'output_tokens': 0}


def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL."""
    try:
        import re
        patterns = [
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
            r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    except:
        return None


def is_playlist_url(url: str) -> bool:
    """Check if URL is a YouTube playlist."""
    try:
        return 'playlist?list=' in url or '/playlist' in url
    except:
        return False


def get_playlist_video_ids(playlist_url: str) -> List[str]:
    """Extract video IDs from a YouTube playlist."""
    try:
        logger.info(f"📋 Extracting video IDs from playlist...")
        
        video_ids = []
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': 'in_playlist',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)
            video_ids = [entry['id'] for entry in info.get('entries', [])]
        
        logger.info(f"✓ Found {len(video_ids)} videos in playlist")
        return video_ids
        
    except Exception as e:
        logger.error(f"✗ Error extracting playlist: {e}")
        return []


def process_youtube_url_chunked(
    api_client: genai.Client,
    url: str,
    output_dir: Path,
    keep_audio: bool = False,
    no_chunking: bool = False
) -> Optional[Dict]:
    """
    Main processing function for a single YouTube URL.
    
    Args:
        api_client: Gemini client
        url: YouTube URL
        output_dir: Output directory
        keep_audio: Whether to keep downloaded audio
        no_chunking: If True, process full audio without chunking; otherwise use chunking
        
    Returns:
        Result dictionary with video metadata and diarization
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {url}")
        logger.info('='*60)
        
        # Extract video ID
        video_id = extract_video_id(url)
        if not video_id:
            logger.error(f"✗ Invalid YouTube URL: {url}")
            return None
        
        # Get metadata
        logger.info("📋 Fetching video metadata...")
        metadata = get_video_metadata(video_id)
        if not metadata:
            metadata = {'title': 'Unknown', 'duration': 0, 'duration_formatted': '00:00:00'}
        
        # Download audio
        logger.info("⏬ Downloading audio...")
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path, title = download_audio(url, temp_dir)
            
            if not audio_path:
                logger.error(f"✗ Failed to download audio")
                return {
                    'video_id': video_id,
                    'url': url,
                    'status': 'failed',
                    'error': 'Failed to download audio',
                    'processed_at': datetime.now().isoformat()
                }
            
            # Validate audio file
            if not os.path.exists(audio_path):
                logger.error(f"✗ Audio file not found: {audio_path}")
                return {
                    'video_id': video_id,
                    'url': url,
                    'status': 'failed',
                    'error': 'Failed to download audio (file not found)',
                    'processed_at': datetime.now().isoformat()
                }
            
            file_size = os.path.getsize(audio_path)
            logger.info(f"✓ Audio file size: {file_size} bytes")
            if file_size == 0:
                logger.error(f"✗ Audio file is empty")
                return {
                    'video_id': video_id,
                    'url': url,
                    'status': 'failed',
                    'error': 'Audio file is empty',
                    'processed_at': datetime.now().isoformat()
                }
            
            # Process audio (with or without chunking)
            if no_chunking:
                logger.info("🔄 Processing full audio without chunking...")
                diarization, costs = process_audio_file_full(api_client, audio_path, video_id, title)
            else:
                logger.info("🔄 Processing with chunking...")
                diarization, costs = process_audio_file_chunked(api_client, audio_path, video_id, title)
            
            if diarization is None:
                return {
                    'video_id': video_id,
                    'url': url,
                    'status': 'failed',
                    'error': 'Failed to process audio',
                    'video_details': metadata,
                    'costs': costs,
                    'processed_at': datetime.now().isoformat()
                }
            
            # Calculate stats
            speakers = set(item.get('speaker', 'Unknown') for item in diarization if isinstance(item, dict))
            num_speakers = len(speakers)
            
            # Calculate costs in USD and INR
            cost_calculation = calculate_cost(costs.get('input_tokens', 0), costs.get('output_tokens', 0))
            
            result = {
                'video_id': video_id,
                'url': url,
                'video_details': metadata,
                'diarization': diarization,
                'num_speakers': num_speakers,
                'speakers': list(speakers),
                'num_segments': len(diarization),
                'processing_mode': 'chunked',
                'costs': {
                    'input_tokens': costs.get('input_tokens', 0),
                    'output_tokens': costs.get('output_tokens', 0),
                    'total_tokens': costs.get('input_tokens', 0) + costs.get('output_tokens', 0),
                    'input_cost_usd': cost_calculation['input_cost_usd'],
                    'output_cost_usd': cost_calculation['output_cost_usd'],
                    'total_cost_usd': cost_calculation['total_cost_usd'],
                    'total_cost_inr': cost_calculation['total_cost_inr']
                },
                'processed_at': datetime.now().isoformat(),
                'status': 'success'
            }
            
            logger.info(f"✓ Successfully processed: {num_speakers} speakers, {len(diarization)} segments")
            logger.info(f"💰 Costs - Input: {costs.get('input_tokens', 0)} tokens, Output: {costs.get('output_tokens', 0)} tokens")
            logger.info(f"💰 Cost: ${cost_calculation['total_cost_usd']:.6f} USD | ₹{cost_calculation['total_cost_inr']:.2f} INR")
            
            return result
            
    except Exception as e:
        logger.error(f"✗ Error processing URL: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'video_id': extract_video_id(url) or 'unknown',
            'url': url,
            'status': 'failed',
            'error': str(e),
            'processed_at': datetime.now().isoformat()
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Extract diarized transcripts from YouTube videos using chunking')
    parser.add_argument('youtube_url', nargs='?', help='YouTube video URL or playlist URL')
    parser.add_argument('--urls', '-u', help='File containing YouTube URLs (one per line)')
    parser.add_argument('--playlist', '-p', help='YouTube playlist URL')
    parser.add_argument('--max-videos', type=int, default=None, help='Maximum number of videos to process from playlist')
    parser.add_argument('--output', '-o', default='extracted_transcripts_chunked', help='Output directory')
    parser.add_argument('--api-key', help='Gemini API key (or set GEMINI_API_KEY env var)')
    parser.add_argument('--keep-audio', action='store_true', help='Keep downloaded audio files')
    parser.add_argument('--no-chunking', '--full-audio', action='store_true', help='Process full audio without chunking')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, help='Number of concurrent workers')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Load environment
    load_dotenv()
    api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.error("✗ GEMINI_API_KEY not found. Set via --api-key or environment variable.")
        return
    
    # Initialize client
    client = genai.Client(api_key=api_key)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get URLs to process
    urls = []
    
    # Handle playlist argument
    if args.playlist:
        playlist_video_ids = get_playlist_video_ids(args.playlist)
        if args.max_videos:
            playlist_video_ids = playlist_video_ids[:args.max_videos]
        urls.extend([f"https://www.youtube.com/watch?v={vid}" for vid in playlist_video_ids])
        logger.info(f"Processing {len(urls)} videos from playlist")
    
    # Handle positional URL (could be video or playlist)
    elif args.youtube_url:
        if is_playlist_url(args.youtube_url):
            playlist_video_ids = get_playlist_video_ids(args.youtube_url)
            if args.max_videos:
                playlist_video_ids = playlist_video_ids[:args.max_videos]
            urls.extend([f"https://www.youtube.com/watch?v={vid}" for vid in playlist_video_ids])
            logger.info(f"Processing {len(urls)} videos from playlist")
        else:
            urls.append(args.youtube_url)
    
    # Handle URLs from file
    if args.urls:
        try:
            with open(args.urls, 'r') as f:
                file_urls = [line.strip() for line in f if line.strip()]
                urls.extend(file_urls)
        except Exception as e:
            logger.error(f"✗ Error reading URLs file: {e}")
            return
    
    if not urls:
        parser.print_help()
        return
    
    # Load existing transcripts to skip only successful videos, but retry failed ones
    transcripts_file = output_dir / 'transcripts.json'
    successful_video_ids = set()
    failed_video_ids = set()
    
    if transcripts_file.exists():
        try:
            with open(transcripts_file, 'r', encoding='utf-8') as f:
                existing_transcripts = json.load(f)
                for t in existing_transcripts:
                    video_id = t.get('video_id')
                    if video_id:
                        if t.get('status') == 'success':
                            successful_video_ids.add(video_id)
                        elif t.get('status') == 'failed':
                            failed_video_ids.add(video_id)
            
            logger.info(f"✓ Found {len(successful_video_ids)} successful + {len(failed_video_ids)} failed transcripts")
        except Exception as e:
            logger.warning(f"⚠ Could not load existing transcripts: {e}")
    
    # Filter URLs: skip successful, retry failed, process new
    urls_to_process = []
    skipped_count = 0
    retry_count = 0
    
    for url in urls:
        video_id = extract_video_id(url)
        if video_id in successful_video_ids:
            logger.info(f"⊘ Skipping (successful): {video_id}")
            skipped_count += 1
        elif video_id in failed_video_ids:
            logger.info(f"🔄 Retrying (previously failed): {video_id}")
            urls_to_process.append(url)
            retry_count += 1
        else:
            urls_to_process.append(url)
    
    if skipped_count > 0 or retry_count > 0:
        logger.info(f"Skipped {skipped_count} | Retrying {retry_count} | Processing {len(urls_to_process) - retry_count} new videos")
    
    if not urls_to_process:
        logger.info("ℹ All videos already processed successfully. Nothing to do.")
        return
    
    logger.info(f"Processing {len(urls_to_process)} URL(s) with {args.workers} worker(s)")
    
    # Process URLs
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_youtube_url_chunked,
                client,
                url,
                output_dir,
                args.keep_audio,
                args.no_chunking
            ): url for url in urls_to_process
        }
        
        for future in as_completed(futures):
            url = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"✗ Exception for {url}: {e}")
    
    # Save results (append to existing)
    if results:
        # Load existing results if they exist
        all_results = []
        existing_count = 0
        if transcripts_file.exists():
            try:
                with open(transcripts_file, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
                    existing_count = len(all_results)
            except Exception as e:
                logger.warning(f"⚠ Could not load existing results: {e}")
        
        # Deduplicate by video_id (keep latest)
        results_by_id = {r.get('video_id'): r for r in all_results}  # Existing results
        for result in results:
            video_id = result.get('video_id')
            if video_id in results_by_id:
                logger.info(f"↻ Updating {video_id} (was {results_by_id[video_id].get('status')} → now {result.get('status')})")
            results_by_id[video_id] = result  # New results override old ones
        
        # Convert back to list
        all_results = list(results_by_id.values())
        new_additions = len(all_results) - existing_count
        
        # Save combined results
        with open(transcripts_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved {len(results)} new + {existing_count - len(results) + new_additions} existing = {len(all_results)} total transcripts to {transcripts_file}")
        
        # Calculate total costs
        total_input_tokens = 0
        total_output_tokens = 0
        for result in results:
            if result.get('status') == 'success':
                costs = result.get('costs', {})
                total_input_tokens += costs.get('input_tokens', 0)
                total_output_tokens += costs.get('output_tokens', 0)
        
        # Save CSV summary (rewrite entire file to avoid duplicates)
        summary_file = output_dir / 'summary.csv'
        
        with open(summary_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['Video ID', 'Title', 'Duration', 'URL', 'Speakers', 'Segments', 'Input Tokens', 'Output Tokens', 'Cost (USD)', 'Cost (INR)', 'Status', 'Processed At'])
            
            # Write all results (deduped from all_results)
            for result in all_results:
                if result.get('status') == 'success':
                    video_details = result.get('video_details', {})
                    costs = result.get('costs', {})
                    writer.writerow([
                        result.get('video_id'),
                        video_details.get('title'),
                        video_details.get('duration_formatted'),
                        result.get('url'),
                        result.get('num_speakers', 0),
                        result.get('num_segments', 0),
                        costs.get('input_tokens', 0),
                        costs.get('output_tokens', 0),
                        f"${costs.get('total_cost_usd', 0):.6f}",
                        f"₹{costs.get('total_cost_inr', 0):.2f}",
                        result.get('status'),
                        result.get('processed_at')
                    ])
        
        logger.info(f"✓ Saved summary to {summary_file}")
    
    logger.info(f"\n{'='*60}")
    logger.info("Processing complete!")
    logger.info(f"Output directory: {output_dir}")
    if results:
        successful = sum(1 for r in results if r.get('status') == 'success')
        logger.info(f"Successful: {successful}/{len(results)}")
        
        # Calculate total costs
        total_cost_usd = sum(r.get('costs', {}).get('total_cost_usd', 0) for r in results if r.get('status') == 'success')
        total_cost_inr = sum(r.get('costs', {}).get('total_cost_inr', 0) for r in results if r.get('status') == 'success')
        
        logger.info(f"💰 Total API Costs:")
        logger.info(f"   Input tokens: {total_input_tokens}")
        logger.info(f"   Output tokens: {total_output_tokens}")
        logger.info(f"   Total tokens: {total_input_tokens + total_output_tokens}")
        logger.info(f"   Total cost: ${total_cost_usd:.6f} USD | ₹{total_cost_inr:.2f} INR")
    logger.info('='*60)


if __name__ == '__main__':
    main()
