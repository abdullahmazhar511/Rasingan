"""
YouTube Transcript Extractor - Direct YouTube URL Method
Uses Gemini API's native YouTube URL support for direct video processing.
Based on working sample.py pattern.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import warnings

import yt_dlp
from google import genai
from google.genai import types
from dotenv import load_dotenv

os.environ['GEMINI_API_KEY'] = "AIzaSyBlKnujodqvelyTn5zwtOOI8yrr6GsH5ec"
MODEL = "gemini-3-flash-preview"

# Suppress warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
MAX_WORKERS = 2

# Pricing (per million tokens)
INPUT_COST_PER_MILLION = 0.50  # USD
OUTPUT_COST_PER_MILLION = 3.00  # USD
USD_TO_INR = 83.0


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


def format_duration(seconds: int) -> str:
    """Format duration in seconds to HH:MM:SS."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL."""
    try:
        if 'youtu.be' in url:
            return url.split('/')[-1].split('?')[0]
        elif 'youtube.com' in url:
            return url.split('v=')[1].split('&')[0]
    except Exception:
        pass
    return None


def parse_diarization_response(response_text: str) -> List[Dict]:
    """Parse JSON diarization response from Gemini."""
    try:
        # Try to extract JSON array from response
        if '[' in response_text and ']' in response_text:
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            json_str = response_text[json_start:json_end]
            return json.loads(json_str)
    except Exception as e:
        logger.error(f"✗ Error parsing response: {e}")
    return []


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
                'channel_id': info.get('channel_id', 'Unknown'),
            }
            
            return metadata
    except Exception as e:
        logger.error(f"✗ Error fetching metadata: {e}")
        return None


def is_playlist_url(url: str) -> bool:
    """Check if URL is a playlist URL."""
    return 'playlist?list=' in url


def get_playlist_video_ids(playlist_url: str) -> List[str]:
    """Extract video IDs from playlist."""
    try:
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


def process_youtube_url_direct(api_client: genai.Client, url: str) -> tuple:
    """
    Process YouTube video directly using Gemini API.
    Uses file_data with YouTube URL - no download or upload required.
    
    Args:
        api_client: Gemini client
        url: YouTube video URL
        
    Returns:
        Tuple of (diarization_list, costs_dict)
    """
    try:
        logger.info(f"🤖 Processing YouTube video directly (no download/upload)...")
        
        prompt = """Please transcribe this video and perform speaker diarization.
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
  {"start": "00:00:00", "speaker": "Therapist", "text": "text here"},
  {"start": "00:00:05", "speaker": "Patient", "text": "text here"},
  {"start": "00:00:10", "speaker": "Therapist", "text": "text here"}
]

Rules:
- Use HH:MM:SS format for timestamps
- Identify speakers as either "Patient" or "Therapist" based on context
- If both speakers are therapists, distinguish them appropriately
- There are EXACTLY 2 speakers total
- Each entry should have: start, speaker, text
- Keep the text concise but complete
- Maintain chronological order

Return ONLY the JSON array, no other text."""
        
        # Create content with YouTube URL (working pattern from sample.py)
        response = api_client.models.generate_content(
            model=MODEL,
            contents=types.Content(
                parts=[
                    types.Part(
                        file_data=types.FileData(file_uri=url)
                    ),
                    types.Part(text=prompt)
                ]
            )
        )
        
        # Extract usage data
        cost_data = {
            'input_tokens': getattr(response.usage_metadata, 'prompt_token_count', 0) if hasattr(response, 'usage_metadata') else 0,
            'output_tokens': getattr(response.usage_metadata, 'candidates_token_count', 0) if hasattr(response, 'usage_metadata') else 0,
        }
        
        # Parse response
        diarization = parse_diarization_response(response.text)
        logger.info(f"✓ Processed video: {len(diarization)} segments (tokens: in={cost_data.get('input_tokens', 0)}, out={cost_data.get('output_tokens', 0)})")
        
        return diarization, cost_data
        
    except Exception as e:
        logger.error(f"✗ Error processing video: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def process_youtube_url(
    api_client: genai.Client,
    url: str,
    output_dir: Path
) -> Optional[Dict]:
    """
    Main processing function for a single YouTube URL.
    
    Args:
        api_client: Gemini client
        url: YouTube URL
        output_dir: Output directory
        
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
        logger.info("📹 Fetching video metadata...")
        metadata = get_video_metadata(video_id)
        if not metadata:
            logger.error(f"✗ Failed to get video metadata")
            return {
                'video_id': video_id,
                'url': url,
                'status': 'failed',
                'error': 'Failed to get video metadata',
                'processed_at': datetime.now().isoformat()
            }
        
        logger.info(f"✓ Video: {metadata['title']}")
        logger.info(f"  Duration: {metadata['duration_formatted']}")
        logger.info(f"  Uploader: {metadata['uploader']}")
        
        # Process video directly
        diarization, costs = process_youtube_url_direct(api_client, url)
        
        if diarization is None:
            return {
                'video_id': video_id,
                'url': url,
                'status': 'failed',
                'error': 'Failed to process video',
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
            'speakers': list(speakers),
            'num_speakers': num_speakers,
            'diarization': diarization,
            'tokens': {
                'input': costs.get('input_tokens', 0),
                'output': costs.get('output_tokens', 0)
            },
            'costs': {
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
    parser = argparse.ArgumentParser(description='Extract diarized transcripts from YouTube videos using direct URL method')
    parser.add_argument('youtube_url', nargs='?', help='YouTube video URL or playlist URL')
    parser.add_argument('--urls', '-u', help='File containing YouTube URLs (one per line)')
    parser.add_argument('--playlist', '-p', help='YouTube playlist URL')
    parser.add_argument('--max-videos', type=int, default=None, help='Maximum number of videos to process from playlist')
    parser.add_argument('--output', '-o', default='extracted_transcripts_direct', help='Output directory')
    parser.add_argument('--api-key', help='Gemini API key (or set GEMINI_API_KEY env var)')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, help='Number of concurrent workers')
    
    args = parser.parse_args()
    
    # Setup API client
    api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.error("✗ Gemini API key not found. Set GEMINI_API_KEY environment variable or use --api-key")
        return
    
    client = genai.Client(api_key=api_key)
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect URLs
    urls = []
    
    # From playlist argument
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
                logger.info(f"Processing {len(file_urls)} URLs from file")
        except Exception as e:
            logger.error(f"✗ Error reading URLs file: {e}")
            return
    
    if not urls:
        parser.print_help()
        return
    
    logger.info(f"Processing {len(urls)} URL(s) with {args.workers} worker(s)")
    
    # Process URLs
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_youtube_url,
                client,
                url,
                output_dir
            ): url for url in urls
        }
        
        for future in as_completed(futures):
            url = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"✗ Exception for {url}: {e}")
    
    # Save results
    if results:
        transcripts_file = output_dir / 'transcripts.json'
        with open(transcripts_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved {len(results)} transcripts to {transcripts_file}")
        
        # Create CSV summary
        csv_file = output_dir / 'summary.csv'
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Video ID', 'Title', 'Duration', 'URL', 'Speakers',
                'Input Tokens', 'Output Tokens', 'Cost (USD)', 'Cost (INR)',
                'Status', 'Extracted At', 'Error'
            ])
            
            for result in results:
                if result['status'] == 'success':
                    details = result.get('video_details', {})
                    tokens = result.get('tokens', {})
                    costs = result.get('costs', {})
                    
                    writer.writerow([
                        result.get('video_id', ''),
                        details.get('title', ''),
                        details.get('duration_formatted', ''),
                        result.get('url', ''),
                        ', '.join(result.get('speakers', [])),
                        tokens.get('input', 0),
                        tokens.get('output', 0),
                        f"${costs.get('total_cost_usd', 0):.6f}",
                        f"₹{costs.get('total_cost_inr', 0):.2f}",
                        'success',
                        result.get('processed_at', ''),
                        ''
                    ])
                else:
                    writer.writerow([
                        result.get('video_id', ''),
                        '',
                        '',
                        result.get('url', ''),
                        '',
                        '',
                        '',
                        '',
                        '',
                        'failed',
                        result.get('processed_at', ''),
                        result.get('error', '')
                    ])
        
        logger.info(f"✓ Saved summary to {csv_file}")
        
        # Print summary
        successful = [r for r in results if r['status'] == 'success']
        if successful:
            total_input = sum(r.get('tokens', {}).get('input', 0) for r in successful)
            total_output = sum(r.get('tokens', {}).get('output', 0) for r in successful)
            total_cost_usd = sum(r.get('costs', {}).get('total_cost_usd', 0) for r in successful)
            total_cost_inr = sum(r.get('costs', {}).get('total_cost_inr', 0) for r in successful)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"Successfully processed: {len(successful)}/{len(results)}")
            logger.info(f"Total input tokens: {total_input:,}")
            logger.info(f"Total output tokens: {total_output:,}")
            logger.info(f"Total cost: ${total_cost_usd:.6f} USD | ₹{total_cost_inr:.2f} INR")


if __name__ == '__main__':
    main()
