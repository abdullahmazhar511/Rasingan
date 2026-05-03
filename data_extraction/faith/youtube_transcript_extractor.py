"""
Extract transcripts from YouTube playlist videos using Gemini AI.

This module provides functionality to:
1. Fetch video IDs from a YouTube playlist
2. Extract transcripts from each video
3. Process transcripts using Gemini API for insights/extraction
4. Save results to files
"""

import os
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging
import mimetypes

from google import genai
import yt_dlp

os.environ['GEMINI_API_KEY'] = "AIzaSyBlKnujodqvelyTn5zwtOOI8yrr6GsH5ec"
MODEL="gemini-3-flash-preview"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Gemini 3 Flash Batch pricing (per million tokens) - from https://ai.google.dev/gemini-api/docs/pricing
INPUT_COST_PER_MILLION = 0.50  # USD - Batch pricing for text/image/video
OUTPUT_COST_PER_MILLION = 3.00  # USD - Batch pricing
USD_TO_INR = 83.0  # Exchange rate

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


class YouTubePlaylistTranscriptExtractor:
    """Extract and process transcripts from YouTube playlists using Gemini."""
    
    def __init__(
        self, 
        gemini_api_key: Optional[str] = None, 
        output_dir: str = "extracted_transcripts",
        use_video: bool = True
    ):
        """
        Initialize the extractor.
        
        Args:
            gemini_api_key: Gemini API key. If None, will use GEMINI_API_KEY env var.
            output_dir: Directory to save extracted data.
            use_video: If True, download and process full video. If False, extract audio only.
        """
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not provided and not found in environment variables")
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=self.gemini_api_key)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_video = use_video
        self.processing_mode = "video" if use_video else "audio"
        
        self.transcripts_data = []
        self.processed_data = []
        
        logger.info(f"Initialized extractor in {self.processing_mode.upper()} mode")
    
    def get_playlist_video_ids(self, playlist_url: str) -> List[str]:
        """
        Extract video IDs from a YouTube playlist.
        
        Args:
            playlist_url: URL of the YouTube playlist
            
        Returns:
            List of video IDs
        """
        logger.info(f"Extracting video IDs from playlist: {playlist_url}")
        
        video_ids = []
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': 'in_playlist',
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(playlist_url, download=False)
                video_ids = [entry['id'] for entry in info.get('entries', [])]
            logger.info(f"Found {len(video_ids)} videos in playlist")
            return video_ids
        except Exception as e:
            logger.error(f"Error extracting playlist: {e}")
            return []
    
    def download_media(self, video_id: str, media_dir: str = "temp_media") -> Optional[str]:
        """
        Download media (video or audio) from a YouTube video.
        
        Args:
            video_id: YouTube video ID
            media_dir: Directory to store temporary media files
            
        Returns:
            Path to downloaded media file, or None if failed
        """
        Path(media_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            if self.use_video:
                # Download video in mp4 format
                media_path = os.path.join(media_dir, f"{video_id}.mp4")
                
                ydl_opts = {
                    'format': 'best[ext=mp4]',
                    'outtmpl': os.path.join(media_dir, f'{video_id}'),
                    'quiet': False,
                    'no_warnings': False,
                }
                
                logger.info(f"Downloading video for {video_id}...")
            else:
                # Download and extract audio to m4a format
                media_path = os.path.join(media_dir, f"{video_id}.m4a")
                
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'm4a',
                        'preferredquality': '192',
                    }],
                    'outtmpl': os.path.join(media_dir, f'{video_id}'),
                    'quiet': False,
                    'no_warnings': False,
                }
                
                logger.info(f"Downloading audio for {video_id}...")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"Attempting to download from: {video_url}")
                info = ydl.extract_info(video_url, download=True)
                logger.info(f"Successfully extracted info for {video_id}")
            
            logger.info(f"✓ Downloaded {self.processing_mode} for {video_id}")
            return media_path
            
        except Exception as e:
            logger.error(f"✗ Error downloading {self.processing_mode} for {video_id}: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def get_video_metadata(self, video_id: str) -> Optional[Dict]:
        """
        Extract video metadata (title, duration, etc.) from YouTube.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dict with video metadata or None if failed
        """
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
                    'duration': info.get('duration', 0),  # in seconds
                    'duration_formatted': self._format_duration(info.get('duration', 0)),
                    'uploader': info.get('uploader', 'Unknown'),
                    'upload_date': info.get('upload_date', 'Unknown'),
                    'view_count': info.get('view_count', 0),
                    'description': info.get('description', '')[:500],  # First 500 chars
                }
                
                logger.info(f"✓ Retrieved metadata for {video_id}: {metadata['title']}")
                return metadata
                
        except Exception as e:
            logger.error(f"✗ Error getting video metadata for {video_id}: {e}")
            return None
    
    def _format_duration(self, seconds: int) -> str:
        """Format duration in seconds to HH:MM:SS format."""
        if not seconds:
            return "00:00:00"
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"
    
    def extract_transcript(self, video_id: str) -> Optional[Dict]:
        """
        Extract and diarize transcript from a YouTube video using Gemini.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dict with video metadata, diarization in JSON format, and transcript details
        """
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        try:
            # Get video metadata first
            logger.info(f"Fetching metadata for video {video_id}...")
            metadata = self.get_video_metadata(video_id)
            if not metadata:
                logger.warning(f"Could not fetch metadata for {video_id}, continuing with processing...")
                metadata = {'title': 'Unknown', 'duration': 0, 'duration_formatted': '00:00:00'}
            
            # Use Gemini to transcribe and diarize directly from URL
            logger.info(f"Processing video {video_id} with Gemini...")
            logger.info(f"Video URL: {video_url}")
            
            diarization_prompt = f"""
            Please transcribe this YouTube video and perform speaker diarization.
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
            - Use HH:MM:SS format for timestamps
            - Identify speakers as either "Patient" or "Therapist" based on context
            - If both speakers are therapists, distinguish them appropriately
            - There are EXACTLY 2 speakers total
            - Each entry should have: start, speaker, text
            - Keep the text concise but complete
            - Maintain chronological order
            
            Video URL: {video_url}
            
            Return ONLY the JSON array, no other text.
            """
            
            response = self.client.models.generate_content(
                model=MODEL,
                contents=diarization_prompt
            )
            
            response_text = response.text.strip()
            
            # Extract usage data for cost tracking
            cost_data = {
                'input_tokens': getattr(response.usage_metadata, 'prompt_token_count', 0) if hasattr(response, 'usage_metadata') else 0,
                'output_tokens': getattr(response.usage_metadata, 'candidates_token_count', 0) if hasattr(response, 'usage_metadata') else 0,
            }
            
            # Try to parse JSON from response
            try:
                # Remove markdown code blocks if present
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                
                diarization = json.loads(response_text.strip())
                logger.info(f"✓ Parsed diarization with {len(diarization)} segments")
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON response, using raw text")
                diarization = response_text
            
            # Calculate total speakers
            if isinstance(diarization, list):
                speakers = set(item.get('speaker', 'Unknown') for item in diarization if isinstance(item, dict))
                num_speakers = len(speakers)
            else:
                num_speakers = 0
            
            # Calculate costs in USD and INR
            cost_calculation = calculate_cost(cost_data['input_tokens'], cost_data['output_tokens'])
            
            result = {
                'video_id': video_id,
                'url': video_url,
                'video_details': metadata,
                'diarization': diarization,  # JSON format if successful
                'num_speakers': num_speakers,
                'costs': {
                    'input_tokens': cost_data['input_tokens'],
                    'output_tokens': cost_data['output_tokens'],
                    'total_tokens': cost_data['input_tokens'] + cost_data['output_tokens'],
                    'input_cost_usd': cost_calculation['input_cost_usd'],
                    'output_cost_usd': cost_calculation['output_cost_usd'],
                    'total_cost_usd': cost_calculation['total_cost_usd'],
                    'total_cost_inr': cost_calculation['total_cost_inr']
                },
                'processing_mode': 'url',
                'extracted_at': datetime.now().isoformat(),
                'status': 'success'
            }
            
            logger.info(f"✓ Extracted diarization from {video_id} ({num_speakers} speakers)")
            logger.info(f"💰 Costs - Input: {cost_data['input_tokens']} tokens, Output: {cost_data['output_tokens']} tokens")
            logger.info(f"💰 Cost: ${cost_calculation['total_cost_usd']:.6f} USD | ₹{cost_calculation['total_cost_inr']:.2f} INR")
            return result
            
        except Exception as e:
            logger.error(f"✗ Error processing {video_id} with Gemini: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'video_id': video_id,
                'status': 'failed',
                'error': str(e),
                'extracted_at': datetime.now().isoformat()
            }
    
    def process_transcript_with_gemini(
        self, 
        diarization: Optional[List[Dict]], 
        video_id: str,
        extraction_prompt: Optional[str] = None
    ) -> Dict:
        """
        Process diarization using Gemini for extraction/analysis.
        
        Args:
            diarization: Diarization data (list of dicts or string)
            video_id: YouTube video ID
            extraction_prompt: Custom prompt for Gemini. If None, uses default summary prompt.
            
        Returns:
            Dict with processed content
        """
        if not extraction_prompt:
            extraction_prompt = """
            Please analyze the following YouTube video transcript and provide:
            1. A concise summary (2-3 sentences)
            2. Key topics discussed (as a list)
            3. Main takeaways (as a list)
            4. Any notable quotes or insights
            
            Transcript:
            """
        
        try:
            # Convert diarization to text format if it's a list
            if isinstance(diarization, list):
                transcript_text = ""
                for item in diarization:
                    if isinstance(item, dict):
                        speaker = item.get('speaker', 'Unknown')
                        text = item.get('text', '')
                        timestamp = item.get('start', '')
                        transcript_text += f"[{timestamp}] {speaker}: {text}\n"
                    else:
                        transcript_text += str(item) + "\n"
            else:
                transcript_text = str(diarization) if diarization else ""
            
            # Truncate transcript if too long (Gemini has token limits)
            max_transcript_length = 10000
            truncated_transcript = transcript_text[:max_transcript_length]
            if len(transcript_text) > max_transcript_length:
                truncated_transcript += "\n[... transcript truncated ...]"
            
            full_prompt = extraction_prompt + "\n" + truncated_transcript
            
            response = self.client.models.generate_content(
                model=MODEL,
                contents=full_prompt
            )
            
            logger.info(f"✓ Processed {video_id} with Gemini")
            
            return {
                'video_id': video_id,
                'gemini_response': response.text,
                'processed_at': datetime.now().isoformat(),
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"✗ Error processing {video_id} with Gemini: {e}")
            return {
                'video_id': video_id,
                'status': 'failed',
                'error': str(e),
                'processed_at': datetime.now().isoformat()
            }
    
    def extract_from_playlist(
        self, 
        playlist_url: str,
        extraction_prompt: Optional[str] = None,
        process_with_gemini: bool = True,
        max_videos: Optional[int] = None
    ) -> Dict:
        """
        Main method to extract transcripts from entire playlist.
        
        Args:
            playlist_url: URL of YouTube playlist
            extraction_prompt: Custom prompt for Gemini processing
            process_with_gemini: Whether to process transcripts with Gemini
            max_videos: Maximum number of videos to process. If None, process all videos.
            
        Returns:
            Summary dict with results
        """
        logger.info("=" * 60)
        logger.info("Starting YouTube Playlist Transcript Extraction")
        logger.info("=" * 60)
        
        # Step 1: Get video IDs
        all_video_ids = self.get_playlist_video_ids(playlist_url)
        if not all_video_ids:
            logger.error("No videos found in playlist")
            return {'status': 'failed', 'error': 'No videos found'}
        
        # Limit videos if max_videos is specified
        video_ids = all_video_ids[:max_videos] if max_videos else all_video_ids
        total_available = len(all_video_ids)
        total_to_process = len(video_ids)
        
        # Step 2: Extract transcripts
        if max_videos:
            logger.info(f"\nExtracting transcripts from {total_to_process} videos (of {total_available} available)...")
        else:
            logger.info(f"\nExtracting transcripts from {total_to_process} videos...")
        for i, video_id in enumerate(video_ids, 1):
            logger.info(f"[{i}/{len(video_ids)}] Processing {video_id}")
            
            transcript_data = self.extract_transcript(video_id)
            if transcript_data:
                self.transcripts_data.append(transcript_data)
                
                # Step 3: Process with Gemini if requested and transcript is available
                if process_with_gemini and transcript_data.get('status') == 'success':
                    processed = self.process_transcript_with_gemini(
                        transcript_data.get('diarization'),
                        video_id,
                        extraction_prompt
                    )
                    self.processed_data.append(processed)
        
        # Step 4: Save results
        self._save_results()
        
        # Return summary
        successful_transcripts = sum(1 for t in self.transcripts_data if t.get('status') == 'success')
        summary = {
            'status': 'completed',
            'total_videos_in_playlist': total_available,
            'videos_processed': total_to_process,
            'successful_transcripts': successful_transcripts,
            'failed_transcripts': total_to_process - successful_transcripts,
            'processed_with_gemini': len(self.processed_data),
            'output_files': {
                'transcripts': str(self.output_dir / 'transcripts.json'),
                'processed': str(self.output_dir / 'processed_gemini_analysis.json'),
                'summary_csv': str(self.output_dir / 'summary.csv')
            }
        }
        
        logger.info("\n" + "=" * 60)
        logger.info(f"✓ Extraction complete!")
        if max_videos:
            logger.info(f"  - Videos in playlist: {total_available}")
            logger.info(f"  - Videos processed: {total_to_process}")
        else:
            logger.info(f"  - Total videos: {total_to_process}")
        logger.info(f"  - Successful: {successful_transcripts}")
        logger.info(f"  - Failed: {total_to_process - successful_transcripts}")
        logger.info(f"  - Processed with Gemini: {len(self.processed_data)}")
        
        # Calculate and display total costs
        total_input_tokens = 0
        total_output_tokens = 0
        for transcript in self.transcripts_data:
            if transcript.get('status') == 'success':
                costs = transcript.get('costs', {})
                total_input_tokens += costs.get('input_tokens', 0)
                total_output_tokens += costs.get('output_tokens', 0)
        
        logger.info(f"  💰 Total API Costs:")
        logger.info(f"     - Input tokens: {total_input_tokens}")
        logger.info(f"     - Output tokens: {total_output_tokens}")
        logger.info(f"     - Total tokens: {total_input_tokens + total_output_tokens}")
        logger.info(f"  - Output directory: {self.output_dir}")
        logger.info("=" * 60)
        
        return summary
    
    def _save_results(self):
        """Save extracted and processed data to files."""
        # Save raw transcripts
        transcripts_file = self.output_dir / 'transcripts.json'
        with open(transcripts_file, 'w', encoding='utf-8') as f:
            json.dump(self.transcripts_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved transcripts to {transcripts_file}")
        
        # Save Gemini processed data
        if self.processed_data:
            processed_file = self.output_dir / 'processed_gemini_analysis.json'
            with open(processed_file, 'w', encoding='utf-8') as f:
                json.dump(self.processed_data, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Saved Gemini analysis to {processed_file}")
        
        # Save CSV summary
        self._save_summary_csv()
    
    def _save_summary_csv(self):
        """Save summary as CSV for easy viewing."""
        summary_file = self.output_dir / 'summary.csv'
        
        with open(summary_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Video ID', 'Title', 'Duration', 'URL', 'Speakers', 'Input Tokens', 'Output Tokens', 'Cost (USD)', 'Cost (INR)', 'Status', 'Extracted At', 'Error'])
            
            for transcript in self.transcripts_data:
                video_details = transcript.get('video_details', {})
                title = video_details.get('title', 'Unknown')
                duration = video_details.get('duration_formatted', '00:00:00')
                num_speakers = transcript.get('num_speakers', 0)
                costs = transcript.get('costs', {})
                input_tokens = costs.get('input_tokens', 0)
                output_tokens = costs.get('output_tokens', 0)
                total_cost_usd = costs.get('total_cost_usd', 0)
                total_cost_inr = costs.get('total_cost_inr', 0)
                
                writer.writerow([
                    transcript.get('video_id', ''),
                    title,
                    duration,
                    transcript.get('url', ''),
                    num_speakers,
                    input_tokens,
                    output_tokens,
                    f"${total_cost_usd:.6f}",
                    f"₹{total_cost_inr:.2f}",
                    transcript.get('status', ''),
                    transcript.get('extracted_at', ''),
                    transcript.get('error', '')
                ])
        
        logger.info(f"✓ Saved summary to {summary_file}")


def main():
    """Extract transcripts from YouTube playlist using command-line arguments."""
    
    parser = argparse.ArgumentParser(
        description='Extract transcripts from YouTube playlist videos using Gemini AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process default playlist
  python youtube_transcript_extractor.py
  
  # Process custom playlist
  python youtube_transcript_extractor.py "https://www.youtube.com/playlist?list=PLxxxx"
  
  # Process only first 5 videos in audio mode
  python youtube_transcript_extractor.py --max-videos 5 --audio-only
  
  # Custom output directory and disable Gemini processing
  python youtube_transcript_extractor.py "https://www.youtube.com/playlist?list=PLxxxx" --output transcripts/ --no-gemini
  
  # Process with custom extraction prompt
  python youtube_transcript_extractor.py --prompt "Extract all mentioned resources"
        """
    )
    
    parser.add_argument(
        'playlist_url',
        nargs='?',
        default='https://www.youtube.com/playlist?list=PLzTb3s1HQCQXL3EbHx3OaIG0G7XuXh9Da',
        help='YouTube playlist URL (default: PLzTb3s1HQCQXL3EbHx3OaIG0G7XuXh9Da)'
    )
    
    parser.add_argument(
        '--audio-only',
        action='store_true',
        default=False,
        help='Process audio only (default: process full video for better diarization)'
    )
    
    parser.add_argument(
        '--max-videos',
        type=int,
        default=None,
        help='Maximum number of videos to process from playlist (default: all)'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        default='extracted_transcripts',
        help='Output directory for results (default: extracted_transcripts)'
    )
    
    parser.add_argument(
        '--no-gemini',
        action='store_true',
        default=False,
        help='Skip Gemini processing, only extract transcripts'
    )
    
    parser.add_argument(
        '--prompt',
        '-p',
        default=None,
        help='Custom extraction prompt for Gemini processing'
    )
    
    parser.add_argument(
        '--api-key',
        default=None,
        help='Gemini API key (default: GEMINI_API_KEY env variable)'
    )
    
    args = parser.parse_args()
    
    # Validate playlist URL
    if 'youtube.com' not in args.playlist_url and 'youtu.be' not in args.playlist_url:
        parser.error('Invalid YouTube URL')
    
    # Get API key
    gemini_api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        parser.error('GEMINI_API_KEY not provided. Set environment variable or use --api-key')
    
    # Default extraction prompt
    default_prompt = """
    Analyze this content and extract:
    1. Main topic and purpose
    2. Key points discussed
    3. Important statistics or data mentioned
    4. Recommendations or conclusions
    5. Any resources or links mentioned
    """
    
    extraction_prompt = args.prompt or default_prompt
    use_video = not args.audio_only
    process_with_gemini = not args.no_gemini
    
    # Initialize extractor
    extractor = YouTubePlaylistTranscriptExtractor(
        gemini_api_key=gemini_api_key,
        output_dir=args.output,
        use_video=use_video
    )
    
    # Extract from playlist
    results = extractor.extract_from_playlist(
        playlist_url=args.playlist_url,
        extraction_prompt=extraction_prompt if process_with_gemini else None,
        process_with_gemini=process_with_gemini,
        max_videos=args.max_videos
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
