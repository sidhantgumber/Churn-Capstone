"""
Text-to-Speech Module using Edge TTS
Converts customer insights to high-quality speech
"""

import edge_tts
import asyncio
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSGenerator:
    """Text-to-Speech generator using Edge TTS"""
    
    def __init__(self, voice="en-US-AriaNeural"):
        """
        Initialize TTS with voice selection
        
        Popular voices:
        - en-US-AriaNeural (Professional female)
        - en-US-JennyNeural (Conversational female)  
        - en-US-GuyNeural (Professional male)
        - en-US-DavisNeural (Casual male)
        """
        self.voice = voice
        self.output_dir = Path("audio_output")
        self.output_dir.mkdir(exist_ok=True)
    
    async def text_to_speech_async(self, text: str, output_filename: str = None) -> str:
        """
        Convert text to speech asynchronously
        
        Args:
            text: Text to convert to speech
            output_filename: Optional custom filename
            
        Returns:
            Path to generated audio file
        """
        try:
            if not output_filename:
                output_filename = "customer_insights_audio.mp3"
            
            output_path = self.output_dir / output_filename
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(str(output_path))
            
            logger.info(f"Audio generated successfully: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"TTS Error: {str(e)}")
            raise Exception(f"Failed to generate audio: {str(e)}")
    
    def text_to_speech(self, text: str, output_filename: str = None) -> str:
        """
        Synchronous wrapper for text_to_speech_async
        
        Args:
            text: Text to convert to speech
            output_filename: Optional custom filename
            
        Returns:
            Path to generated audio file
        """
        return asyncio.run(self.text_to_speech_async(text, output_filename))
    
    def read_file_and_generate_audio(self, file_path: str, output_filename: str = None) -> str:
        """
        Read text file and convert to speech
        
        Args:
            file_path: Path to text file
            output_filename: Optional custom audio filename
            
        Returns:
            Path to generated audio file
        """
        try:
            # Read the insights file
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            # Clean up the text for better speech
            cleaned_text = self.clean_text_for_speech(text_content)
            
            # Generate audio
            return self.text_to_speech(cleaned_text, output_filename)
            
        except FileNotFoundError:
            raise Exception(f"File not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error reading file and generating audio: {str(e)}")
    
    def clean_text_for_speech(self, text: str) -> str:
        """
        Clean text to make it more suitable for text-to-speech
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text optimized for speech
        """

        text = text.replace("="*80, "")
        text = text.replace("="*50, "")
        text = text.replace("CUSTOMER INTELLIGENCE INSIGHTS", "Customer Intelligence Insights.")
        text = text.replace("**", "")  
        text = text.replace("##", "")  
        text = text.replace("•", "Point:")
        text = text.replace("-", "Point:")
        
        import re
        text = re.sub(r'\b(\d+)\.\s*\*\*([^*]+)\*\*', r'Point \1: \2.', text)
        
        text = re.sub(r'\n\s*\n', '. ', text)
        text = re.sub(r'\s+', ' ', text)
        
        text = text.replace("$", "dollars ")
        text = text.replace("%", " percent")
        text = text.replace("&", " and ")
        
        return text.strip()
    
    def get_available_voices(self) -> list:
        """
        Get list of available Edge TTS voices
        
        Returns:
            List of available voice names
        """
        try:
            return [
                "en-US-AriaNeural",
                "en-US-JennyNeural", 
                "en-US-GuyNeural",
                "en-US-DavisNeural",
                "en-US-JasonNeural",
                "en-US-NancyNeural"
            ]
        except Exception as e:
            logger.error(f"Error getting voices: {str(e)}")
            return ["en-US-AriaNeural"]


def generate_audio_from_text(text: str, voice: str = "en-US-AriaNeural", output_file: str = None) -> str:
    """
    Convenience function to generate audio from text
    
    Args:
        text: Text to convert
        voice: Voice to use
        output_file: Output filename
        
    Returns:
        Path to generated audio file
    """
    tts = TTSGenerator(voice)
    return tts.text_to_speech(text, output_file)

def generate_audio_from_file(file_path: str, voice: str = "en-US-AriaNeural", output_file: str = None) -> str:
    """
    Convenience function to generate audio from text file
    
    Args:
        file_path: Path to text file
        voice: Voice to use  
        output_file: Output filename
        
    Returns:
        Path to generated audio file
    """
    tts = TTSGenerator(voice)
    return tts.read_file_and_generate_audio(file_path, output_file)

# if __name__ == "__main__":
#     tts = TTSGenerator()
#     sample_text = """
#     Customer Intelligence Insights:
    
#     1. Sales Performance: Total sales of $462 million show strong performance with an increasing trend.
    
#     2. Customer Segments: The largest segment is Dormant Regulars at 28.5% of customers, requiring re-engagement strategies.
    
#     3. Churn Risk: At 20.1%, the churn rate represents medium risk requiring immediate attention.
#     """
    
#     try:
#         print("Testing TTS generation...")
#         audio_path = tts.text_to_speech(sample_text, "test_insights.mp3")
#         print(f"Audio generated: {audio_path}")
        
#         # Test file reading if insights file exists
#         insights_file = "customer_insights_mistral.txt"
#         if os.path.exists(insights_file):
#             print("Testing file-based TTS...")
#             file_audio_path = tts.read_file_and_generate_audio(insights_file, "insights_from_file.mp3")
#             print(f"File audio generated: {file_audio_path}")
#         else:
#             print(f"Insights file not found: {insights_file}")
            
#     except Exception as e:
#         print(f"TTS test failed: {e}")