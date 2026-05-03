```# Therapeutic Agent Pipeline

A sophisticated multi-agent system that simulates a therapeutic session with:
- **Agent 1 (Primary Therapist)**: Conducts therapy with the patient
- **Agent 2 (Senior Therapist Supervisor)**: Reviews and provides coaching feedback
- **Patient**: Roleplays a real scenario based on Reddit posts

## System Architecture

### Components

#### 1. **Patient Agent** (`patient.py`)
- Takes a Reddit post as context for what they're experiencing
- Responds authentically to the therapist's questions
- Maintains conversation history
- Uses LLaMA 3.1 for realistic patient responses

#### 2. **Agent 1 - Primary Therapist** (`agent_1.py`)
- Conducts the therapy session
- Uses person-centered, empathetic approach
- Adapts responses based on session phase:
  - **Opening**: Establish rapport and safety
  - **Assessment**: Gather information and understand situation
  - **Intervening**: Help patient find solutions
  - **Closing**: Summarize and plan next steps

#### 3. **Agent 2 - Senior Therapist Supervisor** (`agent_2.py`)
- Reviews Agent 1's responses in real-time
- Evaluates against therapeutic best practices
- Provides constructive feedback using checklist:
  - **Empathy & Validation**: Did therapist acknowledge feelings?
  - **Questioning Technique**: Were open-ended questions used?
  - **Safety Assessment**: Were risks identified appropriately?
  - **Active Listening**: Did therapist build on what was said?
  - **Clinical Judgment**: Was response appropriate?
  - **Session Structure**: Does response move session forward?

#### 4. **Pipeline Orchestrator** (`pipeline.py`)
- Manages the full workflow
- Coordinates between agents
- Handles session saving and formatting
- Generates transcripts and feedback summaries

## Installation

### Requirements
- Python 3.10+
- PyTorch with CUDA support
- Transformers library
- 24GB+ GPU memory (for LLaMA 3.1 8B model)

### Setup

```bash
# Navigate to the directory
cd /home/umairai/faithfulness_emnlp/Rasingan/final_pipeline

# Install dependencies
pip install torch transformers huggingface-hub

# Authenticate with Hugging Face (for LLaMA access)
huggingface-cli login
```

## Usage

### Quick Start - Command Line

```bash
# Run with a predefined scenario
python run.py --scenario anxiety_workplace

# Run in interactive mode
python run.py --interactive

# List all available scenarios
python run.py --list-scenarios
```

### Available Scenarios

1. **anxiety_workplace** - Professional anxiety and fear of judgment
2. **depression_isolation** - Depression, anhedonia, social withdrawal
3. **relationship_conflict** - Long-term relationship deterioration
4. **grief_loss** - Processing loss of a loved one
5. **self_esteem_perfectionism** - Self-criticism and impostor syndrome
6. **family_trauma** - Childhood trauma and trust issues

### Advanced Usage - Python Script

```python
from pipeline import TherapyPipeline
from reddit_posts import get_reddit_post, get_patient_profile

# Setup
scenario = "anxiety_workplace"
reddit_posts = get_reddit_post(scenario)
patient_profile = get_patient_profile(scenario)

# Create pipeline
pipeline = TherapyPipeline(
    reddit_posts=reddit_posts,
    patient_profile=patient_profile,
    output_dir="sessions",
    max_turns=10
)

# Run session
results = pipeline.run_session(
    enable_supervisor=True,
    enable_suggestions=True,
    verbose=True
)

# Access results
transcript = pipeline.get_transcript()
feedback = pipeline.get_supervisor_feedback()
stats = pipeline.get_session_stats()
```

### Custom Reddit Posts & Profiles

```bash
# Use custom Reddit post and patient profile
python run.py --scenario anxiety_workplace \
            --custom-post my_post.json \
            --custom-profile my_profile.json
```

**Format of custom_post.json:**
```json
{
  "post": "Your custom Reddit post content here..."
}
```

**Format of custom_profile.json:**
```json
{
  "name": "John",
  "age": 35,
  "occupation": "Engineer",
  "family_structure": "Married, 2 children",
  "primary_concern": "Work stress",
  "history": "First therapy"
}
```

### Command Line Options

```
--scenario {anxiety_workplace, depression_isolation, ...}
                          Choose a predefined scenario
--max-turns N             Maximum conversation turns (default: 8)
--no-supervisor          Disable supervisor feedback
--no-suggestions         Don't show suggestions during session
--output-dir DIR         Directory for session files (default: sessions)
--interactive           Run in interactive mode
--list-scenarios        Show available scenarios
--quiet                 Suppress detailed output
--custom-post FILE      Use custom Reddit post
--custom-profile FILE   Use custom patient profile
```

## Output Files

Each session generates:

### 1. **Full Session Transcript** (`session_YYYYMMDD_HHMMSS.json`)
```json
{
  "metadata": {
    "session_id": "20260417_143020",
    "patient_profile": {...},
    "duration_turns": 14
  },
  "transcript": [
    {
      "turn": 0,
      "role": "therapist",
      "message": "Welcome...",
      "phase": "opening"
    },
    ...
  ],
  "supervisor_feedback": [
    {
      "patient_message": "...",
      "therapist_response": "...",
      "feedback": "..."
    },
    ...
  ]
}
```

### 2. **Summary Report** (`summary_YYYYMMDD_HHMMSS.txt`)
Human-readable transcript with supervisor feedback summary

## Therapeutic Checklist (Agent 2)

Agent 2 evaluates responses using these criteria:

### Empathy & Validation
- ☐ Did the therapist acknowledge the patient's emotions?
- ☐ Were feelings validated without judgment?
- ☐ Did the therapist use reflective listening?
- ☐ Was there genuine warmth and compassion?

### Questioning Technique
- ☐ Did the therapist ask open-ended questions?
- ☐ Was only one main question asked?
- ☐ Were yes/no questions minimized?
- ☐ Did the question encourage deeper exploration?

### Safety & Risk Assessment
- ☐ Were safety concerns identified if relevant?
- ☐ Was risk assessment appropriate?
- ☐ Was the patient reassured without minimizing risks?
- ☐ Were crisis resources mentioned if needed?

### Active Listening
- ☐ Did the therapist listen to the full message?
- ☐ Were key points acknowledged?
- ☐ Did the response build on what was said?
- ☐ Were assumptions avoided?

### Clinical Judgment
- ☐ Was the response clinically appropriate?
- ☐ Did the therapist avoid giving unsolicited advice?
- ☐ Were boundaries maintained?
- ☐ Was the pacing appropriate?

### Session Structure
- ☐ Did the response move the session forward?
- ☐ Was there appropriate transition between topics?
- ☐ Were agenda items being addressed?
- ☐ Was the response length appropriate?

## Example Session Flow

```
1. OPENING
   Therapist: "Welcome! Tell me what brings you in today..."
   Patient: "I've been struggling with anxiety at work..."
   
2. ASSESSMENT
   Therapist: "Can you tell me more about what happens..."
   Patient: "My heart races before meetings..."
   [Supervisor feedback on questioning technique]
   
3. INTERVENING
   Therapist: "It sounds like you're worried about judgment..."
   Patient: "Yes, exactly. I feel like everyone can see..."
   [Supervisor feedback on empathy and validation]
   
4. CLOSING
   Therapist: "Let's talk about what you might try..."
   Patient: "That sounds helpful, thank you..."
   [Session summary saved]
```

## Session Statistics

After each session, the pipeline provides:

```
Total Exchanges: 14
Therapist Turns: 7
Patient Turns: 7
Avg Therapist Response: 45 words
Avg Patient Response: 78 words
Supervisor Feedbacks: 7
```

## Customization

### Adding New Scenarios

Edit `reddit_posts.py`:

```python
REDDIT_POSTS["my_scenario"] = [
    """Your Reddit post content here..."""
]

PATIENT_PROFILES["my_scenario"] = {
    "name": "Name",
    "age": 30,
    "gender": "Male",
    "occupation": "Job",
    "family_structure": "Family info",
    "primary_concern": "Main concern",
    "history": "Therapy history"
}
```

### Modifying Therapeutic Approach

Edit `agent_1.py` to change:
- Therapeutic orientation (currently person-centered)
- Phase-specific strategies
- Response tone and style
- Question approach

### Customizing Supervisor Checklist

Edit `agent_2.py` to add/modify:
- Checklist categories
- Evaluation criteria
- Feedback prompts
- Supervision style

## Performance Notes

### GPU Memory Requirements
- Base model load: ~18GB
- Generation buffer: ~2GB
- Total: ~24GB VRAM recommended

### Generation Parameters
- Temperature: 0.7 (default, balances creativity and consistency)
- Max tokens: 150-400 depending on phase
- Top-p: 0.9 (diverse but focused)

### Session Duration
- Typical 8-turn session: 3-5 minutes
- With supervisor feedback: 5-8 minutes
- Depends on GPU speed and generation length

## Troubleshooting

### Out of Memory Error
```bash
# Reduce max tokens or use smaller model
python run.py --scenario anxiety_workplace --no-supervisor
```

### Slow Generation
- Check GPU utilization: `nvidia-smi`
- Consider using quantized models
- Reduce max_turns

### Poor Patient Response Quality
- Increase temperature for more diversity
- Improve Reddit post context
- Check patient system prompt in patient.py

## API Reference

### TherapyPipeline

```python
pipeline = TherapyPipeline(
    reddit_posts: List[str],
    patient_profile: Optional[Dict] = None,
    output_dir: str = "sessions",
    max_turns: int = 10
)

# Methods
pipeline.run_session(
    enable_supervisor: bool = True,
    enable_suggestions: bool = True,
    verbose: bool = True
) -> Dict

pipeline.get_transcript() -> List[Dict]
pipeline.get_supervisor_feedback() -> List[Dict]
pipeline.get_session_stats() -> Dict
pipeline.export_session(format: str = "json") -> str
```

### Agent1_PrimaryTherapist

```python
therapist = Agent1_PrimaryTherapist(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    session_type: str = "counseling"
)

therapist.generate_opening() -> str
therapist.respond_to_patient(
    patient_message: str, 
    session_phase: str = "assessment"
) -> str
```

### Agent2_SeniorTherapist

```python
supervisor = Agent2_SeniorTherapist(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
)

supervisor.evaluate_response(
    patient_message: str,
    therapist_response: str,
    session_context: str = ""
) -> Dict

supervisor.get_checklist_category_evaluation(
    category: str,
    patient_message: str,
    therapist_response: str
) -> Dict
```

### Patient

```python
patient = Patient(
    reddit_post: List[str],
    patient_profile: Optional[Dict] = None,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
)

patient.generate_response(
    therapist_message: str,
    max_tokens: int = 256,
    temperature: float = 0.7
) -> str
```

## Research Applications

This pipeline can be used for:

1. **Training & Evaluation**: Assess therapist performance using consistent patient behavior
2. **LLM Benchmarking**: Evaluate different models on therapeutic competency
3. **Dialogue Quality**: Study therapeutic conversation patterns
4. **Feedback Generation**: Research automatic supervisor feedback systems
5. **Care Simulation**: Create synthetic therapy datasets for analysis

## License

This pipeline is for research purposes.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{therapeutic_pipeline_2026,
  title={Therapeutic Agent Pipeline: Multi-Agent Therapy Simulation},
  author={Your Name},
  year={2026},
  url={url}
}
```

## Support & Issues

For issues or questions:
1. Check troubleshooting section
2. Review example usage in `run.py`
3. Check session logs in `sessions/` directory
4. Verify GPU/CUDA setup with `nvidia-smi`

## Future Enhancements

- [ ] Support for multiple LLM backends (GPT-4, Claude, etc.)
- [ ] Real-time visualization dashboard
- [ ] Therapy modality variants (CBT, DBT, psychodynamic)
- [ ] Multi-patient group therapy sessions
- [ ] Voice input/output integration
- [ ] Session analysis and insights generation
- [ ] Clinical outcomes tracking

```
