# UI Screenshots Guide

This guide provides step-by-step instructions for capturing high-quality screenshots of the Typed-RAG Streamlit demo application for documentation and presentation purposes.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Setup and Launch](#setup-and-launch)
- [Screenshots to Capture](#screenshots-to-capture)
- [Best Practices](#best-practices)
- [Screenshot Specifications](#screenshot-specifications)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software
- **Web Browser**: Chrome, Firefox, or Safari (Chrome recommended)
- **Screenshot Tool**: 
  - macOS: Built-in (Cmd+Shift+4) or Screenshot app
  - Windows: Snipping Tool or Snip & Sketch (Win+Shift+S)
  - Linux: GNOME Screenshot, Flameshot, or Spectacle
- **Streamlit App**: Typed-RAG demo (`app.py`)

### Environment Setup
Ensure you have:
1. Python environment activated
2. Dependencies installed (`pip install -r requirements.txt`)
3. API keys configured in `.env` file
4. Vector store index built (FAISS or Pinecone)

---

## Setup and Launch

### Step 1: Start the Streamlit App

```bash
# Navigate to project root
cd Typed-Rag

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Launch Streamlit app
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### Step 2: Configure Browser Window

For consistent, professional screenshots:

1. **Set Window Size**: 1920×1080 (Full HD) or 1280×800 (standard)
2. **Zoom Level**: 100% (Cmd+0 / Ctrl+0)
3. **Hide Bookmarks Bar**: For cleaner screenshots (Cmd+Shift+B / Ctrl+Shift+B)

**Using Chrome DevTools** (recommended for exact sizing):
1. Press `F12` to open DevTools
2. Click "Toggle device toolbar" (Cmd+Shift+M / Ctrl+Shift+M)
3. Select "Responsive" and set to 1920×1080
4. Click "..." → "Hide device frame"

---

## Screenshots to Capture

### Screenshot 1: Query Interface (Default State)

**Purpose**: Show the initial state of the query interface

**Steps**:
1. Navigate to "🔍 Query Interface" tab (should be default)
2. Ensure no question is entered yet (fresh state)
3. Configuration panel should show:
   - System: "Typed-RAG" selected
   - Model: "gemini-2.0-flash-lite"
   - Data Source: "Wikipedia" selected
4. Example questions visible
5. Welcome message displayed on the right panel

**Screenshot Details**:
- **File Name**: `query_interface_default.png`
- **Location**: `docs/screenshots/`
- **Content**: Full page view showing left config panel and right welcome panel

**Tips**:
- Scroll to top to ensure all content is visible
- Capture both columns in full

---

### Screenshot 2: Query Interface (Processing)

**Purpose**: Show the Typed-RAG pipeline in action

**Steps**:
1. In the "Query Interface" tab
2. Enter example question: `"Why did the campus police establish in 1958?"`
3. Select "Typed-RAG" system
4. Click "🚀 Ask Question" button
5. Wait for results to fully load (all steps should be expanded)
6. Capture when complete, showing:
   - **Step 1**: Classification result (Question Type: "Reason")
   - **Step 2**: Decomposition (multiple aspects/sub-queries)
   - **Step 3**: Retrieval & Generation (aspect answers with sources)
   - **Final Answer**: Aggregated response
   - **Metrics**: Question Type, Aspects, Total Docs

**Screenshot Details**:
- **File Name**: `query_interface_typed_rag.png`
- **Location**: `docs/screenshots/`
- **Content**: Full pipeline visualization with all expanders open

**Tips**:
- Ensure all expander sections are expanded
- Scroll to show final answer at bottom
- May need multiple screenshots if content is too long

---

### Screenshot 3: Query Interface (System Comparison)

**Purpose**: Show different system responses side-by-side (optional, requires multiple runs)

**Steps**:
1. Run same question with "LLM-Only" system → Capture screenshot
2. Run same question with "RAG Baseline" system → Capture screenshot
3. Run same question with "Typed-RAG" system → Capture screenshot

**Screenshot Details**:
- **File Names**: 
  - `query_interface_llm_only.png`
  - `query_interface_rag_baseline.png`
  - `query_interface_typed_rag.png`
- **Location**: `docs/screenshots/`
- **Content**: Answer section and timing metrics

---

### Screenshot 4: Evaluation Results Tab

**Purpose**: Show evaluation metrics and performance tables

**Steps**:
1. Click on "📊 Evaluation Results" tab
2. Wait for data to load
3. Ensure visible:
   - **Linkage Evaluation**: MRR/MPR comparison table
   - **System metrics**: Comparison across systems
   - **Classifier Evaluation**: Accuracy and per-type performance
   - **Per-Type Performance Table**: Precision, Recall, F1 scores
4. Scroll to show all tables if needed

**Screenshot Details**:
- **File Name**: `evaluation_results.png`
- **Location**: `docs/screenshots/`
- **Content**: All evaluation metrics and tables

**Tips**:
- If content is long, take multiple screenshots:
  - `evaluation_results_linkage.png` - MRR/MPR metrics
  - `evaluation_results_classifier.png` - Classifier performance
- Ensure all table data is readable

---

### Screenshot 5: Ablation Study Tab

**Purpose**: Show ablation study results and component impact

**Steps**:
1. Click on "🔬 Ablation Study" tab
2. Wait for data to load
3. Ensure visible:
   - **Performance Summary**: Table with 4 variants (Full, No Classification, No Decomposition, No Retrieval)
   - **Success Rate** and **Avg Latency** for each variant
   - **Quality Metrics (MRR & MPR)**: Comparison across variants
   - **Key Insights**: Impact analysis
4. Capture all sections

**Screenshot Details**:
- **File Name**: `ablation_study.png`
- **Location**: `docs/screenshots/`
- **Content**: All ablation study visualizations and insights

**Tips**:
- Ensure all metrics are visible
- If needed, split into:
  - `ablation_study_summary.png` - Performance summary
  - `ablation_study_metrics.png` - Quality metrics

---

### Screenshot 6: Pipeline Steps (Detailed)

**Purpose**: Show detailed view of each pipeline step

**Steps for Classification**:
1. In Query Interface, process a question
2. Expand "📋 Step 1: Question Classification" only
3. Capture detailed view showing:
   - Original question
   - Classified type
   - Type description

**Screenshot Details**:
- **File Name**: `pipeline_step_classification.png`
- **Location**: `docs/screenshots/`

**Steps for Decomposition**:
1. Expand "🔀 Step 2: Multi-Aspect Decomposition" only
2. Capture showing:
   - Number of aspects generated
   - Each aspect with its sub-query
   - Aspect cards clearly visible

**Screenshot Details**:
- **File Name**: `pipeline_step_decomposition.png`
- **Location**: `docs/screenshots/`

**Steps for Retrieval**:
1. Expand "🔍 Step 3: Retrieval & Generation" only
2. Show at least one aspect with:
   - Retrieved documents (expand document list)
   - Document metadata (title, score, source)
   - Aspect answer

**Screenshot Details**:
- **File Name**: `pipeline_step_retrieval.png`
- **Location**: `docs/screenshots/`

---

## Best Practices

### Lighting and Display

1. **Consistent Lighting**: Take all screenshots in similar conditions
2. **Display Settings**: 
   - 100% brightness recommended
   - Night shift/blue light filter OFF
   - Dark mode: Optional (choose one and stick with it)

### Framing and Composition

1. **Remove Distractions**:
   - Hide browser bookmarks bar
   - Close unnecessary tabs
   - Hide desktop notifications
   - Full screen or hide OS taskbar/dock

2. **Content Centering**:
   - Ensure main content is centered
   - Leave minimal whitespace on sides
   - Don't cut off important text

3. **Readability**:
   - All text should be crisp and legible
   - Minimum font size: 9-10pt visible in screenshot
   - Check zoom level (100% recommended)

### File Quality

1. **Format**: PNG (lossless) - DO NOT use JPEG
2. **Resolution**: 
   - Standard: 1920×1080 or 1280×800
   - High-res: 2560×1440 or 3840×2160 (for presentations)
3. **File Size**: 
   - Optimize if >2MB: Use tools like TinyPNG or ImageOptim
   - Keep under 500KB for web use

---

## Screenshot Specifications

### Recommended Settings

| Setting | Value | Notes |
|---------|-------|-------|
| **Resolution** | 1920×1080 | Standard Full HD |
| **Format** | PNG | Lossless compression |
| **Color Depth** | 24-bit RGB | True color |
| **DPI** | 72-96 | Screen resolution |
| **Compression** | Lossless | PNG default |

### File Naming Convention

Use descriptive, lowercase names with underscores:

```
[component]_[feature]_[variant].png

Examples:
- query_interface_default.png
- query_interface_typed_rag.png
- evaluation_results_linkage.png
- ablation_study_summary.png
- pipeline_step_classification.png
```

### Directory Structure

```
docs/screenshots/
├── query_interface_default.png
├── query_interface_typed_rag.png
├── query_interface_llm_only.png
├── query_interface_rag_baseline.png
├── evaluation_results.png
├── evaluation_results_linkage.png
├── evaluation_results_classifier.png
├── ablation_study.png
├── ablation_study_summary.png
├── ablation_study_metrics.png
├── pipeline_step_classification.png
├── pipeline_step_decomposition.png
└── pipeline_step_retrieval.png
```

---

## Troubleshooting

### Issue: App Not Loading Data

**Problem**: Evaluation Results or Ablation Study tabs show "No evaluation results found"

**Solution**:
```bash
# Ensure evaluation has been run
python scripts/run_ablation_study.py --input data/wiki_nfqa/dev6.jsonl --output results/ablation/

# Verify results files exist
ls -lh results/ablation/
ls -lh results/*.json
```

### Issue: Slow Loading

**Problem**: App takes too long to process questions

**Solution**:
1. Use cached results (run questions multiple times)
2. Use smaller models if available
3. Reduce document corpus size
4. Check API rate limits

### Issue: Content Too Long for Screenshot

**Problem**: Cannot fit all content in one screenshot

**Solution**:
1. **Option A**: Take multiple screenshots and stitch together
2. **Option B**: Use browser extensions for full-page screenshots:
   - Chrome: "Full Page Screen Capture"
   - Firefox: Built-in (Ctrl+Shift+S → "Save full page")
3. **Option C**: Capture key sections separately

### Issue: Poor Image Quality

**Problem**: Text is blurry or pixelated

**Solution**:
1. Ensure browser zoom is at 100%
2. Use PNG format (not JPEG)
3. Check display resolution settings
4. Use high-DPI/Retina display if available
5. Capture at higher resolution (2560×1440)

### Issue: Inconsistent Styling

**Problem**: Screenshots look different (colors, fonts, sizes)

**Solution**:
1. Use same browser for all screenshots
2. Same zoom level (100%)
3. Same window size
4. Take all screenshots in one session
5. Don't switch between light/dark mode

---

## Advanced Tips

### Creating Annotated Screenshots

Add annotations to highlight features:

1. **Tools**:
   - macOS: Preview (Markup toolbar)
   - Windows: Paint or Paint 3D
   - Cross-platform: GIMP, Inkscape, Figma

2. **Annotation Style**:
   - Use red or orange for highlights
   - Add numbered callouts for steps
   - Use arrows to point to specific features
   - Add text boxes for explanations

### Creating Animated GIFs

For dynamic demonstrations:

1. **Tools**:
   - macOS: Kap (https://getkap.co/)
   - Windows: ScreenToGif
   - Cross-platform: OBS Studio + ffmpeg

2. **Settings**:
   - Frame rate: 10-15 fps (sufficient for UI)
   - Duration: 10-30 seconds
   - Size: 800×600 or 1280×720
   - Format: GIF or WebM

### Creating Video Demos

For comprehensive walkthroughs:

1. **Tools**:
   - OBS Studio (free, cross-platform)
   - QuickTime (macOS)
   - Windows Game Bar (Windows)

2. **Settings**:
   - Resolution: 1920×1080
   - Frame rate: 30 fps
   - Format: MP4 (H.264)
   - Audio: Optional voiceover

---

## Checklist

Before finalizing screenshots:

- [ ] All screenshots taken at consistent resolution
- [ ] All screenshots use PNG format
- [ ] File names follow naming convention
- [ ] All text is readable and not cut off
- [ ] No personal information visible (API keys, usernames, etc.)
- [ ] Consistent browser UI (same zoom, window size)
- [ ] Screenshots organized in `docs/screenshots/` directory
- [ ] Screenshots referenced in documentation (README, papers, etc.)
- [ ] File sizes optimized (<500KB each)
- [ ] Git `.gitignore` updated if screenshots shouldn't be committed

---

## Example Screenshot Session

Complete workflow (30-45 minutes):

```bash
# 1. Prepare environment
cd Typed-Rag
source venv/bin/activate
streamlit run app.py

# 2. Open browser and configure
# - Set window to 1920×1080
# - Zoom to 100%
# - Hide bookmarks bar

# 3. Capture screenshots in order:
#    a. Query Interface (default)
#    b. Query Interface (Typed-RAG result)
#    c. Evaluation Results tab
#    d. Ablation Study tab
#    e. Individual pipeline steps (optional)

# 4. Save to docs/screenshots/
# 5. Optimize file sizes if needed
# 6. Update documentation references
```

---

## Additional Resources

- **Streamlit Documentation**: https://docs.streamlit.io/
- **Screenshot Tools**: 
  - macOS: https://support.apple.com/guide/mac-help/take-screenshots-mh26782/mac
  - Windows: https://support.microsoft.com/en-us/windows/use-snipping-tool-to-capture-screenshots-00246869-1843-655f-f220-97299b865f6b
- **Image Optimization**: https://tinypng.com/, https://imageoptim.com/

---

**Last Updated**: January 2025  
**Maintained By**: Typed-RAG Team  
**Questions**: See [EVALUATION.md](../EVALUATION.md) or create an issue
