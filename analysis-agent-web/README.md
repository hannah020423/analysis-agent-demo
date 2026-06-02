# Analysis Agent Web Demo

React + Node.js UI for a Python-based cell image Analysis Agent.

## Structure

```text
analysis-agent-web/
├─ frontend/   React UI
├─ backend/    Node.js Express API
├─ python/     Put analysis_agent.py here
├─ uploads/    Uploaded datasets
└─ outputs/    Python analysis outputs
```

## 1. Prepare Python script

Copy your final Python analysis code into:

```text
python/analysis_agent.py
```

The backend expects this command to work:

```bash
python python/analysis_agent.py --no_gui --input_dir uploads/<run>/images --output_dir outputs/<run>
```

With GT masks:

```bash
python python/analysis_agent.py --no_gui --input_dir uploads/<run>/images --gt_dir uploads/<run>/gt_masks --output_dir outputs/<run>
```

## 2. Start backend

```bash
cd backend
npm install
npm start
```

Backend runs at:

```text
http://localhost:5000
```

## 3. Start frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at:

```text
http://localhost:5173
```

## Agent role mapping

- Planner: Creates analysis plan
- Execution: Runs Python/OpenCV segmentation
- Metrics: Saves CSV and figures
- Reasoning: Reads results and summarizes
- Critic: Detects abnormal outputs and recommends re-analysis

Current version connects UI → Node.js → Python analysis script.
LLM and CrewAI can be attached after this MVP is stable.
