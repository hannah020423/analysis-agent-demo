# Python analysis script

Put your analysis code here as `analysis_agent.py`.

Required CLI interface for the Node.js backend:

```bash
python analysis_agent.py --no_gui --input_dir <image_folder> --output_dir <output_folder>
```

Optional GT mask:

```bash
python analysis_agent.py --no_gui --input_dir <image_folder> --gt_dir <gt_folder> --output_dir <output_folder>
```

Expected output folder structure:

```text
output/
  masks/
  csv/
  figures/
  reports/analysis_agent_report.txt
```
