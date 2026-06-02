import express from "express";
import cors from "cors";
import multer from "multer";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import { spawn } from "child_process";
import OpenAI from "openai";
import dotenv from "dotenv";
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ROOT_DIR = path.resolve(__dirname, "..");
const FRONTEND_DIST = path.join(ROOT_DIR, "frontend", "dist");

const UPLOAD_ROOT = path.join(ROOT_DIR, "uploads");
const OUTPUT_ROOT = path.join(ROOT_DIR, "outputs");

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});



const app = express();
const PORT = process.env.PORT || 5000;



const PYTHON_SCRIPT = process.env.PYTHON_SCRIPT || path.join(ROOT_DIR, "python", "crew_pipeline.py");
const PYTHON_BIN = process.env.PYTHON_BIN || "python3";

app.use(cors());
app.use(express.json());
app.use("/outputs", express.static(OUTPUT_ROOT));
app.use(express.static(FRONTEND_DIST));

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function createRunId() {
  const now = new Date();
  const stamp = now.toISOString().replace(/[:.]/g, "-");
  return `run_${stamp}`;
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const runId = req.body.runId || req.headers["x-run-id"];
    const fieldDir = file.fieldname === "gtMasks" ? "gt_masks" : "images";
    const uploadDir = path.join(UPLOAD_ROOT, runId, fieldDir);
    ensureDir(uploadDir);
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueName = `${Date.now()}-${Math.random().toString(36).slice(2)}-${file.originalname}`;
    cb(null, Buffer.from(uniqueName, "latin1").toString("utf8"));
  }
});

const upload = multer({ storage });

app.get("/api/health", (req, res) => {
  res.json({
    success: true,
    runId,
    outputDir,
    figures,
    masks,
    ruleReport,
    crewReport,
  });
});

app.post("/api/run-id", (req, res) => {
  const runId = createRunId();
  const userOrder = req.body.userOrder || "";
  ensureDir(path.join(UPLOAD_ROOT, runId, "images"));
  ensureDir(path.join(UPLOAD_ROOT, runId, "gt_masks"));
  ensureDir(path.join(OUTPUT_ROOT, runId));
  res.json({ runId });
});

app.post(
  "/api/analyze",
  upload.fields([
    { name: "images", maxCount: 1000 },
    { name: "gtMasks", maxCount: 1000 }
  ]),
  async (req, res) => {
    console.log("[API] /api/analyze called");
    const userOrder = req.body.userOrder || "";
    const runId = req.body.runId;
    if (!runId) return res.status(400).json({ error: "runId is required" });

    const inputDir = path.join(UPLOAD_ROOT, runId, "images");
    const gtDir = path.join(UPLOAD_ROOT, runId, "gt_masks");
    const outputDir = path.join(OUTPUT_ROOT, runId);

    const hasGt = fs.existsSync(gtDir) && fs.readdirSync(gtDir).length > 0;
        
    restoreFolderStructure(inputDir, req.files?.images || [], req.body.imagePaths, "images");
    restoreFolderStructure(gtDir, req.files?.gtMasks || [], req.body.gtPaths, "gt_masks");

    if (!fs.existsSync(PYTHON_SCRIPT)) {
      return res.status(500).json({
        error: "Python analysis script not found",
        detail: `Put your analysis_agent.py at: ${PYTHON_SCRIPT}`
      });
    }

    const args = [PYTHON_SCRIPT, "--input_dir", inputDir, "--output_dir", outputDir, "--user_order", userOrder,];
    if (hasGt) args.push("--gt_dir", gtDir);

    console.log("[RUN ID]", runId);
    console.log("[INPUT DIR]", inputDir);
    console.log("[GT DIR]", gtDir);
    console.log("[OUTPUT DIR]", outputDir);
    console.log("[PYTHON SCRIPT]", PYTHON_SCRIPT);
    console.log("[ARGS]", args);                      

    const logs = [];
    const py = spawn(PYTHON_BIN, args, {
      cwd: ROOT_DIR,
      shell: false,
      env: {
        ...process.env,
        PYTHONIOENCODING: "utf-8",
        PYTHONUTF8: "1",
        OPENAI_API_KEY: process.env.OPENAI_API_KEY,
      },
    });

    py.stdout.on("data", (data) => {
      const text = data.toString();
      logs.push(text);
      console.log("[PYTHON]", text);
    });

    py.stderr.on("data", (data) => {
      const text = data.toString();
      logs.push(text);
      console.error("[PYTHON ERROR]", text);
    });
    py.on("close", async (code) => {
      if (code !== 0) {
        return res.status(500).json({
          error: "Python analysis failed",
          code,
          logs: logs.join("")
        });
      }

      


      // OpenAI 결과를 report 파일로 저장
      const reportsDir = path.join(outputDir, "reports");

      const ruleReportPath = path.join(
        reportsDir,
        "analysis_agent_report.txt"
      );

      const crewReportPath = path.join(
        reportsDir,
        "crew_agent_report.txt"
      );

      const ruleReport = fs.existsSync(ruleReportPath)
        ? fs.readFileSync(ruleReportPath, "utf8")
        : "";

      const crewReport = fs.existsSync(crewReportPath)
        ? fs.readFileSync(crewReportPath, "utf8")
        : "";

      // =========================
      // Response
      // =========================
      const result = collectOutputs(runId, outputDir);

      res.json({
        ok: true,
        runId,
        logs: logs.join(""),
        ruleReport,
        crewReport,
        ...result
      });
    });
  }
);

function csvToMarkdownTable(csvText) {
  const lines = csvText.trim().split(/\r?\n/);
  if (lines.length < 2) return csvText;

  const headers = lines[0].split(",");
  const rows = lines.slice(1).map((line) => line.split(","));

  const formatNumber = (value) => {
    const num = Number(value);
    if (!Number.isNaN(num)) return num.toFixed(4);
    return value;
  };

  const table = [];
  table.push(`| ${headers.join(" | ")} |`);
  table.push(`| ${headers.map(() => "---").join(" | ")} |`);

  rows.forEach((row) => {
    table.push(`| ${row.map(formatNumber).join(" | ")} |`);
  });

  return table.join("\n");
}

function fileUrl(runId, filePath) {
  const rel = path.relative(path.join(OUTPUT_ROOT, runId), filePath).replaceAll(path.sep, "/");
  return `/outputs/${runId}/${rel}`;
}

function collectOutputs(runId, outputDir) {
  const figuresDir = path.join(outputDir, "figures");
  const masksDir = path.join(outputDir, "masks");
  const csvDir = path.join(outputDir, "csv");
  const reportsDir = path.join(outputDir, "reports");
  const followupsDir = path.join(outputDir, "followups");


  const figures = listPngFilesRecursive(figuresDir).map((p) => ({
    name: path.relative(figuresDir, p).replace(/\\/g, "/"),
    url: fileUrl(runId, p),
  }));

  const csvFiles = fs.existsSync(csvDir)
    ? fs.readdirSync(csvDir).filter((f) => /\.csv$/i.test(f)).map((f) => ({ name: f, url: fileUrl(runId, path.join(csvDir, f)) }))
    : [];

  const reportFiles = fs.existsSync(reportsDir)
    ? fs.readdirSync(reportsDir).filter((f) => /\.(txt|json)$/i.test(f)).map((f) => ({ name: f, url: fileUrl(runId, path.join(reportsDir, f)) }))
    : [];

  let reportText = "";
  const reportPath = path.join(reportsDir, "analysis_agent_report.txt");
  if (fs.existsSync(reportPath)) reportText = fs.readFileSync(reportPath, "utf8");

  const maskFolders = fs.existsSync(masksDir)
    ? fs.readdirSync(masksDir).filter((f) => fs.statSync(path.join(masksDir, f)).isDirectory())
    : [];

  const masks = listPngFilesRecursive(masksDir)
    .slice(0, 30)
    .map((p) => ({
      name: path.relative(masksDir, p).replace(/\\/g, "/"),
      url: fileUrl(runId, p),
    }));

    const followupFiles = fs.existsSync(followupsDir)
      ? fs.readdirSync(followupsDir)
          .filter((f) => /\.(csv|txt)$/i.test(f))
          .map((f) => ({
            name: f,
            url: fileUrl(runId, path.join(followupsDir, f)),
          }))
      : [];

  return { figures, csvFiles, reportFiles, reportText, masks, followupFiles, };
}

function restoreFolderStructure(baseDir, files, relPaths, rootNameToRemove) {
  if (!files || !relPaths) return;

  const paths = Array.isArray(relPaths) ? relPaths : [relPaths];

  files.forEach((file, idx) => {
    let relPath = paths[idx];
    if (!relPath) return;

    relPath = relPath.replace(/\\/g, "/");

    // 예: images/81/1_1.jpg → 81/1_1.jpg
    const parts = relPath.split("/");
    if (parts[0] === rootNameToRemove) {
      parts.shift();
    }

    const safeRelPath = parts.join("/");

    // 파일명이 1_1.jpg만 들어온 경우는 이동할 필요 없음
    if (!safeRelPath || safeRelPath === file.filename) {
      return;
    }

    const targetPath = path.join(baseDir, safeRelPath);

    if (path.resolve(targetPath) === path.resolve(file.path)) {
      return;
    }

    if (!fs.existsSync(file.path)) {
      console.warn("[restoreFolderStructure] source file not found:", file.path);
      return;
    }

    ensureDir(path.dirname(targetPath));

    if (fs.existsSync(targetPath)) {
      fs.unlinkSync(targetPath);
    }

    fs.renameSync(file.path, targetPath);
  });
}

function listPngFilesRecursive(dir) {
  const results = [];

  if (!fs.existsSync(dir)) return results;

  const items = fs.readdirSync(dir, { withFileTypes: true });

  for (const item of items) {
    const fullPath = path.join(dir, item.name);

    if (item.isDirectory()) {
      results.push(...listPngFilesRecursive(fullPath));
    } else if (item.isFile() && item.name.toLowerCase().endsWith(".png")) {
      results.push(fullPath);
    }
  }

  return results;
}

app.post("/api/ask-result", async (req, res) => {
  const { runId, question } = req.body;

  if (!runId) {
    return res.status(400).json({ error: "runId is required" });
  }

  if (!question || !question.trim()) {
    return res.status(400).json({ error: "question is required" });
  }

  const outputDir = path.join(OUTPUT_ROOT, runId);
  const followupScript = path.join(ROOT_DIR, "python", "followup_pipeline.py");

  if (!fs.existsSync(outputDir)) {
    return res.status(404).json({ error: "Output directory not found" });
  }

  if (!fs.existsSync(followupScript)) {
    return res.status(500).json({
      error: "followup_pipeline.py not found",
      detail: followupScript,
    });
  }

  const args = [
    followupScript,
    "--output_dir",
    outputDir,
    "--question",
    question,
  ];

  const logs = [];

  const py = spawn(PYTHON_BIN, args, {
    cwd: ROOT_DIR,
    shell: false,
    env: {
      ...process.env,
      PYTHONIOENCODING: "utf-8",
      PYTHONUTF8: "1",
      OPENAI_API_KEY: process.env.OPENAI_API_KEY,
    },
  });

  py.stdout.on("data", (data) => {
    const text = data.toString();
    logs.push(text);
    console.log("[FOLLOWUP PYTHON]", text);
  });

  py.stderr.on("data", (data) => {
    const text = data.toString();
    logs.push(text);
    console.error("[FOLLOWUP PYTHON ERROR]", text);
  });

  py.on("close", (code) => {
    if (code !== 0) {
      return res.status(500).json({
        error: "Follow-up CrewAI failed",
        code,
        logs: logs.join(""),
      });
    }

    const followupsDir = path.join(outputDir, "followups");

    const files = fs.existsSync(followupsDir)
      ? fs.readdirSync(followupsDir)
          .filter((f) => f.endsWith(".txt"))
          .sort()
      : [];

    const latestFile = files.length
      ? path.join(followupsDir, files[files.length - 1])
      : null;

    const answer = latestFile && fs.existsSync(latestFile)
      ? fs.readFileSync(latestFile, "utf8")
      : logs.join("");

    const result = collectOutputs(runId, outputDir);

    res.json({
      ok: true,
      runId,
      question,
      answer,
      followupFiles: result.followupFiles,
      logs: logs.join(""),
    });
  });
});

app.get("*", (req, res) => {
  res.sendFile(path.join(FRONTEND_DIST, "index.html"));
});

app.listen(PORT, () => {
  console.log(`Analysis Agent backend running on http://localhost:${PORT}`);
});
