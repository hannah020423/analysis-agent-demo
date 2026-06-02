import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Upload, Play, FileText, Image as ImageIcon, Database, BrainCircuit, CheckCircle2 } from "lucide-react";
import "./style.css";

const API_BASE = import.meta.env.VITE_API_BASE || "";

function App() {
  const [images, setImages] = useState([]);
  const [gtMasks, setGtMasks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [userOrder, setUserOrder] = useState("");
  const [followupQuestion, setFollowupQuestion] = useState("");
  const [followupAnswer, setFollowupAnswer] = useState("");
  const [asking, setAsking] = useState(false);

  const imageCount = useMemo(() => images.length, [images]);
  const gtCount = useMemo(() => gtMasks.length, [gtMasks]);

  const handleImageFiles = (e) => {
    const selected = Array.from(e.target.files || []);
    setImages(selected);
  };

  const handleGtFiles = (e) => {
    const selected = Array.from(e.target.files || []);
    setGtMasks(selected);
  };

  const handleAnalyze = async () => {
    if (images.length === 0) {
      setError("분석할 세포 이미지 파일을 먼저 선택하세요.");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const runRes = await fetch(`${API_BASE}/api/run-id`, { method: "POST" });
      const runText = await runRes.text();

      if (!runRes.ok) {
        throw new Error(runText);
      }

      const { runId } = JSON.parse(runText);

      const form = new FormData();
      form.append("userOrder", userOrder);
      form.append("runId", runId);
      images.forEach((file) => {
        form.append("images", file);
        form.append("imagePaths", file.webkitRelativePath || file.name);
      });

      gtMasks.forEach((file) => {
        form.append("gtMasks", file);
        form.append("gtPaths", file.webkitRelativePath || file.name);
      });

      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        headers: { "x-run-id": runId },
        body: form
      });

      const text = await res.text();

      let data;
      try {
        data = JSON.parse(text);
      } catch {
        throw new Error(text.slice(0, 1000));
      }

      if (!res.ok) {
        throw new Error(data.logs || data.detail || data.error || "분석 실패");
      }
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  async function handleAskResult() {
  if (!result?.runId) {
    alert("먼저 dataset 분석을 실행해야 합니다.");
    return;
  }

  if (!followupQuestion.trim()) {
    alert("추가 질문을 입력하세요.");
    return;
  }

  setAsking(true);
  setFollowupAnswer("");

  try {
    const res = await fetch(`${API_BASE}/api/ask-result`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        runId: result.runId,
        question: followupQuestion,
      }),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Failed to ask result");
    }

    setFollowupAnswer(data.answer);
    setResult((prev) => ({
      ...prev,
      followupFiles: data.followupFiles || prev.followupFiles || [],
    }));
  } catch (err) {
    setFollowupAnswer(`Error: ${err.message}`);
  } finally {
    setAsking(false);
  }
}

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="badge">Cloud-based Research Agent MVP</p>
          <h1>Cell Image Analysis Agent</h1>
          <p className="subtitle">ROI 자동 탐색, OTSU/Adaptive/Manual segmentation, 정량 분석, Reasoning/Critic Report를 실행하는 데모 UI입니다.</p>
        </div>
        <div className="agent-flow">
          <FlowItem icon={<Upload />} label="Input" />
          <span>→</span>
          <FlowItem icon={<Play />} label="Execution" />
          <span>→</span>
          <FlowItem icon={<BrainCircuit />} label="Reasoning" />
          <span>→</span>
          <FlowItem icon={<CheckCircle2 />} label="Critic" />
        </div>
      </header>

      <main className="grid">
        <section className="card upload-card">
          <h2>1. Dataset Upload</h2>
          <div style={{ marginTop: 20 }}>
            <h2>Research Order</h2>

            <textarea
              value={userOrder}
              onChange={(e) => setUserOrder(e.target.value)}
              placeholder={`예: Dice와 IoU를 중심으로 가장 안정적인 segmentation 방법을 분석해줘.

              요청:
              1. Otsu, Adaptive, Manual threshold 방법의 장단점을 비교하세요.
              2. Dice와 IoU 기준으로 GT mask와 가장 유사한 segmentation 방법을 설명하세요.
              3. R²와 slope 기준으로 정량적 일관성이 가장 높은 방법을 설명하세요.
              4. threshold 민감도 또는 이상 결과가 있다면 설명하세요.
              5. 연구자가 다음 단계에서 확인해야 할 점을 제안하세요.`}
              rows={5}
              style={{
                width: "100%",
                padding: "10px",
                fontSize: "16px",
              }}
            />
          </div>
          <p className="muted">이미지 파일을 여러 장 선택하세요. GT mask가 있으면 함께 업로드할 수 있습니다.</p>
          <label className="dropzone">
            <Upload />
            <span>Cell images 선택</span>
            <small>png, jpg, jpeg, tif, bmp</small>
            <input
              type="file"
              webkitdirectory=""
              directory=""
              multiple
              onChange={handleImageFiles}
            />
          </label>
          <div className="file-count">선택된 이미지: <b>{imageCount}</b>개</div>

          <label className="dropzone secondary">
            <Database />
            <span>GT masks 선택 선택사항</span>
            <small>원본과 같은 파일명 또는 _mask, _gt 파일명 권장</small>
            <input
              type="file"
              webkitdirectory=""
              directory=""
              multiple
              onChange={handleGtFiles}
            />
          </label>
          <div className="file-count">선택된 GT: <b>{gtCount}</b>개</div>

          <button className="run-btn" onClick={handleAnalyze} disabled={loading}>
            {loading ? "Analysis Agent 실행 중..." : "Analyze Dataset"}
          </button>
          {error && <pre className="error">{error}</pre>}
        </section>

        <section className="card">
          <h2>2. Agent Role</h2>
          <div className="role-list">
            <Role title="Planner" text="사용자의 분석 목표를 실행 계획으로 변환" />
            <Role title="Execution" text="Python/OpenCV 기반 ROI, 전처리, segmentation 수행" />
            <Role title="Metrics" text="면적, area ratio, R² 등 정량 지표 계산" />
            <Role title="Reasoning" text="method별 결과 비교 및 해석 report 생성" />
            <Role title="Critic" text="이상 결과, threshold 민감도, 재실행 필요 여부 판단" />
          </div>
        </section>
      </main>

      {result && (
        <ResultView
          result={result}
          followupQuestion={followupQuestion}
          setFollowupQuestion={setFollowupQuestion}
          followupAnswer={followupAnswer}
          asking={asking}
          handleAskResult={handleAskResult}
        />
      )}
    </div>
  );
}

function FlowItem({ icon, label }) {
  return <div className="flow-item">{icon}<span>{label}</span></div>;
}

function Role({ title, text }) {
  return <div className="role"><b>{title}</b><span>{text}</span></div>;
}

function ResultView({ 
  result,
  followupQuestion,
  setFollowupQuestion,
  followupAnswer,
  asking,
  handleAskResult,
 }) {
  return (
    <section className="results">
      <div className="card wide">
        <h2>3. Analysis Result</h2>
        <p className="muted">Run ID: {result.runId}</p>
        <div className="links">
          {result.csvFiles?.map((f) => <a key={f.name} href={`${API_BASE}${f.url}`} target="_blank">CSV: {f.name}</a>)}
          {result.reportFiles?.map((f) => <a key={f.name} href={`${API_BASE}${f.url}`} target="_blank">Report: {f.name}</a>)}
        </div>
      </div>

      <div className="card wide">
        <h3><ImageIcon size={18}/> Figures</h3>
        <div className="image-grid">
          {result.figures?.slice(0, 8).map((img) => (
            <figure key={img.name}>
              <img src={`${API_BASE}${img.url}`} alt={img.name} />
              <figcaption>{img.name}</figcaption>
            </figure>
          ))}
        </div>
      </div>

      <div className="card wide">
        <h3><ImageIcon size={18}/> Masks Preview</h3>
        <div className="image-grid small">
          {result.masks?.slice(0, 12).map((img) => (
            <figure key={img.name}>
              <img src={`${API_BASE}${img.url}`} alt={img.name} />
              <figcaption>{img.name}</figcaption>
            </figure>
          ))}
        </div>
      </div>

      {/* <div className="card wide">
        <h3><FileText size={18}/> Rule-based Report</h3>

        <pre className="report">
          {result.ruleReport || "No rule-based report"}
        </pre>
      </div> */}

      <div className="card wide">
        <h3><FileText size={18}/> LLM / CrewAI Report</h3>

        <pre className="report">
          {result.crewReport || "No CrewAI report"}
        </pre>
      </div>

      <div className="card wide">
        <h3><FileText size={18}/> Ask Follow-up Question</h3>

        <textarea
          value={followupQuestion}
          onChange={(e) => setFollowupQuestion(e.target.value)}
          placeholder="예: R²와 slope 기준으로 다시 분석해줘 / manual_30과 otsu_feedback을 논문 결과 문장처럼 비교해줘"
          rows={4}
          style={{
            width: "100%",
            padding: "12px",
            borderRadius: "10px",
            border: "1px solid #ddd",
            resize: "vertical",
          }}
        />

        <button
          className="run-btn"
          onClick={handleAskResult}
          disabled={asking}
          style={{ marginTop: "12px" }}
        >
          {asking ? "Asking Agent..." : "Ask Agent"}
        </button>

        {followupAnswer && (
          <pre className="report" style={{ marginTop: "16px" }}>
            {followupAnswer}
          </pre>
        )}
        {result?.followupFiles?.length > 0 && (
        <div className="links" style={{ marginTop: "12px" }}>
          {result.followupFiles.map((f) => (
            <a key={f.name} href={`${API_BASE}${f.url}`} target="_blank">
              Follow-up file: {f.name}
            </a>
          ))}
        </div>
      )}
      </div>

      <details className="card wide logs">
        <summary>Execution Logs</summary>
        <pre>{result.logs}</pre>
      </details>
    </section>
  );
}



createRoot(document.getElementById("root")).render(<App />);
