from pathlib import Path
import argparse
import time
import pandas as pd

from dotenv import load_dotenv
from crewai import Agent, Task, Crew

load_dotenv()


def read_text_if_exists(path: Path, limit: int | None = None) -> str:
    if not path.exists():
        return f"[파일 없음] {path}"

    text = path.read_text(encoding="utf-8-sig")

    if limit:
        return text[:limit]

    return text

def create_patient_summary_csv(output_path: Path) -> str:
    csv_path = output_path / "csv" / "analysis_results.csv"
    followup_dir = output_path / "followups"
    followup_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    summary = (
        df.groupby(["patient_id", "method"])
        .agg(
            mean_r2=("r2", "mean"),
            mean_dice=("dice", "mean"),
            mean_iou=("iou", "mean"),
            n=("image", "count"),
        )
        .reset_index()
    )

    save_path = followup_dir / "patient_method_summary.csv"
    summary.to_csv(save_path, index=False, encoding="utf-8-sig")

    return str(save_path)


def run_followup(output_dir: str, question: str) -> str:
    output_path = Path(output_dir)

    csv_dir = output_path / "csv"
    reports_dir = output_path / "reports"
    followup_dir = output_path / "followups"
    followup_dir.mkdir(parents=True, exist_ok=True)

    analysis_results_path = csv_dir / "analysis_results.csv"
    regression_summary_path = csv_dir / "regression_summary.csv"
    rule_report_path = reports_dir / "analysis_agent_report.txt"
    crew_report_path = reports_dir / "crew_agent_report.txt"


    regression_summary = read_text_if_exists(regression_summary_path)
    analysis_results = read_text_if_exists(analysis_results_path, limit=12000)
    generated_csv_text = ""
    generated_csv_path = ""

    q = question.lower()

    if (
        "환자" in question
        and ("r²" in q or "r2" in q or "dice" in q or "다이스" in question)
    ):
        generated_csv_path = create_patient_summary_csv(output_path)
        generated_csv_text = Path(generated_csv_path).read_text(encoding="utf-8-sig")
    rule_report = read_text_if_exists(rule_report_path, limit=6000)
    previous_crew_report = read_text_if_exists(crew_report_path, limit=6000)

    planner = Agent(
        role="Follow-up Planner Agent",
        goal="사용자의 추가 질문 의도를 분석하고 필요한 분석 관점을 계획한다.",
        backstory="Biomedical image analysis 결과를 다양한 연구 관점으로 재해석하는 Planner Agent.",
        verbose=True,
    )

    analysis = Agent(
        role="Follow-up Analysis Agent",
        goal="기존 CSV 결과를 기반으로 사용자의 추가 질문에 답한다.",
        backstory="Dice, IoU, R², slope, area ratio를 기반으로 segmentation 결과를 해석하는 Agent.",
        verbose=True,
    )

    critic = Agent(
        role="Follow-up Critic Agent",
        goal="Analysis Agent의 답변이 CSV 수치와 일치하는지 검토한다.",
        backstory="정량 결과 해석의 오류, 과장, 수치 불일치를 점검하는 검증 Agent.",
        verbose=True,
    )

    planner_task = Task(
        description=f"""
사용자의 추가 질문을 분석하고, 어떤 기준으로 기존 분석 결과를 재해석해야 하는지 계획하세요.

중요:
- 새 segmentation을 실행하지 마세요.
- 새 알고리즘을 제안하지 마세요.
- 기존 CSV와 report를 기반으로만 계획하세요.

[사용자 추가 질문]
{question}

[regression_summary.csv]
{regression_summary}

출력:
- 질문 의도
- 필요한 지표
- 분석 순서
""",
        expected_output="추가 질문에 대한 분석 계획",
        agent=planner,
    )

    analysis_task = Task(
        description=f"""
Planner Agent의 계획을 바탕으로 기존 분석 결과를 해석하세요.

중요:
- 이미지를 다시 분석하지 마세요.
- Python 코드를 새로 작성하지 마세요.
- 아래 CSV와 report에 있는 수치만 근거로 사용하세요.
- 한국어로 답변하세요.
- 사용자가 CSV 파일 생성을 요청했더라도, CSV 생성 결과만 말하지 말고 반드시 분석 해석도 함께 작성하세요.
- 생성된 CSV에서 어떤 patient_id 또는 method가 높은지 설명하세요.
- CSV 파일이 생성된 경우, 파일명도 함께 안내하세요.

[사용자 추가 질문]
{question}

[regression_summary.csv]
{regression_summary}

[analysis_results.csv 일부]
{analysis_results}

[새로 생성된 환자별 요약 CSV]
{generated_csv_text}

[생성된 CSV 경로]
{generated_csv_path}


[rule-based report]
{rule_report}

[previous CrewAI report]
{previous_crew_report}
""",
    expected_output="사용자 추가 질문에 대한 한국어 분석 답변. CSV 생성 요청이 포함된 경우 생성된 CSV의 핵심 결과 해석도 포함하세요.",
    agent=analysis,
    context=[planner_task],
)

    critic_task = Task(
        description=f"""
Analysis Agent의 답변을 검토하세요.

검토 기준:
- Dice/IoU 최고 method를 잘못 말하지 않았는지
- R²와 slope 해석이 CSV와 일치하는지
- threshold 민감도 해석이 과장되지 않았는지
- 사용자의 질문에 직접 답했는지

중요:
- "검토 결과", "답변이 적절합니다" 같은 평가 문장으로 시작하지 마세요.
- 사용자 질문에 대한 최종 분석 보고서처럼 작성하세요.
- Critic의 역할은 내부 검토이며, 최종 출력에는 자연스럽게 보완된 분석 결과만 남기세요.


[regression_summary.csv]
{regression_summary}

최종 답변에는 필요한 보완 의견만 간단히 추가하세요.
""",
        expected_output="Analysis Agent의 답변을 보완한 최종 한국어 답변. 검토했다는 표현보다 사용자 질문에 대한 최종 분석 결과를 작성하세요.",
        agent=critic,
        context=[analysis_task],
    )

    crew = Crew(
        agents=[planner, analysis, critic],
        tasks=[planner_task, analysis_task, critic_task],
        verbose=True,
    )

    crew.kickoff()

    reasoning_result = analysis_task.output.raw if analysis_task.output else ""
    critic_result = critic_task.output.raw if critic_task.output else ""

    answer = f"""
    [Reasoning Result]

    {reasoning_result}

    --------------------------------------------------

    [Critic Result]

    {critic_result}
    """.strip()

    save_path = followup_dir / f"followup_{int(time.time())}.txt"
    save_path.write_text(
        f"[Question]\n{question}\n\n[Answer]\n{answer}",
        encoding="utf-8",
    )

    print("\n================ Follow-up CrewAI Complete ================")
    print(f"Follow-up report saved: {save_path}")
    print(answer)

    return answer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--question", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_followup(
        output_dir=args.output_dir,
        question=args.question,
    )