from pathlib import Path
import argparse

from dotenv import load_dotenv
from crewai import Agent, Task, Crew

from analysis_agent import run_demo
import os

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

load_dotenv()


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return f"[파일 없음] {path}"
    return path.read_text(encoding="utf-8-sig")


def run_crew(
    input_dir: str,
    output_dir: str,
    gt_dir: str | None,
    user_order: str,
):
    output_path = Path(output_dir)

    # 1. 기존 Python 분석 workflow 먼저 실행
    run_demo(
        input_dir=Path(input_dir),
        output_dir=output_path,
        gt_dir=Path(gt_dir) if gt_dir else None,
        user_order=user_order,
    )

    # 2. 분석 결과 CSV 읽기
    regression_summary_path = output_path / "csv" / "regression_summary.csv"
    analysis_results_path = output_path / "csv" / "analysis_results.csv"
    report_path = output_path / "reports" / "analysis_agent_report.txt"

    regression_summary_text = read_text_if_exists(regression_summary_path)
    analysis_results_preview = read_text_if_exists(analysis_results_path)[:4000]
    rule_report_text = read_text_if_exists(report_path)

    # 3. CrewAI Agent 정의
    planner = Agent(
        role="Planner Agent",
        goal="사용자 연구 오더를 해석하고 분석 결과 해석 방향을 정한다.",
        backstory="Biomedical image analysis workflow를 설계하는 연구 보조 Agent.",
        verbose=True,
    )

    analysis = Agent(
        role="Analysis Agent",
        goal="이미 생성된 segmentation 정량 분석 결과를 해석한다.",
        backstory="Python/OpenCV 기반 분석 결과를 읽고 method별 성능을 비교하는 Agent.",
        verbose=True,
    )

    reasoning = Agent(
        role="Reasoning Agent",
        goal="Dice, IoU, R², slope를 기반으로 가장 안정적인 segmentation 방법을 제안한다.",
        backstory="Biomedical segmentation 결과를 논문 관점에서 해석하는 Agent.",
        verbose=True,
    )

    critic = Agent(
        role="Critic Agent",
        goal="이상 결과와 재분석 필요성을 검토한다.",
        backstory="분석 결과의 신뢰성과 한계를 검토하는 Agent.",
        verbose=True,
    )

    # 4. Task 정의: 새 코드 생성 금지, 결과 해석만 수행
    planner_task = Task(
        description=f"""
    사용자 Research Order를 해석하여 분석 항목을 직접 설계하세요.

    중요:
    - 아래 CSV 데이터를 보고 어떤 기준으로 분석해야 할지 정하세요.
    - 분석 항목은 3~5개로 구성하세요.
    - 사용자가 Dice/IoU를 요청하면 Dice/IoU를 우선하세요.
    - 사용자가 threshold를 요청하면 threshold 민감도를 우선하세요.
    - 사용자가 환자별 분석을 요청하면 patient_id 기준 분석을 우선하세요.

    [사용자 Research Order]
    {user_order}

    [regression_summary.csv]
    {regression_summary_text}

    출력 형식:
    1. 분석 항목
    2. 분석 항목
    3. 분석 항목
    ...
    """,
        expected_output="사용자 오더에 맞춘 분석 항목 목록",
        agent=planner,
    )
    analysis_task = Task(
        description=f"""
    다음은 이미 Python 분석 workflow가 실행된 결과입니다.

    중요 지시:
    - 새로운 Python 코드를 작성하지 마세요.
    - 새로운 segmentation 알고리즘을 제안하지 마세요.
    - 아래 제공된 CSV와 rule-based report만 해석하세요.
    - Planner Agent가 만든 분석 항목을 기준으로 보고서를 작성하세요.

    [사용자 Research Order]
    {user_order}

    [regression_summary.csv]
    {regression_summary_text}

    [analysis_results.csv 일부 미리보기]
    {analysis_results_preview}

    [rule-based report]
    {rule_report_text}

    Planner Agent가 설계한 분석 항목을 따라
    한국어로 세포 이미지 segmentation 결과를 해석하세요.
    """,
        expected_output="Planner가 설계한 항목에 따른 한국어 segmentation 결과 해석 보고서",
        agent=analysis,
        context=[planner_task],
    )

    critic_task = Task(
        description=f"""
    Analysis Agent의 해석 결과를 검토하세요.

    검토 기준:
    - 수치 해석이 CSV와 일치하는지 확인하세요.
    - Dice/IoU 최고 method를 잘못 말하지 않았는지 확인하세요.
    - R²와 slope 해석이 과장되지 않았는지 확인하세요.
    - 이상 결과와 재분석 필요성을 명확히 정리하세요.

    [regression_summary.csv]
    {regression_summary_text}

    최종적으로 연구자가 신뢰할 수 있는 형태로
    간단한 검토 의견을 작성하세요.
    """,
        expected_output="분석 결과에 대한 검토 및 보완 의견",
        agent=critic,
        context=[analysis_task],
    )


    crew = Crew(
        agents=[planner, analysis, reasoning, critic],
        tasks=[planner_task, analysis_task, critic_task],
        verbose=True,
    )

    crew.kickoff()

    reasoning_result = analysis_task.output.raw if analysis_task.output else ""
    critic_result = critic_task.output.raw if critic_task.output else ""

    final_report = f"""
    [Reasoning Result]

    {reasoning_result}

    --------------------------------------------------

    [Critic Result]

    {critic_result}
    """.strip()

    crew_report_path = output_path / "reports" / "crew_agent_report.txt"
    crew_report_path.write_text(final_report, encoding="utf-8")

    print("\n================ CrewAI Analysis Complete ================")
    print(f"CrewAI report saved: {crew_report_path}")
    print(final_report)

    return final_report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gt_dir", default=None)
    parser.add_argument("--user_order", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_crew(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        gt_dir=args.gt_dir,
        user_order=args.user_order,
    )