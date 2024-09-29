import pandas as pd
from jinja2 import Template
import base64
import io
import warnings

from src.evaluation.plotting.score_report import sort_df

warnings.filterwarnings("ignore", "dropping on a non-lexsorted multi-index")


def plot_to_base64(plot):
    img = io.BytesIO()
    plot.figure.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode("utf-8")


def group_to_html(group, group_id, plot_placeholders):
    return f"""
        <h2>Scores by {group_id}</h2>
        <p>{group["explanation"]}</p>
        <table>
            {plots_to_html(group["plots"], group_id, plot_placeholders)}
            {tables_to_html(group["tables"], group_id)}
        </table>"""


def plots_to_html(plot_base64_group, plot_group_id, plot_placeholders):
    assert len(plot_base64_group) == 2

    html_template = "<tr>"
    for plot_idx, plot in enumerate(plot_base64_group):
        plot_id = f"plot_{plot_group_id}_{plot_idx}"
        html_template += f"""
        <td align="center" style="vertical-align: top;">
            <img src="data:image/png;base64,{{{{ {plot_id} }}}}">
        </td>"""
        plot_placeholders[plot_id] = plot

    return html_template + "</tr>"


def tables_to_html(tables, group_id):
    assert len(tables) == 2

    html_template = "<tr>"
    for table in tables:
        sort_df(table, group_id, reset_index=True)
        if "index" in table.columns:  # unclear when this happens but it does...
            table.drop("index", axis=1, inplace=True)
        html_template += f'<td align="center" style="vertical-align: top;"> {table.reset_index(drop=True).to_html(index=False)} </td>'

    return html_template + "</tr>"


def review_sample_to_html(review_sample):
    return (
        review_sample.reset_index(drop=True)
        .to_html(justify="center", index=False)
        .replace("\\n", "<br><br>")
    )


def save_report(
    label_id: str,
    reference_label_ids: list[str],
    metric_id: str,
    total_scores_dict: dict,
    usage_options_count_df: pd.DataFrame,
    tp_score: dict,
    groups: dict[list],
    folder: str,
    review_sample: dict[str, pd.DataFrame],
):
    plot_placeholders = {}

    # Convert plot to base64-encoded image
    for group_name, group in groups.items():
        for plot_idx, plot in enumerate(group["plots"]):
            groups[group_name]["plots"][plot_idx] = plot_to_base64(plot)

    # Convert table to HTML
    classification_table = (
        usage_options_count_df.sort_values(by=["usage_class"], ascending=False)
        .rename(columns={metric_id: "count"})
        .to_html(index=False)
    )
    # Define an HTML template using Jinja2
    html_template = """
    <html>
    <head>
        <title>Score Report</title>
    </head>
    <body>
        <h1>Score Report</h1>
        <p>This document shows the performance of <b>{{label_id}}</b>, scored against the reference <b>{{reference_label_ids}}</b>, based on the metric <b>{{metric_id}}</b>.</p>
        <details>
            <summary>How to read this report</summary>
            <p>Besides the overall score, we differentiate between four different classes for each review:</p>
            <ul>
                <li><b>True Positives</b> (TP):\t{{label_id}} identifies 1+ usage options and the reference label agrees</li>
                <li><b>True Negatives</b> (TN): {{label_id}} identifies 0 usage options and the reference label agrees</li>
                <li><b>False Positives</b> (FP): {{label_id}} identifies 1+ usage options but the reference label disagrees</li>
                <li><b>False Negatives</b> (FN): {{label_id}}  identifies 0 usage options but the reference label disagrees</li>
            </ul>
            <p>To be clear: These four classes make no statement about how similar usage options are; they just specify their (non)existence.</p>
        </details>

        <h2>Overall score</h2>
        <p>{{metric_id}} (mean): <b>{{total_score_mean}}</b></p>
        <p>{{metric_id}} (std): {{total_score_std}}</p>

        <h2>Scores per Class</h2>
        <div>{{classification_table}}</div><br>
        <div>
            Score for the <b>True Positive</b> Labels:<br>
            Mean: <b>{{tp_score_mean}}</b><br>
            Std: {{tp_score_std}}
        </div>
    """

    for group_id in groups.keys():
        html_template += group_to_html(groups[group_id], group_id, plot_placeholders)

    # Add reviews
    html_template += """<h2>Review Samples</h2>
    <p>Below are some examples of reviews that were classified as <b>False Positives</b> (FP) or <b>False Negatives</b> (FN). Then, {{tp_sample_size}} randomly selected <b>True Positives</b> (TP) reviews follow.</p>
    <h3>False Positives</h3>
    {{reviews_fp}}
    <h3>False Negatives</h3>
    {{reviews_fn}}
    <h3>True Positives</h3>
    {{reviews_tp}}
    <h3>Hard Reviews</h3>
    These are reviews that where manually selected because they are hard to classify.
    {{reviews_hard}}
    </body></html>
"""

    rendered_html = Template(html_template).render(
        label_id=label_id,
        reference_label_ids=", ".join(reference_label_ids),
        metric_id=metric_id,
        total_score_mean=total_scores_dict["mean"],
        total_score_std=total_scores_dict["std"],
        tp_score_mean=tp_score["mean"],
        tp_score_std=tp_score["std"],
        classification_table=classification_table,
        reviews_fp=review_sample_to_html(review_sample["FP"]),
        reviews_fn=review_sample_to_html(review_sample["FN"]),
        reviews_tp=review_sample_to_html(review_sample["TP"]),
        reviews_hard=review_sample_to_html(review_sample["HARD"]),
        tp_sample_size=len(review_sample["TP"]),
        **plot_placeholders,
    )

    print(f"Saving report to {folder}")
    with open(f"{folder}/report.html", "w") as file:
        file.write(rendered_html)
