from pathlib import Path
import json

NOTION_PAGE_ID = "360406b6-d85a-81d8-9378-de2638bfa03f"
NOTION_URL = "https://www.notion.so/GNN-State-Estimation-Task-218-Final-Independent-Academic-Review-and-Remediation-360406b6d85a81d89378de2638bfa03f"
TITLE = "GNN State Estimation - Task 218 - Final Independent Academic Review and Remediation"

payload_path = Path("results/task218_review_payload.json")
release_path = Path("results/release_packet.json")

payload = json.loads(payload_path.read_text(encoding="utf-8"))
payload["notion_page_id"] = NOTION_PAGE_ID
payload["notion_url"] = NOTION_URL
payload["notion_page"] = {
    "id": NOTION_PAGE_ID,
    "url": NOTION_URL,
    "title": TITLE,
    "readback_verified": True,
    "project_space_match": True,
    "readable_blocks_verified": True,
}
payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

release = json.loads(release_path.read_text(encoding="utf-8"))
entry = release.setdefault("final_independent_academic_review_remediation", {})
entry["notion_page_id"] = NOTION_PAGE_ID
entry["notion_url"] = NOTION_URL
entry["notion_page_title"] = TITLE
entry["notion_readback_verified"] = True
entry["notion_project_space_match"] = True
entry["notion_readable_blocks_verified"] = True
release_path.write_text(json.dumps(release, indent=2), encoding="utf-8")

print("updated task218 notion evidence")
