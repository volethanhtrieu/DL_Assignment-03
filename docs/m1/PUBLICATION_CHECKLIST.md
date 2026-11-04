# M1 publication checklist

Overall: **PARTIAL - publication blocked by GitHub integration permissions**. Both authorized blob-create requests returned HTTP 403, Resource not accessible by integration. No commit was created and no branch update/push was requested. Local preparation is not deployment evidence.

| Requirement | Status | Evidence |
|---|---|---|
| Current handbook, repositories and M1 artifacts inspected | PASS | Part I, Sections 7.6 and 24-32; pinned repository snapshots and current proposal. |
| Actual AI tool identity | PASS | ChatGPT Work/Codex; underlying model/build unverified; one proposal disclosure cell changed, other content preserved. |
| Primary counts, protocol and provenance | PASS | 7,938 = 6,001/986/951; 141 exclusions; pinned sources and DATASET_LOCK.json. |
| M1 progress and limitations | PASS | No training/approval/M2/M3 completion asserted; archive and 9/9 minority limits disclosed. |
| Derived evidence and reproducibility code prepared | PASS | Public metadata/manifests, replay script, aggregate EDA, prior measured check records. |
| Raw data and sensitive content excluded | PASS | Allowlisted files; no raw tweets/images, Parquet, large archives or credentials. |
| Local tests, PDF and navigation checks | PASS | 270 static/file/path checks and 16 local HTTP requests passed; primary CSVs rebuilt byte-identically. Corrected PDF is 7 pages; only one disclosure value changed. Browser visual check was not run because Chromium is absent. These checks do not establish public deployment. |
| GitHub commits/push | FAIL | Both repositories reject writes with HTTP 403: Resource not accessible by integration. API permissions report push=false. No new commits or push; existing history remains unchanged. |
| Public Pages, PDF access and navigation | PARTIAL | Existing main landing page was retrieved publicly. New M1 content/PDF were not deployed or verified live. A3 has_pages=false; the prepared canonical A3 page is under the existing main site. Local HTTP/PDF/navigation checks passed, but are not public checks. |
| Student final review and LMS actions | PARTIAL | Nguyễn Huỳnh Nam Quốc remains responsible; review/registration/naming/packaging/upload not attested here. |

The earlier compliance PDF is a pre-publication record and is preserved as historical evidence. M2/M3 model results, ablations, checkpoints, slides and video are not M1 submission blockers. Prior approval is waived as reported by the group; no formal approval is invented. No LMS submission is performed by this publication task.

## Observed remote state

- A3 base and observed unchanged main head: `2640214b2c45b0acea677de2167fe281c0396d6a`.
- Landing repository base and observed unchanged main head: `6a810f4e1fa2827a0af046a34e72b1c7cef30780`.
- No force push, model training or LMS submission. Both GitHub write failures are provider permission errors, not instructor dataset-approval requirements.
- Reconnect/configure GitHub with write permission to **both** named repositories to complete publication, then verify commits and the new public Pages/PDF links. A3-specific Pages enablement is not assumed; its content is prepared for the existing main site's assignment-03 path.
- Visual browser checks were not run because Chromium is absent. The proposal's changed page was visually inspected; pages 1-6 are byte-identical rendered images to the previously verified version.
