let currentJobId = null;
let timer = null;

const fileInput = document.getElementById("file");
const fileInfo = document.getElementById("fileInfo");

const runBtn = document.getElementById("runBtn");
const runDefaultBtn = document.getElementById("runDefaultBtn");

const statusEl = document.getElementById("status");
const detailEl = document.getElementById("detail");
const progressEl = document.getElementById("progress");
const barEl = document.getElementById("bar");
const jobidEl = document.getElementById("jobid");

const downloadsEl = document.getElementById("downloads");
const tableWrapEl = document.getElementById("tableWrap");
const summaryEl = document.getElementById("summary");
const tbody = document.querySelector("#tbl tbody");

const dlUseCases = document.getElementById("dl_use_cases");
const dlLabeled = document.getElementById("dl_labeled");
const dlMap = document.getElementById("dl_map");
const dlScores = document.getElementById("dl_scores");

fileInput.addEventListener("change", () => {
  const f = fileInput.files[0];
  fileInfo.textContent = f ? ("Selected: " + f.name) : "";
});

runBtn.addEventListener("click", async () => {
  await runJob({useDefault: false});
});

runDefaultBtn.addEventListener("click", async () => {
  await runJob({useDefault: true});
});

function setStatus(status, detail, progress){
  statusEl.textContent = status;
  detailEl.textContent = detail || "";
  progressEl.textContent = progress ?? 0;
  barEl.style.width = (progress ?? 0) + "%";
}

function clearResults(){
  downloadsEl.style.display = "none";
  tableWrapEl.style.display = "none";
  summaryEl.style.display = "none";
  tbody.innerHTML = "";
}

async function runJob({useDefault}){
  clearResults();
  setStatus("starting", "Submitting job...", 1);

  const f = fileInput.files[0];
  const form = new FormData();

  // If user clicks Run (not default) and chose a file, upload it.
  if (!useDefault && f){
    form.append("file", f);
  }

  const resp = await fetch("/api/run", { method: "POST", body: form });
  const data = await resp.json();

  if (!resp.ok){
    setStatus("error", data.error || "Unknown error", 0);
    return;
  }

  currentJobId = data.job_id;
  jobidEl.textContent = currentJobId;

  setStatus("queued", "Job queued. Polling...", 2);

  if (timer) clearInterval(timer);
  timer = setInterval(poll, 1200);
}

async function poll(){
  if (!currentJobId) return;

  const resp = await fetch("/api/status/" + currentJobId);
  const data = await resp.json();

  if (!resp.ok){
    setStatus("error", data.error || "Status error", 0);
    clearInterval(timer);
    return;
  }

  // Show real error messages if any
  if (data.status === "error"){
    setStatus("error", data.error || data.detail || "Unknown error", data.progress || 0);
    clearInterval(timer);
    return;
  }

  setStatus(data.status, data.detail, data.progress);

  if (data.status === "done"){
    clearInterval(timer);
    renderResult(currentJobId, data.result);
  }
}

function renderResult(jobId, result){
  downloadsEl.style.display = "block";

  dlUseCases.href = `/runs/${jobId}/use_cases.csv`;
  dlLabeled.href  = `/runs/${jobId}/labeled_rows.csv`;
  dlMap.href      = `/runs/${jobId}/use_case_map.json`;
  dlScores.href   = `/runs/${jobId}/k_silhouette_scores.json`;

  summaryEl.style.display = "block";
  summaryEl.innerHTML =
    `<b>Summary:</b> rows=${result.rows}, k=${result.k}, use_cases=${result.use_cases}`;

  tbody.innerHTML = "";
  (result.preview || []).forEach(row => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(row["Use Case"] ?? "")}</td>
      <td>${escapeHtml(row["Description"] ?? "")}</td>
      <td>${row["Prompts"] ?? ""}</td>
      <td>${row["Visitors"] ?? ""}</td>
      <td>${row["Accounts"] ?? ""}</td>
      <td>${row["Retention"] ?? ""}</td>
      <td>${row["% Rage Prompts"] ?? ""}</td>
      <td>${row["cluster_id"] ?? ""}</td>
    `;
    tbody.appendChild(tr);
  });

  tableWrapEl.style.display = "block";
}

function escapeHtml(s){
  return String(s)
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}
