(() => {
  const state = {
    exercises: [],
    selectedExercise: "squat",
    analysisMode: "single_rep",
    selectedFile: null,
    localPreviewUrl: null,
    preparedJobId: null,
    previewSeq: 0,
    previewPromise: null,
    dashboardLoaded: false,
    webcamTimer: null,
    webcamMirror: true,
    wcFullscreen: false,
    wcExercise: "squat",
    wcTargetReps: 10,
    wcUseAllFolds: false,
    wcConfirmed: false,
    wcActive: false,
  };

  const $ = (id) => document.getElementById(id);

  function escapeHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function formatBytes(bytes) {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  // -------------------- TAB SWITCHING --------------------
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;
      document.querySelectorAll(".tab-btn").forEach((b) => {
        const active = b === btn;
        b.classList.toggle("active", active);
        b.setAttribute("aria-selected", active ? "true" : "false");
      });
      document.querySelectorAll(".tab-panel").forEach((panel) => {
        const match = panel.id === `panel-${target}`;
        panel.classList.toggle("active", match);
        panel.hidden = !match;
      });
      if (target !== "webcam") {
        exitWebcamFullscreen();
      }
      if (target === "dashboard" && !state.dashboardLoaded) {
        loadDashboard();
      }
    });
  });

  // -------------------- EXERCISES LOAD --------------------
  async function loadExercises() {
    try {
      const resp = await fetch("/api/exercises");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      state.exercises = data.exercises || [];
      renderExerciseGrid("exercise-grid", state.selectedExercise, (exId) => {
        state.selectedExercise = exId;
      });
      renderExerciseGrid("wc-exercise-grid", state.wcExercise, (exId) => {
        if (state.wcActive) return;
        state.wcExercise = exId;
        state.wcConfirmed = false;
        updateWebcamConfirmUI();
      });
      updateWebcamConfirmUI();
      if (!state.dashboardLoaded) {
        loadDashboard();
      }
    } catch (err) {
      const msg = `Không kết nối được máy chủ bài tập (${err.message}). Hãy đảm bảo server đang chạy tại http://127.0.0.1:8000 (chạy run_web.bat) rồi tải lại trang.`;
      showUploadError(msg);
      if ($("wc-error")) {
        $("wc-error").textContent = msg;
        $("wc-error").classList.remove("hidden");
      }
    }
  }

  function renderExerciseGrid(containerId, activeId, onSelect) {
    const root = $(containerId);
    root.innerHTML = "";
    state.exercises.forEach((ex) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `ex-btn${ex.id === activeId ? " active" : ""}`;
      btn.innerHTML = `<strong>${escapeHtml(ex.name_vi)}</strong><span>${escapeHtml(ex.driver_vi)}</span>`;
      btn.addEventListener("click", () => {
        root.querySelectorAll(".ex-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        onSelect(ex.id);
      });
      root.appendChild(btn);
    });
  }

  // -------------------- UPLOAD DROPZONE --------------------
  const dropzone = $("dropzone");
  const fileInput = $("video-input");
  const btnAnalyze = $("btn-analyze");

  function showPreviewStatus(msg, isError = false) {
    const el = $("preview-status");
    if (!msg) {
      el.textContent = "";
      el.className = "preview-status hidden";
      return;
    }
    el.textContent = msg;
    el.className = `preview-status${isError ? " error" : ""}`;
  }

  async function requestServerPreview(file, seq) {
    if (seq !== state.previewSeq) return;
    showPreviewStatus("Trình duyệt không giải mã được codec gốc (.MOV/HEVC) — đang chuyển sang H.264 để xem trước...");
    const fd = new FormData();
    fd.append("file", file);
    try {
      const resp = await fetch("/api/preview-video", { method: "POST", body: fd });
      const data = await resp.json();
      if (seq !== state.previewSeq) return;
      if (!resp.ok) {
        throw new Error(data.detail || "Không tạo được bản xem trước.");
      }
      state.preparedJobId = data.preview_job_id;
      const vid = $("local-preview-video");
      if (state.localPreviewUrl) {
        URL.revokeObjectURL(state.localPreviewUrl);
        state.localPreviewUrl = null;
      }
      vid.onloadedmetadata = null;
      vid.onerror = null;
      vid.onplay = null;
      vid.onloadeddata = () => {
        if (seq === state.previewSeq) showPreviewStatus("");
      };
      vid.src = `${data.preview_url}?t=${Date.now()}`;
      vid.load();
      showPreviewStatus("");
    } catch (err) {
      if (seq === state.previewSeq) {
        showPreviewStatus(err.message, true);
      }
    } finally {
      if (seq === state.previewSeq) {
        state.previewPromise = null;
      }
    }
  }

  function setSelectedFile(file) {
    if (!file) return;
    hideUploadError();
    state.selectedFile = file;
    state.preparedJobId = null;
    state.previewSeq += 1;
    const seq = state.previewSeq;
    state.previewPromise = null;
    showPreviewStatus("");

    if (state.localPreviewUrl) {
      URL.revokeObjectURL(state.localPreviewUrl);
      state.localPreviewUrl = null;
    }

    const vid = $("local-preview-video");
    const ext = (file.name.split(".").pop() || "").toLowerCase();
    const mime = (file.type || "").toLowerCase();
    const alwaysTranscodePreview = ["mov", "avi", "mkv"].includes(ext) || mime.includes("quicktime");

    $("file-name").textContent = file.name;
    $("file-size").textContent = formatBytes(file.size);
    $("dropzone-empty").classList.add("hidden");
    $("dropzone-preview").classList.remove("hidden");
    btnAnalyze.disabled = false;

    if (alwaysTranscodePreview) {
      vid.onloadedmetadata = null;
      vid.onloadeddata = null;
      vid.onerror = null;
      vid.onplay = null;
      vid.removeAttribute("src");
      vid.load();
      state.previewPromise = requestServerPreview(file, seq);
      return;
    }

    state.localPreviewUrl = URL.createObjectURL(file);
    let serverRequested = false;
    const triggerFallback = () => {
      if (serverRequested || seq !== state.previewSeq) return;
      serverRequested = true;
      state.previewPromise = requestServerPreview(file, seq);
    };

    vid.onloadedmetadata = () => {
      if (!vid.videoWidth || !vid.videoHeight) {
        triggerFallback();
      }
    };
    vid.onerror = () => triggerFallback();
    vid.onplay = () => {
      setTimeout(() => {
        if (seq !== state.previewSeq || vid.paused) return;
        const q = typeof vid.getVideoPlaybackQuality === "function" ? vid.getVideoPlaybackQuality() : null;
        if (q && q.totalVideoFrames === 0) {
          triggerFallback();
        }
      }, 450);
    };

    vid.src = state.localPreviewUrl;
    vid.load();
  }

  $("dropzone-empty").addEventListener("click", () => fileInput.click());
  $("btn-change-file").addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && !state.selectedFile) {
      e.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) setSelectedFile(fileInput.files[0]);
  });

  ["dragenter", "dragover"].forEach((ev) => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.add("drag-over");
    });
  });
  ["dragleave", "drop"].forEach((ev) => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag-over");
    });
  });
  dropzone.addEventListener("drop", (e) => {
    const f = e.dataTransfer?.files?.[0];
    if (f) setSelectedFile(f);
  });

  document.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.analysisMode = btn.dataset.mode;
    });
  });

  function showUploadError(msg) {
    const box = $("upload-error");
    box.textContent = msg;
    box.classList.remove("hidden");
  }

  function hideUploadError() {
    $("upload-error").classList.add("hidden");
  }

  function buildAnalyzeFormData(usePrepared) {
    const formData = new FormData();
    if (usePrepared && state.preparedJobId) {
      formData.append("prepared_job_id", state.preparedJobId);
    } else if (state.selectedFile) {
      formData.append("file", state.selectedFile);
    }
    formData.append("exercise", state.selectedExercise);
    formData.append("analysis_mode", state.analysisMode);
    formData.append("use_all_folds", $("toggle-all-folds").checked ? "true" : "false");
    return formData;
  }

  btnAnalyze.addEventListener("click", async () => {
    if (!state.selectedFile) return;
    hideUploadError();
    btnAnalyze.disabled = true;
    $("loading-box").classList.remove("hidden");
    $("result-section").classList.add("hidden");

    try {
      if (state.previewPromise) {
        await state.previewPromise;
      }
      let resp = await fetch("/api/analyze-upload", {
        method: "POST",
        body: buildAnalyzeFormData(Boolean(state.preparedJobId)),
      });
      if (resp.status === 410 && state.preparedJobId) {
        state.preparedJobId = null;
        resp = await fetch("/api/analyze-upload", {
          method: "POST",
          body: buildAnalyzeFormData(false),
        });
      }
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.detail || `Lỗi máy chủ (${resp.status})`);
      }
      renderAnalysisResult(data);
    } catch (err) {
      showUploadError(err.message);
    } finally {
      $("loading-box").classList.add("hidden");
      btnAnalyze.disabled = false;
    }
  });

  // -------------------- RENDER RESULT --------------------
  function renderAnalysisResult(data) {
    const sec = $("result-section");
    const vid = $("result-video");
    const dl = $("btn-download-video");
    const cacheBust = `${data.video_url}?t=${Date.now()}`;
    vid.src = cacheBust;
    vid.load();
    dl.href = `${data.video_url}?download=1`;
    dl.setAttribute("download", `forgai_hud_${data.job_id}.mp4`);

    if (data.mode === "single_rep") {
      renderSingleRepDetails(data);
      renderDriverCurveChart(data);
    } else {
      $("single-curve-card").classList.add("hidden");
      renderMultiRepDetails(data);
    }
    sec.classList.remove("hidden");
    sec.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function buildGaugeSvg(scorePct, isIncorrect) {
    const r = 44;
    const c = 2 * Math.PI * r;
    const clamped = Math.max(0, Math.min(100, Number(scorePct) || 0));
    const offset = c * (1 - clamped / 100);
    const stroke = isIncorrect ? "#ef4444" : "#22c55e";
    return `
      <div class="gauge-wrap" role="img" aria-label="Độ tin cậy ĐÚNG ${clamped.toFixed(1)}%">
        <svg width="110" height="110" viewBox="0 0 110 110">
          <circle cx="55" cy="55" r="${r}" fill="none" stroke="rgba(148,163,184,0.18)" stroke-width="10" />
          <circle cx="55" cy="55" r="${r}" fill="none" stroke="${stroke}" stroke-width="10"
            stroke-linecap="round" stroke-dasharray="${c.toFixed(1)}" stroke-dashoffset="${offset.toFixed(1)}"
            transform="rotate(-90 55 55)" />
        </svg>
        <div class="gauge-center">
          <span class="gauge-score">${clamped.toFixed(1)}%</span>
          <span class="gauge-lbl">Tin cậy ĐÚNG</span>
        </div>
      </div>`;
  }

  function renderSingleRepDetails(d) {
    const cls = d.is_incorrect ? "incorrect" : "correct";
    const statusIcon = d.is_incorrect
      ? `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" aria-hidden="true"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`
      : `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" aria-hidden="true"><polyline points="20 6 9 17 4 12"/></svg>`;

    const branches = [
      { name: "BiomechanicalRF (103-d E.3 + Prior Shift)", val: d.branches.rf },
      { name: "CNN-BiLSTM + FiLM (166K params)", val: d.branches.cnn_bilstm },
      { name: "Transformer + FiLM (96K params)", val: d.branches.transformer },
      { name: "Hybrid Ensemble (0.60 / 0.25 / 0.15)", val: d.branches.hybrid },
    ];
    const branchHtml = branches
      .map((b) => {
        const pct = (b.val * 100).toFixed(1);
        const bCls = b.val >= d.threshold ? "incorrect" : "correct";
        return `
          <div>
            <div class="branch-row-head">
              <span>${escapeHtml(b.name)}</span>
              <strong class="num-tabular">P(Sai) = ${pct}%</strong>
            </div>
            <div class="bar-track"><div class="bar-fill ${bCls}" style="width:${pct}%"></div></div>
          </div>`;
      })
      .join("");

    const jointsHtml = (d.highlight_joints_names || []).length
      ? `<div class="joint-chips">${d.highlight_joints_names.map((j) => `<span class="joint-chip">${escapeHtml(j)}</span>`).join("")}</div>`
      : "";

    const top3Html = (d.top3_candidates || []).length
      ? `
        <div class="card">
          <div class="card-header"><h2>Top-3 Nguyên nhân Cơ sinh học Tiềm ẩn</h2></div>
          <div class="table-wrap">
            <table class="data-table">
              <thead><tr><th>#</th><th>Tên lỗi tiếng Việt</th><th>Độ tin cậy</th><th>Luật vật lý</th></tr></thead>
              <tbody>
                ${d.top3_candidates
                  .map(
                    (c, idx) => `<tr>
                      <td>${idx + 1}</td>
                      <td>${escapeHtml(c.vietnamese_name)} <span class="empty-muted">(${escapeHtml(c.error_reason)})</span></td>
                      <td>${(c.score * 100).toFixed(1)}%</td>
                      <td>${(c.rule_score * 100).toFixed(1)}%</td>
                    </tr>`
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
        </div>`
      : "";

    const tm = d.timings_ms || {};
    const scoreNum = Number(d.form_score_pct) || 0;
    const foldsArr = d.folds_used || [];
    const modelSummary = foldsArr.length >= 5
      ? "Hội đồng 5 mô hình (0..4)"
      : `1 mô hình nhanh (#${foldsArr[0] ?? 2})`;
    let confNote = "";
    let confCls = "good";
    if (!d.is_incorrect) {
      if (scoreNum >= 75) {
        confNote = "Mức tin cậy: Cao (động tác chuẩn rõ ràng)";
        confCls = "good";
      } else {
        confNote = "Mức tin cậy: Sát ranh giới 50/50 — nghiêng về Đúng nhưng nên kiểm tra lại góc khớp";
        confCls = "borderline";
      }
    } else if (scoreNum <= 25) {
      confNote = "Mức tin cậy: Cao (phát hiện sai lệch kỹ thuật rõ ràng)";
      confCls = "fault";
    } else {
      confNote = "Mức tin cậy: Sát ranh giới 50/50 — nghiêng về Sai nhẹ, đối chiếu bằng chứng bên dưới";
      confCls = "borderline";
    }

    $("result-details-col").innerHTML = `
      <div class="card verdict-card ${cls}">
        ${buildGaugeSvg(d.form_score_pct, d.is_incorrect)}
        <div>
          <span class="status-badge ${cls}">${statusIcon}<span>${escapeHtml(d.label_vi)}</span></span>
          <h3 style="margin:0;font-size:1.08rem;">${escapeHtml(d.exercise_vi)}</h3>
          <span class="confidence-note ${confCls}">${escapeHtml(confNote)}</span>
          <div class="verdict-meta num-tabular">
            <span>Thời lượng: <strong>${d.duration_sec}s</strong> (${d.num_frames} frames)</span>
            <span>Bắt khớp: <strong>${(d.det_rate * 100).toFixed(1)}%</strong></span>
            <span>AI lõi: <strong>${tm.total_ai_ms ?? "-"} ms/rep</strong></span>
            <span>Mô hình chấm: <strong>${escapeHtml(modelSummary)}</strong></span>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header"><h2>Chẩn đoán Cơ sinh học &amp; Gợi ý Huấn luyện viên</h2></div>
        <div class="diag-box">
          <div class="diag-row ${d.is_incorrect ? "fault" : ""}">
            <strong>Chẩn đoán chính</strong>
            <span>${escapeHtml(d.vietnamese_error_name)} (<code>${escapeHtml(d.predicted_error_code)}</code>)</span>
            ${jointsHtml}
          </div>
          <div class="diag-row">
            <strong>Bằng chứng đo đạc</strong>
            <span>${escapeHtml(d.measured_evidence)}</span>
          </div>
          <div class="diag-row advice">
            <strong>Gợi ý chỉnh sửa kỹ thuật</strong>
            <span>${escapeHtml(d.coaching_advice)}</span>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header"><h2>Xác suất Nghi ngờ Sai của 4 Mô hình (≥ 50% là Sai)</h2></div>
        <p class="card-subnote">Thang đo <strong>P(Sai)</strong> là xác suất mô hình nghi ngờ có lỗi (0% = rất chuẩn, 100% = chắc chắn lỗi). Mốc <strong>50%</strong> là ranh giới phân đôi 50/50 giữa Đúng và Sai (không phải mức điểm 5/10 là đạt) — dưới 50% tức AI nghiêng về kết luận ĐÚNG.</p>
        <div class="branch-list">${branchHtml}</div>
      </div>

      ${top3Html}
    `;
  }

  function renderMultiRepDetails(d) {
    const reps = d.reps || [];
    const avgTxt = d.avg_form_score_pct !== null ? `${d.avg_form_score_pct}%` : "-";
    const repCardsHtml = reps.length
      ? reps
          .map(
            (r) => `
            <div class="rep-card ${r.is_incorrect ? "incorrect" : "correct"}">
              <div class="rep-card-head">
                <strong>Rep #${r.rep_index} (${r.start_sec}s – ${r.end_sec}s · ${r.duration_sec}s)</strong>
                <span class="status-badge ${r.is_incorrect ? "incorrect" : "correct"}">
                  ${escapeHtml(r.label_vi)} · Tin cậy Đúng ${r.form_score_pct}%
                </span>
              </div>
              <div class="rep-card-sub">
                <div><strong>Chẩn đoán:</strong> ${escapeHtml(r.vietnamese_error_name)} — ${escapeHtml(r.measured_evidence)}</div>
                <div style="margin-top:3px;"><strong>Gợi ý:</strong> ${escapeHtml(r.coaching_advice)}</div>
              </div>
            </div>`
          )
          .join("")
      : `<div class="alert-banner alert-error">Không phát hiện được chu kỳ rep hoàn chỉnh nào. Nếu video chỉ có 1 rep ngắn hoặc bị cắt đầu/cuối, hãy chuyển sang chế độ <strong>Video 1 Rep</strong>.</div>`;

    $("result-details-col").innerHTML = `
      <div class="card">
        <div class="card-header">
          <h2>Tổng hợp Cả Hiệp Tập — ${escapeHtml(d.exercise_vi)}</h2>
          <span class="hint-pill num-tabular">${d.duration_sec}s · ${d.timings_ms.end_to_end_per_frame_ms} ms/frame</span>
        </div>
        <div class="multi-summary-grid">
          <div class="stat-mini"><span>Tổng Rep đếm được</span><strong>${d.total_reps}</strong></div>
          <div class="stat-mini"><span>Rep Đúng</span><strong style="color:var(--accent-green)">${d.correct_reps}</strong></div>
          <div class="stat-mini"><span>Rep Sai</span><strong style="color:var(--accent-red)">${d.incorrect_reps}</strong></div>
          <div class="stat-mini"><span>TB Tin cậy Đúng</span><strong>${avgTxt}</strong></div>
        </div>
      </div>

      <div class="card">
        <div class="card-header"><h2>Chi tiết Từng Rep Trong Hiệp</h2></div>
        <div class="rep-cards">${repCardsHtml}</div>
      </div>
    `;
  }

  function renderDriverCurveChart(d) {
    const curve = d.driver_curve || [];
    const card = $("single-curve-card");
    if (curve.length < 2) {
      card.classList.add("hidden");
      return;
    }
    const minV = Math.min(...curve);
    const maxV = Math.max(...curve);
    const span = Math.max(maxV - minV, 1e-3);
    const W = 600;
    const H = 140;
    const padX = 28;
    const padY = 18;

    const pts = curve
      .map((val, i) => {
        const x = padX + (i / (curve.length - 1)) * (W - padX * 2);
        const y = H - padY - ((val - minV) / span) * (H - padY * 2);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");

    const unit = (d.driver_key || "").includes("dist") ? "" : "°";
    $("curve-title").textContent = `Quỹ đạo ${d.driver_key || "khớp chủ đạo"} qua ${curve.length} frames`;
    $("curve-stats").textContent = `Min: ${minV.toFixed(1)}${unit}  |  Max: ${maxV.toFixed(1)}${unit}  |  ROM: ${span.toFixed(1)}${unit}`;

    const stroke = d.is_incorrect ? "#f59e0b" : "#22c55e";
    $("curve-svg-wrap").innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" aria-label="Biểu đồ quỹ đạo góc khớp theo frame">
        <line x1="${padX}" y1="${padY}" x2="${W - padX}" y2="${padY}" stroke="rgba(148,163,184,0.18)" stroke-dasharray="4 4"/>
        <line x1="${padX}" y1="${H / 2}" x2="${W - padX}" y2="${H / 2}" stroke="rgba(148,163,184,0.12)" stroke-dasharray="4 4"/>
        <line x1="${padX}" y1="${H - padY}" x2="${W - padX}" y2="${H - padY}" stroke="rgba(148,163,184,0.18)" stroke-dasharray="4 4"/>
        <polyline fill="none" stroke="${stroke}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" points="${pts}" />
        <text x="4" y="${padY + 4}" fill="#94a3b8" font-size="10">${maxV.toFixed(0)}${unit}</text>
        <text x="4" y="${H - padY + 4}" fill="#94a3b8" font-size="10">${minV.toFixed(0)}${unit}</text>
      </svg>`;
    card.classList.remove("hidden");
  }

  // -------------------- TAB 2: WEBCAM (CHỌN BÀI -> CHỌN REP -> XÁC NHẬN -> BẮT ĐẦU) --------------------
  function getWcExerciseName() {
    const found = (state.exercises || []).find((x) => x.id === state.wcExercise);
    return found ? found.name_vi : state.wcExercise;
  }

  function syncRepSelection(val) {
    if (state.wcActive) return;
    let n = parseInt(val, 10);
    if (Number.isNaN(n)) n = 10;
    n = Math.max(0, Math.min(50, n));
    state.wcTargetReps = n;
    state.wcConfirmed = false;
    $("wc-target-reps-input").value = String(n);
    document.querySelectorAll("#wc-rep-presets .rep-chip").forEach((chip) => {
      chip.classList.toggle("active", Number(chip.dataset.reps) === n);
    });
    updateWebcamConfirmUI();
  }

  function updateWebcamConfirmUI() {
    const repTxt = state.wcTargetReps > 0 ? `Mục tiêu: ${state.wcTargetReps} rep` : "Tập tự do (không giới hạn)";
    const aiTxt = state.wcUseAllFolds ? "Hội đồng 5 mô hình AI" : "1 mô hình nhanh";
    const camTxt = $("wc-camera-index").value === "1" ? "Camera #1" : "Camera #0";
    $("wc-confirm-summary").textContent = `${getWcExerciseName()} · ${repTxt} · ${aiTxt} · ${camTxt}`;

    const badge = $("wc-confirm-badge");
    const box = $("wc-confirm-box");
    const setupFields = $("wc-setup-fields");
    const btnConfirm = $("btn-wc-confirm");
    const btnToggle = $("btn-wc-toggle");
    const btnReset = $("btn-wc-reset");
    const btnFsToggle = $("btn-fs-toggle");
    const btnFsReset = $("btn-fs-reset");
    const btnFsMirror = $("btn-fs-mirror");

    if ($("wc-fs-exercise")) {
      $("wc-fs-exercise").textContent = getWcExerciseName();
    }
    if (btnFsMirror) {
      btnFsMirror.textContent = `Lật gương: ${state.webcamMirror ? "Bật" : "Tắt"}`;
      btnFsMirror.classList.toggle("active-outline", state.webcamMirror);
    }

    if (state.wcActive) {
      badge.textContent = "Đang trong hiệp";
      badge.className = "status-pill status-live";
      box.classList.add("confirmed");
      setupFields.classList.add("wc-locked");
      btnConfirm.disabled = true;
      btnConfirm.textContent = "Đã khóa khi đang chạy";
      btnToggle.disabled = false;
      btnToggle.dataset.active = "1";
      btnToggle.innerHTML = "<span>Dừng Camera</span>";
      btnReset.disabled = false;
      if (btnFsToggle) {
        btnFsToggle.disabled = false;
        btnFsToggle.textContent = "Dừng Camera";
      }
      if (btnFsReset) btnFsReset.disabled = false;
      return;
    }

    btnToggle.dataset.active = "0";
    btnToggle.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <circle cx="12" cy="12" r="10" />
        <polygon points="10 8 16 12 10 16 10 8" />
      </svg>
      <span>Bắt đầu Camera</span>`;
    btnReset.disabled = true;
    btnConfirm.disabled = false;
    setupFields.classList.remove("wc-locked");
    if (btnFsReset) btnFsReset.disabled = true;
    if (btnFsToggle) btnFsToggle.textContent = "Bắt đầu Camera";

    if (state.wcConfirmed) {
      badge.textContent = "Đã xác nhận ✓";
      badge.className = "status-pill status-live";
      box.classList.add("confirmed");
      btnConfirm.textContent = "Hủy xác nhận";
      btnToggle.disabled = false;
      if (btnFsToggle) btnFsToggle.disabled = false;
    } else {
      badge.textContent = "Chưa xác nhận";
      badge.className = "status-pill status-pending";
      box.classList.remove("confirmed");
      btnConfirm.textContent = "Xác nhận cấu hình";
      btnToggle.disabled = true;
      if (btnFsToggle) btnFsToggle.disabled = true;
    }

    if (!$("webcam-badge").classList.contains("status-live")) {
      const idleReps = state.wcTargetReps > 0 ? `0 / ${state.wcTargetReps} rep` : "0 rep (Tự do)";
      $("wc-rep-count").textContent = idleReps;
      if ($("wc-fs-reps")) $("wc-fs-reps").textContent = idleReps;
    }
  }

  async function startWebcam() {
    if (!state.wcConfirmed) {
      $("wc-error").textContent = "Hãy chọn bài, chọn số rep và bấm Xác nhận cấu hình trước khi bắt đầu.";
      $("wc-error").classList.remove("hidden");
      return;
    }
    $("wc-error").classList.add("hidden");
    $("wc-summary-banner").classList.add("hidden");
    if ($("wc-fs-summary")) $("wc-fs-summary").classList.add("hidden");
    if ($("wc-stage-banner")) $("wc-stage-banner").classList.add("hidden");
    $("wc-last-result").classList.add("empty-muted");
    $("wc-last-result").textContent = "Hãy vào tư thế chuẩn bị và giữ ổn định ~0.5 giây trước khi tập...";
    if ($("wc-fs-last-result")) {
      $("wc-fs-last-result").className = "wc-fs-last-result";
      $("wc-fs-last-result").textContent = "Hãy vào tư thế chuẩn bị và giữ ổn định ~0.5 giây trước khi tập...";
    }
    $("wc-history").innerHTML = "";

    const camIdx = Number($("wc-camera-index").value || 0);
    const resp = await fetch("/api/webcam/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        exercise: state.wcExercise,
        target_reps: state.wcTargetReps,
        use_all_folds: state.wcUseAllFolds,
        camera_index: camIdx,
        mirror: state.webcamMirror,
      }),
    });
    const st = await resp.json();
    if (!resp.ok) {
      $("wc-error").textContent = st.detail || "Không bật được camera.";
      $("wc-error").classList.remove("hidden");
      return;
    }
    state.wcActive = true;
    const img = $("webcam-stream-img");
    img.src = `/api/webcam/stream?t=${Date.now()}`;
    img.classList.remove("hidden");
    $("webcam-placeholder").classList.add("hidden");
    if (state.webcamTimer) clearInterval(state.webcamTimer);
    state.webcamTimer = setInterval(pollWebcamStatus, 400);
    renderWebcamStatus(st);
  }

  async function stopWebcam() {
    if (state.webcamTimer) {
      clearInterval(state.webcamTimer);
      state.webcamTimer = null;
    }
    const resp = await fetch("/api/webcam/stop", { method: "POST" });
    const st = await resp.json();
    state.wcActive = false;
    $("webcam-stream-img").src = "";
    $("webcam-stream-img").classList.add("hidden");
    $("webcam-placeholder").classList.remove("hidden");
    renderWebcamStatus(st);
  }

  async function pollWebcamStatus() {
    try {
      const resp = await fetch("/api/webcam/status");
      const st = await resp.json();
      renderWebcamStatus(st);
    } catch (_) {}
  }

  function renderWebcamStatus(st) {
    const active = Boolean(st.active);
    const completed = Boolean(st.completed);
    state.wcActive = active;

    if (!active && state.webcamTimer) {
      clearInterval(state.webcamTimer);
      state.webcamTimer = null;
    }

    const badge = $("webcam-badge");
    if (active) {
      badge.textContent = "Đang chạy Live";
      badge.className = "status-pill status-live";
    } else if (completed) {
      badge.textContent = "Đã hoàn thành hiệp ✓";
      badge.className = "status-pill status-live";
    } else {
      badge.textContent = "Chưa bắt đầu";
      badge.className = "status-pill status-idle";
    }

    if (st.error) {
      $("wc-error").textContent = st.error;
      $("wc-error").classList.remove("hidden");
      if (!active) {
        $("webcam-stream-img").src = "";
        $("webcam-stream-img").classList.add("hidden");
        $("webcam-placeholder").classList.remove("hidden");
      }
    }

    const cycleText = completed
      ? "HOÀN THÀNH MỤC TIÊU ✓"
      : st.state === "IN_REP"
        ? "ĐANG THỰC HIỆN REP"
        : st.tracking_lost
          ? "MẤT KHUNG XƯƠNG"
          : st.stabilizing
            ? "ĐANG ỔN ĐỊNH KHỚP"
            : st.waiting_ready
              ? "CHỜ TƯ THẾ CHUẨN BỊ"
              : active
                ? "SẴN SÀNG / ĐẾM REP"
                : "CHƯA CHẠY";
    $("wc-state").textContent = cycleText;

    const stageBanner = $("wc-stage-banner");
    if (stageBanner) {
      if (completed && st.summary) {
        stageBanner.className = "wc-stage-banner is-done";
        stageBanner.textContent = `Hoàn thành mục tiêu ${st.summary.total_reps}/${st.summary.target_reps} rep · Xem chi tiết Rep cuối bên dưới`;
      } else if (active && st.tracking_lost) {
        stageBanner.className = "wc-stage-banner is-warn";
        stageBanner.textContent = "Chưa đủ khung xương toàn thân — hãy lùi để thấy rõ vai & hông";
      } else if (active && (st.stabilizing || st.waiting_ready)) {
        stageBanner.className = "wc-stage-banner is-warn";
        stageBanner.textContent = st.stabilizing
          ? "Đang ổn định lại khung xương..."
          : `Vào đúng tư thế chuẩn bị (${st.exercise || ""}) và giữ yên ~0.5 giây`;
      } else {
        stageBanner.className = "wc-stage-banner hidden";
      }
    }

    const tReps = Number(st.target_reps ?? state.wcTargetReps ?? 0);
    const repCnt = Number(st.rep_count || 0);
    const corrCnt = Number(st.correct_count || 0);
    const pctStr = `${Math.round((st.progress || 0) * 100)}%`;
    const repText = tReps > 0
      ? `${repCnt} / ${tReps} rep (${corrCnt} đúng)`
      : `${repCnt} rep (${corrCnt} đúng)`;
    const fpsText = `${st.fps || 0} FPS · ${st.proc_ms || 0} ms`;

    $("wc-progress-bar").style.width = pctStr;
    $("wc-rep-count").textContent = repText;
    $("wc-fps").textContent = fpsText;

    if ($("wc-fs-state")) {
      $("wc-fs-state").textContent = cycleText;
      $("wc-fs-state").classList.toggle("is-warn", Boolean(st.tracking_lost || st.stabilizing));
      $("wc-fs-state").classList.toggle("is-ready", Boolean(!completed && active && st.waiting_ready));
      $("wc-fs-state").classList.toggle("is-done", Boolean(completed));
      $("wc-fs-state").classList.toggle("is-active", Boolean(!completed && st.state === "IN_REP"));
    }
    if ($("wc-fs-progress-bar")) $("wc-fs-progress-bar").style.width = pctStr;
    if ($("wc-fs-reps")) $("wc-fs-reps").textContent = repText;
    if ($("wc-fs-fps")) $("wc-fs-fps").textContent = `${fpsText} · ${st.display_width || 960}px`;

    if (st.summary) {
      const s = st.summary;
      const prefix = s.target_reached
        ? `Hoàn thành đủ mục tiêu ${s.total_reps}/${s.target_reps} rep!`
        : `Đã dừng hiệp (${s.total_reps} rep đã chấm).`;
      const sumText = `${prefix} Đúng: ${s.correct_count} · Sai: ${s.incorrect_count} · TB Tin cậy Đúng: ${s.avg_form_score_pct ?? "-"}%`;
      $("wc-summary-banner").textContent = sumText;
      $("wc-summary-banner").classList.remove("hidden");
      if ($("wc-fs-summary")) {
        $("wc-fs-summary").textContent = sumText;
        $("wc-fs-summary").classList.remove("hidden");
      }
    } else {
      $("wc-summary-banner").classList.add("hidden");
      if ($("wc-fs-summary")) $("wc-fs-summary").classList.add("hidden");
    }

    if (st.last_result) {
      const r = st.last_result;
      const scorePct = (100 - r.p_incorrect * 100).toFixed(1);
      const finalTag = completed ? `<span class="fs-chip is-done" style="padding:2px 8px;font-size:0.75rem;">Rep cuối</span>` : "";
      const cardHtml = `
        <div class="rep-card-head">
          <strong>Rep #${r.rep_index} · ${escapeHtml(r.exercise)} ${finalTag}</strong>
          <span class="status-badge ${r.is_incorrect ? "incorrect" : "correct"}">${escapeHtml(r.label_vi)} (Tin cậy Đúng ${scorePct}%)</span>
        </div>
        <div><strong>Chẩn đoán:</strong> ${escapeHtml(r.vietnamese_error_name)} — ${escapeHtml(r.measured_evidence)}</div>
        <div style="margin-top:4px;color:var(--text-secondary);"><strong>Gợi ý:</strong> ${escapeHtml(r.coaching_advice)}</div>`;
      $("wc-last-result").classList.remove("empty-muted");
      $("wc-last-result").innerHTML = cardHtml;
      if ($("wc-fs-last-result")) {
        $("wc-fs-last-result").className = `wc-fs-last-result ${r.is_incorrect ? "is-incorrect" : "is-correct"}`;
        $("wc-fs-last-result").innerHTML = cardHtml;
      }
    } else if (active && repCnt === 0) {
      const waitMsg = st.tracking_lost
        ? "Chưa phát hiện đủ khung xương (vai & hông) — hãy đứng lùi trọn người vào khung hình..."
        : st.stabilizing
          ? "Đang ổn định lại khớp xương trước khi đếm rep..."
          : st.waiting_ready
            ? "Hãy vào đúng tư thế chuẩn bị và giữ ổn định ~0.5 giây để mở khóa đếm rep..."
            : "Đã khóa tư thế chuẩn bị ✓ — Hãy thực hiện trọn vẹn rep đầu tiên trước camera...";
      $("wc-last-result").classList.add("empty-muted");
      $("wc-last-result").textContent = waitMsg;
      if ($("wc-fs-last-result")) {
        $("wc-fs-last-result").className = "wc-fs-last-result";
        $("wc-fs-last-result").textContent = waitMsg;
      }
    }

    $("wc-history").innerHTML = (st.recent_reps || [])
      .map(
        (h) => `
        <div class="rep-card ${h.is_incorrect ? "incorrect" : "correct"}" style="padding:8px 10px;">
          <div class="rep-card-head" style="margin-bottom:2px;">
            <span>Rep #${h.rep_index} (${h.duration_sec}s)</span>
            <strong style="color:${h.is_incorrect ? "var(--accent-red)" : "var(--accent-green)"}">Đúng ${h.form_score_pct}%</strong>
          </div>
          <div class="rep-card-sub">${escapeHtml(h.vietnamese_error_name)}</div>
        </div>`
      )
      .join("");

    updateWebcamConfirmUI();
  }

  document.querySelectorAll("#wc-rep-presets .rep-chip").forEach((chip) => {
    chip.addEventListener("click", () => syncRepSelection(chip.dataset.reps));
  });
  $("wc-target-reps-input").addEventListener("input", (e) => {
    syncRepSelection(e.target.value);
  });
  $("wc-camera-index").addEventListener("change", () => {
    if (!state.wcActive) {
      state.wcConfirmed = false;
      updateWebcamConfirmUI();
    }
  });
  $("wc-toggle-all-folds").addEventListener("change", (e) => {
    if (!state.wcActive) {
      state.wcUseAllFolds = Boolean(e.target.checked);
      state.wcConfirmed = false;
      updateWebcamConfirmUI();
    }
  });
  $("btn-wc-confirm").addEventListener("click", () => {
    if (state.wcActive) return;
    syncRepSelection($("wc-target-reps-input").value);
    state.wcConfirmed = !state.wcConfirmed;
    $("wc-error").classList.add("hidden");
    updateWebcamConfirmUI();
  });
  function isWebcamFullscreen() {
    const stage = $("webcam-stage");
    if (!stage) return false;
    const nativeFs = document.fullscreenElement || document.webkitFullscreenElement;
    return nativeFs === stage || stage.classList.contains("is-fullscreen");
  }

  function syncFullscreenUI() {
    const activeFs = isWebcamFullscreen();
    state.wcFullscreen = activeFs;
    document.documentElement.classList.toggle("wc-fs-lock", activeFs);
    document.body.classList.toggle("wc-fs-lock", activeFs);
    const btnHdr = $("btn-wc-fullscreen");
    if (btnHdr) {
      const lbl = btnHdr.querySelector("span");
      if (lbl) lbl.textContent = activeFs ? "Thu nhỏ (F)" : "Toàn màn hình (F)";
      btnHdr.classList.toggle("active-outline", activeFs);
    }
  }

  async function enterWebcamFullscreen() {
    const stage = $("webcam-stage");
    if (!stage) return;
    try {
      if (stage.requestFullscreen) {
        await stage.requestFullscreen();
      } else if (stage.webkitRequestFullscreen) {
        stage.webkitRequestFullscreen();
      } else {
        stage.classList.add("is-fullscreen");
      }
    } catch (_) {
      stage.classList.add("is-fullscreen");
    }
    syncFullscreenUI();
  }

  async function exitWebcamFullscreen() {
    const stage = $("webcam-stage");
    const nativeFs = document.fullscreenElement || document.webkitFullscreenElement;
    try {
      if (nativeFs && document.exitFullscreen) {
        await document.exitFullscreen();
      } else if (nativeFs && document.webkitExitFullscreen) {
        document.webkitExitFullscreen();
      }
    } catch (_) {}
    if (stage) stage.classList.remove("is-fullscreen");
    syncFullscreenUI();
  }

  function toggleWebcamFullscreen() {
    if (isWebcamFullscreen()) exitWebcamFullscreen();
    else enterWebcamFullscreen();
  }

  document.addEventListener("fullscreenchange", syncFullscreenUI);
  document.addEventListener("webkitfullscreenchange", syncFullscreenUI);

  if ($("btn-wc-fullscreen")) $("btn-wc-fullscreen").addEventListener("click", toggleWebcamFullscreen);
  if ($("btn-wc-stage-fs")) $("btn-wc-stage-fs").addEventListener("click", toggleWebcamFullscreen);
  if ($("btn-fs-exit")) $("btn-fs-exit").addEventListener("click", exitWebcamFullscreen);
  if ($("btn-fs-toggle")) $("btn-fs-toggle").addEventListener("click", () => $("btn-wc-toggle").click());
  if ($("btn-fs-reset")) $("btn-fs-reset").addEventListener("click", () => $("btn-wc-reset").click());
  if ($("btn-fs-mirror")) $("btn-fs-mirror").addEventListener("click", () => $("btn-wc-mirror").click());

  $("btn-wc-toggle").addEventListener("click", () => {
    if ($("btn-wc-toggle").dataset.active === "1") stopWebcam();
    else startWebcam();
  });
  $("btn-wc-reset").addEventListener("click", () => {
    fetch("/api/webcam/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reset: true }),
    }).then(() => pollWebcamStatus());
  });
  $("btn-wc-mirror").addEventListener("click", () => {
    state.webcamMirror = !state.webcamMirror;
    $("btn-wc-mirror").textContent = `Lật gương: ${state.webcamMirror ? "Bật" : "Tắt"}`;
    $("btn-wc-mirror").classList.toggle("active-outline", state.webcamMirror);
    if ($("btn-fs-mirror")) {
      $("btn-fs-mirror").textContent = `Lật gương: ${state.webcamMirror ? "Bật" : "Tắt"}`;
      $("btn-fs-mirror").classList.toggle("active-outline", state.webcamMirror);
    }
    if (state.wcActive) {
      fetch("/api/webcam/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mirror: state.webcamMirror }),
      });
    }
  });
  updateWebcamConfirmUI();

  // -------------------- TAB 3: DASHBOARD --------------------
  async function loadDashboard() {
    const errBox = $("dash-error");
    const errTxt = $("dash-error-text");
    if (errBox) errBox.classList.add("hidden");
    try {
      const resp = await fetch("/api/dashboard");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const d = await resp.json();
      state.dashboardLoaded = true;

      const cv = d.cv_summary || {};
      const rf = cv.rf || {};
      const hy = cv.hybrid || {};
      const re = d.rule_engine_metrics || {};
      const bench1f = d.benchmark?.core_ai_single_fold || {};
      const audit = d.zero_leakage_audit || {};

      $("dash-kpis").innerHTML = `
      <div class="kpi-card">
        <span>BiomechanicalRF (103-d E.3)</span>
        <strong class="num-tabular">${(Number(rf.macro_f1_mean || 0) * 100).toFixed(2)}%</strong>
        <small>CV Macro-F1 · Acc ${(Number(rf.accuracy_mean || 0) * 100).toFixed(2)}%</small>
      </div>
      <div class="kpi-card">
        <span>Hybrid Ensemble (0.60/0.25/0.15)</span>
        <strong class="num-tabular">${Number(hy.roc_auc_mean || 0).toFixed(3)} AUC</strong>
        <small>Acc ${(Number(hy.accuracy_mean || 0) * 100).toFixed(2)}% ± ${(Number(hy.accuracy_std || 0) * 100).toFixed(2)}%</small>
      </div>
      <div class="kpi-card">
        <span>Kiểm định Zero Data Leakage</span>
        <strong class="num-tabular">${audit.total_oof_clips || 521} mẫu</strong>
        <small>${audit.unique_video_id_groups || 371} nhóm video_id · ${audit.cross_fold_video_id_leaks ?? 0} rò rỉ</small>
      </div>
      <div class="kpi-card">
        <span>Độ trễ Lõi AI (CPU 1-Fold)</span>
        <strong class="num-tabular">${Number(bench1f.per_frame_ai_ms || 0).toFixed(2)} ms/frame</strong>
        <small>${Number(bench1f.total_ai_mean_ms || 0).toFixed(1)} ms/rep · Top-3 Rule ${(Number(re.top3_recall || 0) * 100).toFixed(1)}%</small>
      </div>`;

      const modelRows = ["rf", "cnn_bilstm", "transformer", "hybrid"]
        .filter((k) => cv[k])
        .map((k) => {
          const m = cv[k];
          return `<tr>
          <td><strong>${escapeHtml(m.label || k)}</strong></td>
          <td>${(Number(m.accuracy_mean || 0) * 100).toFixed(2)} ± ${(Number(m.accuracy_std || 0) * 100).toFixed(2)}%</td>
          <td>${(Number(m.balanced_accuracy_mean || 0) * 100).toFixed(2)}%</td>
          <td><strong>${(Number(m.macro_f1_mean || 0) * 100).toFixed(2)}%</strong></td>
          <td>${(Number(m.recall_incorrect_mean || 0) * 100).toFixed(2)}%</td>
          <td>${Number(m.roc_auc_mean || 0).toFixed(3)}</td>
        </tr>`;
        })
        .join("");
      $("tbl-models").innerHTML = `
      <thead><tr><th>Mô hình</th><th>Accuracy</th><th>Balanced Acc</th><th>Macro-F1</th><th>Recall Sai</th><th>ROC-AUC</th></tr></thead>
      <tbody>${modelRows}</tbody>`;

      const b1 = d.benchmark?.core_ai_single_fold;
      const b5 = d.benchmark?.core_ai_five_fold_ensemble;
      const be = d.benchmark?.end_to_end_video_single_fold;
      const bRows = [
        b1 && ["Lõi AI 1-Fold (100 lượt)", `${b1.total_ai_mean_ms} ms/rep`, `${b1.per_frame_ai_ms} ms/frame`, `${b1.core_ai_fps_equivalent} FPS`],
        b5 && ["Lõi AI 5-Fold Ensemble", `${b5.total_ai_mean_ms} ms/rep`, `${b5.per_frame_ai_ms} ms/frame`, `${b5.core_ai_fps_equivalent} FPS`],
        be && ["End-to-End Video (MediaPipe + AI)", `${be.end_to_end_total_ms} ms/rep`, `${be.end_to_end_per_frame_ms} ms/frame`, `${be.fps_equivalent} FPS`],
      ]
        .filter(Boolean)
        .map((r) => `<tr><td><strong>${r[0]}</strong></td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td></tr>`)
        .join("");
      $("tbl-benchmark").innerHTML = `
      <thead><tr><th>Chế độ</th><th>Độ trễ / Rep</th><th>TB / Frame</th><th>FPS tương đương</th></tr></thead>
      <tbody>${bRows}</tbody>`;

      $("tbl-exercises").innerHTML = `
      <thead><tr><th>Bài tập</th><th>Số mẫu (Đúng/Sai)</th><th>RF Macro-F1</th><th>Hybrid Acc</th><th>Hybrid Macro-F1</th><th>Hybrid Recall Sai</th><th>Hybrid AUC</th></tr></thead>
      <tbody>
        ${(d.per_exercise || [])
          .map(
            (e) => `<tr>
              <td><strong>${escapeHtml(e.exercise)}</strong></td>
              <td>${e.n_total} (${e.n_correct}/${e.n_incorrect})</td>
              <td>${(Number(e.rf_macro_f1 || 0) * 100).toFixed(1)}%</td>
              <td>${(Number(e.hybrid_acc || 0) * 100).toFixed(1)}%</td>
              <td><strong>${(Number(e.hybrid_macro_f1 || 0) * 100).toFixed(1)}%</strong></td>
              <td>${(Number(e.hybrid_recall_inc || 0) * 100).toFixed(1)}%</td>
              <td>${Number(e.hybrid_auc || 0).toFixed(3)}</td>
            </tr>`
          )
          .join("")}
      </tbody>`;

      const gal = $("figures-gallery");
      gal.innerHTML = "";
      (d.figures || []).forEach((fig) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "fig-item";
        btn.innerHTML = `<img src="${fig.url}" alt="${escapeHtml(fig.title)}" loading="lazy" /><span>${escapeHtml(fig.title)}</span>`;
        btn.addEventListener("click", () => openLightbox(fig.url, fig.title));
        gal.appendChild(btn);
      });
    } catch (err) {
      state.dashboardLoaded = false;
      if (errTxt) {
        errTxt.textContent = `Không tải được dữ liệu Tổng quan (${err.message}). Hãy kiểm tra máy chủ http://127.0.0.1:8000 (chạy run_web.bat) rồi bấm Thử tải lại.`;
      }
      if (errBox) errBox.classList.remove("hidden");
    }
  }
  if ($("btn-dash-retry")) {
    $("btn-dash-retry").addEventListener("click", loadDashboard);
  }

  // -------------------- LIGHTBOX --------------------
  function openLightbox(url, title) {
    $("lightbox-title").textContent = title;
    $("lightbox-img").src = url;
    $("lightbox-img").alt = title;
    $("lightbox-modal").classList.remove("hidden");
  }
  function closeLightbox() {
    $("lightbox-modal").classList.add("hidden");
  }
  $("btn-close-modal").addEventListener("click", closeLightbox);
  document.querySelector(".modal-backdrop").addEventListener("click", closeLightbox);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeLightbox();
      if (isWebcamFullscreen()) exitWebcamFullscreen();
      return;
    }
    if ((e.key === "f" || e.key === "F") && !e.ctrlKey && !e.metaKey && !e.altKey) {
      const tag = (e.target && e.target.tagName) ? e.target.tagName.toUpperCase() : "";
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      const wcPanel = $("panel-webcam");
      const lightboxOpen = !$("lightbox-modal").classList.contains("hidden");
      if (wcPanel && !wcPanel.hidden && !lightboxOpen) {
        e.preventDefault();
        toggleWebcamFullscreen();
      }
    }
  });

  loadExercises();
})();
