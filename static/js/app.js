// RetinaAssist front-end helpers
document.addEventListener("DOMContentLoaded", () => {
  // Clickable table rows
  document.querySelectorAll("tr.row-link").forEach((row) => {
    row.addEventListener("click", (e) => {
      if (!e.target.closest("a, button, input, select")) window.location = row.dataset.href;
    });
  });

  // Grad-CAM view switcher
  const lens = document.getElementById("lens");
  if (lens) {
    const buttons = document.querySelectorAll("[data-view]");
    buttons.forEach((btn) => btn.addEventListener("click", () => {
      buttons.forEach((b) => {
        b.classList.toggle("active", b === btn);
        b.setAttribute("aria-pressed", b === btn ? "true" : "false");
      });
      const view = btn.dataset.view;
      lens.classList.toggle("compare", view === "compare");
      lens.querySelectorAll("figure").forEach((fig) => {
        const pane = fig.dataset.pane;
        const show = view === "compare" ? (pane === "original" || pane === "overlay") : pane === view;
        fig.classList.toggle("d-none", !show);
      });
    }));
  }

  // Upload form: drag & drop, preview, client-side checks, loading state
  const form = document.getElementById("screeningForm");
  if (form) {
    const input = document.getElementById("image");
    const zone = document.getElementById("dropzone");
    const error = document.getElementById("fileError");
    const empty = document.getElementById("dzEmpty");
    const preview = document.getElementById("dzPreview");
    const img = document.getElementById("previewImg");
    const MAX = 10 * 1024 * 1024;
    const okTypes = ["image/png", "image/jpeg"];
    let objectUrl = null;

    const reset = () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = null;
      img.removeAttribute("src");
      empty.classList.remove("d-none");
      preview.classList.add("d-none");
      zone.classList.remove("has-file");
    };
    const fail = (msg) => {
      error.textContent = msg;
      zone.classList.add("is-invalid");
    };

    const showFile = () => {
      error.textContent = "";
      zone.classList.remove("is-invalid");
      const file = input.files[0];
      if (!file) { reset(); return; }
      if (!okTypes.includes(file.type)) { reset(); fail("Please choose a PNG or JPEG image."); return; }
      if (file.size > MAX) { reset(); fail("The file is larger than 10 MB."); return; }
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = URL.createObjectURL(file);
      img.src = objectUrl;
      document.getElementById("fileName").textContent = file.name;
      const meta = document.getElementById("fileMeta");
      meta.innerHTML = "";
      const chip = (text) => { const s = document.createElement("span"); s.textContent = text; meta.appendChild(s); };
      chip(file.type === "image/png" ? "PNG" : "JPEG");
      chip(`${(file.size / 1024 / 1024).toFixed(2)} MB`);
      img.onload = () => chip(`${img.naturalWidth}×${img.naturalHeight} px`);
      img.onerror = () => { reset(); fail("This file could not be previewed. It may be corrupted."); };
      empty.classList.add("d-none");
      preview.classList.remove("d-none");
      zone.classList.add("has-file");
    };

    input.addEventListener("change", showFile);
    ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, () => zone.classList.add("dragover")));
    ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, () => zone.classList.remove("dragover")));
    document.getElementById("removeFile").addEventListener("click", (e) => {
      e.preventDefault();
      input.value = "";
      error.textContent = "";
      zone.classList.remove("is-invalid");
      reset();
      input.focus();
    });

    form.addEventListener("submit", (e) => {
      const code = document.getElementById("patient_code");
      if (!/^[A-Za-z0-9-]{3,20}$/.test(code.value.trim())) {
        e.preventDefault(); code.classList.add("is-invalid"); code.focus(); return;
      }
      code.classList.remove("is-invalid");
      if (!input.files.length) { e.preventDefault(); fail("Please choose a retinal fundus image."); return; }
      if (error.textContent) { e.preventDefault(); return; }
      const btn = document.getElementById("analyseBtn");
      btn.disabled = true;
      btn.querySelector(".btn-label").classList.add("d-none");
      btn.querySelector(".btn-loading").classList.remove("d-none");
    });
  }
});
