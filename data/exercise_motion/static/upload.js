"use strict";

const form = document.querySelector("#video-form");
const video = document.querySelector("#video");
const preview = document.querySelector("#preview");
const statusLine = document.querySelector("#status");
const result = document.querySelector("#result");
const button = document.querySelector("#analyze");
let previewUrl;

function status(message, error = false) {
  statusLine.textContent = message;
  statusLine.dataset.error = error;
}

video.addEventListener("change", () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  const file = video.files[0];
  previewUrl = file ? URL.createObjectURL(file) : undefined;
  preview.hidden = !file;
  if (file) preview.src = previewUrl;
  else preview.removeAttribute("src");
  result.hidden = true;
  status(file ? "Ready to analyze." : "Choose a video to see the result.");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  result.hidden = true;
  const file = video.files[0];
  if (!file || !file.name.toLowerCase().endsWith(".mp4")) {
    status("Choose one MP4 file.", true);
    return;
  }
  if (file.size > Number(form.dataset.maxBytes)) {
    status("Choose an MP4 file of 100 MB or less.", true);
    return;
  }
  const upload = new FormData(form);
  button.disabled = true;
  video.disabled = true;
  status("Analyzing body landmarks…");
  try {
    const response = await fetch(form.action, {method: "POST", body: upload});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The video could not be analyzed.");
    document.querySelector("#exercise").textContent = data.exercise;
    document.querySelector("#details").textContent =
      data.exercise === "Uncertain"
        ? "The model score was below the recognition threshold."
        : `Model score ${Math.round(data.score * 100)}% · ${data.windows} analysis windows`;
    const list = document.querySelector("#scores");
    list.replaceChildren();
    for (const [label, score] of Object.entries(data.scores).sort((a, b) => b[1] - a[1])) {
      const row = document.createElement("li");
      const text = document.createElement("span");
      text.textContent = `${label} · ${Math.round(score * 100)}%`;
      const meter = document.createElement("meter");
      meter.max = 1;
      meter.value = score;
      meter.setAttribute("aria-label", label);
      row.append(text, meter);
      list.append(row);
    }
    result.hidden = false;
    status("Analysis complete.");
  } catch (error) {
    status(error.message || "The video could not be analyzed.", true);
  } finally {
    button.disabled = false;
    video.disabled = false;
  }
});
