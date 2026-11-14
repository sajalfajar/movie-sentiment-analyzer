// navbar blur on scroll
const nav = document.querySelector("nav");
window.addEventListener("scroll", () => nav.classList.toggle("scrolled", window.scrollY > 10));

// count-up numbers
function countUp(el) {
  const target = parseFloat(el.dataset.count);
  const dec = parseInt(el.dataset.dec || 0);
  const suffix = el.dataset.suffix || "";
  const start = performance.now();
  function step(now) {
    const p = Math.min((now - start) / 1200, 1);
    const val = target * (1 - Math.pow(1 - p, 3));
    el.textContent = val.toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec }) + suffix;
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// real numbers from the model
fetch("/api/metrics").then(r => r.json()).then(data => {
  const best = data.results[data.best];
  const acc = document.getElementById("accStat");
  if (acc) acc.dataset.count = (best.accuracy * 100).toFixed(1);
  document.querySelectorAll("[data-count]").forEach(countUp);

  // table on results page
  const tbl = document.getElementById("tbl");
  if (tbl) {
    let html = "<tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>";
    for (const [name, m] of Object.entries(data.results)) {
      const cls = name === data.best ? "best" : "";
      html += `<tr class="${cls}"><td>${name}${cls ? " 🏆" : ""}</td><td>${m.accuracy}</td><td>${m.precision}</td><td>${m.recall}</td><td>${m.f1}</td></tr>`;
    }
    tbl.innerHTML = html;
  }
}).catch(() => document.querySelectorAll("[data-count]").forEach(countUp));

// example buttons
document.querySelectorAll(".chip").forEach(chip => {
  chip.addEventListener("click", () => {
    document.getElementById("review").value = chip.dataset.text;
  });
});

// analyze
const btn = document.getElementById("btn");
if (btn) {
  btn.addEventListener("click", async () => {
    const text = document.getElementById("review").value.trim();
    if (!text) { alert("Please write a review first."); return; }

    const btnText = document.getElementById("btnText");
    btnText.innerHTML = '<span class="spin"></span> Analyzing...';
    btn.disabled = true;

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text }),
      });
      const data = await res.json();
      const isPos = data.sentiment === "Positive";

      const box = document.getElementById("result");
      box.className = (isPos ? "pos" : "neg");
      void box.offsetWidth; // restart animation
      box.classList.add("show");
      document.getElementById("emoji").textContent = isPos ? "😍" : "😞";
      document.getElementById("label").textContent = data.sentiment + " Review";
      document.getElementById("conf").textContent = "Confidence: " + data.confidence + "%";
      const fill = document.getElementById("fill");
      fill.style.width = "0";
      setTimeout(() => (fill.style.width = data.confidence + "%"), 50);
    } catch (e) {
      alert("Server not running. Start uvicorn first.");
    }
    btnText.textContent = "Analyze Review";
    btn.disabled = false;
  });
}