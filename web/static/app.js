// Single page with hash routes: #/ (new analysis), #/historico, #/analise/<id>.
"use strict";

const NUTRIENTS = [
  { key: "added_sugars", value: "added_sugars_g", name: "Açúcares adicionados", unit: "g",
    warning: "ALTO EM AÇÚCAR ADICIONADO", limits: { "100g": 15, "100ml": 7.5 } },
  { key: "saturated_fat", value: "saturated_fat_g", name: "Gorduras saturadas", unit: "g",
    warning: "ALTO EM GORDURA SATURADA", limits: { "100g": 6, "100ml": 3 } },
  { key: "sodium", value: "sodium_mg", name: "Sódio", unit: "mg",
    warning: "ALTO EM SÓDIO", limits: { "100g": 600, "100ml": 300 } },
];

const VERDICTS = {
  compliant: { title: "De acordo com a ANVISA", short: "De acordo" },
  inconclusive: { title: "Não foi possível decidir", short: "Inconclusivo" },
  non_compliant: { title: "Viola as regras da ANVISA", short: "Viola as regras" },
  // Table only: what the front must show, not a compliance decision.
  info: { title: "Alertas exigidos pela tabela", short: "Só a tabela" },
};

const MODES = {
  full: {
    lead: "Envie uma foto da <strong>frente</strong> da embalagem e uma do <strong>verso</strong>, com a tabela nutricional legível.",
    hint: "Escolha as duas fotos para continuar.",
  },
  table_only: {
    lead: "Envie uma foto do <strong>verso</strong>, com a tabela nutricional legível: o resultado mostra quais alertas a frente deve ter.",
    hint: "Escolha a foto do verso para continuar.",
  },
};

const PHOTO_TYPES = {
  table: "Tabela nutricional (formato novo)",
  linear: "Informação nutricional em texto corrido",
  old_format: "Tabela no formato antigo (por porção)",
  no_table: "Sem tabela nutricional",
  uncertain: "Formato não identificado",
};

// Icons carry the verdict's shape, so it never depends on color alone.
const ICONS = {
  compliant: '<circle cx="12" cy="12" r="10"/><path d="M7 12.5l3.2 3.2L17 9"/>',
  inconclusive: '<path d="M12 3 2.5 20h19z"/><path d="M12 10v4.5M12 17.5v.01"/>',
  non_compliant: '<circle cx="12" cy="12" r="10"/><path d="M8.5 8.5l7 7M15.5 8.5l-7 7"/>',
  info: '<circle cx="12" cy="12" r="10"/><path d="M12 11v6M12 7.5v.01"/>',
  yes: '<path d="M5 12.5l4.5 4.5L19 7"/>',
  no: '<path d="M6 12h12"/>',
  unknown: '<path d="M9.2 9a3 3 0 1 1 3.8 2.9c-.6.2-1 .8-1 1.4V14M12 17.5v.01"/>',
  trash: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
  download: '<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>',
};

const icon = (name, cls = "icon") =>
  `<svg class="${cls}" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name]}</svg>`;

const esc = (text) =>
  String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const number = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 1 });
const dateTime = new Intl.DateTimeFormat("pt-BR", { dateStyle: "short", timeStyle: "short" });

const $ = (selector, root = document) => root.querySelector(selector);

// ---------------------------------------------------------------- API ---

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `Erro ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch { /* not JSON */ }
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}

// ------------------------------------------------------------- Routing ---

const VIEWS = ["analyze", "result", "history"];

function show(view) {
  for (const name of VIEWS) $(`#view-${name}`).hidden = name !== view;
  const tab = view === "history" ? "history" : "analyze";
  for (const link of document.querySelectorAll(".tabs a")) {
    if (link.dataset.tab === tab) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
}

async function route() {
  const hash = location.hash || "#/";
  const match = hash.match(/^#\/analise\/([0-9a-f]{32})$/);
  try {
    if (match) {
      show("result");
      $("#result").innerHTML = '<p class="muted">Carregando…</p>';
      renderResult(await api(`/api/analyses/${match[1]}`));
      document.title = "Resultado · Confere Rótulo";
    } else if (hash === "#/historico") {
      show("history");
      $("#history").innerHTML = '<p class="muted">Carregando…</p>';
      renderHistory(await api("/api/analyses"));
      document.title = "Histórico · Confere Rótulo";
    } else {
      show("analyze");
      document.title = "Confere Rótulo";
    }
  } catch (error) {
    $(`#${match ? "result" : "history"}`).innerHTML =
      `<p class="error" role="alert">${esc(error.message)}</p><a class="button" href="#/historico">Voltar ao histórico</a>`;
  }
  // Move focus to the new content, for screen readers and keyboards.
  if (routed) $("#main").focus({ preventScroll: true });
  routed = true;
  window.scrollTo(0, 0);
}
let routed = false;

// ------------------------------------------------------------- Upload ---

const chosen = { front: null, back: null };
let mode = "full";

function setMode(value) {
  mode = value;
  const tableOnly = mode === "table_only";
  $('.slot[data-side="front"]').hidden = tableOnly;
  $(".slots").classList.toggle("single", tableOnly);
  $("#analyze-lead").innerHTML = MODES[mode].lead;
  $("#analyze-hint").textContent = MODES[mode].hint;
  updateReady();
}

function updateReady() {
  const ready = chosen.back && (mode === "table_only" || chosen.front);
  $("#analyze-submit").disabled = !ready;
  $("#analyze-hint").hidden = Boolean(ready);
}

function setPhoto(side, file) {
  const slot = $(`.slot[data-side="${side}"]`);
  const preview = $(".preview", slot);
  const img = $("img", preview);
  if (img.src) URL.revokeObjectURL(img.src);
  chosen[side] = file;
  img.hidden = !file;
  const placeholder = $(".placeholder", preview);
  placeholder.hidden = Boolean(file);
  placeholder.dataset.text ??= placeholder.lastChild.textContent;
  placeholder.lastChild.textContent = placeholder.dataset.text;
  preview.dataset.empty = String(!file);
  if (file) img.src = URL.createObjectURL(file);
  else img.removeAttribute("src");
  updateReady();
}

function resetForm() {
  // Keep the chosen mode: someone checking tables checks several.
  const current = mode;
  $("#analyze-form").reset();
  $(`input[name="mode"][value="${current}"]`).checked = true;
  setPhoto("front", null);
  setPhoto("back", null);
  setMode(current);
  $("#analyze-error").hidden = true;
}

async function submitAnalysis(event) {
  event.preventDefault();
  const tableOnly = mode === "table_only";
  if (!chosen.back || (!tableOnly && !chosen.front)) return;
  const form = new FormData();
  if (!tableOnly) form.append("front", chosen.front);
  form.append("back", chosen.back);
  form.append("name", $('input[name="name"]').value);
  $("#analyze-error").hidden = true;
  $("#busy").hidden = false;
  try {
    const entry = await api("/api/analyses", { method: "POST", body: form });
    resetForm();
    location.hash = `#/analise/${entry.id}`;
  } catch (error) {
    $("#analyze-error").textContent = error.message;
    $("#analyze-error").hidden = false;
  } finally {
    $("#busy").hidden = true;
  }
}

// ------------------------------------------------------------- Result ---

function flag(value, labels) {
  if (value === true) return `<span class="tag yes">${icon("yes")} ${labels[0]}</span>`;
  if (value === false) return `<span class="tag no">${icon("no")} ${labels[1]}</span>`;
  return `<span class="tag unknown">${icon("unknown")} ${labels[2]}</span>`;
}

function warningsTable(result) {
  // Table only: there's no front to compare with.
  const withFront = result.mode !== "table_only";
  const rows = NUTRIENTS.map(({ key, warning }) => {
    const required = result.required_warnings[key];
    const found = result.found_warnings[key];
    const mismatch = withFront && typeof required === "boolean" && typeof found === "boolean" && required !== found;
    return `<tr class="${mismatch ? "mismatch" : ""}">
      <th scope="row">${warning}${mismatch ? '<span class="visually-hidden"> (divergente)</span>' : ""}</th>
      <td>${flag(required, ["Exigido", "Não exigido", "Indefinido"])}</td>
      ${withFront ? `<td>${flag(found, ["Presente", "Ausente", "Não lido"])}</td>` : ""}
    </tr>`;
  }).join("");
  return `<div class="table-wrap"><table>
    <thead><tr><th scope="col">Alerta</th><th scope="col">Pelas regras</th>${withFront ? '<th scope="col">Na frente</th>' : ""}</tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

function nutritionTable(result) {
  const { nutrition } = result;
  const basis = nutrition.basis;
  const per = basis === "100ml" ? "100 ml" : basis === "100g" ? "100 g" : "100 g/ml";
  const rows = NUTRIENTS.map(({ value, name, unit, limits }) => {
    const amount = nutrition[value];
    const limit = basis ? `${number.format(limits[basis])} ${unit}`
      : `${number.format(limits["100g"])} ${unit} (sólido) / ${number.format(limits["100ml"])} ${unit} (líquido)`;
    return `<tr>
      <th scope="row">${name}</th>
      <td class="num">${amount == null ? '<span class="tag unknown">não lido</span>' : `${number.format(amount)} ${unit}`}</td>
      <td class="num">${limit}</td>
    </tr>`;
  }).join("");
  return `<div class="table-wrap"><table>
    <thead><tr><th scope="col">Nutriente</th><th scope="col">Por ${per}</th><th scope="col">Limite do alerta</th></tr></thead>
    <tbody>${rows}</tbody></table></div>
    <p class="muted small" style="margin:.75rem 0 0">Foto do verso: ${esc(PHOTO_TYPES[result.back_photo_type] || result.back_photo_type)}.
    ${basis ? "" : "Base (100 g ou 100 ml) não identificada."}</p>`;
}

function renderResult(entry) {
  const { result, photos } = entry;
  const verdict = VERDICTS[result.verdict];
  const name = entry.name || "Produto sem nome";
  $("#result").innerHTML = `
    ${result.mock ? `<p class="mock"><strong>Resultado simulado.</strong> Os pipelines de leitura ainda não estão conectados: os valores abaixo são fictícios.</p>` : ""}
    <div class="verdict verdict-${result.verdict}" role="status">
      ${icon(result.verdict)}
      <div>
        <h2>${verdict.title}</h2>
        <ul>${result.reasons.map((reason) => `<li>${esc(reason)}</li>`).join("")}</ul>
      </div>
    </div>
    <p class="meta"><strong>${esc(name)}</strong> · ${dateTime.format(new Date(entry.created_at))}</p>

    <section class="card" aria-labelledby="warnings-title">
      <h2 id="warnings-title">Alertas “ALTO EM”</h2>
      ${warningsTable(result)}
    </section>

    <section class="card" aria-labelledby="nutrition-title">
      <h2 id="nutrition-title">Tabela nutricional lida</h2>
      ${nutritionTable(result)}
    </section>

    ${result.notes?.length ? `<section class="card notes" aria-labelledby="notes-title">
      <h2 id="notes-title">Sobre os limites</h2>
      ${result.notes.map((note) => `<p>${esc(note)}</p>`).join("")}
    </section>` : ""}

    <section class="card" aria-labelledby="photos-title">
      <h2 id="photos-title">Fotos enviadas</h2>
      <div class="photos">
        ${["front", "back"].filter((side) => photos[side]).map((side) => `<figure>
          <a href="${photos[side].view}" target="_blank" rel="noopener">
            <img src="${photos[side].view}" alt="Foto da ${side === "front" ? "frente" : "verso"} de ${esc(name)}" loading="lazy">
          </a>
          <figcaption>${side === "front" ? "Frente" : "Verso"}</figcaption>
        </figure>`).join("")}
      </div>
    </section>

    <div class="actions">
      <a class="button primary" href="/api/analyses/${entry.id}/export" download>${icon("download", "")} Exportar JSON</a>
      <a class="button" href="#/">Nova análise</a>
      <button type="button" class="button danger" data-delete="${entry.id}">${icon("trash", "")} Excluir</button>
    </div>`;
}

// ------------------------------------------------------------ History ---

function renderHistory(entries) {
  if (!entries.length) {
    $("#history").innerHTML = `<div class="empty card">
      <p>Nenhuma análise ainda.</p>
      <a class="button primary" href="#/">Analisar um produto</a></div>`;
    return;
  }
  const items = entries.map((entry) => {
    const verdict = VERDICTS[entry.verdict];
    const name = entry.name || "Produto sem nome";
    return `<li class="history-item">
      <div class="thumbs">
        ${["front", "back"].filter((side) => entry.photos[side]).map((side) =>
          `<img src="${entry.photos[side].thumb}" alt="" loading="lazy">`).join("")}
      </div>
      <div class="history-main">
        <a href="#/analise/${entry.id}">${esc(name)}</a>
        <span class="badge ${entry.verdict}">${icon(entry.verdict, "icon")} ${verdict.short}</span>
        <span class="history-date"><time datetime="${entry.created_at}">${dateTime.format(new Date(entry.created_at))}</time></span>
      </div>
      <button type="button" class="button danger" data-delete="${entry.id}" aria-label="Excluir análise de ${esc(name)}">${icon("trash", "")}</button>
    </li>`;
  }).join("");
  $("#history").innerHTML = `<p class="muted">${entries.length} ${entries.length === 1 ? "análise" : "análises"}, da mais recente para a mais antiga.</p>
    <ul class="history-list">${items}</ul>`;
}

async function deleteAnalysis(id) {
  if (!confirm("Excluir esta análise e suas fotos? Não é possível desfazer.")) return;
  try {
    await api(`/api/analyses/${id}`, { method: "DELETE" });
  } catch (error) {
    alert(error.message);
    return;
  }
  if (location.hash === "#/historico") route();
  else location.hash = "#/historico";
}

// -------------------------------------------------------------- Theme ---

const THEMES = ["auto", "light", "dark"];
const THEME_NAMES = { auto: "automático", light: "claro", dark: "escuro" };

function applyTheme(theme) {
  if (theme === "auto") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.dataset.theme = theme;
  const next = THEME_NAMES[THEMES[(THEMES.indexOf(theme) + 1) % 3]];
  $("#theme").setAttribute("aria-label", `Tema: ${THEME_NAMES[theme]} (mudar para ${next})`);
  $("#theme").title = `Tema: ${THEME_NAMES[theme]}`;
}

function storedTheme() {
  try { return localStorage.getItem("theme") || "auto"; } catch { return "auto"; }
}

// --------------------------------------------------------------- Init ---

document.addEventListener("DOMContentLoaded", () => {
  let theme = THEMES.includes(storedTheme()) ? storedTheme() : "auto";
  applyTheme(theme);
  $("#theme").addEventListener("click", () => {
    theme = THEMES[(THEMES.indexOf(theme) + 1) % 3];
    applyTheme(theme);
    try { localStorage.setItem("theme", theme); } catch { /* private mode */ }
  });

  // Formats the browser can't show (e.g. HEIC outside Safari): keep the
  // file, say so instead of a broken image.
  for (const img of document.querySelectorAll(".preview img")) {
    img.addEventListener("error", () => {
      if (!img.src) return;
      img.hidden = true;
      const placeholder = $(".placeholder", img.parentElement);
      placeholder.hidden = false;
      placeholder.lastChild.textContent = " Prévia indisponível neste navegador";
    });
  }
  for (const input of document.querySelectorAll('.slot input[type="file"]')) {
    input.addEventListener("change", () => {
      const file = input.files[0];
      if (!file) return;
      // Early feedback only: the server checks the content again.
      if (file.type && !file.type.startsWith("image/")) {
        const side = input.name === "front" ? "da frente" : "do verso";
        $("#analyze-error").textContent = `Foto ${side}: o arquivo escolhido não é uma imagem.`;
        $("#analyze-error").hidden = false;
        input.value = "";
        return;
      }
      $("#analyze-error").hidden = true;
      setPhoto(input.name, file);
    });
  }
  for (const radio of document.querySelectorAll('input[name="mode"]')) {
    radio.addEventListener("change", () => setMode(radio.value));
  }
  setMode($('input[name="mode"]:checked').value);
  $("#analyze-form").addEventListener("submit", submitAnalysis);
  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-delete]");
    if (button) deleteAnalysis(button.dataset.delete);
  });

  window.addEventListener("hashchange", route);
  route();
});
