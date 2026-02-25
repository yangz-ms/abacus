(function() {
  "use strict";

  /* ============================================================
   * I18N DATA — from external i18n.js, with built-in fallback
   * ============================================================ */
  var I18N = window.ABACUS_I18N || {
    languages: [{ code: "en", flag: "us", label: "English" }],
    translations: {
      en: {
        title: "Abacus",
        tagline: "From arithmetic to algebra \u2014 beautifully rendered",
        calculate: "Calculate",
        input_label: "Input",
        result_label: "Result",
        error_label: "Error",
        placeholder: "Enter an expression\u2026",
        footer: "Powered by {0}",
        conn_error: "Connection error: ",
        group_expression: "Expression",
        group_solver: "Solver"
      }
    }
  };

  var LANGUAGES    = I18N.languages;
  var TRANSLATIONS = I18N.translations;
  var FLAG_URL     = "https://flagcdn.com/w40/";

  /* ============================================================
   * DEFAULT CALCULATORS (fallback when API is down)
   * ============================================================ */
  var DEFAULT_CALCS = [];

  /* ============================================================
   * STATE
   * ============================================================ */
  var calcs = DEFAULT_CALCS.slice();
  var currentCalc = calcs[0];
  var currentLang = localStorage.getItem("abacus-lang") || "en";
  var currentTheme = document.documentElement.getAttribute("data-theme") || "dark";

  /* ============================================================
   * HELPERS
   * ============================================================ */
  function t(key, fallback) {
    var lang = TRANSLATIONS[currentLang] || TRANSLATIONS.en;
    return lang[key] || TRANSLATIONS.en[key] || fallback || key;
  }

  function escapeHtml(s) {
    var d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function flagUrl(code) {
    return FLAG_URL + code + ".png";
  }

  function _mergeCalcI18n(calculators) {
    for (var i = 0; i < calculators.length; i++) {
      var c = calculators[i];
      var key = c.id + "_desc";
      // Set English desc from short_desc
      if (TRANSLATIONS.en) {
        TRANSLATIONS.en[key] = c.desc;
      }
      if (!c.i18n) continue;
      for (var lang in c.i18n) {
        if (c.i18n.hasOwnProperty(lang) && TRANSLATIONS[lang]) {
          TRANSLATIONS[lang][key] = c.i18n[lang];
        }
      }
    }
  }

  /* ============================================================
   * DOM REFS
   * ============================================================ */
  var langDropdown    = document.getElementById("langDropdown");
  var langBtn         = document.getElementById("langBtn");
  var langFlag        = document.getElementById("langFlag");
  var langLabel       = document.getElementById("langLabel");
  var langMenu        = document.getElementById("langMenu");
  var themeToggle     = document.getElementById("themeToggle");
  var selectorScroll  = document.getElementById("selectorScroll");
  var calcDescription = document.getElementById("calcDescription");
  var exprInput       = document.getElementById("exprInput");
  var calcBtn         = document.getElementById("calcBtn");
  var examplesRow     = document.getElementById("examplesRow");
  var outputSection   = document.getElementById("outputSection");
  var outputContent   = document.getElementById("outputContent");
  var footerText      = document.getElementById("footerText");

  /* ============================================================
   * THEME
   * ============================================================ */
  function applyTheme() {
    document.documentElement.setAttribute("data-theme", currentTheme);
    themeToggle.textContent = currentTheme === "dark" ? "\u2600" : "\u263E";
    localStorage.setItem("abacus-theme", currentTheme);
  }

  themeToggle.addEventListener("click", function() {
    currentTheme = currentTheme === "dark" ? "light" : "dark";
    applyTheme();
  });

  applyTheme();

  /* ============================================================
   * LANGUAGE DROPDOWN
   * ============================================================ */
  function buildLangMenu() {
    langMenu.innerHTML = "";
    for (var i = 0; i < LANGUAGES.length; i++) {
      (function(lang) {
        var opt = document.createElement("div");
        opt.className = "lang-option";
        if (lang.code === currentLang) opt.className += " active";
        opt.setAttribute("data-lang", lang.code);

        var img = document.createElement("img");
        img.className = "flag-icon";
        img.src = flagUrl(lang.flag);
        img.alt = lang.code.toUpperCase();

        var span = document.createElement("span");
        span.textContent = lang.label;

        opt.appendChild(img);
        opt.appendChild(span);
        opt.addEventListener("click", function() {
          setLanguage(lang.code);
          langMenu.classList.remove("open");
        });
        langMenu.appendChild(opt);
      })(LANGUAGES[i]);
    }
  }

  function updateLangButton() {
    var lang = null;
    for (var i = 0; i < LANGUAGES.length; i++) {
      if (LANGUAGES[i].code === currentLang) { lang = LANGUAGES[i]; break; }
    }
    if (!lang) lang = LANGUAGES[0];
    langFlag.src = flagUrl(lang.flag);
    langFlag.alt = lang.code.toUpperCase();
    langLabel.textContent = lang.label;
    var opts = langMenu.querySelectorAll(".lang-option");
    for (var j = 0; j < opts.length; j++) {
      if (opts[j].getAttribute("data-lang") === currentLang) {
        opts[j].className = "lang-option active";
      } else {
        opts[j].className = "lang-option";
      }
    }
  }

  langBtn.addEventListener("click", function(e) {
    e.stopPropagation();
    langMenu.classList.toggle("open");
  });

  document.addEventListener("click", function(e) {
    if (!langDropdown.contains(e.target)) {
      langMenu.classList.remove("open");
    }
  });

  function setLanguage(code) {
    currentLang = code;
    localStorage.setItem("abacus-lang", currentLang);
    applyTranslations();
    buildPills();
  }

  /* ============================================================
   * I18N
   * ============================================================ */
  function applyTranslations() {
    var elems = document.querySelectorAll("[data-i18n]");
    for (var i = 0; i < elems.length; i++) {
      elems[i].textContent = t(elems[i].getAttribute("data-i18n"));
    }
    var placeholders = document.querySelectorAll("[data-i18n-placeholder]");
    for (var j = 0; j < placeholders.length; j++) {
      placeholders[j].placeholder = t(placeholders[j].getAttribute("data-i18n-placeholder"));
    }
    document.title = t("title");
    document.documentElement.lang = currentLang;
    updateLangButton();
    updateFooter();
  }

  function updateFooter() {
    var template = t("footer");
    footerText.innerHTML = template.replace("{0}", '<span class="accent">Python</span>');
  }


  /* ============================================================
   * PILLS & CALCULATOR SELECTION
   * ============================================================ */

  // Attach click handlers to EXISTING static pills and example chips
  // so the page works before any JS rebuild occurs.
  function attachHandlers() {
    var pills = selectorScroll.querySelectorAll(".calc-pill");
    for (var i = 0; i < pills.length; i++) {
      (function(pill) {
        var calcId = pill.getAttribute("data-id");
        pill.addEventListener("click", function() {
          var c = findCalc(calcId);
          if (c) selectCalc(c);
        });
      })(pills[i]);
    }
    var chips = examplesRow.querySelectorAll(".example-chip");
    for (var j = 0; j < chips.length; j++) {
      (function(chip) {
        chip.addEventListener("click", function() {
          exprInput.value = chip.textContent;
          doCalculate();
        });
      })(chips[j]);
    }
  }

  function findCalc(id) {
    for (var i = 0; i < calcs.length; i++) {
      if (calcs[i].id === id) return calcs[i];
    }
    return null;
  }

  // Full rebuild — only called after fetch succeeds or language changes.
  function buildPills() {
    if (calcs.length === 0) return;
    selectorScroll.innerHTML = "";
    var seenGroups = {};
    for (var i = 0; i < calcs.length; i++) {
      (function(c) {
        // Insert a labeled divider the first time a group appears
        if (!seenGroups[c.group]) {
          seenGroups[c.group] = true;
          var divider = document.createElement("div");
          divider.className = "group-divider";
          var label = document.createElement("span");
          label.textContent = t("group_" + c.group, c.group);
          divider.appendChild(label);
          selectorScroll.appendChild(divider);
        }
        var pill = document.createElement("button");
        pill.className = "calc-pill";
        if (currentCalc && currentCalc.id === c.id) pill.className += " active";
        pill.textContent = c.desc || c.name;
        pill.setAttribute("data-id", c.id);
        pill.addEventListener("click", function() { selectCalc(c); });
        selectorScroll.appendChild(pill);
      })(calcs[i]);
    }
    if (currentCalc) {
      calcDescription.textContent = t(currentCalc.id + "_desc", currentCalc.desc);
      renderExamples(currentCalc.examples || []);
    }
  }

  function selectCalc(c) {
    currentCalc = c;
    var pills = selectorScroll.querySelectorAll(".calc-pill");
    for (var i = 0; i < pills.length; i++) {
      if (pills[i].getAttribute("data-id") === c.id) {
        pills[i].className = "calc-pill active";
      } else {
        pills[i].className = "calc-pill";
      }
    }
    calcDescription.textContent = t(c.id + "_desc", c.desc);
    renderExamples(c.examples || []);
    outputSection.classList.remove("visible");
    outputContent.innerHTML = "";
    exprInput.value = "";
    exprInput.focus();
  }

  function renderExamples(examples) {
    examplesRow.innerHTML = "";
    for (var i = 0; i < examples.length; i++) {
      (function(ex) {
        var chip = document.createElement("button");
        chip.className = "example-chip";
        chip.textContent = ex;
        chip.addEventListener("click", function() {
          exprInput.value = ex;
          doCalculate();
        });
        examplesRow.appendChild(chip);
      })(examples[i]);
    }
  }

  /* ============================================================
   * CALCULATE
   * ============================================================ */
  function setLoading(on) {
    if (on) {
      calcBtn.className = "btn-calc loading";
    } else {
      calcBtn.className = "btn-calc";
    }
  }

  function renderKatex(el, tex) {
    try {
      katex.render(tex, el, { displayMode: true, throwOnError: false, trust: true });
    } catch (e) {
      el.textContent = tex;
    }
  }

  function showResult(inputTex, outputTex) {
    outputContent.innerHTML =
      '<div class="output-cards">' +
        '<div class="output-card input-card">' +
          '<div class="label">' + escapeHtml(t("input_label")) + '</div>' +
          '<div class="katex-container"><div class="katex-render" id="katexInput"></div></div>' +
        '</div>' +
        '<div class="output-card result-card">' +
          '<div class="label">' + escapeHtml(t("result_label")) + '</div>' +
          '<div class="katex-container"><div class="katex-render" id="katexResult"></div></div>' +
        '</div>' +
      '</div>';
    renderKatex(document.getElementById("katexInput"), inputTex);
    renderKatex(document.getElementById("katexResult"), outputTex);
    outputSection.classList.add("visible");
    setTimeout(scaleKatexContainers, 0);
  }

  function showError(msg) {
    outputContent.innerHTML =
      '<div class="error-card">' +
        '<div class="label">' + escapeHtml(t("error_label")) + '</div>' +
        '<div>' + escapeHtml(msg) + '</div>' +
      '</div>';
    outputSection.classList.add("visible");
  }

  function doCalculate() {
    var expr = exprInput.value.trim();
    if (!expr) return;

    setLoading(true);
    outputSection.classList.remove("visible");

    fetch("/api/calculate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ calculator: currentCalc.id, expression: expr })
    })
    .then(function(res) { return res.json(); })
    .then(function(data) {
      setLoading(false);
      if (data.error) {
        showError(data.error);
      } else {
        showResult(data.input_tex || "", data.output_tex || "");
      }
    })
    ["catch"](function(err) {
      setLoading(false);
      showError(t("conn_error") + err.message);
    });
  }

  /* ============================================================
   * KATEX AUTO-SCALING
   * ============================================================ */
  function scaleKatexContainers() {
    var containers = document.querySelectorAll(".katex-container");
    for (var i = 0; i < containers.length; i++) {
      var container = containers[i];
      container.style.transform = "";
      container.style.height = "";

      var scrollW = container.scrollWidth;
      var clientW = container.clientWidth;
      if (scrollW > clientW + 1 && clientW > 0) {
        var ratio = clientW / scrollW;
        if (ratio < 0.5) ratio = 0.5;
        container.style.transformOrigin = "left center";
        container.style.transform = "scale(" + ratio + ")";
        container.style.height = (container.scrollHeight * ratio) + "px";
      }
    }
  }

  var resizeTimer;
  window.addEventListener("resize", function() {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(scaleKatexContainers, 150);
  });

  /* ============================================================
   * EVENTS
   * ============================================================ */
  calcBtn.addEventListener("click", doCalculate);
  exprInput.addEventListener("keydown", function(e) {
    if (e.key === "Enter") {
      e.preventDefault();
      doCalculate();
    }
  });

  /* ============================================================
   * INIT
   * ============================================================ */
  buildLangMenu();
  applyTranslations();

  // Wire up click handlers on the static HTML pills and example chips
  // so the page is fully interactive without any DOM rebuild.
  attachHandlers();

  // Fetch calculators from API and populate the UI dynamically.
  fetch("/api/calculators")
    .then(function(res) { return res.json(); })
    .then(function(data) {
      if (!data || !data.length) return;
      calcs = [];
      for (var i = 0; i < data.length; i++) {
        var d = data[i];
        calcs.push({
          id: d.id || d.name,
          name: d.name,
          desc: d.short_desc || d.description,
          group: d.group || "expression",
          examples: d.examples || [],
          i18n: d.i18n || {}
        });
      }
      _mergeCalcI18n(calcs);
      currentCalc = calcs[0];
      buildPills();
    })
    ["catch"](function() {
      // API unavailable — static HTML stays as-is
    });

})();
