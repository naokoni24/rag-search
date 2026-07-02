(() => {
  const PAGE_SIZE = 20;

  function normalize(value) {
    return (value || "").toString().toLocaleLowerCase("ja-JP");
  }

  function parseDate(value) {
    const match = (value || "").match(/^(\d{4})\/(\d{2})\/(\d{2})\s+(\d{2}):(\d{2})$/);
    if (!match) return 0;
    const [, year, month, day, hour, minute] = match.map(Number);
    return new Date(year, month - 1, day, hour, minute).getTime();
  }

  function getCardData(card) {
    return {
      filename: card.dataset.filename || card.querySelector(".doc-name")?.textContent.trim() || "",
      date: card.dataset.date || card.querySelector(".doc-date")?.textContent.trim() || "",
      originalIndex: Number(card.dataset.originalIndex || 0),
    };
  }

  function compareCards(a, b, state) {
    const cardA = getCardData(a);
    const cardB = getCardData(b);
    let result = 0;

    if (state.sortKey === "date") {
      result = parseDate(cardA.date) - parseDate(cardB.date);
    } else {
      result = cardA.filename.localeCompare(cardB.filename, "ja-JP", {
        numeric: true,
        sensitivity: "base",
      });
    }

    if (result === 0) {
      result = cardA.originalIndex - cardB.originalIndex;
    }
    return state.sortDirection === "asc" ? result : -result;
  }

  function updateSortButtons(panel, state) {
    panel.querySelectorAll("[data-sort-key]").forEach((button) => {
      const active = button.dataset.sortKey === state.sortKey;
      button.classList.toggle("is-active", active);
      button.classList.toggle("is-desc", active && state.sortDirection === "desc");
      button.setAttribute("aria-pressed", String(active));
      button.setAttribute(
        "aria-label",
        `${button.dataset.sortLabel || button.textContent.trim()}を${active && state.sortDirection === "asc" ? "降順" : "昇順"}に並べ替え`
      );
    });
  }

  function renderPager(panel, state, totalPages) {
    const pager = panel.querySelector("[data-doc-pager]");
    if (!pager) return;
    pager.innerHTML = "";
    pager.hidden = totalPages <= 1;
    if (totalPages <= 1) return;

    const createButton = (label, page, options = {}) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "doc-page-btn";
      button.textContent = label;
      button.disabled = options.disabled || false;
      button.classList.toggle("is-current", options.current || false);
      if (options.current) {
        button.setAttribute("aria-current", "page");
      }
      button.addEventListener("click", () => {
        state.page = page;
        applyState(panel, state);
      });
      return button;
    };

    pager.appendChild(createButton("< 前へ", Math.max(1, state.page - 1), { disabled: state.page === 1 }));
    for (let page = 1; page <= totalPages; page += 1) {
      pager.appendChild(createButton(String(page), page, { current: page === state.page }));
    }
    pager.appendChild(createButton("次へ >", Math.min(totalPages, state.page + 1), { disabled: state.page === totalPages }));
  }

  function updateSelection(panel) {
    const form = panel.matches(".doc-list-form") ? panel : panel.querySelector(".doc-list-form");
    if (!form) return;

    const checkboxes = [...form.querySelectorAll('input[name="filenames"]')];
    const checked = checkboxes.filter((checkbox) => checkbox.checked);
    const deleteButton = form.querySelector("#delete-btn");
    const selectionText = form.querySelector("[data-selection-summary]");
    const selectAll = form.querySelector("[data-select-all]");

    if (deleteButton) {
      deleteButton.hidden = checked.length === 0;
    }
    if (selectionText) {
      selectionText.textContent = checked.length > 0 ? `${checked.length}件選択中` : "";
    }
    if (selectAll) {
      selectAll.checked = checkboxes.length > 0 && checked.length === checkboxes.length;
      selectAll.indeterminate = checked.length > 0 && checked.length < checkboxes.length;
    }
  }

  function applyState(panel, state) {
    const list = panel.querySelector(".doc-list");
    if (!list) return;

    const cards = [...list.querySelectorAll(".doc-card")];
    const filter = normalize(state.filter);
    const sortedCards = cards.sort((a, b) => compareCards(a, b, state));
    sortedCards.forEach((card) => list.appendChild(card));

    const filteredCards = sortedCards.filter((card) => normalize(getCardData(card).filename).includes(filter));
    const totalPages = Math.max(1, Math.ceil(filteredCards.length / PAGE_SIZE));
    state.page = Math.min(Math.max(1, state.page), totalPages);
    const pageStart = (state.page - 1) * PAGE_SIZE;
    const pageCards = new Set(filteredCards.slice(pageStart, pageStart + PAGE_SIZE));

    sortedCards.forEach((card) => {
      card.hidden = !pageCards.has(card);
    });

    const empty = panel.querySelector("[data-doc-empty-filter]");
    if (empty) {
      empty.hidden = filteredCards.length !== 0;
    }

    const count = panel.querySelector("[data-doc-count]");
    if (count) {
      const suffix = totalPages > 1 ? ` / ${state.page}ページ目` : "";
      count.textContent = `全${cards.length}件中 ${filteredCards.length}件を表示${suffix}`;
    }

    updateSortButtons(panel, state);
    renderPager(panel, state, totalPages);
    updateSelection(panel);
  }

  function initDocList(panel) {
    if (!panel || panel.dataset.docListReady === "true") return;
    panel.dataset.docListReady = "true";

    const list = panel.querySelector(".doc-list");
    if (!list) return;

    const state = {
      sortKey: "date",
      sortDirection: "desc",
      filter: "",
      page: 1,
    };

    [...list.querySelectorAll(".doc-card")].forEach((card, index) => {
      card.dataset.originalIndex = String(index);
    });

    panel.querySelectorAll("[data-sort-key]").forEach((button) => {
      button.addEventListener("click", () => {
        const key = button.dataset.sortKey;
        if (state.sortKey === key) {
          state.sortDirection = state.sortDirection === "asc" ? "desc" : "asc";
        } else {
          state.sortKey = key;
          state.sortDirection = key === "date" ? "desc" : "asc";
        }
        state.page = 1;
        applyState(panel, state);
      });
    });

    const filterInput = panel.querySelector("[data-doc-filter]");
    if (filterInput) {
      filterInput.addEventListener("input", () => {
        state.filter = filterInput.value;
        state.page = 1;
        applyState(panel, state);
      });
    }

    const form = panel.matches(".doc-list-form") ? panel : panel.querySelector(".doc-list-form");
    if (form) {
      form.addEventListener("change", () => updateSelection(panel));
      const selectAll = form.querySelector("[data-select-all]");
      if (selectAll) {
        selectAll.addEventListener("change", () => {
          form.querySelectorAll('input[name="filenames"]').forEach((checkbox) => {
            checkbox.checked = selectAll.checked;
          });
          updateSelection(panel);
        });
      }
    }

    applyState(panel, state);
  }

  function initAll(root = document) {
    if (root.matches?.("[data-doc-list-panel]")) {
      initDocList(root);
    }
    root.querySelectorAll("[data-doc-list-panel]").forEach(initDocList);
  }

  window.DocList = { init: initAll };

  document.addEventListener("DOMContentLoaded", () => initAll());
  document.body.addEventListener("htmx:afterSettle", (event) => initAll(event.detail.target));
  document.body.addEventListener("htmx:afterSwap", (event) => initAll(event.detail.target));
})();
